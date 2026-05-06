// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT
//
// Adapted from sail-object-store::hugging_face (LakeSail, MIT) — kept the
// path-parsing and metadata logic; trimmed `lazy_static` and `async_stream`
// in favour of `OnceLock` + collect-then-stream so manifold doesn't grow
// extra macro-helper deps.

//! HuggingFace Hub object store.
//!
//! Implements [`object_store::ObjectStore`] over the
//! [Hugging Face Hub](https://huggingface.co/docs/hub) dataset/model file API.
//! Read paths (`get`, `list`) are wired; write paths return
//! `NotImplemented` because the HF Hub uploads use a different API surface
//! (LFS multipart) that this adapter intentionally does not expose.
//!
//! ## Path convention
//!
//! Object paths are formatted as
//! `{username}/{dataset}[@{revision}]/{path/to/file}`. For example:
//!
//! - `JanosAudran/financial-reports-sec/data/full_index.parquet`
//! - `bigcode/the-stack-v2-dedup@v2/data/python/train-00000-of-00007.parquet`
//!
//! ## Reuse
//!
//! Used by Sail (foundation) and Fathom — SPARC (financial filings ingest)
//! to read HF parquet datasets through the standard object-store interface,
//! so DataFusion / parquet readers don't need an HF-specific code path.

use std::fmt::Display;
use std::path::PathBuf;
use std::sync::OnceLock;

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use futures::stream;
use futures::stream::BoxStream;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiError, ApiRepo};
use hf_hub::{Repo, RepoType};
use object_store::local::LocalFileSystem;
use futures::StreamExt;
use object_store::path::{self, Path};
use object_store::{
    CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use regex_lite::Regex;

const STORE_NAME: &str = "Hugging Face object store";

fn hf_path_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        // {username}/{dataset}[@{revision}]/{path}
        Regex::new(r"^(?P<username>[^/]+)/(?P<dataset>[^/@]+)(@(?P<revision>[^/]+))?/(?P<path>.*)$")
            .expect("static HF path regex compiles")
    })
}

#[derive(Debug, thiserror::Error)]
enum HuggingFaceError {
    #[error("Hugging Face API error: {0}")]
    Api(String),
    #[error("date/time parse error: {0}")]
    DateTime(#[from] chrono::format::ParseError),
    #[error("HTTP request error: {0}")]
    Request(String),
    #[error("HTTP error {status} for {url}")]
    Http { status: u16, url: String },
    #[error("missing response header: {0}")]
    MissingHeader(&'static str),
    #[error("invalid path: {0}")]
    InvalidPath(PathBuf),
}

impl HuggingFaceError {
    fn from_api(err: ApiError) -> Self {
        // hf-hub's ApiError wraps a reqwest::Error variant transitively, but
        // the underlying reqwest crate may differ from manifold's. We flatten
        // to a string here so the rest of the module isn't entangled with
        // hf-hub's reqwest re-export.
        if let ApiError::RequestError(req_err) = &err {
            if let Some(status) = req_err.status() {
                let url = req_err
                    .url()
                    .map(|u| u.path().to_string())
                    .unwrap_or_default();
                return Self::Http {
                    status: status.as_u16(),
                    url,
                };
            }
        }
        Self::Api(err.to_string())
    }

    /// Catches errors from hf-hub's transitive reqwest version, which is a
    /// different type than manifold's own reqwest. Stringifies and inspects
    /// the message for "404"/"NotFound" so we can still surface NotFound
    /// cleanly to ObjectStore consumers.
    fn from_send_error<E: std::fmt::Display>(err: E) -> Self {
        let s = err.to_string();
        if s.contains("404") || s.to_lowercase().contains("not found") {
            return Self::Http {
                status: 404,
                url: String::new(),
            };
        }
        Self::Request(s)
    }
}

impl From<HuggingFaceError> for object_store::Error {
    fn from(value: HuggingFaceError) -> Self {
        match value {
            HuggingFaceError::Http { status, url } if status == 404 => object_store::Error::NotFound {
                path: url,
                source: format!("HTTP 404").into(),
            },
            HuggingFaceError::Http { status, url } => object_store::Error::Generic {
                store: STORE_NAME,
                source: format!("HTTP {status} for {url}").into(),
            },
            HuggingFaceError::Api(s) | HuggingFaceError::Request(s) => {
                object_store::Error::Generic {
                    store: STORE_NAME,
                    source: s.into(),
                }
            }
            HuggingFaceError::DateTime(e) => object_store::Error::Generic {
                store: STORE_NAME,
                source: Box::new(e),
            },
            HuggingFaceError::MissingHeader(name) => object_store::Error::Generic {
                store: STORE_NAME,
                source: format!("missing required header: {name}").into(),
            },
            HuggingFaceError::InvalidPath(path) => object_store::Error::InvalidPath {
                source: path::Error::InvalidPath { path },
            },
        }
    }
}

#[derive(Debug)]
struct HuggingFacePath {
    username: String,
    dataset: String,
    revision: Option<String>,
    path: String,
}

impl HuggingFacePath {
    fn parse(path: &Path) -> object_store::Result<Self> {
        let captures = hf_path_pattern()
            .captures(path.as_ref())
            .ok_or_else(|| HuggingFaceError::InvalidPath(path.as_ref().into()))?;
        let username = captures
            .name("username")
            .ok_or_else(|| HuggingFaceError::InvalidPath(path.as_ref().into()))?
            .as_str()
            .to_string();
        let dataset = captures
            .name("dataset")
            .ok_or_else(|| HuggingFaceError::InvalidPath(path.as_ref().into()))?
            .as_str()
            .to_string();
        let revision = captures.name("revision").map(|m| m.as_str().to_string());
        let path = captures
            .name("path")
            .ok_or_else(|| HuggingFaceError::InvalidPath(path.as_ref().into()))?
            .as_str()
            .to_string();
        Ok(Self {
            username,
            dataset,
            revision,
            path,
        })
    }

    fn parse_optional(path: Option<&Path>) -> object_store::Result<Self> {
        match path {
            Some(p) => Self::parse(p),
            None => Err(HuggingFaceError::InvalidPath(PathBuf::default()).into()),
        }
    }

    fn repo(&self) -> Repo {
        let repo_id = format!("{}/{}", self.username, self.dataset);
        match &self.revision {
            Some(r) => Repo::with_revision(repo_id, RepoType::Dataset, r.clone()),
            None => Repo::new(repo_id, RepoType::Dataset),
        }
    }

    fn base_path(&self) -> String {
        match &self.revision {
            Some(r) => format!("{}/{}@{}", self.username, self.dataset, r),
            None => format!("{}/{}", self.username, self.dataset),
        }
    }

    fn matches(&self, filename: &str) -> bool {
        filename.starts_with(&self.path)
    }
}

/// Read-only object store backed by the Hugging Face Hub.
pub struct HuggingFaceObjectStore {
    api: Api,
    local: LocalFileSystem,
}

impl std::fmt::Debug for HuggingFaceObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // hf_hub::api::tokio::Api doesn't implement Debug.
        f.debug_struct("HuggingFaceObjectStore").finish()
    }
}

impl HuggingFaceObjectStore {
    /// Builds a HuggingFace store using credentials from the environment
    /// (`HF_TOKEN`, `HF_HOME`, etc. — all standard `hf-hub` variables).
    pub fn try_new() -> object_store::Result<Self> {
        // ApiBuilder::new() already honours HF_HOME / HF_TOKEN environment
        // variables; there is no separate `from_env`.
        Ok(Self {
            api: ApiBuilder::new()
                .with_progress(false)
                .build()
                .map_err(HuggingFaceError::from_api)?,
            local: LocalFileSystem::new(),
        })
    }

    async fn get_meta(
        api: &Api,
        repo: &ApiRepo,
        base_path: &str,
        filename: &str,
    ) -> object_store::Result<ObjectMeta> {
        let location = Path::parse(format!("{base_path}/{filename}"))?;
        let response = api
            .client()
            .head(repo.url(filename))
            .send()
            .await
            .map_err(HuggingFaceError::from_send_error)?
            .error_for_status()
            .map_err(HuggingFaceError::from_send_error)?;
        // Use string-keyed lookups to avoid the http-version mismatch between
        // hf-hub's transitive reqwest and manifold's reqwest re-export.
        let headers = response.headers();

        let size = headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or(HuggingFaceError::MissingHeader("content-length"))?;

        let last_modified = headers
            .get("last-modified")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| DateTime::parse_from_rfc2822(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|| Utc.timestamp_nanos(0));

        let e_tag = headers
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        Ok(ObjectMeta {
            location,
            last_modified,
            size,
            e_tag,
            version: None,
        })
    }
}

impl Display for HuggingFaceObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HuggingFaceObjectStore")
    }
}

#[async_trait]
impl ObjectStore for HuggingFaceObjectStore {
    async fn put_opts(
        &self,
        _location: &Path,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(object_store::Error::NotImplemented {
            operation: "put_opts".to_string(),
            implementer: STORE_NAME.to_string(),
        })
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(object_store::Error::NotImplemented {
            operation: "put_multipart_opts".to_string(),
            implementer: STORE_NAME.to_string(),
        })
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let GetOptions {
            if_match,
            if_none_match,
            if_modified_since,
            if_unmodified_since,
            range,
            version,
            head,
            extensions: _,
        } = options;
        if if_match.is_some()
            || if_none_match.is_some()
            || if_modified_since.is_some()
            || if_unmodified_since.is_some()
            || version.is_some()
        {
            return Err(object_store::Error::NotImplemented {
                operation: "get_opts (conditional/version)".to_string(),
                implementer: STORE_NAME.to_string(),
            });
        }
        let path = HuggingFacePath::parse(location)?;
        let repo = self.api.repo(path.repo());
        if head {
            let meta =
                Self::get_meta(&self.api, &repo, &path.base_path(), path.path.as_str()).await?;
            return Ok(GetResult {
                payload: GetResultPayload::Stream(Box::pin(stream::empty())),
                meta,
                range: 0..0,
                attributes: Default::default(),
            });
        }
        // The hf-hub `get` resolves to a cached local file path, downloading
        // on cache miss. The whole file is materialised — for parquet
        // readers that only touch the footer this over-fetches; that
        // limitation is inherited from hf-hub itself.
        let local_path = repo
            .get(path.path.as_str())
            .await
            .map_err(HuggingFaceError::from_api)?;
        self.local
            .get_opts(
                &Path::from_filesystem_path(local_path)?,
                GetOptions {
                    range,
                    ..GetOptions::default()
                },
            )
            .await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let path = match HuggingFacePath::parse_optional(prefix) {
            Ok(p) => p,
            Err(e) => return Box::pin(stream::once(async { Err(e) })),
        };
        let api = self.api.clone();
        // Collect into a Vec inside one future, then unwrap into a stream —
        // keeps us off `async_stream` since HF's list is a single API call
        // returning all siblings at once.
        let collect = async move {
            let repo = api.repo(path.repo());
            let info = repo.info().await.map_err(HuggingFaceError::from_api)?;
            let mut metas: Vec<ObjectMeta> = Vec::new();
            for sibling in info.siblings {
                let filename = sibling.rfilename.as_str();
                if path.matches(filename) {
                    metas.push(Self::get_meta(&api, &repo, &path.base_path(), filename).await?);
                }
            }
            Ok::<Vec<ObjectMeta>, object_store::Error>(metas)
        };
        Box::pin(stream::once(collect).flat_map(
            |result: object_store::Result<Vec<ObjectMeta>>| -> BoxStream<'static, object_store::Result<ObjectMeta>> {
                match result {
                    Ok(metas) => stream::iter(metas.into_iter().map(Ok)).boxed(),
                    Err(e) => stream::once(async move { Err(e) }).boxed(),
                }
            },
        ))
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&Path>,
    ) -> object_store::Result<ListResult> {
        Err(object_store::Error::NotImplemented {
            operation: "list_with_delimiter".to_string(),
            implementer: STORE_NAME.to_string(),
        })
    }

    async fn copy_opts(
        &self,
        _from: &Path,
        _to: &Path,
        _options: CopyOptions,
    ) -> object_store::Result<()> {
        Err(object_store::Error::NotImplemented {
            operation: "copy_opts".to_string(),
            implementer: STORE_NAME.to_string(),
        })
    }

    fn delete_stream(
        &self,
        _locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        Box::pin(stream::once(async {
            Err(object_store::Error::NotImplemented {
                operation: "delete_stream".to_string(),
                implementer: STORE_NAME.to_string(),
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_username_dataset_path() {
        let p = Path::parse("user/dataset/data/train-00000.parquet").unwrap();
        let parsed = HuggingFacePath::parse(&p).unwrap();
        assert_eq!(parsed.username, "user");
        assert_eq!(parsed.dataset, "dataset");
        assert_eq!(parsed.revision, None);
        assert_eq!(parsed.path, "data/train-00000.parquet");
    }

    #[test]
    fn parses_revision_in_path() {
        let p = Path::parse("user/dataset@v2/data/train-00000.parquet").unwrap();
        let parsed = HuggingFacePath::parse(&p).unwrap();
        assert_eq!(parsed.revision.as_deref(), Some("v2"));
        assert_eq!(parsed.path, "data/train-00000.parquet");
    }

    #[test]
    fn rejects_paths_without_dataset_segment() {
        let p = Path::parse("just-a-username").unwrap();
        assert!(HuggingFacePath::parse(&p).is_err());
    }

    #[test]
    fn base_path_includes_revision_when_present() {
        let p = Path::parse("u/d@v1/x").unwrap();
        let parsed = HuggingFacePath::parse(&p).unwrap();
        assert_eq!(parsed.base_path(), "u/d@v1");
        let p2 = Path::parse("u/d/x").unwrap();
        let parsed2 = HuggingFacePath::parse(&p2).unwrap();
        assert_eq!(parsed2.base_path(), "u/d");
    }
}
