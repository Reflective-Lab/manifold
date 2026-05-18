---
source: mixed
type: reference
---
# Release Commands

Mirror of foundation's
[Building/Release Commands](https://github.com/Reflective-Lab/converge/blob/main/kb/Building/Release%20Commands.md).
The same four gates that gate the Converge release gate this extension.

| Command | Purpose | Output |
|---|---|---|
| `just security-audit` | Supply-chain audit (advisories, licenses, bans) | `target/security/` |
| `just coverage` | Workspace line coverage; ≥ 80% floor | `target/coverage/` |
| `just performance-profile` | Criterion baseline / regression compare | `target/criterion/`, `kb/Baselines/` |
| `just soak` | Bounded long-running stability validation | `target/soak/` |

## Coverage floor

Per
[Extension Release Checklist §4](https://github.com/Reflective-Lab/converge/blob/main/kb/Standards/Extension%20Release%20Checklist.md#4-coverage-floor):

- Workspace ≥ 80%, per crate ≥ 80%.
- No regression below the previously-recorded percentage for this release line.
- Approved exclusions (transport servers, CLI shells) are listed in
  the local coverage policy note with a one-paragraph justification.

## Performance baseline

Always pass `PERF_BASELINE` when running `just performance-profile`. Use
the release tag as the baseline name (`v0.1.0`, `v1.0.0`, …). The first
run for a baseline `--save-baseline`s; subsequent runs `--baseline`
compare.

## Soak

`SOAK_DURATION_MIN` controls the wall-clock budget. 5 min ≈ CI grade.
For a release, run `SOAK_DURATION_MIN=15` locally and capture the log.

## The release ritual

```bash
just release-check
```

Runs all five gates in order. All must be green before tagging.
