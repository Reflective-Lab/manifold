# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Report through GitHub Security Advisories for the repository, or email
[kenneth@reflective.se](mailto:kenneth@reflective.se).

You should receive a response within 48 hours.

## Security Notes

- Credentials and secrets must stay outside committed config.
- Adapter builders should validate URI schemes and feature availability.
- Object-store and database permissions are operator responsibilities.
- Vector and experience-store adapters must not silently widen tenant or
  correlation scopes.

## Operator Responsibility

Operators are responsible for IAM, bucket/database permissions, TLS, retention,
audit logging, and provider-specific security configuration.
