# Security & Credential Hygiene Policy

## BYOK Credential Handling
- **Hello 3DNeko** never ships embedded API keys, tokens, or paid account credentials.
- Configure external tool executables and API key file paths locally using environment variables (`.env`, excluded from git via `.gitignore`).
- Logs and failure reports pass through `pipeline.guards.redact_secrets()` to mask API tokens and signed URL query strings.
- Blender subprocess scripts operate locally; never load untrusted `.blend` files with auto-run scripts enabled.

## Reporting Issues
If you discover an unintended credential exposure, path leak, or security flaw, please avoid posting live tokens or sensitive paths in public issues and report details directly to the project maintainers.
