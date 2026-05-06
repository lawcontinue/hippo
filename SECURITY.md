# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Hippo, please report it responsibly:

- **Email**: Open a GitHub Issue with the tag `security` (we'll convert to a private advisory if needed)
- **Response time**: We aim to acknowledge within 48 hours and provide a fix within 7 days

## Known Limitations

Hippo provides **L1 safety measures** (behavioral constraints like loop detection). It does **not** provide:
- Content safety filtering (use a separate content filter on top)
- Authentication beyond basic API keys
- Protection against adversarial prompts (jailbreaks)

## API Security Checklist

When deploying Hippo API in production:

- [ ] Set `api_keys` in config to restrict access
- [ ] Bind to `127.0.0.1` (not `0.0.0.0`) unless behind a reverse proxy
- [ ] Configure CORS headers if accessing from browsers (or disable with `--no-cors`)
- [ ] Enable `loop_detect` to prevent degenerate outputs
- [ ] Set `max_tokens` limits to prevent resource exhaustion
- [ ] Run behind a reverse proxy (nginx/caddy) for TLS termination

## Versions

| Version | Supported |
| ------- | --------- |
| 0.2.x   | ✅ |
| < 0.2   | ❌ |
