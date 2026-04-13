# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not open a public issue**.

Instead, send an email to: **[INSERT SECURITY EMAIL]**

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if known)

### What to Expect

- We will acknowledge receipt within 48 hours
- We will provide a detailed response within 7 days
- We will schedule a fix and release a security patch
- Credit will be given in the release notes

## Security Best Practices

### For Users

1. **API Keys**: Never share your API keys publicly
2. **Network**: Bind to `127.0.0.1` only (not `0.0.0.0`) when running locally
3. **Updates**: Keep Hippo updated to the latest version
4. **Models**: Only download models from trusted sources (HuggingFace)

### For Developers

1. **Input Validation**: All user inputs must be validated
2. **Dependencies**: Regularly update dependencies for security patches
3. **Secrets**: Never commit secrets or API keys to the repository
4. **Code Review**: All code must be reviewed before merging

### Known Security Considerations

1. **Model Loading**: Hippo loads GGUF models from `~/.hippo/models/`
   - Users should only download models from trusted sources
   - Malicious models could execute arbitrary code

2. **Network Exposure**: By default, Hippo binds to `127.0.0.1:11434`
   - Do not change to `0.0.0.0` unless behind a firewall
   - Use a reverse proxy (nginx) for production deployments

3. **Authentication**: Hippo does not include built-in authentication
   - Use API keys for `/api/pull` and `/api/delete` endpoints
   - Deploy behind a reverse proxy with auth for production

## Security Audits

- **First Audit**: Planned for v0.2.0 release
- **Frequency**: Before every major version

## Dependencies

Hippo depends on:
- `fastapi` - Web framework
- `llama-cpp-python` - LLM inference
- `pyyaml` - Configuration
- `requests` - HTTP client
- `typer` - CLI framework
- `rich` - TUI rendering

Security alerts for these dependencies are monitored via GitHub Dependabot.

---

**Thank you for helping keep Hippo secure!** 🦛
