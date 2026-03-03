# Security Policy

## Reporting Security Vulnerabilities

The Tenstorrent team takes security vulnerabilities seriously. We appreciate your
efforts to responsibly disclose your findings.

### How to Report a Vulnerability

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them using one of the following methods:

#### Option 1: GitHub Security Advisories (Preferred)

1. Navigate to the [Security Advisories page](https://github.com/tenstorrent/tt-inference-server/security/advisories)
2. Click "Report a vulnerability"
3. Fill out the form with details about the vulnerability
4. Submit the report

This is our preferred method as it allows for secure, private communication.

#### Option 2: Email

Send an email to: **ospo@tenstorrent.com**

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)
- Your contact information

### What to Include in Your Report

To help us understand and address the issue quickly, please include:

- **Type of vulnerability** (e.g., buffer overflow, SQL injection, cross-site scripting)
- **Full paths of source file(s)** related to the vulnerability
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions to reproduce** the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the vulnerability** and how an attacker might exploit it
- **Any special configuration required** to reproduce the issue

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report
   within 3 business days.

2. **Initial Assessment**: We will perform an initial assessment within 7 business
   days to determine the severity and impact of the vulnerability.

3. **Regular Updates**: We will keep you informed about our progress as we work
   to address the vulnerability.

4. **Resolution**: We will work to resolve the vulnerability as quickly as
   possible. Timeline depends on severity:
   - **Critical:** Within 30 days
   - **High:** Within 60 days
   - **Medium:** Within 90 days
   - **Low:** Best effort basis

5. **Disclosure**: Once the vulnerability is resolved, we will coordinate with
   you on public disclosure timing.

### Responsible Disclosure Guidelines

We ask that you:

- **Give us reasonable time** to investigate and fix the vulnerability before
  public disclosure
- **Do not exploit the vulnerability** beyond what is necessary to demonstrate it
- **Do not access, modify, or delete data** that doesn't belong to you
- **Do not perform attacks** that could harm the availability or integrity of
  our services
- **Do not conduct social engineering attacks** against Tenstorrent employees or
  contractors

### Recognition

We believe in giving credit where credit is due. If you report a valid security
vulnerability:

- We will acknowledge your contribution in our security advisories (unless you
  prefer to remain anonymous)
- We will mention your name in release notes when the fix is published (with
  your permission)
- For significant vulnerabilities, we may provide a Tenstorrent swag pack as a
  token of our appreciation

### Safe Harbor

We support safe harbor for security researchers who:

- Make a good faith effort to avoid privacy violations, destruction of data, and
  interruption or degradation of our services
- Only interact with accounts you own or with explicit permission of the account
  holder
- Do not exploit a security issue you discover for any reason (including
  demonstrating additional risk)
- Report vulnerabilities promptly
- Give us reasonable time to address vulnerabilities prior to public disclosure

If you follow these guidelines, we will:

- Not pursue or support legal action against you
- Work with you to understand and resolve the issue quickly
- Publicly acknowledge your responsible disclosure (if desired)

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| dev     | :white_check_mark: |
| Latest release | :white_check_mark: |
| Older releases | :x: |

We strongly recommend using the latest stable release to benefit from all
security updates.

## Security Best Practices for Users

When deploying tt-inference-server, we recommend:

### Network Security

- **Deploy behind a firewall** or in a private network
- **Use TLS/SSL** for all network communications in production
- **Implement rate limiting** to prevent abuse
- **Use authentication** for all API endpoints

### Environment Security

- **Keep dependencies updated** by regularly rebuilding with latest versions
- **Use minimal containers** in production deployments
- **Scan container images** for vulnerabilities before deployment
- **Limit container privileges** using non-root users where possible

### Secrets Management

- **Never commit secrets** to version control
- **Use environment variables** for sensitive configuration
- **Rotate credentials regularly**
- **Use secret management systems** (e.g., HashiCorp Vault, AWS Secrets Manager)
  for production deployments

### Model Security

- **Verify model checksums** before deployment
- **Use trusted model sources** (official Hugging Face models, verified sources)
- **Be cautious with custom models** from untrusted sources
- **Monitor model inputs/outputs** for unexpected behavior

### Logging and Monitoring

- **Enable audit logging** for security-relevant events
- **Monitor for suspicious activity** (unusual request patterns, errors)
- **Set up alerts** for security-critical events
- **Regularly review logs** for potential security incidents

## Known Security Considerations

### Python Interpreter Embedding (C++ Server)

The C++ server embeds a Python interpreter via pybind11:

- **Thread Safety**: Uses GIL (Global Interpreter Lock) for thread safety
- **Resource Management**: Proper cleanup on shutdown is essential
- **Environment Variables**: Validates `TT_PYTHON_PATH`, `TT_METAL_HOME` before use

### Model Runner Security

- **Input Validation**: All model inputs are validated before processing
- **Resource Limits**: Configurable limits on batch size, token length
- **Error Handling**: Comprehensive error handling prevents crashes

### API Security

- **Input Sanitization**: All API inputs are validated
- **Rate Limiting**: Configurable rate limiting available
- **Error Messages**: Error messages avoid leaking sensitive information

## Security Updates

Security updates will be announced through:

1. **GitHub Security Advisories**: https://github.com/tenstorrent/tt-inference-server/security/advisories
2. **Release Notes**: Mentioned in GitHub releases
3. **Repository Notifications**: Watch the repository for security alerts

## Questions?

If you have questions about this security policy or general security concerns
that don't require private disclosure, please:

- Open a [GitHub Discussion](https://github.com/tenstorrent/tt-inference-server/discussions)
- File a public [GitHub Issue](https://github.com/tenstorrent/tt-inference-server/issues)
  (for non-sensitive questions only)

---

**Last Updated:** March 2026

Thank you for helping keep tt-inference-server and our community safe!
