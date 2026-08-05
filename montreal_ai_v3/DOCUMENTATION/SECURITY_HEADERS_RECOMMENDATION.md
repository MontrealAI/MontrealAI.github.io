# Security headers recommendation

GitHub Pages does not provide repository-level control over every HTTP response header. The HTML release includes a restrictive meta Content Security Policy and `no-referrer`. If a controlled reverse proxy or another production host is used, configure and test at least:

- `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload` only after confirming every subdomain is HTTPS-ready;
- `Content-Security-Policy` at the response layer;
- `X-Content-Type-Options: nosniff`;
- `Referrer-Policy: no-referrer`;
- `Permissions-Policy` disabling unneeded sensors and APIs;
- `Cross-Origin-Opener-Policy` and `Cross-Origin-Resource-Policy` where compatible;
- frame protection through CSP `frame-ancestors 'none'`.

Do not move private execution, payments, passwords, wallet custody, confidential customer material or signing keys into GitHub Pages.
