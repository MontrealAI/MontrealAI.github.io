# Recommended Security Headers

GitHub Pages and IPFS gateway headers are partly controlled by the hosting surface. The HTML files contain a restrictive meta Content Security Policy where practical. On a controlled host, add response headers such as:

- `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
- `Content-Security-Policy` tailored to the exact RPC, wallet, frame, image, and support endpoints
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy` disabling unused sensors and APIs
- `Cross-Origin-Opener-Policy` / `Cross-Origin-Resource-Policy` after wallet and iframe compatibility testing
- `frame-ancestors` through CSP on controlled hosting
- `Cache-Control: no-store` for legal manifests, access pages, or rapidly changing configuration where appropriate

Do not copy a header set blindly. Test MetaMask Connect, embedded frames, IPFS gateways, mobile handoff, and the local application before activation.
