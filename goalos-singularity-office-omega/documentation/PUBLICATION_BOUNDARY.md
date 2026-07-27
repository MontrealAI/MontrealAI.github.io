# Public Publication Boundary

The public GitHub Pages package may contain only public-safe code, schemas, signed public records, public documentation, research papers and public visual assets.

It must not contain:

- publisher private signing keys;
- customer secrets or privileged material;
- API credentials;
- live institutional tokens;
- internal incident records;
- confidential contracts;
- production connector credentials.

The optional protected runtime is distributed as source and must be deployed separately with TLS, exact origin allowlists, short-lived bearer credentials, secret management, rate limiting, no prompt logging by default and deployment-specific review.
