# Deployment

## GitHub Pages reference deployment

Upload the contents of the GitHub Pages ZIP to a repository root, enable GitHub Pages through GitHub Actions and run `.github/workflows/deploy-pages.yml`.

This provides the same client-side holder gate pattern as the UVSI3 release. All static files remain publicly retrievable.

## Genuine protected delivery

1. Put the application bundle at a private object-storage or origin URL.
2. Deploy `access-gateway/` to Cloudflare Workers.
3. Configure:

```text
ALLOWED_ORIGINS
PRIVATE_APP_ORIGIN
ETHEREUM_RPC_URL
SESSION_SECRET                 secret
PRIVATE_ORIGIN_SECRET          secret
```

4. Serve only the access shell publicly.
5. After receipt verification, establish the HttpOnly session through `/api/session` and use `/app/` as the protected entry point.

## Authority gateway

```bash
cd authority-gateway
npm test
npm start
```

Do not attach a production external-action adapter until legal authority, rights, monitoring, incident response, revocation and rollback have been independently reviewed and tested.
