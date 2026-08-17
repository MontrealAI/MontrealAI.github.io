# GitHub Browser Upload Guide

The direct-wallet edition is designed for GitHub's browser uploader. Every repository file is below 25 MiB.

## What changed

The encrypted Complete Publication Suite is stored as six independently authenticated AES-256-GCM parts:

```text
complete_suite.part-001-of-006.enc
...
complete_suite.part-006-of-006.enc
```

The protected reader automatically downloads, verifies, decrypts and reassembles the six parts into the original:

```text
Successor_Omega_Illustrated_Journey_v1_0_0_COMPLETE_SUITE.zip
```

Do not merge, rename or edit the parts manually.

## Browser upload

1. Extract the GitHub Pages ZIP locally.
2. Create or open the repository `MONTREALAI/successor-omega-illustrated-journey`.
3. Choose **Add file → Upload files**.
4. Drag the extracted repository contents into the upload surface, preserving the folders.
5. Commit the upload.
6. Under **Settings → Pages**, select **GitHub Actions**.

The package contains fewer than 100 files, and its largest file is below 20 MiB.

## Safer alternative

GitHub Desktop or the Git command line also works. The browser-safe split remains useful because it avoids a single large Git object and makes failed downloads retryable one part at a time.
