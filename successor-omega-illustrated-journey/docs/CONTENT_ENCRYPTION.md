# Publication Packaging and Integrity

Protected publication assets remain packaged with AES-256-GCM, unique IVs, authenticated data and exact plaintext/ciphertext SHA-256 records. The large Complete Suite is divided into six independently authenticated parts below GitHub's browser-upload limit.

After direct wallet eligibility verification, the browser downloads, authenticates, opens and hash-verifies the selected asset locally. The multipart Complete Suite is automatically reassembled into the original ZIP.

## Important boundary

This is a public GitHub Pages deployment. The browser-delivery material is necessarily public and therefore is not a server-side secrecy boundary. Encryption here provides authenticated packaging, integrity verification and a smoother gated interface—not cryptographic exclusivity against a technically determined visitor.
