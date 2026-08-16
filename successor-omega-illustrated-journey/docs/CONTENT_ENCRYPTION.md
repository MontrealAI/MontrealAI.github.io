# Content Encryption

Protected assets use AES-256-GCM with unique 96-bit IVs and 128-bit authentication tags. The public manifest records ciphertext and plaintext SHA-256 hashes, sizes, MIME types and authenticated data, but never the key.

Ordinary PDFs remain one encrypted object. The larger Complete Publication Suite uses six independently encrypted parts. Each part has its own IV, authenticated data, ciphertext hash, plaintext hash and exact byte count. The browser:

1. downloads one part at a time;
2. verifies its ciphertext hash;
3. decrypts it locally after current eligibility is verified;
4. verifies its plaintext hash;
5. assembles the ordered plaintext parts into the original ZIP;
6. checks the exact final byte count before delivery.

This design keeps every public repository file below GitHub's browser-upload limit without exposing readable publication files.
