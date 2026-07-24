# Clean-Room Verification Protocol

A release archive passes only when it is extracted into a newly created empty directory and the following controls succeed:

1. ZIP CRC and compressed-data integrity;
2. exactly the expected publication root structure;
3. every path in `MANIFEST.json` exists, with matching byte size and SHA-256 digest;
4. every entry in `SHA256SUMS.txt` matches independently recomputed content;
5. `index.html`, `404.html`, the named v13 standalone and the distributed standalone are byte-identical;
6. embedded intelligence contains 70 jurisdiction nodes, 463 governed routes, 476 sources and 20 transaction categories;
7. the v13 paper contains 22 pages;
8. the complete owner-mode and production QA result contains no failed control;
9. no publication file exceeds 25 MB;
10. no unresolved local link, external script dependency, external stylesheet dependency or `/api/access/` dependency is present.

The archive SHA-256 is recorded separately after final packaging. Clean-room verification validates artifact integrity, not future legal, fiscal, regulatory or transaction correctness.
