# Release notes — v10.0.0-GHP1

Release date: 2026-07-24

## Purpose

Purpose-built GitHub Pages edition for the current direct owner of an exact one-label `club.agi.eth` subdomain.

## Changes from AGC1

- Removed all `/api/access/*`, Worker, KV, R2, cookie and private-package dependencies from the static edition.
- Replaced private archive delivery with a browser-local Owner Access Center.
- Added exact direct-owner verification for unwrapped Registry ownership and official Mainnet Name Wrapper ownership.
- Restricted wrapped-name recognition to the official Ethereum Mainnet ENS Name Wrapper at `0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401`; the Sepolia wrapper address `0x0635513f179D50A207757E05759CbD106d7dFcE8` and all unrecognized wrappers are rejected.
- Added a domain-bound signed access statement and `goalos-agi-club-owner-access/2.0` receipt.
- Added five-minute ownership revalidation, focus checks, session restoration checks and immediate relocking on wallet or network change.
- Added canonical GitHub Pages path binding.
- Preserved the complete GoalOS institution and all AGC1 interface identifiers.
- Added root-level `.nojekyll` deployment packaging, final browser QA, link validation, manifests, checksums and clean-room verification.

## Static boundary

The release controls ordinary use and creates contractual/evidentiary provenance. Because GitHub Pages is static and public, it does not provide source confidentiality or unbypassable server-side access control.
