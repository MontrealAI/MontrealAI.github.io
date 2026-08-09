# AGI Club Live-Wallet Acceptance Matrix

**Product:** GoalOS Singularity Navigator Ω  
**Edition:** AGI Club GitHub Pages Institution — v3.1.0-SN5-LR1  
**Static-release revision:** 2026-07-26-DSR1  
**Canonical deployment:** `https://montrealai.github.io/goalos-singularity-navigator-omega-agi-club-owner-access/`  
**Ethereum network:** Mainnet (`0x1`)  
**ENS Registry:** `0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e`  
**Official Name Wrapper:** `0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401`

## Purpose

This matrix separates two forms of evidence:

1. **Deterministic release-fixture evidence** — completed automatically before packaging, without using a private wallet.
2. **Live current-owner evidence** — executed after publication by a current AGI Club owner through the real wallet, real Ethereum Mainnet provider and deployed canonical origin.

The deterministic suite establishes the behavior of the static release under controlled Ethereum responses. A genuine live-wallet signature cannot honestly be manufactured in an offline build environment. The included interactive acceptance instrument records, hashes and optionally wallet-signs the post-publication evidence.

## A. Deterministic release-fixture matrix — completed

| ID | Case | Expected result | Packaged result |
|---|---|---|---|
| FX-01 | Exact one-label unwrapped name; Registry owner equals connected wallet | Unlock | **PASS** |
| FX-02 | Exact one-label wrapped name; official Mainnet wrapper; wrapped owner equals connected wallet; current expiry | Unlock | **PASS** |
| FX-03 | Registry owner is another wallet | Deny | **PASS** |
| FX-04 | Wrapped name is expired | Deny | **PASS** |
| FX-05 | Sepolia wrapper is presented on Ethereum Mainnet | Deny | **PASS** |
| FX-06 | Nested label is entered | Deny before RPC | **PASS** |
| FX-07 | Root `club.agi.eth` is entered | Reject by one-label constitution | **PASS by design** |
| FX-08 | Account changes after unlock | Relock | **PASS** |
| FX-09 | Network changes after unlock | Relock | **PASS** |
| FX-10 | Wallet disconnects after unlock | Relock | **PASS** |
| FX-11 | Owner manually locks | Relock | **PASS** |
| FX-12 | Mobile 390 px | No horizontal overflow | **PASS** |
| FX-13 | Canonical 204-page SN5 MasterClass is exposed in the MasterClass surface | Open current PDF | **PASS** |
| FX-14 | Archived 170-page MasterClass remains preserved but is not an active application link | Preserve / do not present as current | **PASS** |

## B. Live production acceptance matrix — operator execution record

The following matrix is complete as a release instrument. Its **Live result** cells are intentionally not pre-certified: they must be completed using real current-owner wallets after deployment. Use `tools/LIVE_WALLET_ACCEPTANCE_MATRIX.html` to record evidence, export JSON/Markdown and optionally sign the record.

| ID | Live case | Required evidence | Expected | Live result |
|---|---|---|---|---|
| LW-01 | Current direct unwrapped owner | Screenshot; exact name; wallet; block; EIP-712/personal-sign mode | Unlock | TO BE EXECUTED |
| LW-02 | Current direct wrapped owner through official Mainnet wrapper | Screenshot; exact name; wallet; wrapper; expiry; block; signature | Unlock | TO BE EXECUTED |
| LW-03 | Non-owner wallet against a valid AGI Club name | Screenshot; wallet; exact denial reason | Deny | TO BE EXECUTED |
| LW-04 | Nested name | Screenshot showing local rejection before ownership read | Deny | TO BE EXECUTED |
| LW-05 | Root `club.agi.eth` | Screenshot showing one-label rejection | Deny | TO BE EXECUTED |
| LW-06 | Wrong namespace (`*.agent.agi.eth`) | Screenshot showing suffix rejection | Deny | TO BE EXECUTED |
| LW-07 | Transferred name / former owner | Before/after owner evidence and denial | Deny former owner | TO BE EXECUTED WHEN ASSET AVAILABLE |
| LW-08 | Expired wrapped name | Expiry evidence and denial | Deny | TO BE EXECUTED WHEN ASSET AVAILABLE |
| LW-09 | Account change during active session | Video/screenshot sequence and relock timestamp | Relock | TO BE EXECUTED |
| LW-10 | Network change during active session | Video/screenshot sequence and relock timestamp | Relock | TO BE EXECUTED |
| LW-11 | Wallet disconnect during active session | Video/screenshot sequence and relock timestamp | Relock | TO BE EXECUTED |
| LW-12 | Manual lock | Screenshot before/after | Relock | TO BE EXECUTED |
| LW-13 | Ten-minute idle timeout | Timestamped screenshots or screen recording | Relock | TO BE EXECUTED |
| LW-14 | Thirty-minute absolute expiry | Timestamped screenshots or screen recording | Relock | TO BE EXECUTED |
| LW-15 | Protected export after ownership change | Export attempt and revalidation denial | Block export and relock | TO BE EXECUTED |
| LW-16 | EIP-712 unavailable | Wallet evidence of `personal_sign` fallback and access receipt | Fallback succeeds if all ownership gates pass | TO BE EXECUTED WITH COMPATIBLE WALLET |
| LW-17 | Mobile current-owner entry | Mobile screenshot; no overflow; successful unlock | Unlock / responsive | TO BE EXECUTED |
| LW-18 | Standalone HTML current-owner entry | File URL or local origin evidence; wallet behavior and boundary note | Unlock subject to wallet/provider support | TO BE EXECUTED |
| LW-19 | Offline/PWA reopening after prior asset cache | Offline screenshot; source-boundary note | Public assets load; fresh ownership verification still requires provider | TO BE EXECUTED |
| LW-20 | Current MasterClass link | Opened PDF metadata showing 204 pages and v3.1.0-SN5-LR1 | Open canonical MasterClass | TO BE EXECUTED |

## Activation gate

The **definitive static release** is complete when all fixture, packaging and content controls pass. **Live production activation** should additionally obtain signed evidence for LW-01, LW-02 (where a wrapped owner is available), LW-09, LW-10, LW-11, LW-15 and LW-20. Cases requiring a transferred or expired name are evidentiary when such an asset is available and must not be fabricated.

## Required operator record

Record at minimum:

- release SHA-256;
- canonical origin;
- browser and wallet versions;
- chain ID;
- exact qualifying name;
- connected wallet;
- verification block;
- ownership mode and wrapper when applicable;
- signature mode;
- timestamps;
- screenshots or video hashes;
- pass/fail determination;
- tester name or pseudonymous role;
- wallet-signed matrix digest where feasible.

## Exact boundary

A static GitHub Pages release cannot pre-sign with a user’s private wallet. This document therefore distinguishes completed deterministic assurance from real-wallet operational evidence. No live-wallet PASS is claimed without the corresponding wallet, timestamp and evidence.
