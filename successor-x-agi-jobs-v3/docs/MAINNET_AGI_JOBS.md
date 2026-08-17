# Actual AGI Jobs on Ethereum Mainnet

The browser reads contract code, `nextJobId()` and event logs directly through the connected EIP-1193 wallet provider.

## Reconstructed events

- `JobCreated`
- `JobApplied`
- `CheckpointSubmitted`
- `JobCompletionRequested`
- `JobValidated`
- `JobDisapproved`
- `JobDisputed`
- `JobCompleted`
- `JobEmployerRefunded`
- `JobExpired`
- `JobCancelled`

The job state shown in the UI is a deterministic reduction of the ordered event stream. Etherscan links are evidence navigation, not a substitute for chain state.

## Mainnet writes

Writes are disabled until the user explicitly enables transaction mode. The interface estimates gas and requests a final confirmation before calling `eth_sendTransaction`.

The access gate itself never sends a transaction.
