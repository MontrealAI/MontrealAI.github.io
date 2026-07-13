# Exact utility semantics used by the simulation

## BondMath.computeValidatorBond

```solidity
bond = Math.mulDiv(payout, bps, 10_000);
if (bond < minBond) bond = minBond;
if (bond > maxBond) bond = maxBond;
if (bond > payout) bond = payout;
```

## BondMath.computeAgentBond

```solidity
bond = Math.mulDiv(payout, bps, 10_000);
if (bond < minBond) bond = minBond;
if (durationLimit != 0) {
    uint256 durationPremium = Math.mulDiv(bond, duration, durationLimit);
    bond += durationPremium;
}
if (maxBond != 0 && bond > maxBond) bond = maxBond;
if (bond > payout) bond = payout;
```

## ReputationMath.computeReputationPoints

```solidity
if (!repEligible) return 0;
uint256 completionTime = completionRequestedAt > assignedAt ? completionRequestedAt - assignedAt : 0;
uint256 payoutUnits = payout / 1e15;
uint256 timeBonus = duration > completionTime ? (duration - completionTime) / 10000 : 0;
uint256 base = Math.log2(1 + payoutUnits);
if (timeBonus > base) timeBonus = base;
reputationPoints = base + timeBonus;
```

## TransferUtils.safeTransferFromExact

The recipient balance is read before and after `transferFrom`. Any balance delta other than the exact requested amount reverts with `TransferFailed`.
