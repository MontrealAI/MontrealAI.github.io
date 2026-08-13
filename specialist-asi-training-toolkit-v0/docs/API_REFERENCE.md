# API reference

All protected POST requests require:

```text
X-GoalOS-App: goalos-uvsi3-v8-8
Content-Type: application/json
```

## GET `/api/health`

Returns version, service and claim boundary.

## GET `/api/catalog`

Returns missions, executable reference algorithms, verifier classes and institutional assets.

## GET `/api/mission-constitution?mission_id=...`

Returns mission features, router result, verifier bundle, curriculum and sample task cards.

## POST `/api/compile`

Validates and registers a finite custom Mission Gym specification.

## POST `/api/toolkit-cycle`

Runs formation, exact release freezing, protected fresh and transfer evaluation, Proof-Adjusted Alpha selection, quality-diversity archive and Chronicle lineage.

Example:

```json
{
  "mission_id": "mission_alpha_planning",
  "formation_episodes": 120,
  "fresh_cases": 80,
  "transfer_cases": 40,
  "seed": 20260813
}
```
