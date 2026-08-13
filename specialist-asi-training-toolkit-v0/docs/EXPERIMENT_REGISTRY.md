# Experiment Registry

The local Toolkit creates a SQLite registry at `runs/goalos_experiment_registry.sqlite3`.

It records:

- exact run identity;
- mission;
- champion;
- verdict;
- manifest hash;
- assurance events;
- artifact lineage.

Use `GET /api/registry` or `POST /api/registry/export` to export the current institutional history. The registry is not a substitute for enterprise audit infrastructure, but it makes the reference implementation traceable and resumable.
