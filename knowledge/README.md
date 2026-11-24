# Experiment Knowledge & Ledger Schema (`knowledge/`)

By default, `.gitignore` keeps personal run logs (`*.jsonl` and `EXPERIMENT_SCORECARD.md`) on the local machine so private model filenames, user workspace paths, and experimental tuning artifacts are not inadvertently published.

To inspect or rebuild a clean local scorecard after running benchmarks:

```powershell
python run_pipeline.py experiments
```

Files created locally during closed-loop runs:
- `experiment_ledger.jsonl` — 7-metric unified record (autonomous completion rate, human interventions, auto-rework count, runtime seconds, quality score, failure reasons, deterministic `config_hash` + `seed`)
- `EXPERIMENT_SCORECARD.md` — human-readable Markdown summary table
- `learned_recipes.jsonl`, `failure_lessons.jsonl`, `reusable_capabilities.jsonl` — 4-outcome feedback records
