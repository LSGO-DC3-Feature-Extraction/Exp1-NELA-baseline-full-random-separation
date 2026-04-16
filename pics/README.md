# Pics Report Pipeline

This directory contains an isolated reporting pipeline for experiment results under `../record`.

## Structure

- `build_report_data.py`: scan `../record` and build normalized CSV tables
- `plot_report.py`: render per-problem report figures from normalized CSV tables
- `common.py`: shared parsing, labeling, and plotting helpers
- `data/`: generated CSV tables
- `output/`: generated figures, grouped by problem id

## Data Sources

The pipeline only reads:

- `../record/<problem_id>/exp_*/output.json`
- `../record/<problem_id>/exp_*/log.csv`
- `../record/<problem_id>/exp_*/history_epoch.txt`
- `../record/<problem_id>/exp_*/history_slice_*.txt`

## Output Tables

- `data/experiment_summary.csv`
- `data/epoch_progress.csv`
- `data/slice_progress.csv`
- `data/vector_progress.csv`
- `output/index.csv`

## Run

```powershell
python .\pics\build_report_data.py
python .\pics\plot_report.py
```

Or generate everything in one pass:

```powershell
python .\pics\plot_report.py --build-data
```

## Main Figures Per Problem

- `01_final_ranking.png`
- `02_grouped_comparison.png`
- `03_epoch_progress.png`
- `04_vector_distance_evolution.png`
- `05_vector_path_ratio.png`

Legacy figures that are no longer generated:

- `04_time_comparison.png`
- `05_top3_slice_curve.png`
- `06_gain_vs_baseline.png`

## Notes

- Lower `optimized_y` is treated as better.
- `vector_progress.csv` uses sparse `fes_anchor = 100, 200, ..., 2000`.
- Full-vector reconstruction uses all slice histories plus `test_problem/func/slices.json`.
- Each anchor uses step-hold semantics: for each slice, take the last known state with `fes <= fes_anchor`.
- `path_length_ratio` compares cumulative traveled distance to the straight-line distance from the start state to the current state.
- Figures are emitted as PNG with `matplotlib`.
- All generated artifacts stay inside `./pics`.
