# Statistical analysis

These scripts assess differences between model results in **taxonomy discovery** and **ontology-to-ontology generation**. Task-specific paired randomization tests produce p-values; the general Holm utility adjusts existing p-values from either task for multiple comparisons.

## Scripts

| Script | Purpose and output |
| --- | --- |
| [p_value_for_taxonomy_discovery.py](p_value_for_taxonomy_discovery.py) | Compare two sets of predicted parent-child relations against a fixed gold reference using a two-sided Monte Carlo randomization test on F1. Reassign discordant relations between models and save an **unadjusted p-value** and comparison statistics to TXT. |
| [p_value_for_ontology_to_ontology.py](p_value_for_ontology_to_ontology.py) | Compare annotated generation error rates for six configurations against Base. Exchange complete response bundles per input prompt and enumerate all assignments. Export raw and Holm-adjusted p-values to Excel for seven error categories and their average. |
| [holm_correction_of_p_value.py](holm_correction_of_p_value.py) | Interactively select an Excel workbook, worksheet, and raw p-value column. Apply Holm adjustment, insert the adjusted column, and save a copy with the `_holm` suffix. **Works with both tasks.** |

## Requirements and execution

```bash
python3 -m pip install numpy openpyxl statsmodels

# Taxonomy discovery: JSON lists containing parent/child records.
python3 p_value_for_taxonomy_discovery.py \
  --gold GOLD_JSON --model-a MODEL_A_JSON --model-b MODEL_B_JSON \
  --output RESULTS_TXT

# Ontology-to-ontology: workbook with the required evaluation sheets.
python3 p_value_for_ontology_to_ontology.py INPUT_XLSX -o RESULTS_XLSX

# General Holm adjustment: follow the interactive prompts.
python3 holm_correction_of_p_value.py
```

Ontology-to-ontology inputs must contain `base`, `q1_e1`, `q1_e2`, `q1+q2_e1`, `q1+q2_e2`, `full_e1`, and `full_e2`, with the columns listed in the script. Exact prompts define the paired blocks; six responses per input are expected by default (`--repeats`).

## Using Holm with either task

The utility operates on **existing raw p-values**, independently of the task. For taxonomy discovery, first collect the p-values from the TXT reports into an Excel column. For ontology-to-ontology, the task-specific script already provides Holm-adjusted values, so additional adjustment is normally unnecessary. Do not adjust previously adjusted p-values again.

Each selected column is treated as **one comparison family**. Prepare the table accordingly; the utility does not infer the family from the task or worksheet. For example, comparisons against Base and comparisons against the best model can be kept in separate columns and adjusted separately.

Two implementation details matter:

- Automatic self-comparison exclusions depend on `Modelo` and `Ep.` headers: Base is excluded for `p_base`, and Top 10% epoch 1 for `p_mejor` or `p_best`. With different headers or labels, remove self-comparison markers before adjustment. Empty cells are skipped; other included values must be numeric and finite in `[0, 1]`.
- The ontology-to-ontology script adjusts **eight separate families of six comparisons**, one per indicator, retaining unavailable comparisons internally with p=1 while displaying `N/A`. The general utility skips empty cells and uses only the included numeric p-values. These procedures need not give identical results when comparisons are unavailable; `N/A` text must not be supplied as a numeric p-value.

Both tests condition on the evaluated benchmark and checkpoints. Their exchange units differ: individual discordant relations for taxonomy discovery, complete prompt-level response bundles for ontology-to-ontology. Neither estimates variability across training runs.
