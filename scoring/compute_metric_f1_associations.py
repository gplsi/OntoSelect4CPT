"""Correlate token-weighted ontology metrics with F1 at epochs 1 and 2.

Each Excel workbook represents one dataset. For every numeric metric, compute
its mean weighted by the token count of each ontology:

    weighted_mean = sum(tokens_i * metric_i) / sum(tokens_i)

Missing or non-finite metric values are excluded from both the numerator and
the denominator for that metric. Token weights must be finite and positive.
Calculate Pearson and Spearman correlations with F1_e1 and F1_e2, together
with their two-sided p-values.

A metric is suitable when all four coefficients are finite, nonzero, and have
the same sign. Its association score is the mean of their absolute values.
P-values are reported but are not used to filter metrics.

Requirements:
    Python 3.10 or later; numpy, pandas, openpyxl, and scipy.
    python3 -m pip install numpy pandas openpyxl scipy

Execution:
    python3 correlate_metrics_with_f1.py INPUT_DIRECTORY \
        --pattern "*random*.xlsx" \
        --output OUTPUT_XLSX

Filenames must contain random followed immediately by a dataset number,
for example random1_metrics.xlsx. Each dataset must have exactly one workbook.
The default input worksheet is DetalleOntologías and the default token weight
column is NumTokensLlama32_3B. Override these using --sheet and --weight-column.
Numeric metric columns must match across workbooks, including their order.

By default, use the ten experiment scores in DEFAULT_F1. Supply --scores with
a CSV or Excel file containing dataset, F1_e1, and F1_e2 to use other results.
The workbook dataset numbers and score table must match exactly.

Export all correlations, suitable metrics, dataset-level weighted means,
and an analysis summary. Existing sheet names and column identifiers are
retained for compatibility with the analysis workflow.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.stats import pearsonr, spearmanr


# Provided scores for the ten deterministic experiments.
DEFAULT_F1 = {
    1: {"F1_e1": 0.496, "F1_e2": 0.445},
    2: {"F1_e1": 0.456, "F1_e2": 0.451},
    3: {"F1_e1": 0.494, "F1_e2": 0.456},
    4: {"F1_e1": 0.477, "F1_e2": 0.533},
    5: {"F1_e1": 0.471, "F1_e2": 0.517},
    6: {"F1_e1": 0.457, "F1_e2": 0.454},
    7: {"F1_e1": 0.483, "F1_e2": 0.389},
    8: {"F1_e1": 0.425, "F1_e2": 0.405},
    9: {"F1_e1": 0.452, "F1_e2": 0.456},
    10: {"F1_e1": 0.487, "F1_e2": 0.449},
}

DEFAULT_METADATA = {"Dominio", "Ontología", "Archivo", "RutaRelativa"}
DEFAULT_WEIGHT_COLUMN = "NumTokensLlama32_3B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate token-weighted dataset means with F1 at epochs 1 and 2."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing the dataset Excel workbooks.")
    parser.add_argument("--pattern", default="*.xlsx", help="File pattern within input_dir.")
    parser.add_argument("--sheet", default="DetalleOntologías")
    parser.add_argument(
        "--weight-column",
        default=DEFAULT_WEIGHT_COLUMN,
        help=f"Column used for weighting (default: {DEFAULT_WEIGHT_COLUMN}).",
    )
    parser.add_argument("--output", type=Path, default=Path("correlaciones_metricas_f1.xlsx"))
    parser.add_argument(
        "--scores",
        type=Path,
        help="Optional CSV or Excel file with dataset, F1_e1, and F1_e2 columns. Defaults to the built-in F1 scores.",
    )
    return parser.parse_args()


def dataset_number(filename: str) -> int:
    match = re.search(r"random(\d+)", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract the dataset number from: {filename}")
    return int(match.group(1))


def load_scores(path: Path | None) -> pd.DataFrame:
    if path is None:
        frame = pd.DataFrame.from_dict(DEFAULT_F1, orient="index").reset_index(names="dataset")
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path)

    required = {"dataset", "F1_e1", "F1_e2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    frame = frame[["dataset", "F1_e1", "F1_e2"]].copy()
    frame["dataset"] = pd.to_numeric(frame["dataset"], errors="raise").astype(int)
    frame["F1_e1"] = pd.to_numeric(frame["F1_e1"], errors="raise")
    frame["F1_e2"] = pd.to_numeric(frame["F1_e2"], errors="raise")
    if frame["dataset"].duplicated().any():
        raise ValueError("The score table contains duplicate datasets.")
    return frame.sort_values("dataset").reset_index(drop=True)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Calculate a weighted mean using valid value-weight pairs only."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid = numeric_values.notna() & np.isfinite(numeric_values)
    if not valid.any():
        return np.nan
    valid_weights = weights.loc[valid]
    denominator = float(valid_weights.sum())
    if denominator <= 0:
        return np.nan
    return float(np.average(numeric_values.loc[valid], weights=valid_weights))


def load_dataset_means(
    input_dir: Path,
    pattern: str,
    sheet: str,
    weight_column: str,
) -> pd.DataFrame:
    paths = sorted(input_dir.glob(pattern), key=lambda p: dataset_number(p.name))
    if not paths:
        raise FileNotFoundError(f"No files matching the pattern {pattern!r} in {input_dir}")

    records: list[dict[str, object]] = []
    seen: set[int] = set()
    expected_metrics: list[str] | None = None

    for path in paths:
        number = dataset_number(path.name)
        if number in seen:
            raise ValueError(f"Multiple files found for random{number}: check --pattern.")
        seen.add(number)

        frame = pd.read_excel(path, sheet_name=sheet)
        if weight_column not in frame.columns:
            raise ValueError(f"Token count column {weight_column!r} is missing from {path.name}.")

        weights = pd.to_numeric(frame[weight_column], errors="coerce")
        invalid_weights = weights.isna() | ~np.isfinite(weights) | (weights <= 0)
        if invalid_weights.any():
            rows = (np.flatnonzero(invalid_weights.to_numpy()) + 2).tolist()
            raise ValueError(
                f"Missing, non-numeric, infinite, or non-positive weights in {path.name}; "
                f"Excel rows: {rows[:10]}"
            )

        numeric = [
            column
            for column in frame.columns
            if column not in DEFAULT_METADATA
            and column != weight_column
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if expected_metrics is None:
            expected_metrics = numeric
        elif numeric != expected_metrics:
            missing = sorted(set(expected_metrics) - set(numeric))
            extra = sorted(set(numeric) - set(expected_metrics))
            raise ValueError(f"Incompatible metric columns in {path.name}. Missing={missing}; extra={extra}")

        record: dict[str, object] = {"dataset": number, "archivo": path.name}
        record.update({column: weighted_mean(frame[column], weights) for column in numeric})
        records.append(record)

    return pd.DataFrame(records).sort_values("dataset").reset_index(drop=True)


def safe_correlations(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float, int]:
    valid = x.notna() & y.notna()
    xv = x[valid].astype(float)
    yv = y[valid].astype(float)
    n = len(xv)
    if n < 3 or xv.nunique() < 2 or yv.nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, n
    pearson = pearsonr(xv, yv)
    spearman = spearmanr(xv, yv)
    return float(pearson.statistic), float(pearson.pvalue), float(spearman.statistic), float(spearman.pvalue), n


def same_nonzero_sign(values: list[float]) -> tuple[bool, str]:
    if not all(np.isfinite(value) for value in values):
        return False, "undetermined"
    signs = np.sign(values)
    if np.any(signs == 0) or len(set(signs.tolist())) != 1:
        return False, "inconsistent"
    return True, "positive" if signs[0] > 0 else "negative"


def compute_results(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records = []
    for metric in metrics:
        p_e1, pp_e1, s_e1, sp_e1, n_e1 = safe_correlations(data[metric], data["F1_e1"])
        p_e2, pp_e2, s_e2, sp_e2, n_e2 = safe_correlations(data[metric], data["F1_e2"])
        coefficients = [p_e1, s_e1, p_e2, s_e2]
        suitable, direction = same_nonzero_sign(coefficients)
        score = float(np.mean(np.abs(coefficients))) if suitable else np.nan
        records.append(
            {
                "Métrica": metric,
                "Apta": suitable,
                "Dirección": direction,
                "Association_score": score,
                "Pearson_e1": p_e1,
                "p_Pearson_e1": pp_e1,
                "Spearman_e1": s_e1,
                "p_Spearman_e1": sp_e1,
                "Pearson_e2": p_e2,
                "p_Pearson_e2": pp_e2,
                "Spearman_e2": s_e2,
                "p_Spearman_e2": sp_e2,
                "n_e1": n_e1,
                "n_e2": n_e2,
            }
        )
    result = pd.DataFrame(records)
    result["_orden_apta"] = result["Apta"].astype(int)
    result = result.sort_values(
        ["_orden_apta", "Association_score", "Métrica"],
        ascending=[False, False, True],
        na_position="last",
    ).drop(columns="_orden_apta")
    return result.reset_index(drop=True)


def style_output(path: Path) -> None:
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 24
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for column in range(1, sheet.max_column + 1):
            header = str(sheet.cell(1, column).value or "")
            sheet.column_dimensions[get_column_letter(column)].width = min(max(len(header) + 2, 12), 34)

    for name in ("Todas", "Aptas"):
        sheet = workbook[name]
        for row in sheet.iter_rows(min_row=2, min_col=4, max_col=12):
            for cell in row:
                cell.number_format = "0.0000"
        sheet.conditional_formatting.add(
            f"D2:D{sheet.max_row}",
            ColorScaleRule(
                start_type="num", start_value=0, start_color="F8696B",
                mid_type="num", mid_value=0.5, mid_color="FFEB84",
                end_type="num", end_value=1, end_color="63BE7B",
            ),
        )
    workbook.save(path)


def main() -> None:
    args = parse_args()
    scores = load_scores(args.scores)
    means = load_dataset_means(
        args.input_dir,
        args.pattern,
        args.sheet,
        args.weight_column,
    )

    data = means.merge(scores, on="dataset", how="outer", validate="one_to_one", indicator=True)
    unmatched = data.loc[data["_merge"] != "both", ["dataset", "_merge"]]
    if not unmatched.empty:
        raise ValueError("Dataset numbers and score entries do not match:\n" + unmatched.to_string(index=False))
    data = data.drop(columns="_merge").sort_values("dataset").reset_index(drop=True)

    metrics = [c for c in means.columns if c not in {"dataset", "archivo"}]
    results = compute_results(data, metrics)
    suitable = results[results["Apta"]].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Todas", index=False)
        suitable.to_excel(writer, sheet_name="Aptas", index=False)
        data[["dataset", "archivo", "F1_e1", "F1_e2", *metrics]].to_excel(
            writer, sheet_name="Medias_ponderadas", index=False
        )
        pd.DataFrame(
            {
                "Item": [
                    "Datasets analyzed",
                    "Metrics analyzed",
                    "Suitable metrics",
                    "Dataset aggregation",
                    "Weight column",
                    "Suitability criterion",
                    "Use of p-values",
                ],
                "Value": [
                    len(data),
                    len(metrics),
                    len(suitable),
                    "Token-weighted mean; no unweighted arithmetic mean is calculated",
                    args.weight_column,
                    "Same sign for Pearson and Spearman at epochs 1 and 2",
                    "Reported for information; not used to filter metrics",
                ],
            }
        ).to_excel(writer, sheet_name="Resumen", index=False)

    style_output(args.output)
    print(f"Datasets: {len(data)}")
    print(f"Metrics analyzed: {len(metrics)}")
    print(f"Suitable metrics: {len(suitable)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
