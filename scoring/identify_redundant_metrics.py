"""Unsupervised removal of redundant metrics from an ontology Excel workbook.

Selection rules
---------------
1. Exclude metrics with no variation (all NaN or one constant non-null value).
2. While any pair exceeds the strict absolute Pearson correlation threshold,
   select the pair with the highest absolute correlation. Remove the metric
   with the greater mean absolute correlation with the other active metrics,
   excluding the paired metric from both means. Resolve ties by retaining the
   column that appears first in the original workbook.
3. F1 is not used as a target or selection criterion. Metrics are selected
   without any target variable.

Pearson correlations use pairwise complete, non-null observations.
Undefined correlations are not treated as zero. If only one metric in a pair
has a computable redundancy score, that metric is removed.

Requirements:
    Python 3.9 or later; numpy, pandas, and openpyxl.
    python3 -m pip install numpy pandas openpyxl

Execution:
    python3 select_correlated_metrics.py INPUT_XLSX \
        --output-dir OUTPUT_DIRECTORY \
        --threshold 0.85

The default input sheet is DetalleOntologías. Use --sheet to override it and
--metadata to specify identifier columns excluded from correlation analysis.
All remaining columns must be numeric. The default metadata names, workbook
sheet names, output filenames, and default output directory are retained for
compatibility with the existing workflow.

Outputs: a workbook containing selected columns and a selection summary,
a correlation matrix workbook with shared-observation counts, a detailed
selection report, and a comma-separated list of removed columns. The latter
can be passed to remove_metrics_from_excels.py using --metrics-txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


DEFAULT_METADATA = ["Dominio", "Ontología", "Archivo", "RutaRelativa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove redundant metrics without using a target variable."
    )
    parser.add_argument("input", type=Path, help="Input Excel workbook.")
    parser.add_argument("--sheet", default="DetalleOntologías", help="Worksheet containing one row per ontology.")
    parser.add_argument("--output-dir", type=Path, default=Path("salida_seleccion_metricas"))
    parser.add_argument("--threshold", type=float, default=0.85, help="Remove a metric when |r| is strictly greater than this threshold.")
    parser.add_argument(
        "--metadata",
        nargs="*",
        default=DEFAULT_METADATA,
        help="Identifier columns to retain and exclude from correlation analysis.",
    )
    return parser.parse_args()


def shared_counts(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.notna().astype(np.int64)
    counts = valid.T.dot(valid)
    counts.index = df.columns
    counts.columns = df.columns
    return counts


def high_pairs(
    corr: pd.DataFrame,
    counts: pd.DataFrame,
    columns: Iterable[str],
    threshold: float,
) -> list[tuple[str, str, float, int]]:
    cols = list(columns)
    pairs: list[tuple[str, str, float, int]] = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.at[left, right]
            n = int(counts.at[left, right])
            if pd.notna(value) and abs(float(value)) > threshold:
                pairs.append((left, right, float(value), n))
    pairs.sort(key=lambda x: (-abs(x[2]), x[0], x[1]))
    return pairs


def redundancy_score(
    metric: str,
    partner: str,
    active: list[str],
    corr: pd.DataFrame,
) -> tuple[float, int]:
    values = [
        abs(float(corr.at[metric, other]))
        for other in active
        if other not in {metric, partner}
        and pd.notna(corr.at[metric, other])
    ]
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def choose_elimination(
    left: str,
    right: str,
    active: list[str],
    corr: pd.DataFrame,
    original_order: dict[str, int],
) -> tuple[str, str, float, float, int, int, str]:
    left_score, left_n = redundancy_score(left, right, active, corr)
    right_score, right_n = redundancy_score(right, left, active, corr)

    if np.isnan(left_score) and not np.isnan(right_score):
        eliminated, retained, reason = right, left, "only the removed metric had a computable global redundancy score"
    elif np.isnan(right_score) and not np.isnan(left_score):
        eliminated, retained, reason = left, right, "only the removed metric had a computable global redundancy score"
    elif not np.isnan(left_score) and not np.isnan(right_score) and not np.isclose(left_score, right_score, rtol=1e-12, atol=1e-12):
        eliminated, retained = (left, right) if left_score > right_score else (right, left)
        reason = "higher mean absolute correlation with the other active metrics"
    else:
        # Deterministic tie-break: retain the column appearing first in the workbook.
        retained, eliminated = sorted((left, right), key=lambda x: original_order[x])
        reason = "tie; retained the column appearing first in the original workbook"

    return eliminated, retained, left_score, right_score, left_n, right_n, reason


def fmt(value: float) -> str:
    return "NaN" if np.isnan(value) else f"{value:.6f}"


def style_workbook(path: Path, correlation_matrix: bool = False) -> None:
    """Apply readable formatting without changing the data."""
    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "B2" if correlation_matrix else "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        sheet.row_dimensions[1].height = 24
        for col_idx in range(1, sheet.max_column + 1):
            header = str(sheet.cell(1, col_idx).value or "")
            width = min(max(len(header) + 2, 12), 32)
            sheet.column_dimensions[get_column_letter(col_idx)].width = width

    if correlation_matrix:
        corr_sheet = workbook["Correlaciones"]
        corr_sheet.conditional_formatting.add(
            f"B2:{get_column_letter(corr_sheet.max_column)}{corr_sheet.max_row}",
            ColorScaleRule(
                start_type="num", start_value=-1, start_color="F8696B",
                mid_type="num", mid_value=0, mid_color="FFEB84",
                end_type="num", end_value=1, end_color="63BE7B",
            ),
        )
        for row in corr_sheet.iter_rows(min_row=2, min_col=2):
            for cell in row:
                cell.number_format = "0.000"
    workbook.save(path)


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(args.input, sheet_name=args.sheet)
    missing_metadata = [c for c in args.metadata if c not in raw.columns]
    if missing_metadata:
        raise ValueError(f"Missing metadata columns: {missing_metadata}")

    candidate_names = [c for c in raw.columns if c not in args.metadata]
    non_numeric = [c for c in candidate_names if not pd.api.types.is_numeric_dtype(raw[c])]
    if non_numeric:
        raise ValueError(
            "Non-numeric columns not declared as metadata: " + ", ".join(non_numeric)
        )

    all_nan = [c for c in candidate_names if raw[c].notna().sum() == 0]
    constant = [c for c in candidate_names if c not in all_nan and raw[c].nunique(dropna=True) <= 1]
    informative = [c for c in candidate_names if c not in set(all_nan + constant)]
    if len(informative) < 2:
        raise ValueError("Not enough informative metrics to calculate correlations.")

    metrics = raw[informative].apply(pd.to_numeric, errors="coerce")
    corr = metrics.corr(method="pearson")
    counts = shared_counts(metrics)
    initial_pairs = high_pairs(corr, counts, informative, args.threshold)

    active = list(informative)
    original_order = {name: i for i, name in enumerate(candidate_names)}
    decisions: list[dict[str, object]] = []

    while True:
        pairs = high_pairs(corr, counts, active, args.threshold)
        if not pairs:
            break
        left, right, r_value, n_shared = pairs[0]
        eliminated, retained, left_score, right_score, left_n, right_n, reason = choose_elimination(
            left, right, active, corr, original_order
        )
        decisions.append(
            {
                "step": len(decisions) + 1,
                "left": left,
                "right": right,
                "r": r_value,
                "n_shared": n_shared,
                "left_score": left_score,
                "right_score": right_score,
                "left_score_n": left_n,
                "right_score_n": right_n,
                "retained": retained,
                "eliminated": eliminated,
                "reason": reason,
            }
        )
        active.remove(eliminated)

    redundant_removed = [str(d["eliminated"]) for d in decisions]
    removed = all_nan + constant + redundant_removed
    selected_columns = list(args.metadata) + active
    selected = raw[selected_columns]

    selected_path = args.output_dir / "metricas_seleccionadas.xlsx"
    report_path = args.output_dir / "correlaciones_superiores_085.txt"
    removed_path = args.output_dir / "columnas_eliminadas.txt"
    matrix_path = args.output_dir / "matriz_correlaciones.xlsx"

    with pd.ExcelWriter(selected_path, engine="openpyxl") as writer:
        selected.to_excel(writer, sheet_name="DetalleOntologías", index=False)
        summary = pd.DataFrame(
            {
                "Item": [
                    "Candidate metrics",
                    "Non-varying metrics removed",
                    "Metrics removed due to redundancy",
                    "Selected metrics",
                    "Strict threshold |r| >",
                ],
                "Value": [
                    len(candidate_names),
                    len(all_nan) + len(constant),
                    len(redundant_removed),
                    len(active),
                    args.threshold,
                ],
            }
        )
        summary.to_excel(writer, sheet_name="ResumenSelección", index=False)

    with pd.ExcelWriter(matrix_path, engine="openpyxl") as writer:
        corr.to_excel(writer, sheet_name="Correlaciones")
        counts.to_excel(writer, sheet_name="N_compartidas")

    style_workbook(selected_path)
    style_workbook(matrix_path, correlation_matrix=True)

    final_active = set(active)
    lines = [
        "CORRELATED METRIC SELECTION",
        "=" * 80,
        f"File: {args.input.name}",
        f"Worksheet: {args.sheet}",
        f"Criterion: |r| > {args.threshold}, using all shared valid observations",
        "F1 was not used to select metrics.",
        "",
        "NON-VARYING METRICS EXCLUDED BEFORE ANALYSIS",
        "-" * 80,
        f"All NaN: {', '.join(all_nan) if all_nan else 'None'}",
        f"Constant: {', '.join(constant) if constant else 'None'}",
        "",
        f"ALL INITIAL PAIRS WITH |r| > {args.threshold}",
        "-" * 80,
    ]
    if not initial_pairs:
        lines.append("No pairs above the threshold were found.")
    else:
        for i, (left, right, r_value, n_shared) in enumerate(initial_pairs, 1):
            left_status = "SELECTED" if left in final_active else "REMOVED"
            right_status = "SELECTED" if right in final_active else "REMOVED"
            if left in final_active and right not in final_active:
                selected_from_pair = left
            elif right in final_active and left not in final_active:
                selected_from_pair = right
            elif left not in final_active and right not in final_active:
                selected_from_pair = "Neither remains at the end; both were removed in separate decisions"
            else:
                selected_from_pair = "Both remain (check configuration)"
            lines.extend(
                [
                    f"{i}. {left}  <->  {right}",
                    f"   r = {r_value:.6f}; |r| = {abs(r_value):.6f}; n = {n_shared}",
                    f"   Final status: {left} [{left_status}] | {right} [{right_status}]",
                    f"   Final representative of the pair: {selected_from_pair}",
                ]
            )

    lines.extend(["", "ITERATIVE DECISIONS", "-" * 80])
    if not decisions:
        lines.append("No metrics needed to be removed due to redundancy.")
    else:
        for d in decisions:
            lines.extend(
                [
                    f"Step {d['step']}: {d['left']}  <->  {d['right']}",
                    f"   r = {d['r']:.6f}; |r| = {abs(float(d['r'])):.6f}; n = {d['n_shared']}",
                    f"   Global redundancy {d['left']}: {fmt(float(d['left_score']))} (n={d['left_score_n']} correlations)",
                    f"   Global redundancy {d['right']}: {fmt(float(d['right_score']))} (n={d['right_score_n']} correlations)",
                    f"   RETAINED AT THIS STEP: {d['retained']}",
                    f"   REMOVED: {d['eliminated']}",
                    f"   Reason: {d['reason']}",
                ]
            )

    lines.extend(
        [
            "",
            "FINAL RESULT",
            "-" * 80,
            f"Selected metrics ({len(active)}): {', '.join(active)}",
            f"Removed metrics ({len(removed)}): {', '.join(removed)}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    removed_path.write_text(",".join(removed) + "\n", encoding="utf-8")

    print(f"Candidate metrics: {len(candidate_names)}")
    print(f"Selected metrics: {len(active)}")
    print(f"Removed due to lack of variation: {len(all_nan) + len(constant)}")
    print(f"Removed due to redundancy: {len(redundant_removed)}")
    print(f"Initial pairs above the threshold: {len(initial_pairs)}")
    for path in (selected_path, report_path, removed_path, matrix_path):
        print(path.resolve())


if __name__ == "__main__":
    main()
