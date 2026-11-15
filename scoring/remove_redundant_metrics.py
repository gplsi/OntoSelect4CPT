"""Remove selected metric columns from copies of Excel workbooks.

Read comma-separated column names from the first line of a UTF-8 text file.
Process all .xlsx files directly inside the input directory, excluding Excel
lock files. Delete matching columns from the selected worksheet and save the
modified workbooks to the output directory.

Requirements:
    Python 3.9 or later and openpyxl.
    python3 -m pip install openpyxl

Usage:
    python3 remove_metrics_from_excels.py \
        --input-dir ./randoms \
        --metrics-txt ./metrics_to_remove.txt \
        --output-dir ./cleaned_randoms

The first line of the metrics file must contain comma-separated names, e.g.:
    AvgDepth,MaxDepth,NumIndividuals

Use --sheet to select a worksheet and --allow-missing to tolerate missing
metrics. The default worksheet name is DetalleOntologías. If --output-dir is
omitted, the output directory is <input-dir>/sin_metricas_redundantes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openpyxl import load_workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create copies of all .xlsx files in a directory, removing metric "
            "columns named in the first line of a text file."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the original Excel workbooks.")
    parser.add_argument("--metrics-txt", type=Path, required=True, help="Text file containing comma-separated column names.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input-dir>/sin_metricas_redundantes.",
    )
    parser.add_argument(
        "--sheet",
        default="DetalleOntologías",
        help="Worksheet from which to remove columns (default: DetalleOntologías).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Continue if a metric is missing from a workbook; otherwise, fail that workbook.",
    )
    return parser.parse_args()


def read_metrics(txt_path: Path) -> list[str]:
    if not txt_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {txt_path}")

    with txt_path.open("r", encoding="utf-8-sig") as file:
        first_line = file.readline().strip()

    if not first_line:
        raise ValueError("The first line of the metrics file is empty.")

    metrics = [item.strip() for item in first_line.split(",")]
    if any(not metric for metric in metrics):
        raise ValueError("The metrics file contains an empty column name.")

    duplicates = sorted({metric for metric in metrics if metrics.count(metric) > 1})
    if duplicates:
        raise ValueError(f"Duplicate metrics in the text file: {', '.join(duplicates)}")

    return metrics


def find_columns_to_delete(worksheet, metrics: list[str]) -> tuple[list[int], list[str]]:
    positions: dict[str, list[int]] = {}
    for cell in worksheet[1]:
        if cell.value is not None:
            positions.setdefault(str(cell.value).strip(), []).append(cell.column)

    duplicated_headers = {name: cols for name, cols in positions.items() if len(cols) > 1 and name in metrics}
    if duplicated_headers:
        details = ", ".join(f"{name} (columns {cols})" for name, cols in duplicated_headers.items())
        raise ValueError(f"The worksheet contains duplicate headers: {details}")

    missing = [metric for metric in metrics if metric not in positions]
    columns = sorted((positions[metric][0] for metric in metrics if metric in positions), reverse=True)
    return columns, missing


def process_workbook(
    input_path: Path,
    output_path: Path,
    metrics: list[str],
    sheet_name: str,
    allow_missing: bool,
) -> tuple[int, list[str]]:
    workbook = load_workbook(input_path)
    if sheet_name not in workbook.sheetnames:
        workbook.close()
        raise ValueError(f"Worksheet '{sheet_name}' not found.")

    worksheet = workbook[sheet_name]
    columns, missing = find_columns_to_delete(worksheet, metrics)
    if missing and not allow_missing:
        workbook.close()
        raise ValueError(f"Metrics not found: {', '.join(missing)}")

    # Delete columns from right to left to preserve the remaining column indices.
    for column_index in columns:
        worksheet.delete_cols(column_index, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        workbook.save(temporary_path)
        workbook.close()
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return len(columns), missing


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    metrics_txt = args.metrics_txt.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "sin_metricas_redundantes"
    )

    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    try:
        metrics = read_metrics(metrics_txt)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    excel_files = sorted(
        path for path in input_dir.glob("*.xlsx")
        if not path.name.startswith("~$") and path.parent != output_dir
    )
    if not excel_files:
        print(f"ERROR: no .xlsx files found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Requested metrics: {len(metrics)}")
    print(f"Excel workbooks found: {len(excel_files)}")
    print(f"Target worksheet: {args.sheet}")

    failures: list[tuple[str, str]] = []
    for input_path in excel_files:
        output_path = output_dir / input_path.name
        try:
            deleted, missing = process_workbook(
                input_path=input_path,
                output_path=output_path,
                metrics=metrics,
                sheet_name=args.sheet,
                allow_missing=args.allow_missing,
            )
            missing_text = f"; missing: {', '.join(missing)}" if missing else ""
            print(f"OK  {input_path.name}: {deleted} columns removed{missing_text}")
        except Exception as error:
            failures.append((input_path.name, str(error)))
            print(f"ERROR  {input_path.name}: {error}", file=sys.stderr)

    if failures:
        print(f"\nFinished with errors in {len(failures)} file(s).", file=sys.stderr)
        return 1

    print(f"\nProcessing complete. Output workbooks: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
