#!/usr/bin/env python3
"""Apply Holm multiple-comparison adjustment to an Excel p-value column.

How it works
------------
Prompt for an Excel file, a worksheet (if more than one exists), and a p-value
column. Treat all valid included p-values in that column as one comparison
family. Insert the adjusted values immediately after the original column,
copy its cell formatting, and update native Excel table ranges and headers.
Save the result beside the input file with a _holm filename suffix.

Empty cells are skipped. The Base row is excluded when adjusting p_base.
When adjusting p_mejor or p_best, the Top 10% epoch 1 self-comparison is
excluded. These exclusions use the existing Modelo and Ep. column names,
which are retained for compatibility with the input workbooks.

Requirements
------------
Python 3 and the following packages:
    python3 -m pip install openpyxl statsmodels

Execution
---------
Run from a terminal:
    python3 adjust_pvalues_holm.py

Then enter:
1. The input Excel file path.
2. The worksheet name, if prompted.
3. The exact name of the p-value column to adjust.

Headers must be in the first row. Included p-values must be finite numeric
values in [0, 1]. The new column is named '<original column> adjusted by Holm'.
Significant adjusted p-values are counted using p < 0.05. The script adjusts
existing p-values; it does not calculate the underlying statistical tests.
No input file paths are hardcoded. An existing _holm output file is overwritten.
"""

from pathlib import Path
import math

from openpyxl import load_workbook
from statsmodels.stats.multitest import multipletests


def main():
    path = Path(
        input("Excel file path: ").strip().strip('"')
    ).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    workbook = load_workbook(path)

    if len(workbook.sheetnames) == 1:
        worksheet = workbook[workbook.sheetnames[0]]
    else:
        print("Available worksheets:", ", ".join(workbook.sheetnames))
        sheet_name = input("Worksheet name: ").strip()
        worksheet = workbook[sheet_name]

    # The generated workbook has its headers in the first row.
    headers = {
        str(cell.value).strip(): cell.column
        for cell in worksheet[1]
        if cell.value is not None
    }

    print("Available columns:", ", ".join(headers))
    name = input("Name of the p-value column to adjust: ").strip()

    if name not in headers:
        raise ValueError(f"Column not found: {name}")

    p_column = headers[name]
    model_column = headers.get("Modelo")
    epoch_column = headers.get("Ep.")

    rows = []
    p_values = []

    for row in range(2, worksheet.max_row + 1):
        model = (
            str(worksheet.cell(row, model_column).value or "").strip()
            if model_column else ""
        )
        epoch = (
            worksheet.cell(row, epoch_column).value
            if epoch_column else None
        )
        value = worksheet.cell(row, p_column).value

        # Exclude the Base self-comparison when adjusting p_base.
        if (
            model.casefold() == "base"
            and name.casefold() == "p_base"
        ):
            continue

        # Exclude the best model's self-comparison.
        if (
            name.casefold() in {"p_mejor", "p_best"}
            and model.replace(" ", "").casefold() == "top10%"
            and str(epoch) in {"1", "1.0"}
        ):
            continue

        if value is None or value == "":
            continue

        if isinstance(value, bool):
            raise ValueError(f"Non-numeric value in row {row}: {value}")

        try:
            p = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Non-numeric p-value in row {row}: {value}"
            )

        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(
                f"p-value outside [0, 1] in row {row}: {value}"
            )

        rows.append(row)
        p_values.append(p)

    if not p_values:
        raise ValueError("No valid p-values to adjust.")

    # One family: all valid included comparisons in the selected column.
    _, adjusted, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="holm",
    )

    new_name = f"{name} adjusted by Holm"

    if new_name in headers:
        raise ValueError(f"Column already exists: {new_name}")

    # Insert the new column immediately after the original one.
    new_column = p_column + 1

    # Update native Excel table ranges.
    from openpyxl.utils.cell import range_boundaries, get_column_letter

    table_ranges = []
    for table in worksheet.tables.values():
        min_col, min_row, max_col, max_row = range_boundaries(table.ref)
        table_ranges.append(
            (table, min_col, min_row, max_col, max_row)
        )

    worksheet.insert_cols(new_column)
    worksheet.cell(1, new_column, new_name)

    from copy import copy

    for row in range(1, worksheet.max_row + 1):
        original = worksheet.cell(row, p_column)
        new = worksheet.cell(row, new_column)
        new._style = copy(original._style)

    for row, adjusted_p in zip(rows, adjusted):
        cell = worksheet.cell(row, new_column, float(adjusted_p))
        cell.number_format = (
            "0.00E+00" if 0 < adjusted_p < 0.001 else "0.000"
        )

    worksheet.column_dimensions[
        get_column_letter(new_column)
    ].width = max(24, len(new_name) + 2)

    # Rebuild the headers of affected tables.
    from openpyxl.worksheet.table import TableColumn

    for table, min_col, min_row, max_col, max_row in table_ranges:
        if new_column <= min_col:
            min_col += 1
            max_col += 1
        elif new_column <= max_col + 1:
            max_col += 1

        table.ref = (
            f"{get_column_letter(min_col)}{min_row}:"
            f"{get_column_letter(max_col)}{max_row}"
        )
        table.tableColumns = [
            TableColumn(
                id=index,
                name=str(worksheet.cell(min_row, column).value),
            )
            for index, column in enumerate(
                range(min_col, max_col + 1), start=1
            )
        ]

        if table.autoFilter is not None:
            table.autoFilter.ref = table.ref

    output = path.with_name(f"{path.stem}_holm{path.suffix}")
    workbook.save(output)

    significant = sum(p < 0.05 for p in adjusted)
    print(f"\nAdjusted comparisons: {len(p_values)}")
    print(f"Significant after Holm adjustment (p < 0.05): {significant}")
    print(f"File saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
