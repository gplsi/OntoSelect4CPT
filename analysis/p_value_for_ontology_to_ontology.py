"""Calculate p-values for ontology-to-ontology generation error rates.

Compare six adapted model configurations against Base using an exact,
two-sided paired randomization test over complete generation bundles.
For each identical input prompt, swap the full bundle of responses between
the two systems, including all seven error counts and triple denominators.
The input prompt is the exchange unit, not the ontology or generated triple.

By default, each input has six generations. With 12 distinct inputs, enumerate
all 2^12 A/B assignments. Up to 20 input blocks are supported.
The statistic is the error rate difference (configuration minus Base);
a lower error rate is better. No gold reference is used: the test compares
existing error annotations in the workbook.

Default aggregation:
    rate = 100 * sum(errors) / sum(triples)

Eight indicators are tested: seven error categories and Avg, the arithmetic
mean of their seven rates (not total errors per triple). Holm adjustment is
applied separately to each indicator: eight families of six comparisons
against Base. It does not provide joint familywise control over the full table.

Requirements
------------
Python 3 and the following packages:
    python3 -m pip install numpy openpyxl

Input workbook
--------------
Required sheets: base, q1_e1, q1_e2, q1+q2_e1, q1+q2_e2, full_e1, full_e2.
Each sheet must contain model, prompt, source, answer, Number of triplets,
and the seven error columns specified in HEADERS. Their original spellings
are retained to match existing evaluation workbooks. Prompts must match
exactly across sheets, with --repeats responses per prompt.

Usage
-----
Replace the generic filenames with your own input and output workbooks:
    python3 p_value_for_ontology_to_ontology.py INPUT_XLSX -o OUTPUT_XLSX

Optional settings:
    --aggregation pooled         Ratio of total errors to total triples (default).
    --aggregation mean-response  Unweighted mean of per-response error rates.
    --missing unavailable        Leave incomplete indicators and p-values unavailable.
    --missing error              Stop when an error count is missing.
    --missing zero               Explicitly treat missing error counts as zero.
    --repeats 6                  Expected generations per input (default: 6).
    --alpha 0.05                 Significance threshold (default: 0.05).

In unavailable mode, Avg is unavailable if any category is incomplete.
Unavailable tests retain their place in each planned family of six using
p=1 internally for Holm adjustment, but are displayed as N/A, not as invented
p-values. Output values are static; rerun after changing the annotations.
The test assumes exchangeability of the two systems' bundles under H0.
Results are conditional on the evaluated benchmark and checkpoints and do
not estimate variability across training runs. No input paths are hardcoded.
"""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# Change these labels if full does not mean Q1-4 in your experiment.
SHEETS = [
    ("base", "Base", "--"),
    ("q1_e1", "Q1", 1), ("q1_e2", "Q1", 2),
    ("q1+q2_e1", "Q1-2", 1), ("q1+q2_e2", "Q1-2", 2),
    ("full_e1", "Q1-4", 1), ("full_e2", "Q1-4", 2),
]
CATEGORIES = ["Syn", "Rep", "TR", "Red", "Amb", "Sem", "Voc", "Avg"]
HEADERS = [
    "Sintactic", "Triplet Rpetition", "Text repetition", "Redundancy",
    "Ambiguity", "Semantic Contradictions", "Vocabulary Errors",
]
LONG_NAMES = [
    "Syntax", "Triplet repetition", "Text repetition", "Redundancy",
    "Ambiguity", "Semantic contradictions", "Vocabulary errors",
    "Mean of the seven error rates",
]


def numeric_count(value, location, missing_policy, warnings):
    if value is None or (isinstance(value, str) and not value.strip()):
        warnings.append(f"Missing count: {location}")
        if missing_policy == "error":
            raise ValueError(f"Missing count at {location}. See --missing.")
        return 0.0 if missing_policy == "zero" else np.nan
    if isinstance(value, (bool, str)):
        raise ValueError(f"Expected a numeric count at {location}: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"Invalid count at {location}: {value!r}")
    return number


def read_input(path, missing_policy, expected_repeats):
    wb = load_workbook(path, data_only=True, read_only=True)
    data, warnings, sources = {}, [], {}
    try:
        for sheet, _, _ in SHEETS:
            if sheet not in wb.sheetnames:
                raise ValueError(f"Required worksheet not found: {sheet}")
            ws = wb[sheet]
            iterator = ws.iter_rows(values_only=True)
            header = next(iterator)
            positions = {str(x).strip(): i for i, x in enumerate(header) if x is not None}
            required = ["model", "prompt", "source", "answer", "Number of triplets"] + HEADERS
            for name in required:
                if name not in positions:
                    raise ValueError(f"Missing column {name!r} in {sheet}")
            groups = {}
            source_missing = 0
            for row_number, row in enumerate(iterator, start=2):
                model, prompt = row[positions["model"]], row[positions["prompt"]]
                # Exclude averages, totals, and empty rows at the bottom of the worksheet.
                if model is None and prompt is None:
                    continue
                if model is None or not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Incomplete identification in {sheet}, row {row_number}")
                # Use EXACT prompt text as the key, not row numbers or response text.
                source = row[positions["source"]]
                if source is None:
                    source_missing += 1
                else:
                    if prompt in sources and sources[prompt] != source:
                        raise ValueError("The same prompt is associated with different ontologies.")
                    sources[prompt] = source
                counts = [numeric_count(
                    row[positions[h]], f"{sheet}!{get_column_letter(positions[h]+1)}{row_number}",
                    missing_policy, warnings,
                ) for h in HEADERS]
                denom = numeric_count(
                    row[positions["Number of triplets"]],
                    f"{sheet}!{get_column_letter(positions['Number of triplets']+1)}{row_number}",
                    "error", warnings,
                )
                if denom <= 0:
                    raise ValueError(f"Non-positive triple count in {sheet}, row {row_number}")
                groups.setdefault(prompt, []).append(np.array(counts + [denom], dtype=float))
            if not groups:
                raise ValueError(f"No responses in {sheet}")
            if source_missing:
                warnings.append(f"{sheet}: {source_missing} responses without source; paired by exact prompt text.")
            for prompt, answers in groups.items():
                if len(answers) != expected_repeats:
                    identifier = hashlib.sha256(prompt.encode()).hexdigest()[:12]
                    raise ValueError(f"{sheet}, input {identifier}: {len(answers)} responses; expected {expected_repeats}.")
            data[sheet] = groups
        keys = sorted(data["base"])
        for sheet, _, _ in SHEETS:
            if set(data[sheet]) != set(keys):
                raise ValueError(f"The inputs in {sheet} do not match Base exactly.")
        if len(keys) > 20:
            raise ValueError("More than 20 blocks: exact enumeration is too large. This script is designed for 12 inputs.")
        return data, keys, sources, warnings
    finally:
        wb.close()


def aggregate(groups, keys, aggregation):
    blocks = []
    for key in keys:
        records = np.asarray(groups[key])
        if aggregation == "pooled":
            blocks.append(records.sum(axis=0))
        else:
            # Numerator = sum of rates (without the factor 100); denominator = responses.
            rates = records[:, :7] / records[:, 7, None]
            blocks.append(np.r_[rates.sum(axis=0), len(records)])
    return np.asarray(blocks)


def scores(totals):
    rates = 100.0 * totals[..., :7] / totals[..., 7, None]
    return np.concatenate([rates, rates.mean(axis=-1, keepdims=True)], axis=-1)


def exact_test(a, b):
    """Swap complete A/B blocks and return exact two-sided p-values."""
    n = len(a)
    number = 1 << n
    observed = scores(a.sum(axis=0)) - scores(b.sum(axis=0))
    available = np.isfinite(observed)
    extreme = np.zeros(8, dtype=np.int64)
    # Handle NaNs ONLY in computations for categories that will not be
    # tested. Their values and p-values remain NaN, including incomplete Avg.
    safe_a, safe_b = np.nan_to_num(a), np.nan_to_num(b)
    safe_total_a, safe_total_b = safe_a.sum(axis=0), safe_b.sum(axis=0)
    for start in range(0, number, 4096):
        ids = np.arange(start, min(start + 4096, number), dtype=np.uint64)
        swap = ((ids[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
        transfer = swap @ (safe_a - safe_b)
        new_a = safe_total_a - transfer
        new_b = safe_total_b + transfer
        delta = scores(new_a) - scores(new_b)
        # Allow for ties caused by floating-point rounding.
        extreme += ((np.abs(delta) >= np.abs(observed) - 1e-10) & available).sum(axis=0)
    p = np.full(8, np.nan)
    p[available] = extreme[available] / number
    # Full enumeration: p=c/2^n. The Monte Carlo formula (c+1)/(R+1) is not used.
    return observed, p


def holm_planned(raw):
    """Adjust ONE family, retaining planned but unavailable comparisons."""
    flat = np.asarray(raw, dtype=float).ravel()
    valid = np.isfinite(flat)
    for_adjustment = np.where(valid, flat, 1.0)
    order = np.argsort(for_adjustment, kind="stable")
    sorted_adjusted = np.maximum.accumulate(
        (len(flat) - np.arange(len(flat))) * for_adjustment[order]
    ).clip(0, 1)
    result = np.empty(len(flat))
    result[order] = sorted_adjusted
    result[~valid] = np.nan
    return result.reshape(np.shape(raw))


def holm_by_indicator(raw):
    """Eight separate families: six comparisons against Base per column.

    Separate adjustment does not imply statistical independence of the data.
    Familywise control applies within each indicator, not jointly to the table.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (6, 8):
        raise ValueError(f"Expected 6 comparisons by 8 indicators; received {raw.shape}")
    return np.column_stack([holm_planned(raw[:, j]) for j in range(8)])


def excel_value(x):
    return float(x) if np.isfinite(x) else "N/A"


def style_table(ws):
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="24476A")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = 34
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if cell.row % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="F0F4F8")
            cell.alignment = Alignment(vertical="top")
    for col in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = 18
    ws.sheet_view.zoomScale = 80


def write_excel(output, values, raw, adjusted, deltas, data, keys, sources, warnings, args):
    wb = Workbook()
    wb.remove(wb.active)
    for name in ["Results", "Holm_table", "Raw_p_table"]:
        ps = adjusted if name == "Holm_table" else raw
        ws = wb.create_sheet(name)
        headers = ["Model/Config.", "Epoch"]
        for category in CATEGORIES:
            if name == "Results":
                headers.extend([f"{category} (%)", f"raw p {category} vs Base", f"p Holm {category} vs Base"])
            else:
                headers.extend([f"{category} (%)", f"p {category} vs Base"])
        ws.append(headers)
        for i, (sheet, label, epoch) in enumerate(SHEETS):
            row = [label, epoch]
            for j in range(8):
                if name == "Results":
                    row.extend([excel_value(values[i, j]),
                                "--" if i == 0 else excel_value(raw[i-1, j]),
                                "--" if i == 0 else excel_value(adjusted[i-1, j])])
                else:
                    row.extend([excel_value(values[i, j]), "--" if i == 0 else excel_value(ps[i-1, j])])
            ws.append(row)
        style_table(ws)
        ws.column_dimensions["A"].width = 20
        ws.column_dimensions["B"].width = 7
        for row in ws.iter_rows(min_row=2):
            for offset, cell in enumerate(row[2:]):
                group_size = 3 if name == "Results" else 2
                is_value = offset % group_size == 0
                is_holm = (name == "Results" and offset % 3 == 2) or (name == "Holm_table" and offset % 2 == 1)
                cell.number_format = "0.000" if is_value else "0.000000"
                if is_holm and isinstance(cell.value, float) and cell.value < args.alpha:
                    cell.fill = PatternFill("solid", fgColor="DDF0DF")
                    cell.font = Font(bold=True, color="17632E")

    ws = wb.create_sheet("Details")
    ws.append(["Sheet", "Model/Config.", "Epoch", "Indicator", "Base (%)", "Configuration (%)",
               "Delta (config.-Base), pp", "raw p", "p Holm", "Holm p < alpha"])
    for i, (sheet, label, epoch) in enumerate(SHEETS[1:]):
        for j, category in enumerate(CATEGORIES):
            q = adjusted[i, j]
            ws.append([sheet, label, epoch, category, excel_value(values[0, j]),
                       excel_value(values[i+1, j]), excel_value(deltas[i, j]),
                       excel_value(raw[i, j]), excel_value(q),
                       "N/A" if not np.isfinite(q) else ("Yes" if q < args.alpha else "No")])
    style_table(ws)
    for row in ws.iter_rows(min_row=2, min_col=5, max_col=9):
        for cell in row:
            cell.number_format = "0.000000" if cell.column >= 8 else "0.000"

    ws = wb.create_sheet("Counts_per_input")
    ws.append(["Sheet", "Input ID", "Ontology", "Responses", "Triples"] + CATEGORIES[:7])
    for sheet, _, _ in SHEETS:
        for key in keys:
            records = np.asarray(data[sheet][key])
            totals = records.sum(axis=0)
            ws.append([sheet, hashlib.sha256(key.encode()).hexdigest()[:12], sources.get(key, "N/A"),
                       len(records), float(totals[7])] + [excel_value(x) for x in totals[:7]])
    style_table(ws)

    ws = wb.create_sheet("Method")
    ws.append(["Parameter", "Value / description"])
    notes = [
        ("Source file", args.input.name),
        ("Comparison", "Each configuration and epoch against Base; no between-epoch comparisons."),
        ("Exchange unit", "Exact input prompt: its complete response bundle, counts, and denominators."),
        ("Blocks / responses per sheet", f"{len(keys)} / {len(keys) * args.repeats}"),
        ("Permutations", f"Exact enumeration of {1 << len(keys)} A/B assignments."),
        ("Statistic", "Two-sided error rate difference (configuration minus Base); lower rates are better."),
        ("Aggregation", "100 * sum of errors / sum of triples" if args.aggregation == "pooled" else "Unweighted mean of 100 * errors/triples per response"),
        ("Avg", "Arithmetic mean of the seven rates, recomputed for each permutation."),
        ("Exact p", "Number of assignments with |delta| >= |observed delta| / total assignments."),
        ("Holm", "Per indicator/column: eight separate families of six comparisons against Base, including Avg."),
        ("Error control", "At alpha=0.05, familywise control at 5% within each indicator, not jointly across the table."),
        ("Unavailable tests", "Not computable. Retain all six planned comparisons per indicator using p=1 internally, but display N/A."),
        ("Alpha", args.alpha),
        ("Missing cells", args.missing),
        ("Labels", "full = Q1-4; q1+q2 = Q1-2. Review if your naming convention differs."),
        ("Timing", "Not included: the input file does not provide training times."),
        ("Assumption", "Exchangeability of both systems response bundles within each input under H0."),
        ("Scope", "Evaluated benchmark and checkpoints; does not estimate training-run variability or treat ontologies as independent units."),
        ("Updates", "Static results; rerun the script after changing annotations."),
    ]
    notes += [(c, explanation) for c, explanation in zip(CATEGORIES, LONG_NAMES)]
    notes += [("Warning", warning) for warning in warnings]
    for note in notes:
        ws.append(note)
    style_table(ws)
    ws.freeze_panes = "A2"
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 110
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row[0].row].height = 32
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Original Excel workbook containing the seven worksheets")
    parser.add_argument("-o", "--output", type=Path, default=Path("pvalues_results.xlsx"))
    parser.add_argument("--aggregation", choices=["pooled", "mean-response"], default="pooled")
    parser.add_argument("--missing", choices=["unavailable", "error", "zero"], default="unavailable")
    parser.add_argument("--repeats", type=int, default=6, help="Expected generations per input (default: 6)")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"File not found: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("The output must not overwrite the input file.")
    if args.repeats < 1 or not 0 < args.alpha < 1:
        parser.error("--repeats must be positive and --alpha must be between 0 and 1.")
    data, keys, sources, warnings = read_input(args.input, args.missing, args.repeats)
    arrays = {s: aggregate(data[s], keys, args.aggregation) for s, _, _ in SHEETS}
    values = np.array([scores(arrays[s].sum(axis=0)) for s, _, _ in SHEETS])
    results = [exact_test(arrays[s], arrays["base"]) for s, _, _ in SHEETS[1:]]
    deltas = np.array([r[0] for r in results])
    raw = np.array([r[1] for r in results])
    adjusted = holm_by_indicator(raw)
    write_excel(args.output, values, raw, adjusted, deltas, data, keys, sources, warnings, args)
    print(f"Inputs: {len(keys)}; responses per input: {args.repeats}; exact permutations: {1 << len(keys)}")
    print(f"Holm per column: 8 families of 6 comparisons; {np.isfinite(raw).sum()} p-values available in total.")
    if warnings:
        print(f"Warnings: {len(warnings)}. See the Method worksheet.")
    print(f"Output workbook: {args.output.resolve()}")


if __name__ == "__main__":
    main()
