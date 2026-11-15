""Calculate a p-value for comparing taxonomy discovery results on F1.

Use a paired relation-level randomization test (Yeh, 2000, Section 3.3) to
compare the parent-child relations predicted by models A and B against a fixed
gold reference. The statistic is F1(B) - F1(A). This is a two-sided Monte Carlo
test; the reported p-value is not adjusted for multiple comparisons.

Relations predicted by both models remain fixed. Relations exclusive to either
model are randomly reassigned to A or B. Under H0, the test assumes independent
exchange of A/B labels for each discordant relation. It does not verify that
assumption or measure variability across training runs.

Requirements
------------
Python 3 and numpy:
    python3 -m pip install numpy

Input format
------------
Each input must be a JSON list of parent-child records:
    [{"parent": "parent concept", "child": "child concept"}]

Relation labels are lowercased without stripping whitespace. In original mode,
the gold denominator uses the number of gold records and duplicate predictions
are rejected. In unique mode, denominators use unique relation sets, including
for the gold reference.

Usage
-----
Replace the generic filenames below with your own input and output files:

    python3 p_value_for_taxonomy_discovery.py \
        --gold GOLD_JSON \
        --model-a MODEL_A_JSON \
        --model-b MODEL_B_JSON \
        --output RESULTS_TXT

Optional settings:
    --mode original     Reproduce the evaluator denominators (default).
    --mode unique       Use unique relation sets for all denominators.
    --shuffles 1048576  Number of random reassignments (default: 1048576).
    --seed 42           Random seed (default: 42).
    --alpha 0.05        Significance threshold (default: 0.05).

The output filename must end in .txt and must differ from the input filenames.
The complete report is printed and saved to the TXT file. If --output is
omitted, the script prompts for an output filename. No file paths are hardcoded.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import numpy as np


def read_edges(path):
    with open(path, encoding='utf-8') as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f'{path}: expected a JSON list')
    edges = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ('parent', 'child')):
            raise ValueError(f'{path}: record {i} must contain string-valued parent/child fields')
        edges.add((row['parent'].lower(), row['child'].lower()))
    return rows, edges


def scores(tp, predicted, gold):
    p = tp / predicted if predicted else 0.0
    r = tp / gold if gold else 0.0
    f = 2 * tp / (predicted + gold) if predicted + gold else 0.0
    return p, r, f


def randomization(a, b, gold, gold_denominator, shuffles, seed):
    common = a & b
    discordant = sorted(a ^ b)
    n = len(discordant)
    correct = np.array([edge in gold for edge in discordant], dtype=np.int64)
    common_tp = len(common & gold)
    total_tp = len(a & gold) + len(b & gold)
    total_predicted = len(a) + len(b)
    observed = scores(len(b & gold), len(b), gold_denominator)[2] - scores(len(a & gold), len(a), gold_denominator)[2]
    rng = np.random.default_rng(seed)
    extreme = 0
    # Bound memory use; each bit reassigns an exclusive relation to B or A.
    batch_size = max(1, min(4096, 16_000_000 // max(n, 1)))
    for start in range(0, shuffles, batch_size):
        size = min(batch_size, shuffles - start)
        assigned_b = rng.integers(0, 2, size=(size, n), dtype=np.int8)
        nb = len(common) + assigned_b.sum(axis=1, dtype=np.int64)
        tb = common_tp + assigned_b @ correct
        na = total_predicted - nb
        ta = total_tp - tb
        fb = np.divide(2.0 * tb, gold_denominator + nb, out=np.zeros(size), where=(gold_denominator + nb) != 0)
        fa = np.divide(2.0 * ta, gold_denominator + na, out=np.zeros(size), where=(gold_denominator + na) != 0)
        extreme += int(np.count_nonzero(np.abs(fb - fa) >= abs(observed) - 1e-12))
    return observed, extreme, (extreme + 1) / (shuffles + 1), n


def run(args):
    gr, g = read_edges(args.gold)
    ar, a = read_edges(args.model_a)
    br, b = read_edges(args.model_b)
    if args.mode == 'original' and (len(ar) != len(a) or len(br) != len(b)):
        raise ValueError('Duplicate predictions found: use --mode unique or define how to reassign duplicates; predictions are not silently deduplicated in original mode.')
    ng = len(gr) if args.mode == 'original' else len(g)
    print(f'Gold: {args.gold.resolve()}')
    print(f'Model A: {args.model_a.resolve()}')
    print(f'Model B: {args.model_b.resolve()}')
    print(f'Mode: {args.mode}; normalization: lower(), without strip()')
    if len(gr) != len(g):
        print(f'WARNING: gold contains {len(gr)} records and {len(g)} unique relations; denominator used: {ng}.')
    for name, edges in [('A', a), ('B', b)]:
        tp = len(edges & g)
        p, r, f = scores(tp, len(edges), ng)
        print(f'{name}: predictions={len(edges)}, TP={tp}, FP={len(edges-g)}, unique FN={len(g-edges)}; P={p:.8f}, R={r:.8f}, F1={f:.8f}')
    delta, extreme, pvalue, n = randomization(a, b, g, ng, args.shuffles, args.seed)
    print(f'\nShared relations: {len(a & b)}; exclusive/discordant relations: {n}')
    print(f'Reassignment space: 2^{n}; shuffles performed: {args.shuffles:,}; seed: {args.seed}')
    print(f'F1 difference (B - A): {delta:+.8f} ({100*delta:+.4f} percentage points)')
    print(f'At-least-as-extreme reassignments: {extreme}; two-sided p=(c+1)/(R+1): {pvalue:.10g}')
    print(f'Approximate Monte Carlo standard error: {np.sqrt(pvalue*(1-pvalue)/args.shuffles):.3g}')
    print(f'Minimum p-value resolution: {1/(args.shuffles+1):.3g}')
    print(f'Significant at alpha={args.alpha}, without multiple-comparison adjustment?: {"Yes" if pvalue < args.alpha else "No"}')
    if delta != 0:
        print(f'Higher observed F1: model {"B" if delta > 0 else "A"}')
    print('The result is conditional on the paired exchangeability assumption; it does not verify that assumption.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True, type=Path)
    parser.add_argument('--model-a', required=True, type=Path)
    parser.add_argument('--model-b', required=True, type=Path)
    parser.add_argument('--shuffles', type=int, default=1048576)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--mode', choices=['original', 'unique'], default='original', help='original reproduces the evaluator denominators; unique also uses sets for the denominators')
    parser.add_argument('--output', type=Path, help='Results TXT path; prompted interactively if omitted')
    args = parser.parse_args()
    if args.shuffles < 1 or not 0 < args.alpha < 1:
        parser.error('shuffles must be positive and alpha must be between 0 and 1')
    if args.output is None:
        try:
            raw = input('Output TXT path (including the filename): ').strip()
        except EOFError:
            parser.error('No interactive input available: specify --output OUTPUT_TXT')
        if not raw:
            parser.error('You must specify an output path')
        args.output = Path(raw).expanduser()
    else:
        args.output = args.output.expanduser()
    if args.output.suffix.lower() != '.txt':
        parser.error('The output file must have a .txt extension')
    if args.output.resolve() in {args.gold.resolve(), args.model_a.resolve(), args.model_b.resolve()}:
        parser.error('The output must not overwrite an input file')
    # Capture the complete report to display and save identical text.
    report = io.StringIO()
    try:
        with contextlib.redirect_stdout(report):
            run(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output_text = report.getvalue()
    print(output_text, end='')
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding='utf-8')
    except OSError as exc:
        parser.error(f'Could not save the TXT file: {exc}')
    print(f'Results saved to: {args.output.resolve()}')


if __name__ == '__main__':
    main()
