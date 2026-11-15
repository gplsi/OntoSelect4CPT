# Random corpus sampling

[generate_random_subsets.py](generate_random_subsets.py) creates random ontology subsets for continual pretraining (CPT) experiments. These subsets provide the experimental datasets used to relate corpus metrics to downstream F1 in the [scoring stage](../scoring/).

## How it works

The script scans the source corpus recursively and treats each non-empty ontology file or fragment as an independent sampling unit. By default, it generates **10 subsets**, each containing approximately **10% of the corpus volume in bytes**.

Each candidate starts from the complete corpus and uses a new integer seed, beginning at `42`. Random selections are refined toward the target volume. Files can occur in multiple subsets, but a candidate is rejected if its shared volume with any accepted subset exceeds **40% of the smaller subset's volume**:

```text
overlap(A, B) = shared_volume(A, B) / min(volume(A), volume(B))
```

The default size tolerance permits subsets between 9.7% and 10.3% of the total corpus volume. Sampling uses file sizes, without reading ontology semantics or token counts.

## Configuration and execution

Requires **Python 3.10+** and only the standard library.

1. Set `ONTOLOGY_SOURCE_DIR` to the corpus produced by [preprocessing](../preprocessing/).
2. Set `OUTPUT_ROOT_DIR` to a separate output directory outside the source tree.
3. Review the sampling constants, including `NUMBER_OF_DATASETS`, `VOLUME_PERCENTAGE`, and `MAX_PAIRWISE_VOLUME_OVERLAP`.
4. Run from this directory:

```bash
python3 generate_random_subsets.py
```

With `CLEAN_PREVIOUS_OUTPUT = True`, the existing output directory is deleted before generation.

## Outputs

Accepted subsets are copied into `random_01` through `random_10`, each containing an `ontologies` directory and a selection manifest. Global CSV reports record source files, accepted subsets and seeds, file membership, pairwise overlap, and rejected candidates.

Compute ontology metrics for each subset and obtain its downstream F1 results before running the scoring workflow. Generation stops with an error if the configured maximum number of candidate attempts is exceeded.
