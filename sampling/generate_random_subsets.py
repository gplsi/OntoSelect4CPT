"""
Generate 10 random ontology datasets directly from a source directory.

Execution
---------
Requires Python 3.10 or later. Uses only the Python standard library;
no additional packages need to be installed.

1. Set ONTOLOGY_SOURCE_DIR to the directory containing the ontology files.
2. Set OUTPUT_ROOT_DIR to a separate output directory outside the source tree.
3. Review the selection settings in CONFIGURATION.
4. Run from a terminal:

   python3 generate_random_subsets.py

The output contains random_01 through random_10, per-dataset manifests,
and CSV reports covering selections, membership, overlap, and rejections.
With CLEAN_PREVIOUS_OUTPUT = True, the existing output directory is deleted
before new datasets are generated. Blank directory settings are rejected.

Requirements
------------
1. Every candidate starts from the complete ontology directory.
2. Every file or 10 MB fragment is treated as an independent ontology.
3. Every candidate uses a different random seed.
4. Each accepted dataset contains approximately 10% of the total volume.
5. The same ontology may appear in multiple accepted datasets.
6. A candidate is rejected if it shares more than 40% of the smaller
dataset's volume with any previously accepted dataset.
7. The process continues until 10 datasets have been accepted.
8. Each accepted dataset is stored in a separate directory.

Overlap definition
------------------
For datasets A and B:

overlap(A, B) =
    shared_volume(A, B)
    / min(volume(A), volume(B))

A candidate is rejected when:

overlap(A, B) > MAX_PAIRWISE_VOLUME_OVERLAP
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ONTOLOGY_SOURCE_DIR = Path("")

OUTPUT_ROOT_DIR = Path("")

NUMBER_OF_DATASETS = 10

# Each accepted dataset should contain approximately 10% of the
# total valid ontology volume.
VOLUME_PERCENTAGE = 0.10

# First candidate seed.
# Every new candidate uses the following integer seed.
BASE_RANDOM_SEED = 42

# Reject a candidate when it shares more than 40% of the volume
# of the smaller dataset with any accepted dataset.
MAX_PAIRWISE_VOLUME_OVERLAP = 0.40

# Number of randomized selections tested for each candidate seed.
# The one closest to the target volume is retained.
SELECTION_ATTEMPTS_PER_CANDIDATE = 500

# Maximum number of candidates generated before stopping.
MAX_CANDIDATE_ATTEMPTS = 10_000

# Maximum allowed size above the target.
#
# If the target is 10% and this value is 0.03:
#
#     maximum = 10% × 1.03 = 10.3% of total volume
OVERSHOOT_TOLERANCE = 0.03

# Minimum accepted size relative to the target.
#
# If the target is 10% and this value is 0.97:
#
#     minimum = 10% × 0.97 = 9.7% of total volume
MINIMUM_TARGET_RATIO = 0.97

ONTOLOGY_EXTENSIONS = {
".ttl",
".rdf",
".owl",
".nt",
".xml",
".n3",
".trig",
".jsonld",
}

# Search recursively inside ONTOLOGY_SOURCE_DIR.
RECURSIVE_SEARCH = True

# Remove the previous output directory before generating new datasets.
CLEAN_PREVIOUS_OUTPUT = True


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_SOURCE_FILES_CSV = (
OUTPUT_ROOT_DIR
/ "source_ontology_files.csv"
)

OUTPUT_SUMMARY_CSV = (
OUTPUT_ROOT_DIR
/ "accepted_random_datasets_summary.csv"
)

OUTPUT_MEMBERSHIP_CSV = (
OUTPUT_ROOT_DIR
/ "ontology_dataset_membership.csv"
)

OUTPUT_PAIRWISE_OVERLAP_CSV = (
OUTPUT_ROOT_DIR
/ "accepted_random_datasets_pairwise_overlap.csv"
)

OUTPUT_REJECTED_CANDIDATES_CSV = (
OUTPUT_ROOT_DIR
/ "rejected_candidates.csv"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def bytes_to_mb(size_bytes: int | float) -> float:
"""Convert bytes to MiB."""
return float(size_bytes) / (1024 * 1024)


def calculate_selection_size(
selected_indices: list[int] | set[int],
size_by_index: dict[int, int],
) -> int:
"""Calculate the total volume represented by some file indices."""
return sum(
    size_by_index[idx]
    for idx in selected_indices
)


def write_csv(
output_path: Path,
rows: list[dict],
fieldnames: list[str],
) -> None:
"""Write dictionaries to a UTF-8 CSV file."""
output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with output_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_ontology_files(
source_dir: Path,
) -> list[dict]:
"""
Discover ontology files directly from the source directory.

Every physical file is treated as an independent ontology.
"""
if not source_dir.is_dir():
    raise FileNotFoundError(
        f"Ontology source directory not found: {source_dir}"
    )

if RECURSIVE_SEARCH:
    candidate_paths = source_dir.rglob("*")
else:
    candidate_paths = source_dir.glob("*")

ontology_paths = sorted(
    path.resolve()
    for path in candidate_paths
    if (
        path.is_file()
        and path.suffix.lower() in ONTOLOGY_EXTENSIONS
    )
)

if not ontology_paths:
    raise RuntimeError(
        f"No ontology files with the configured extensions "
        f"were found in: {source_dir}"
    )

ontology_rows: list[dict] = []

empty_files = 0

for path in ontology_paths:
    size_bytes = path.stat().st_size

    if size_bytes <= 0:
        empty_files += 1
        print(f"SKIP EMPTY FILE: {path}")
        continue

    relative_path = path.relative_to(
        source_dir.resolve()
    )

    ontology_rows.append(
        {
            "ontology_id": len(ontology_rows),
            "filename": path.name,
            "relative_path": str(relative_path),
            "resolved_path": str(path),
            "file_size_bytes": size_bytes,
            "file_size_mb": bytes_to_mb(size_bytes),
        }
    )

if not ontology_rows:
    raise RuntimeError(
        "Ontology files were found, but all of them were empty."
    )

total_size_bytes = sum(
    row["file_size_bytes"]
    for row in ontology_rows
)

print("=" * 80)
print("ONTOLOGY FILE DISCOVERY")
print("=" * 80)
print(f"Source directory:      {source_dir}")
print(f"Valid ontology files:  {len(ontology_rows)}")
print(f"Empty files skipped:   {empty_files}")
print(
    f"Total dataset volume:  "
    f"{bytes_to_mb(total_size_bytes):.2f} MB"
)

return ontology_rows


# ============================================================
# RANDOM VOLUME SELECTION
# ============================================================

def generate_random_candidate_once(
all_indices: list[int],
size_by_index: dict[int, int],
target_size_bytes: int,
maximum_size_bytes: int,
rng: random.Random,
) -> list[int]:
"""
Generate one candidate by shuffling all ontology indices.

Files are added in random order while they fit below the configured
maximum volume.
"""
shuffled_indices = all_indices.copy()
rng.shuffle(shuffled_indices)

selected_indices: list[int] = []
selected_size_bytes = 0

for idx in shuffled_indices:
    ontology_size = size_by_index[idx]

    proposed_size = (
        selected_size_bytes
        + ontology_size
    )

    if proposed_size > maximum_size_bytes:
        continue

    selected_indices.append(idx)
    selected_size_bytes = proposed_size

    if selected_size_bytes >= target_size_bytes:
        break

return selected_indices


def improve_candidate_with_addition_or_swap(
selected_indices: list[int],
all_indices: list[int],
size_by_index: dict[int, int],
target_size_bytes: int,
maximum_size_bytes: int,
) -> list[int]:
"""
Improve a candidate's volume using a single addition or replacement.

This only tries to reduce the absolute distance from the target.
"""
selected = list(dict.fromkeys(selected_indices))
selected_set = set(selected)

current_size = calculate_selection_size(
    selected,
    size_by_index,
)

best_selection = selected.copy()
best_error = abs(
    target_size_bytes
    - current_size
)

unselected_indices = [
    idx
    for idx in all_indices
    if idx not in selected_set
]

# Try adding one ontology.
for add_idx in unselected_indices:
    new_size = (
        current_size
        + size_by_index[add_idx]
    )

    if new_size > maximum_size_bytes:
        continue

    new_error = abs(
        target_size_bytes
        - new_size
    )

    if new_error < best_error:
        best_error = new_error
        best_selection = selected + [add_idx]

# Try replacing one selected ontology with one unselected ontology.
for remove_idx in selected:
    size_without_removed = (
        current_size
        - size_by_index[remove_idx]
    )

    for add_idx in unselected_indices:
        new_size = (
            size_without_removed
            + size_by_index[add_idx]
        )

        if new_size > maximum_size_bytes:
            continue

        new_error = abs(
            target_size_bytes
            - new_size
        )

        if new_error < best_error:
            best_error = new_error

            best_selection = [
                idx
                for idx in selected
                if idx != remove_idx
            ]

            best_selection.append(add_idx)

return best_selection


def select_random_subset_by_volume(
ontology_rows: list[dict],
target_size_bytes: int,
seed: int,
) -> list[int]:
"""
Generate one random candidate close to the target volume.

Every candidate starts from the complete source dataset.
"""
rng = random.Random(seed)

all_indices = [
    row["ontology_id"]
    for row in ontology_rows
]

size_by_index = {
    row["ontology_id"]: row["file_size_bytes"]
    for row in ontology_rows
}

maximum_size_bytes = round(
    target_size_bytes
    * (1.0 + OVERSHOOT_TOLERANCE)
)

minimum_size_bytes = round(
    target_size_bytes
    * MINIMUM_TARGET_RATIO
)

best_selection: list[int] = []
best_size_bytes = 0
best_score: tuple[int, int] | None = None

for _ in range(SELECTION_ATTEMPTS_PER_CANDIDATE):
    candidate = generate_random_candidate_once(
        all_indices=all_indices,
        size_by_index=size_by_index,
        target_size_bytes=target_size_bytes,
        maximum_size_bytes=maximum_size_bytes,
        rng=rng,
    )

    if not candidate:
        continue

    candidate = improve_candidate_with_addition_or_swap(
        selected_indices=candidate,
        all_indices=all_indices,
        size_by_index=size_by_index,
        target_size_bytes=target_size_bytes,
        maximum_size_bytes=maximum_size_bytes,
    )

    candidate_size_bytes = calculate_selection_size(
        candidate,
        size_by_index,
    )

    candidate_error = abs(
        target_size_bytes
        - candidate_size_bytes
    )

    reaches_minimum = (
        candidate_size_bytes
        >= minimum_size_bytes
    )

    score = (
        0 if reaches_minimum else 1,
        candidate_error,
    )

    if best_score is None or score < best_score:
        best_score = score
        best_selection = candidate.copy()
        best_size_bytes = candidate_size_bytes

    if reaches_minimum and candidate_error == 0:
        break

if not best_selection:
    raise RuntimeError(
        f"No random candidate could be generated "
        f"with seed {seed}."
    )

if best_size_bytes < minimum_size_bytes:
    raise RuntimeError(
        f"Candidate generated with seed {seed} contains "
        f"{bytes_to_mb(best_size_bytes):.2f} MB, below the "
        f"minimum permitted volume of "
        f"{bytes_to_mb(minimum_size_bytes):.2f} MB."
    )

return best_selection


# ============================================================
# OVERLAP CALCULATION
# ============================================================

def calculate_pairwise_volume_overlap(
first_indices: list[int],
second_indices: list[int],
size_by_index: dict[int, int],
) -> dict:
"""
Calculate pairwise volume overlap.

Main overlap measure:

    shared volume / volume of smaller dataset
"""
first_set = set(first_indices)
second_set = set(second_indices)

intersection = first_set & second_set
union = first_set | second_set

first_size_bytes = calculate_selection_size(
    first_set,
    size_by_index,
)

second_size_bytes = calculate_selection_size(
    second_set,
    size_by_index,
)

shared_size_bytes = calculate_selection_size(
    intersection,
    size_by_index,
)

smaller_size_bytes = min(
    first_size_bytes,
    second_size_bytes,
)

overlap_smaller_ratio = (
    shared_size_bytes / smaller_size_bytes
    if smaller_size_bytes > 0
    else 0.0
)

overlap_first_ratio = (
    shared_size_bytes / first_size_bytes
    if first_size_bytes > 0
    else 0.0
)

overlap_second_ratio = (
    shared_size_bytes / second_size_bytes
    if second_size_bytes > 0
    else 0.0
)

jaccard_file_ratio = (
    len(intersection) / len(union)
    if union
    else 0.0
)

return {
    "first_size_bytes": first_size_bytes,
    "second_size_bytes": second_size_bytes,
    "shared_size_bytes": shared_size_bytes,
    "shared_file_count": len(intersection),
    "overlap_smaller_ratio": overlap_smaller_ratio,
    "overlap_first_ratio": overlap_first_ratio,
    "overlap_second_ratio": overlap_second_ratio,
    "jaccard_file_ratio": jaccard_file_ratio,
}


def compare_candidate_with_accepted_datasets(
candidate_indices: list[int],
accepted_datasets: list[list[int]],
size_by_index: dict[int, int],
) -> tuple[bool, list[dict]]:
"""
Compare a candidate with every previously accepted dataset.
"""
comparisons: list[dict] = []
candidate_is_valid = True

for accepted_position, accepted_indices in enumerate(
    accepted_datasets,
    start=1,
):
    overlap = calculate_pairwise_volume_overlap(
        first_indices=candidate_indices,
        second_indices=accepted_indices,
        size_by_index=size_by_index,
    )

    threshold_exceeded = (
        overlap["overlap_smaller_ratio"]
        > MAX_PAIRWISE_VOLUME_OVERLAP
    )

    if threshold_exceeded:
        candidate_is_valid = False

    comparisons.append(
        {
            "accepted_dataset_number": accepted_position,
            "accepted_dataset": (
                f"random_{accepted_position:02d}"
            ),
            "shared_file_count": (
                overlap["shared_file_count"]
            ),
            "shared_size_bytes": (
                overlap["shared_size_bytes"]
            ),
            "shared_size_mb": bytes_to_mb(
                overlap["shared_size_bytes"]
            ),
            "overlap_percentage_of_smaller": (
                overlap["overlap_smaller_ratio"] * 100
            ),
            "overlap_percentage_of_candidate": (
                overlap["overlap_first_ratio"] * 100
            ),
            "overlap_percentage_of_accepted": (
                overlap["overlap_second_ratio"] * 100
            ),
            "jaccard_by_file": (
                overlap["jaccard_file_ratio"]
            ),
            "threshold_exceeded": threshold_exceeded,
        }
    )

return candidate_is_valid, comparisons


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_accepted_random_datasets(
ontology_rows: list[dict],
) -> tuple[
list[list[int]],
list[int],
list[dict],
]:
"""
Generate candidates until NUMBER_OF_DATASETS datasets are accepted.
"""
size_by_index = {
    row["ontology_id"]: row["file_size_bytes"]
    for row in ontology_rows
}

total_size_bytes = sum(
    size_by_index.values()
)

target_size_bytes = round(
    total_size_bytes
    * VOLUME_PERCENTAGE
)

accepted_datasets: list[list[int]] = []
accepted_seeds: list[int] = []
rejected_rows: list[dict] = []

candidate_attempt = 0

print("\n" + "=" * 80)
print("GENERATING RANDOM DATASETS")
print("=" * 80)
print(f"Required datasets:          {NUMBER_OF_DATASETS}")
print(
    f"Target volume:              "
    f"{bytes_to_mb(target_size_bytes):.2f} MB"
)
print(
    f"Target dataset percentage:  "
    f"{VOLUME_PERCENTAGE * 100:.2f}%"
)
print(
    f"Maximum pairwise overlap:   "
    f"{MAX_PAIRWISE_VOLUME_OVERLAP * 100:.2f}%"
)

while len(accepted_datasets) < NUMBER_OF_DATASETS:
    candidate_attempt += 1

    if candidate_attempt > MAX_CANDIDATE_ATTEMPTS:
        raise RuntimeError(
            f"Maximum number of candidate attempts reached: "
            f"{MAX_CANDIDATE_ATTEMPTS}. "
            f"Only {len(accepted_datasets)} datasets were accepted."
        )

    seed = (
        BASE_RANDOM_SEED
        + candidate_attempt
        - 1
    )

    candidate_indices = select_random_subset_by_volume(
        ontology_rows=ontology_rows,
        target_size_bytes=target_size_bytes,
        seed=seed,
    )

    candidate_size_bytes = calculate_selection_size(
        candidate_indices,
        size_by_index,
    )

    candidate_percentage = (
        candidate_size_bytes
        / total_size_bytes
        * 100
    )

    print("\n" + "-" * 80)
    print(
        f"Candidate {candidate_attempt} | seed {seed}"
    )
    print(
        f"Selected files:      {len(candidate_indices)}"
    )
    print(
        f"Selected volume:     "
        f"{bytes_to_mb(candidate_size_bytes):.2f} MB"
    )
    print(
        f"Percentage of total: "
        f"{candidate_percentage:.4f}%"
    )

    if not accepted_datasets:
        accepted_datasets.append(candidate_indices)
        accepted_seeds.append(seed)

        print("ACCEPTED as random_01.")
        continue

    candidate_is_valid, comparisons = (
        compare_candidate_with_accepted_datasets(
            candidate_indices=candidate_indices,
            accepted_datasets=accepted_datasets,
            size_by_index=size_by_index,
        )
    )

    for comparison in comparisons:
        status = (
            "REJECT"
            if comparison["threshold_exceeded"]
            else "OK"
        )

        print(
            f"Compared with "
            f"{comparison['accepted_dataset']}: "
            f"{comparison['overlap_percentage_of_smaller']:.2f}% "
            f"shared volume | {status}"
        )

    if candidate_is_valid:
        accepted_datasets.append(candidate_indices)
        accepted_seeds.append(seed)

        dataset_number = len(accepted_datasets)

        print(
            f"ACCEPTED as random_{dataset_number:02d}."
        )

    else:
        exceeded_comparisons = [
            comparison
            for comparison in comparisons
            if comparison["threshold_exceeded"]
        ]

        maximum_overlap_comparison = max(
            comparisons,
            key=lambda item: (
                item["overlap_percentage_of_smaller"]
            ),
        )

        rejected_rows.append(
            {
                "candidate_attempt": candidate_attempt,
                "candidate_seed": seed,
                "candidate_file_count": len(
                    candidate_indices
                ),
                "candidate_size_bytes": candidate_size_bytes,
                "candidate_size_mb": bytes_to_mb(
                    candidate_size_bytes
                ),
                "candidate_percentage_of_total": (
                    candidate_percentage
                ),
                "number_of_threshold_violations": len(
                    exceeded_comparisons
                ),
                "maximum_overlap_dataset": (
                    maximum_overlap_comparison[
                        "accepted_dataset"
                    ]
                ),
                "maximum_overlap_percentage_of_smaller": (
                    maximum_overlap_comparison[
                        "overlap_percentage_of_smaller"
                    ]
                ),
                "maximum_shared_size_mb": (
                    maximum_overlap_comparison[
                        "shared_size_mb"
                    ]
                ),
                "maximum_shared_file_count": (
                    maximum_overlap_comparison[
                        "shared_file_count"
                    ]
                ),
                "overlap_threshold_percentage": (
                    MAX_PAIRWISE_VOLUME_OVERLAP * 100
                ),
            }
        )

        print("REJECTED. Generating another candidate.")

print("\n" + "=" * 80)
print("GENERATION COMPLETED")
print("=" * 80)
print(f"Accepted datasets:   {len(accepted_datasets)}")
print(f"Rejected candidates: {len(rejected_rows)}")
print(f"Total attempts:      {candidate_attempt}")

return (
    accepted_datasets,
    accepted_seeds,
    rejected_rows,
)


# ============================================================
# REPORT GENERATION
# ============================================================

def calculate_memberships(
datasets: list[list[int]],
) -> tuple[dict[int, list[int]], Counter]:
"""Calculate in which accepted datasets each file appears."""
memberships: dict[int, list[int]] = {}

for dataset_number, indices in enumerate(
    datasets,
    start=1,
):
    for ontology_idx in indices:
        memberships.setdefault(
            ontology_idx,
            [],
        ).append(dataset_number)

appearance_counts = Counter(
    {
        ontology_idx: len(dataset_numbers)
        for ontology_idx, dataset_numbers
        in memberships.items()
    }
)

return memberships, appearance_counts


def build_pairwise_overlap_rows(
datasets: list[list[int]],
size_by_index: dict[int, int],
) -> list[dict]:
"""Build one report row for every pair of accepted datasets."""
rows: list[dict] = []

for first_position in range(len(datasets)):
    for second_position in range(
        first_position + 1,
        len(datasets),
    ):
        overlap = calculate_pairwise_volume_overlap(
            first_indices=datasets[first_position],
            second_indices=datasets[second_position],
            size_by_index=size_by_index,
        )

        rows.append(
            {
                "dataset_a": (
                    f"random_{first_position + 1:02d}"
                ),
                "dataset_b": (
                    f"random_{second_position + 1:02d}"
                ),
                "shared_file_count": (
                    overlap["shared_file_count"]
                ),
                "shared_size_bytes": (
                    overlap["shared_size_bytes"]
                ),
                "shared_size_mb": bytes_to_mb(
                    overlap["shared_size_bytes"]
                ),
                "overlap_percentage_of_smaller": (
                    overlap["overlap_smaller_ratio"] * 100
                ),
                "overlap_percentage_of_dataset_a": (
                    overlap["overlap_first_ratio"] * 100
                ),
                "overlap_percentage_of_dataset_b": (
                    overlap["overlap_second_ratio"] * 100
                ),
                "jaccard_by_file": (
                    overlap["jaccard_file_ratio"]
                ),
            }
        )

return rows


# ============================================================
# SAVING DATASETS
# ============================================================

def prepare_output_directory() -> None:
"""Prepare a clean output directory."""
if (
    CLEAN_PREVIOUS_OUTPUT
    and OUTPUT_ROOT_DIR.exists()
):
    print(
        f"Removing previous output directory: "
        f"{OUTPUT_ROOT_DIR}"
    )

    shutil.rmtree(OUTPUT_ROOT_DIR)

OUTPUT_ROOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def save_single_dataset(
ontology_rows_by_id: dict[int, dict],
selected_indices: list[int],
dataset_number: int,
seed: int,
total_size_bytes: int,
appearance_counts: Counter,
) -> dict:
"""Save one accepted dataset."""
dataset_name = f"random_{dataset_number:02d}"

dataset_dir = (
    OUTPUT_ROOT_DIR
    / dataset_name
)

ontology_output_dir = (
    dataset_dir
    / "ontologies"
)

dataset_dir.mkdir(
    parents=True,
    exist_ok=True,
)

ontology_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

selected_rows = [
    ontology_rows_by_id[idx].copy()
    for idx in selected_indices
]

rng = random.Random(seed)
rng.shuffle(selected_rows)

selected_size_bytes = sum(
    row["file_size_bytes"]
    for row in selected_rows
)

selected_percentage = (
    selected_size_bytes
    / total_size_bytes
    * 100
)

manifest_rows: list[dict] = []

print("\n" + "-" * 80)
print(f"Saving {dataset_name}")
print("-" * 80)
print(f"Seed:                 {seed}")
print(f"Files:                {len(selected_rows)}")
print(
    f"Volume:               "
    f"{bytes_to_mb(selected_size_bytes):.2f} MB"
)
print(
    f"Percentage of total:  "
    f"{selected_percentage:.4f}%"
)

for row in selected_rows:
    source_path = Path(
        row["resolved_path"]
    )

    relative_path = Path(
        row["relative_path"]
    )

    destination_path = (
        ontology_output_dir
        / relative_path
    )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_path,
        destination_path,
    )

    appearance_count = appearance_counts.get(
        row["ontology_id"],
        0,
    )

    manifest_rows.append(
        {
            **row,
            "dataset": dataset_name,
            "random_seed": seed,
            "appearance_count_across_datasets": (
                appearance_count
            ),
            "appears_in_multiple_datasets": (
                appearance_count > 1
            ),
            "copied_path": str(destination_path),
        }
    )

    print(
        f"COPY: {row['relative_path']} "
        f"({row['file_size_mb']:.3f} MB)"
    )

manifest_path = (
    dataset_dir
    / f"selected_files_{dataset_name}.csv"
)

write_csv(
    output_path=manifest_path,
    rows=manifest_rows,
    fieldnames=[
        "ontology_id",
        "filename",
        "relative_path",
        "resolved_path",
        "file_size_bytes",
        "file_size_mb",
        "dataset",
        "random_seed",
        "appearance_count_across_datasets",
        "appears_in_multiple_datasets",
        "copied_path",
    ],
)

return {
    "dataset": dataset_name,
    "random_seed": seed,
    "number_of_files": len(selected_rows),
    "selected_size_bytes": selected_size_bytes,
    "selected_size_mb": bytes_to_mb(
        selected_size_bytes
    ),
    "percentage_of_total_dataset": selected_percentage,
    "target_percentage": VOLUME_PERCENTAGE * 100,
    "absolute_error_from_target_percentage": abs(
        selected_percentage
        - VOLUME_PERCENTAGE * 100
    ),
    "files_appearing_in_multiple_datasets": sum(
        1
        for row in selected_rows
        if appearance_counts.get(
            row["ontology_id"],
            0,
        ) > 1
    ),
    "output_directory": str(dataset_dir),
}


# ============================================================
# MAIN
# ============================================================

def main() -> None:
if ONTOLOGY_SOURCE_DIR == Path(".") or OUTPUT_ROOT_DIR == Path("."):
    raise ValueError(
        "Set ONTOLOGY_SOURCE_DIR and OUTPUT_ROOT_DIR before running."
    )

prepare_output_directory()

ontology_rows = discover_ontology_files(
    ONTOLOGY_SOURCE_DIR
)

ontology_rows_by_id = {
    row["ontology_id"]: row
    for row in ontology_rows
}

size_by_index = {
    row["ontology_id"]: row["file_size_bytes"]
    for row in ontology_rows
}

total_size_bytes = sum(
    size_by_index.values()
)

write_csv(
    output_path=OUTPUT_SOURCE_FILES_CSV,
    rows=ontology_rows,
    fieldnames=[
        "ontology_id",
        "filename",
        "relative_path",
        "resolved_path",
        "file_size_bytes",
        "file_size_mb",
    ],
)

(
    accepted_datasets,
    accepted_seeds,
    rejected_rows,
) = generate_accepted_random_datasets(
    ontology_rows
)

memberships, appearance_counts = (
    calculate_memberships(
        accepted_datasets
    )
)

summary_rows: list[dict] = []

for dataset_number, (
    selected_indices,
    seed,
) in enumerate(
    zip(
        accepted_datasets,
        accepted_seeds,
    ),
    start=1,
):
    summary_row = save_single_dataset(
        ontology_rows_by_id=ontology_rows_by_id,
        selected_indices=selected_indices,
        dataset_number=dataset_number,
        seed=seed,
        total_size_bytes=total_size_bytes,
        appearance_counts=appearance_counts,
    )

    summary_rows.append(summary_row)

write_csv(
    output_path=OUTPUT_SUMMARY_CSV,
    rows=summary_rows,
    fieldnames=[
        "dataset",
        "random_seed",
        "number_of_files",
        "selected_size_bytes",
        "selected_size_mb",
        "percentage_of_total_dataset",
        "target_percentage",
        "absolute_error_from_target_percentage",
        "files_appearing_in_multiple_datasets",
        "output_directory",
    ],
)

membership_rows: list[dict] = []

for ontology_idx, dataset_numbers in memberships.items():
    row = ontology_rows_by_id[ontology_idx]

    membership_rows.append(
        {
            **row,
            "appearance_count": len(dataset_numbers),
            "appears_in_multiple_datasets": (
                len(dataset_numbers) > 1
            ),
            "datasets": ", ".join(
                f"random_{dataset_number:02d}"
                for dataset_number in dataset_numbers
            ),
        }
    )

write_csv(
    output_path=OUTPUT_MEMBERSHIP_CSV,
    rows=membership_rows,
    fieldnames=[
        "ontology_id",
        "filename",
        "relative_path",
        "resolved_path",
        "file_size_bytes",
        "file_size_mb",
        "appearance_count",
        "appears_in_multiple_datasets",
        "datasets",
    ],
)

overlap_rows = build_pairwise_overlap_rows(
    datasets=accepted_datasets,
    size_by_index=size_by_index,
)

write_csv(
    output_path=OUTPUT_PAIRWISE_OVERLAP_CSV,
    rows=overlap_rows,
    fieldnames=[
        "dataset_a",
        "dataset_b",
        "shared_file_count",
        "shared_size_bytes",
        "shared_size_mb",
        "overlap_percentage_of_smaller",
        "overlap_percentage_of_dataset_a",
        "overlap_percentage_of_dataset_b",
        "jaccard_by_file",
    ],
)

write_csv(
    output_path=OUTPUT_REJECTED_CANDIDATES_CSV,
    rows=rejected_rows,
    fieldnames=[
        "candidate_attempt",
        "candidate_seed",
        "candidate_file_count",
        "candidate_size_bytes",
        "candidate_size_mb",
        "candidate_percentage_of_total",
        "number_of_threshold_violations",
        "maximum_overlap_dataset",
        "maximum_overlap_percentage_of_smaller",
        "maximum_shared_size_mb",
        "maximum_shared_file_count",
        "overlap_threshold_percentage",
    ],
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Output root:        {OUTPUT_ROOT_DIR}")
print(f"Accepted datasets:  {len(accepted_datasets)}")
print(f"Rejected candidates:{len(rejected_rows)}")
print(f"Summary:            {OUTPUT_SUMMARY_CSV}")
print(f"Overlap report:     {OUTPUT_PAIRWISE_OVERLAP_CSV}")
print(f"Membership report:  {OUTPUT_MEMBERSHIP_CSV}")

print("\nAccepted dataset summary")
print("-" * 80)

for row in summary_rows:
    print(
        f"{row['dataset']} | "
        f"seed={row['random_seed']} | "
        f"files={row['number_of_files']} | "
        f"size={row['selected_size_mb']:.2f} MB | "
        f"total={row['percentage_of_total_dataset']:.4f}%"
    )


if __name__ == "__main__":
main()
