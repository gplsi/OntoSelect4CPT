# Ontology preprocessing

These scripts prepare a Turtle ontology corpus for continual pretraining (CPT). They remove structures matching held-out taxonomy relations and split large ontologies into smaller files for subsequent metric computation and corpus selection.

## Requirements

Python 3.10+ and:

```bash
python3 -m pip install rdflib tqdm
```

## Workflow

Run contamination removal on the original ontologies, then fragment the cleaned corpus so that matching and dependency traversal take place before structures are distributed across files.

| Step | Script | Purpose and outputs |
| --- | --- | --- |
| 1 | [remove_test_contamination.py](remove_test_contamination.py) | Match explicit `rdfs:subClassOf` relations against Task C test pairs using normalized labels, synonyms, and IRI local names. Remove both matched entities, recursively dependent subjects, and outgoing blank-node structures. Write cleaned TTL files and CSV reports of matches, removed nodes, and per-ontology summaries. Files without matches are copied unchanged. |
| 2 | [fragment_ontologies.py](fragment_ontologies.py) | Split files larger than the configured threshold into subject-based structural blocks grouped toward a target size of 10 MiB. Carry namespaces and ontology metadata into the fragments. Smaller files are parsed and reserialized without fragmentation. Write the resulting TTL corpus and `fragmentation_summary.csv`. |

## Configuration and execution

Both scripts are configured through constants near the top of the file.

1. In `remove_test_contamination.py`, configure the test-pair JSON files in `TASKC_SOURCES`, `INPUT_TTL_FOLDER`, `OUTPUT_TTL_FOLDER`, and the three CSV report paths. Test-pair JSON entries must contain `parent` and `child` fields.
2. In `fragment_ontologies.py`, set `INPUT_DIR` to the cleaned corpus directory and configure `OUTPUT_DIR`. Adjust `THRESHOLD_MB` and `TARGET_FRAGMENT_MB` if needed; both default to `10.0`.
3. Run from this directory:

```bash
python3 remove_test_contamination.py
python3 fragment_ontologies.py
```

Set all paths explicitly and use separate output directories outside their input trees. Both scripts search recursively for `.ttl` files and retain relative directory structure. Fragment sizes are approximate: structural blocks are kept together, and an oversized block is placed in its own fragment.

The final corpus provides the ontology or fragment files used by the metric computation, random dataset generation, and scoring stages.
