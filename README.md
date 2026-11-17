# OntoSelect4CPT

**Task-guided selection of ontology corpora for continual pretraining of language models.**

OntoSelect4CPT provides tools to characterize Turtle ontology datasets, explore how their properties relate to downstream performance, and score files for corpus selection. It combines ontology, structural, textual, and semantic metrics with empirical results from **continual pretraining (CPT)** followed by **fine-tuning (FT)**.

The methodology was applied separately to **MatOnto** and **OBI** for taxonomy discovery: identifying directed subclass relations between concepts. Corpus usefulness is measured by the F1 obtained after CPT and supervised FT, so the resulting selection criteria depend on the model and task.

## Pipeline and code

| Stage | Purpose | Code |
| --- | --- | --- |
| 1. Prepare the corpus | Filter structures matching held-out taxonomy relations and fragment large Turtle files into manageable selection units. | [Contamination removal](preprocessing/remove_test_contamination.py) · [Fragmentation](preprocessing/fragment_ontologies.py) |
| 2. Characterize the data | Compute ontology embeddings, similarity to a task description, ontology/text metrics, and token counts for each file or fragment. | [Embeddings](metrics/compute_embeddings.py) · [Semantic similarity](metrics/compute_semantic_similarity.py) · [Ontology and text metrics](metrics/compute_ontology_and_text_metrics.py) |
| 3. Generate experimental subsets | Create ten random subsets, each targeting approximately 10% of corpus volume in bytes, with controlled pairwise overlap. | [Random sampling](sampling/generate_random_subsets.py) |
| 4. Obtain downstream F1 | Run independent CPT from the same base model for each subset, then apply the same supervised FT procedure and evaluate taxonomy discovery. | [LMTK — deterministic branch](https://github.com/gplsi/LMTK/tree/deterministic) for CPT; **FT repository: LINK TO BE ADDED** |
| 5. Learn metric–F1 associations | Remove redundant metrics, aggregate file-level values using token-weighted means, and compute Pearson and Spearman correlations with F1 after one and two CPT epochs. | [Identify redundancy](scoring/identify_redundant_metrics.py) · [Remove redundant metrics](scoring/remove_redundant_metrics.py) · [Compute associations](scoring/compute_metric_f1_associations.py) |
| 6. Score and select data | Normalize metrics with consistent association signs, orient them toward higher F1, and combine them using association-based weights. Rank files and select a corpus under a token budget. | [Score corpus files](scoring/score_corpus_files.py). Token-budget selection is a subsequent step; its script is not currently included. |
| 7. Evaluate guided selections | Repeat CPT → FT → evaluation and compare against random subsets, the full corpus, and the model fine-tuned without CPT. Assess F1 differences statistically. | [Taxonomy discovery test](analysis/p_value_for_taxonomy_discovery.py) · [Holm correction](analysis/holm_correction_of_p_value.py) |

The association stage retains metrics whose Pearson and Spearman coefficients share the same sign across both epochs. Their mean absolute correlation determines their weight in the file score. Guided selections are then evaluated through new training runs to test whether these criteria improve the performance–cost trade-off.

## Getting started

Use **Python 3.10+** and follow the configuration, dependencies, and input/output formats documented in each directory:

- [Preprocessing](preprocessing/README.md)
- [Metrics and their definitions](metrics/README.md)
- [Random sampling](sampling/README.md)
- [Corpus scoring](scoring/README.md)
- [Statistical analysis](analysis/README.md)

CPT and supervised FT run in external repositories; their measured F1 results are supplied to the scoring workflow. The analysis directory also includes a [paired randomization test for ontology-to-ontology generation](analysis/p_value_for_ontology_to_ontology.py).
