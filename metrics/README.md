# Ontology, text, and semantic metrics

This directory describes each Turtle ontology or fragment through its size, schema, taxonomy, population, annotations, and similarity to a task description. The resulting workbooks feed the [scoring workflow](../scoring/), which identifies redundant metrics, learns their association with downstream F1, and scores corpus files.

## Scripts and execution order

| Order | Script | Role and outputs |
| --- | --- | --- |
| 1 | [compute_embeddings.py](compute_embeddings.py) | Extract filtered annotation literals and compute one embedding per ontology file. Write filename-keyed embeddings and annotation JSON files. |
| 2 | [compute_semantic_similarity.py](compute_semantic_similarity.py) | Embed a task description, compare it with the ontology embeddings using cosine similarity, and write similarity scores, quartiles, and summary statistics. |
| 3 | [compute_ontology_and_text_metrics.py](compute_ontology_and_text_metrics.py) | Analyze each TTL file independently, count Llama tokens, import similarity scores from step 2, and export detailed metrics and summaries to Excel. |

All three scripts use configuration constants near the top of their files. Some script docstrings retain earlier filenames; the commands below use the current repository names.

```bash
python3 -m pip install pandas rdflib openpyxl transformers torch numpy tqdm einops

python3 compute_embeddings.py
python3 compute_semantic_similarity.py
python3 compute_ontology_and_text_metrics.py
```

Use Python 3.10+ for the full repository workflow. Configure the following before running:

| Script | Required configuration |
| --- | --- |
| `compute_embeddings.py` | `INPUT_FOLDER`, `OUTPUT_FILE_EMB`, and `OUTPUT_FILE_LITS`. Optional output overrides: `--output_emb` and `--output_lits`. |
| `compute_semantic_similarity.py` | `TASK_FILE`, `ONTOLOGY_EMBEDDINGS_FILE` pointing to step 1's embeddings JSON, and `OUTPUT_FILE`. |
| `compute_ontology_and_text_metrics.py` | `BASEPATH`, `OUTPUT_FILE` for the `.xlsx` workbook, and `SIMILARITY_JSON` pointing to step 2's output. Optionally set `EXCLUDE_PATHS`. |

Create output parent directories in advance. Embedding and similarity computation use CUDA when available, otherwise CPU, and default to `nomic-ai/nomic-embed-text-v1.5`; keep the model consistent across both stages. Model loading uses `trust_remote_code=True`. The ontology/text analyzer loads only the Llama tokenizer, not Llama model weights, and does not require a GPU. Set `HF_TOKEN` if tokenizer access requires authentication.

The embedding script scans only the immediate input directory; the ontology/text analyzer scans recursively. For a corpus with subdirectories, generate and combine embeddings for all relevant files, keeping filenames unique. Similarity lookup uses the filename, not its relative path.

## Calculation conventions

The definitions below document the current implementation, including historical column names. They describe each **file or fragment**, which need not contain a complete ontology.

- $G$ is the RDFLib graph of distinct RDF triples. Identical triples are deduplicated when parsed. Text metrics operate on the original serialization and retain repeated text.
- $T$ is the decoded file text. Unless specified otherwise, the analyzer reads it as UTF-8 with undecodable bytes ignored.
- $\mathcal C$ is the set of identified URI classes, with $C=|\mathcal C|$. Classes include URI subjects explicitly typed as `owl:Class`, `rdfs:Class`, or `rdf:Class`, plus eligible URI objects of `rdf:type` after the filters in `identify_classes`. Merely occurring in `rdfs:subClassOf`, `rdfs:domain`, or `rdfs:range` does not add a class.
- $\mathcal O$, $\mathcal D$, and $\mathcal A$ are URI properties explicitly declared as object, datatype, and annotation properties. $\mathcal F$ and $\mathcal F^{-1}$ are URI subjects explicitly typed as functional and inverse-functional properties.
- $E=\{(c,p):(c,\texttt{rdfs:subClassOf},p)\in G,\ c\in\mathcal C,\ p\text{ is a URI}\}$, with $H=|E|$. The internal taxonomy uses $E_C=\{(c,p)\in E:p\in\mathcal C\}$.
- Within $E_C$, $\mathrm{par}(c)$, $\mathrm{child}(c)$, and $\mathrm{sib}(c)$ are direct parents, children, and distinct siblings. Siblings share at least one parent; a class is not its own sibling. Roots $\mathcal R$ have no internal parents; leaves $\mathcal L$ have no internal children. An isolated class is both.
- Counts use explicit triples. No OWL reasoner is run, imports are not resolved, and property associations are not inherited.
- Unless specified otherwise, a ratio with a zero denominator returns `0.0`.

With the default configuration, `owl:Thing` and `owl:Nothing` are excluded, anonymous superclasses are omitted, and OWL restrictions do not contribute to non-taxonomic relationship counts. The formulas below describe those defaults.

## Metric groups

The metrics are organized into eight groups:

1. [File volume and representation size](#1-file-volume-and-representation-size)
2. [Schema inventory and axiom counts](#2-schema-inventory-and-axiom-counts)
3. [Taxonomy shape, inheritance, and paths](#3-taxonomy-shape-inheritance-and-paths)
4. [Non-taxonomic connectivity](#4-non-taxonomic-connectivity)
5. [Property coverage and schema elaboration](#5-property-coverage-and-schema-elaboration)
6. [Individuals and class population](#6-individuals-and-class-population)
7. [Textual vocabulary, repetition, and literal coverage](#7-textual-vocabulary-repetition-and-literal-coverage)
8. [Semantic richness, quality, and task alignment](#8-semantic-richness-quality-and-task-alignment)

## 1. File volume and representation size

### File size — `TamañoMB`

**Internal:** `size_mb`, supplied to `build_row`.

$$S=\mathrm{round}\left(\frac{\text{file bytes}}{1024^2},2\right).$$

Physical file size rounded to two decimal places. Despite the column name, the unit is **MiB**, not decimal MB.

### Character count — `NumCaracteres`

**Internal:** `char_count`, supplied to `build_row`.

$$N_{\mathrm{char}}=|T|.$$

Length of the UTF-8-decoded file, including spaces, line breaks, comments, and declarations. Decoding uses `errors='ignore'`. This counts Python string characters, not bytes.

### Llama token count — `NumTokensLlama32_3B`

**Internal:** `num_tokens_llama` in `build_row`; populated by `count_llama_tokens` in `main`.

$$N_{\mathrm{Llama}}=|\mathrm{encode}_{\mathrm{Llama}}(T)|.$$

Count identifiers produced by the fast tokenizer for `meta-llama/Llama-3.2-3B` over the complete file, with `add_special_tokens=False` and `truncation=False`. Here decoding uses `errors='replace'`. This is separate from both the embedding tokenizer and the regex tokenization used for Brunet's index.

## 2. Schema inventory and axiom counts

### Classes — `NumClases`

**Internal:** `ttl_info['num_classes']`.

$$C=|\mathcal C|.$$

Number of distinct URI classes identified under the rules above. The result depends on declarations and eligible `rdf:type` uses present in this particular file.

### Axioms — `NumAxioms`

**Internal:** `ttl_info['num_axioms']`.

$$N_{\mathrm{ax}}=|G|.$$

Number of distinct RDF triples. An abstract OWL axiom may require several RDF triples, so this is not an OWL axiom count in the abstract syntax.

### Logical axioms — `NumLogicalAxioms`

**Internal:** `ttl_info['num_logical_axioms']`.

$$N_{\mathrm{log}}=|\{(s,p,o)\in G:p\notin\mathcal A_{\mathrm{filter}}\}|.$$

Count triples whose predicate is absent from the fixed `ANNOTATION_PROPS` filter inside `count_axioms`. That filter covers known labels, comments, definitions, synonyms, and metadata predicates. This operational approximation does not automatically exclude every property declared as `owl:AnnotationProperty` and does not perform logical axiom classification.

### Object properties — `NumObjectProperties`

**Internal:** `ttl_info['num_object_properties']`.

$$N_{\mathrm{obj}}=|\mathcal O|.$$

Distinct URI subjects explicitly typed as `owl:ObjectProperty`, irrespective of usage frequency.

### Datatype properties — `NumDatatypeProperties`

**Internal:** `ttl_info['num_datatype_properties']`.

$$N_{\mathrm{data}}=|\mathcal D|.$$

Distinct URI subjects explicitly typed as `owl:DatatypeProperty`.

### Annotation properties — `NumAnnotationProperties`

**Internal:** `ttl_info['num_annotation_properties']`.

$$N_{\mathrm{ann}}=|\mathcal A|.$$

Distinct URI subjects explicitly typed as `owl:AnnotationProperty`. Using `rdfs:label` without declaring it does not increase this count.

### Total distinct properties — `NumPropertiesCount`

**Internal:** `ttl_info['num_properties_count']`.

$$N_{\mathrm{prop}}=|\mathcal O\cup\mathcal D\cup\mathcal A|.$$

A URI declared under several property types counts once. Consequently, this total can differ from the sum of the three preceding counts.

## 3. Taxonomy shape, inheritance, and paths

This group distinguishes the broader edge set $E$ from the internal taxonomy $E_C$. Except for inheritance richness and the taxonomic relation count, the metrics in this section use $E_C$.

### Mean direct subclasses per class — `SC`

**Internal:** `ttl_info['SC']`.

$$SC=\frac{\sum_{c\in\mathcal C}|\mathrm{child}(c)|}{C}=\frac{|E_C|}{C}.$$

Average direct child count. Both ends of each counted edge must belong to $\mathcal C$.

### Inheritance richness — `IR_InheritanceRichness`

**Internal:** `ttl_info['IR_inheritance_richness']`.

$$IR=\frac{H}{C}.$$

Explicit subclass relations per identified class. Unlike `SC`, the parent may be an external URI not included in $\mathcal C$. The two values coincide when every counted parent is internal.

### Taxonomic relations — `HTaxonomicRelations`

**Internal:** `ttl_info['H_taxonomic_subClassOf']`.

$$H=|E|.$$

Count explicit `rdfs:subClassOf` triples with an identified child and a URI parent. No additional transitive relations or anonymous superclasses are counted.

### Roots — `RootNodes`

**Internal:** `ttl_info['abs_root_cardinality']`.

$$N_{\mathrm{root}}=|\mathcal R|=|\{c\in\mathcal C:|\mathrm{par}(c)|=0\}|.$$

Classes without an internal direct parent. A class with only an external URI parent is therefore an internal root.

### Leaves — `LeafNodes`

**Internal:** `ttl_info['abs_leaf_cardinality']`.

$$N_{\mathrm{leaf}}=|\mathcal L|=|\{c\in\mathcal C:|\mathrm{child}(c)|=0\}|.$$

Classes without an internal direct child. Isolated classes contribute to both the root and leaf counts.

### Absolute sibling cardinality — `AbsSiblingCardinality`

**Internal:** `ttl_info['abs_sibling_cardinality']`.

$$S_{\mathrm{sib}}=\sum_{c\in\mathcal C}|\mathrm{sib}(c)|.$$

Sum of distinct sibling counts per class. A sibling pair contributes twice, once from each endpoint. Sharing several parents does not multiply that pair's contribution.

### Absolute depth — `AbsDepth`

**Internal:** `ttl_info['abs_depth']`.

Let $\Pi(c)$ contain all paths from any root to $c$, and let $|\pi|$ be path length in edges. For an acyclic taxonomy:

$$D_{\mathrm{abs}}=\sum_{\ell\in\mathcal L}\sum_{\pi\in\Pi(\ell)}|\pi|.$$

Sum lengths of all root-to-leaf paths. Multiple inheritance can create several paths to the same leaf.

### Average depth — `AvgDepth`

**Internal:** `ttl_info['avg_depth']`.

$$D_{\mathrm{avg}}=\frac{D_{\mathrm{abs}}}{\sum_{\ell\in\mathcal L}|\Pi(\ell)|}.$$

Mean root-to-leaf path length, averaged **over paths**, rather than over classes or leaves. An isolated root contributes one zero-length path.

### Maximum depth — `MaxDepth`

**Internal:** `ttl_info['max_depth']`.

$$D_{\max}=\max_{c\in\mathcal C}\max_{\pi\in\Pi(c)}|\pi|.$$

Longest root-to-class path; root depth is zero. If any cycle is detected, the implementation sets this metric to `0.0`.

### Absolute breadth — `AbsBreadth`

**Internal:** `ttl_info['abs_breadth']`.

A multi-source breadth-first traversal starts from all roots. Let $B_k$ be the set first reached at level $k$, with $K$ visited levels:

$$B_{\mathrm{abs}}=\sum_{k=0}^{K-1}|B_k|.$$

Count classes reachable from roots, each once. For a finite acyclic taxonomy this equals $C$.

### Average breadth — `AvgBreadth`

**Internal:** `ttl_info['avg_breadth']`.

$$B_{\mathrm{avg}}=\frac{\sum_{k=0}^{K-1}|B_k|}{K}.$$

Average classes per visited breadth-first level. Each class belongs to the level of its shortest distance from any root.

### Maximum breadth — `MaxBreadth`

**Internal:** `ttl_info['max_breadth']`.

$$B_{\max}=\max_k|B_k|.$$

Size of the most populated visited level.

### Minimum breadth — `MinBreadth`

**Internal:** `ttl_info['min_breadth']`.

$$B_{\min}=\min_k|B_k|.$$

Size of the least populated visited level. All breadth metrics return zero if no level is visited.

### Leaf proportion — `RatioLeafFanoutness`

**Internal:** `ttl_info['ratio_leaf_fanoutness']`.

$$F_{\mathrm{leaf}}=\frac{|\mathcal L|}{C}.$$

Fraction of classes without internal children. This is a leaf proportion rather than a mean branching factor.

### Mean siblings per class — `RatioSiblingFanoutness`

**Internal:** `ttl_info['ratio_sibling_fanoutness']`.

$$F_{\mathrm{sib}}=\frac{S_{\mathrm{sib}}}{C}.$$

Average distinct sibling count per class. Despite “Ratio” in its name, it can exceed one.

### Multiple-inheritance proportion — `Tangledness`

**Internal:** `ttl_info['tangledness']`.

$$T_{\mathrm{multi}}=\frac{|\{c\in\mathcal C:|\mathrm{par}(c)|>1\}|}{C}.$$

Fraction of classes with more than one internal direct parent.

### Multi-parent classes — `NumMultiParentClasses`

**Internal:** `ttl_info['num_multi_parent_classes']`.

$$N_{\mathrm{multi}}=|\{c\in\mathcal C:|\mathrm{par}(c)|>1\}|.$$

Count corresponding to the numerator of `Tangledness`.

### Total number of paths — `TNOP`

**Internal:** `ttl_info['TNOP']`.

$$TNOP=\sum_{c\in\mathcal C}|\Pi(c)|.$$

Count all root-to-class paths, not only root-to-leaf paths. Each root contributes one zero-length path. This formula assumes an acyclic taxonomy.

### Average number of paths — `ANOP`

**Internal:** `ttl_info['ANOP']`.

$$ANOP=\frac{TNOP}{C}.$$

Mean number of root-to-class paths per class.

### Cycle indicator — `CycleDetected`

**Internal:** `ttl_info['cycle_detected']`.

$$I_{\mathrm{cycle}}=\begin{cases}1,&\text{a cycle is detected},\\0,&\text{otherwise}.\end{cases}$$

The recursive path traversal cuts cyclic branches. With cycles present, path counts and depths are partial traversal results rather than complete DAG measures; `MaxDepth` is forced to zero. These values must be interpreted together with the indicator.

## 4. Non-taxonomic connectivity

For an object property $p$, let $\mathrm{dom}_C(p)$ and $\mathrm{ran}_C(p)$ contain its explicitly stated domain and range classes in $\mathcal C$.

### Non-taxonomic relations — `PNonTaxonomic`

**Internal:** `ttl_info['P_nontaxonomic']`.

$$P=\sum_{p\in\mathcal O}|\mathrm{dom}_C(p)|\,|\mathrm{ran}_C(p)|.$$

Count each property–domain–range combination once. A property lacking a valid domain or range contributes zero. Different properties connecting the same classes count separately. Assertions between individuals are not counted.

If `COUNT_RESTRICTION_AXIOMS_AS_RELATIONSHIPS` is enabled, the function additionally counts valid `someValuesFrom` and `allValuesFrom` occurrences directly reached through blank-node superclass expressions; the default disables this addition.

### Mean non-taxonomic relations per class — `NTR`

**Internal:** `ttl_info['NTR']`.

$$NTR=\frac{P}{C}.$$

Average outgoing schema connections per class considered as a domain. A connection does not receive another contribution for its destination class.

### Total considered relations — `RTotalRelations`

**Internal:** `ttl_info['R_total_relationship_assertions']`.

$$R=H+P.$$

Sum taxonomic edges and non-taxonomic schema connections. This is neither the number of RDF triples nor the number of distinct properties.

### Relationship richness — `RR_RelationshipRichness`

**Internal:** `ttl_info['RR_relationship_richness']`.

$$RR=\frac{P}{H+P}.$$

Share of non-taxonomic connections among the relations considered by this implementation.

### Class-to-relation ratio — `CRR_ClassRelationRatio`

**Internal:** `ttl_info['CRR_CoverR']`.

$$CRR=\frac{C}{H+P}.$$

Classes per considered relation. It can exceed one; larger values mean fewer counted relations relative to the class inventory.

## 5. Property coverage and schema elaboration

### Property density — `PD`

**Internal:** `ttl_info['PD']`.

$$PD=\frac{\sum_{c\in\mathcal C}\bigl(d(c)+a(c)\bigr)}{C}.$$

For each class, count distinct linked datatype properties $d(c)$ and annotation properties $a(c)$. A link is established by `rdfs:domain` or direct use as a predicate on that class. Repeated values do not add properties; a link found through both routes counts once within its type.

Annotation candidates are $\mathcal A\cup\texttt{NAME_PROPS_STRICT}\cup\texttt{DESC_PROPS}$, so known annotation predicates need not be declared. Shared properties contribute for every linked class; unlinked ones contribute nothing. A URI belonging to both datatype and annotation sets contributes in both terms. Object properties are not included.

### Attribute richness — `AR_AttributeRichness`

**Internal:** `ttl_info['AR_attribute_richness']`.

Let $D_c$ be datatype properties with explicit domain $c$, and $D_{\varnothing}$ those with no valid domain in $\mathcal C$:

$$AR=\frac{\sum_{c\in\mathcal C}|D_c|+|D_{\varnothing}|}{C}.$$

A property with several valid domains contributes once per domain. Every property without a valid domain contributes once to the additional term. This differs from `PD`, which also includes annotations and direct predicate use.

### Classes with attributes — `ACR_AttributeClassRatio`

**Internal:** `ttl_info['ACR_attribute_class_ratio']`.

$$ACR=\frac{|\{c\in\mathcal C:|D_c|>0\}|}{C}.$$

Fraction of classes explicitly declared as the domain of at least one datatype property. No inherited associations or direct uses without domain declarations are added.

### Classes with equivalences — `ER_EquivalenceRatio`

**Internal:** `ttl_info['ER_equivalence_ratio']`.

$$ER=\frac{|\{c\in\mathcal C:\exists x,\ (c,\texttt{owl:equivalentClass},x)\in G\ \lor\ (x,\texttt{owl:equivalentClass},c)\in G\}|}{C}.$$

Count each class once if it occurs at either end of an explicit equivalence statement. The other end may be an anonymous class expression.

### Logical-axiom-to-class ratio — `AxiomClassRatio`

**Internal:** `ttl_info['AxiomClassRatio']`.

$$AxCR=\frac{N_{\mathrm{log}}}{C}.$$

Logical triples per class, using the same fixed annotation filter and limitations as `NumLogicalAxioms`.

### Inverse relations ratio — `IRR_InverseRelationsRatio`

**Internal:** `ttl_info['IRR_inverse_relations_ratio']`.

Let $\mathcal I_{\mathrm{inv}}$ contain URI subjects or objects of `owl:inverseOf`. The implemented ratio is:

$$IRR=\frac{|(\mathcal O\cap\mathcal I_{\mathrm{inv}})\cup(\mathcal O\cap\mathcal F^{-1})|+|\mathcal D\cap\mathcal F^{-1}|}{|\mathcal O|+|\mathcal D\cap\mathcal F|}.$$

The numerator combines object properties with an explicit inverse or inverse-functional declaration and inverse-functional datatype properties. The denominator combines all object properties and functional datatype properties. It is not `NumPropsWithInverse` divided by the total property count and is not guaranteed to stay below one.

### Functional datatype properties — `NumFunctionalDatatype`

**Internal:** `ttl_info['num_functional_datatype']`.

$$N_{\mathrm{df}}=|\mathcal D\cap\mathcal F|.$$

URI properties explicitly declared as both datatype and functional properties.

### Inverse-functional datatype properties — `NumInverseFunctionalDatatype`

**Internal:** `ttl_info['num_inverse_functional_datatype']`.

$$N_{\mathrm{dif}}=|\mathcal D\cap\mathcal F^{-1}|.$$

Count RDF declarations of both types without checking whether they are admissible under an OWL profile.

### Properties occurring in inverse statements — `NumPropsWithInverse`

**Internal:** `ttl_info['num_props_with_inverse']`.

$$N_{\mathrm{inv}}=|\mathcal I_{\mathrm{inv}}|.$$

Distinct URI endpoints of `owl:inverseOf`. An object-property declaration is not required; one statement between two distinct URIs can contribute two units.

## 6. Individuals and class population

### Individuals — `NumIndividuals`

**Internal:** `ttl_info['num_individuals']`.

$$N_{\mathrm{ind}}=|\mathcal I_{\mathrm{named}}\cup\mathcal I_{\mathrm{rule}}|.$$

$\mathcal I_{\mathrm{named}}$ contains URI subjects explicitly typed as `owl:NamedIndividual`. The additional rule includes URI subjects whose `rdf:type` object is a URI outside the `non_individual_types` set.

**Implementation detail:** that exclusion set contains the identified classes themselves. Therefore, an ordinary `x rdf:type Class` statement with `Class` in $\mathcal C$ does not by itself make `x` a counted individual. Instances not also declared as named individuals can be omitted. This metric should not be read as a complete instance count.

### Non-empty classes — `NumNonEmptyClasses`

**Internal:** `ttl_info['num_non_empty_classes']`.

$$C_+=|\{c\in\mathcal C:\exists s,\ (s,\texttt{rdf:type},c)\in G\}|.$$

Identified classes used as the type of at least one subject. That subject can be a blank node and need not belong to the individual set counted above.

### Average population — `AP_AveragePopulation`

**Internal:** `ttl_info['AP_average_population']`.

$$AP=\frac{N_{\mathrm{ind}}}{C}.$$

Counted individuals per class. This ratio of totals inherits the individual-identification limitation and does not average class-membership assertion counts.

### Class richness — `CR_ClassRichness`

**Internal:** `ttl_info['CR_class_richness']`.

$$CR=\frac{C_+}{C}.$$

Fraction of identified classes with an explicit type assertion. Instances are not propagated to superclasses by inference.

## 7. Textual vocabulary, repetition, and literal coverage

These metrics combine raw-text processing and graph counts. Regexes approximate serialization patterns; they do not replace a Turtle parser. The fixed vocabulary $\mathcal V$ is `SEMANTIC_VOCAB`. Prefix spelling and letter case affect textual matches.

### RDF/OWL vocabulary density — `VocabSpecificDensity`

**Internal:** `ttl_info['vocab_specific_density']`.

Let $\mathcal L_T$ be the list of non-empty, edge-trimmed file lines:

$$VD=\frac{\sum_{l\in\mathcal L_T}\mathrm{occurrences}_{\mathcal V}(l)}{|\mathcal L_T|}.$$

Occurrences of fixed vocabulary terms per non-empty line. Double-quoted literals are masked with line-by-line regexes before counting. Comments and prefix declarations are retained. The result can exceed one and depends on serialization layout; multiline or escaped literals are not handled as a complete Turtle grammar would handle them.

### RDF/OWL vocabulary diversity — `VocabSpecificDiversity`

**Internal:** `ttl_info['vocab_specific_diversity']`.

$$VV=\frac{|\mathcal V_{\mathrm{used}}|}{|\mathcal V|}.$$

Fraction of fixed vocabulary terms appearing as prefixed-name matches anywhere in the file. Literals and comments are not masked here. This does not measure diversity across all words or graph URIs.

### Unique textual block ratio — `LogicalBlockUniquenessRatio`

**Internal:** `ttl_info['logical_block_uniqueness_ratio']`.

Discard empty lines and lines starting with `#` or `@`, join the rest, mask literals, and split at periods followed by whitespace. For the resulting block list $\mathcal B$:

$$U_{\mathrm{block}}=\frac{|\mathrm{set}(\mathcal B)|}{|\mathcal B|}.$$

This is a textual heuristic, not a logical-equivalence check. Each literal occurrence receives a different placeholder, so otherwise identical blocks containing literals can be treated as distinct. Not all comment or prefix syntaxes are removed.

### Unique line ratio — `LineUniquenessRatio`

**Internal:** `ttl_info['line_uniqueness_ratio']`.

$$U_{\mathrm{line}}=\frac{|\mathrm{set}(\mathcal L_T)|}{|\mathcal L_T|}.$$

Distinct lines divided by non-empty lines after trimming their edges. Comments and prefix declarations remain. A value of one indicates no repeated line under this normalization.

### Brunet's index — `BrunetIndex`

**Internal:** `ttl_info['brunet_index']`.

For $N$ regex-derived text tokens and $V$ distinct tokens:

$$W=N^{V^{-0.165}}.$$

`semantic_tokenize` splits each `prefix:name` match into two tokens and extracts other word-character sequences by regex. Case is preserved. It does not use the Llama tokenizer. At fixed $N$, greater vocabulary diversity produces a lower index; if $V=0$, the result is zero.

### Literal-to-total-text character ratio — `AnnotationToStructureRatio`

**Internal:** `ttl_info['annotation_to_structure_ratio']`.

$$ASR=\frac{\sum_{(s,p,o)\in G,\ o\text{ literal}}|\mathrm{str}(o)|}{|T|}.$$

Count string characters in every literal object, irrespective of predicate, then divide by the complete serialized text length. The denominator is not structure-only text. A value is counted per triple and may recur across triples; this is not guaranteed to be a bounded annotation proportion.

### Classes with literal values — `LiteralCoverageRatio`

**Internal:** `ttl_info['literal_coverage_ratio']`.

$$LCR=\frac{|\{c\in\mathcal C:\exists p,o,\ (c,p,o)\in G,\ o\text{ literal},\ |\mathrm{strip}(\mathrm{str}(o))|>0\}|}{C}.$$

Fraction of identified classes with at least one directly associated non-empty literal. Predicates need not be known labels, comments, or declared annotation properties.

## 8. Semantic richness, quality, and task alignment

This group brings together structural richness components, the composite `QS` index, and task semantic similarity.

The richness components come from `semantic_richness_for_all_classes`. They count neighborhood structure, property associations, and annotation-predicate presence. They do not use embeddings or judge semantic correctness. The historical `Annotations` and `Synonyms` suffixes require the distinctions below.

### Mean taxonomy neighborhood — `AvgRi_Relations`

**Internal:** `ttl_info['avg_Ri']`.

$$\overline R=\frac{1}{C}\sum_{c\in\mathcal C}\left(|\mathrm{par}_E(c)|+|\mathrm{child}_E(c)|+|\mathrm{sib}_E(c)|\right).$$

Neighbors are built using $E$, so external URI parents may contribute. Parent, child, and sibling sets are added separately; the same neighbor can contribute under several roles. Siblings are deduplicated within their set.

### Mean associated properties — `AvgAi_Annotations`

**Internal:** `ttl_info['avg_Ai']`.

$$\overline A=\frac{1}{C}\sum_{c\in\mathcal C}|\mathcal P_c|.$$

$\mathcal P_c$ contains object or datatype properties with domain $c$, plus object properties with range $c$. Each property contributes once per class. Despite its column name, this is **not an annotation-property count**.

### Mean annotation-predicate coverage — `AvgSi_Synonyms`

**Internal:** `ttl_info['avg_Si']`.

Let $\mathcal A_{\mathrm{text}}=\texttt{NAME_PROPS_STRICT}\cup\texttt{DESC_PROPS}$:

$$\overline S=\frac{1}{C}\sum_{c\in\mathcal C}\sum_{p\in\mathcal A_{\mathrm{text}}}\mathbf 1[\exists o:(c,p,o)\in G].$$

Average number of known annotation predicate types present per class. The set includes labels, synonyms, comments, and descriptions. It counts predicate types, not synonym values; objects need not be literals or non-empty.

### Combined semantic richness — `SR_SemanticRichness`

**Internal:** `ttl_info['SR_semantic_richness']`.

$$SR=\overline R+\overline A+\overline S.$$

Unweighted sum of the three components without additional normalization. There is no fixed upper bound or independent guarantee of semantic quality. Although the function receives `class_index`, the current calculation does not use that textual index.


### Relative quality score — `QS`

**Internal:** `df['QS']`, computed in `save_excel`.

For each metric $x$ across all file rows in the workbook:

$$\mathrm{norm}(x_f)=\begin{cases}\dfrac{x_f-x_{\min}}{x_{\max}-x_{\min}},&x_{\max}>x_{\min},\\0,&x_{\max}=x_{\min}.\end{cases}$$

$$QS_f=\mathrm{norm}(SC_f)+\mathrm{norm}(NTR_f)+\mathrm{norm}(PD_f).$$

Each component contributes between zero and one, so $QS\in[0,3]$. The sum is not divided by three. Normalization is global across files, not separate by domain, and is recomputed whenever the workbook is saved. Intermediate values are provisional until all files have been processed.

A higher score means relatively higher subclass connectivity, non-taxonomic connectivity, and property density under these definitions. It is not an independent quality validation. This `QS` differs from the downstream `global_correlation_score`: that score uses selected metrics, learned directions, and association weights from the scoring stage.

### Task semantic similarity — `SemanticSimilarity`

**Internal:** `similarity`, supplied to `build_row` after filename lookup in `load_similarity_map`.

The ontology/text analyzer imports this value; the preceding two scripts define how it is produced in this repository.

**Ontology representation.** `compute_embeddings.py` collects distinct literal values from its configured annotation predicates, removes empty strings, URLs, and dates, and splits literals at periods. It groups text using a default chunk size of 8,192 model tokens and overlap of 2,048. Each encoded chunk uses attention-masked mean pooling of its final hidden states. The file embedding is the weighted mean of chunk vectors, with weights proportional to encoded chunk lengths. Files without usable chunks receive `null`.

The literal list written to JSON is sorted, but embedding input is iterated from a set; its order is not explicitly fixed. Overlapping tokens can contribute through more than one chunk. Actual model input is truncated to `MODEL_MAX_TOKENS`; the chunking routine should not be interpreted as a guarantee that no truncation occurs.

**Task representation.** `compute_semantic_similarity.py` reads `TASK_FILE`, splits it into whitespace-delimited **word** chunks, and encodes them with the same default model, `nomic-ai/nomic-embed-text-v1.5`. Default word chunk size and overlap are 8,192 and 2,048. Tokenization truncates inputs at 8,192 model tokens, so a word chunk may be truncated. Token hidden states are averaged per chunk, then chunk vectors are averaged with equal weights. This pooling differs from the length-weighted ontology aggregation.

**Comparison.** For nonzero task and ontology vectors $e_t$ and $e_f$:

$$\mathrm{Sim}(f)=\frac{e_t\cdot e_f}{\|e_t\|_2\,\|e_f\|_2}.$$

This is cosine similarity, theoretically in $[-1,1]$ for valid vectors. Null ontology embeddings are assigned `0.0`; this value is a fallback, not an observed cosine similarity. The code does not explicitly guard against zero-norm vectors or empty task text.

**Quartiles.** Similarity percentiles 75, 50, and 25 assign quartiles `1` through `4`, with `1` the highest. Ties go into the higher-ranked group and may yield unequal counts. Null-embedding fallback zeros participate in the thresholds. Quartile assignments and `_statistics` are stored in the JSON but are not imported as Excel metrics.

**Lookup.** The analyzer reads:

$$\mathrm{SemanticSimilarity}(f)=\mathrm{float}\bigl(\mathrm{JSON}[\mathrm{filename}(f)][\texttt{similarity}]\bigr).$$

A missing filename match leaves the Excel cell empty. Files in different directories sharing a filename use the same JSON entry. The score expresses alignment with the supplied task text and representation pipeline, rather than a universal measure of ontology quality.

## Excel output and aggregation

| Worksheet | Contents |
| --- | --- |
| `DetalleOntologías` | One row per file, with identifiers, detailed metrics, `QS`, and `Error`. |
| `ResumenporDominio` | Per-domain sums, unweighted arithmetic means, and selected extrema. |
| `EstadísticasGenerales` | Corpus-wide file/domain counts, totals, means, and extrema. |

| Identifier or status column | Meaning |
| --- | --- |
| `Dominio` | First relative-path component when the file is in a subdirectory; otherwise `unknown`. |
| `Ontología` | File basename without its extension. |
| `Archivo` | Filename including its extension. |
| `RutaRelativa` | Path relative to `BASEPATH`. |
| `Error` | Analysis or tokenization incident recorded for that row. |

“Total ontologies” counts file rows, not necessarily distinct source ontologies. Fragment totals do not deduplicate shared entities across files.

Summary means are **not token-weighted or class-weighted**. For example, average `SC` is $\sum_f SC_f/n$, rather than $\sum_f|E_{C,f}|/\sum_f C_f$. This differs from the token-weighted means computed in `scoring/compute_metric_f1_associations.py`.

Counts and volumes are generally summed. Domain summaries use the maximum `MaxDepth` and `MaxBreadth` and the minimum `MinBreadth`; `CycleDetected` has no dedicated aggregate. Token counts are summed, while `AvgSC`, `AvgNTR`, `AvgPD`, and `AvgQS` are arithmetic means.

Domain summaries are rounded to three decimal places. General statistics apply their own count/size/mean rounding. Detail rows preserve computed values except for file size, which is already rounded to two decimals. The workbook is rewritten after every processed file.

## Error handling and interpretation

A parsing or ontology-analysis failure returns zero-valued ontology metrics through `empty_analysis` and records the error. A tokenization-only failure leaves the token count empty and records the incident. Broader per-file exceptions create an error row with zero size/character counts and missing similarity.

Error rows are not automatically removed from summaries or `QS` normalization. Pandas means skip missing cells, but error-generated zeros participate. Review `Error` and `CycleDetected` before interpreting results or using the workbook in the sampling/scoring analysis.

These metrics describe complementary aspects of a serialization and its explicit graph. A large value is not uniformly desirable: direction and usefulness depend on the task, corpus composition, and implementation. The downstream scoring workflow estimates those associations empirically.
