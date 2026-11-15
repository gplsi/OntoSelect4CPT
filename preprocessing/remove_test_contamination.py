#!/usr/bin/env python3
"""
Remove Task C test-pair contamination from a Turtle ontology corpus.
Match explicit subclass relations using normalized labels, synonyms, and IRI
local names; remove both matched nodes and their recursive dependency closure.
Write cleaned TTL files and CSV reports of matches, removed nodes, and summaries.

Requirements: Python 3.10+, rdflib; tqdm is optional.
Install: python3 -m pip install rdflib tqdm
Configure TASKC_SOURCES, the input/output folders, and all three CSV report paths.
Use a separate output folder outside the input tree.
Run: python3 clean_taskc_ontologies.py
"""

import csv
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS


# =========================================================
# CONFIGURATION
# =========================================================

# Task C: Taxonomy Discovery (parent-child pairs)
TASKC_SOURCES = {
    "matonto": {
        "pairs": ".../matonto_test_pairs.json",
        "types": ".../matonto_test_types.txt",
    },

    # "obi": {
    #     "pairs": ".../obi_test_pairs.json",
    #     "types": ".../obi_test_types.txt",
    # },

    # "sweet": {
    #     "pairs": ".../sweet_test_pairs.json",
    #     "types": ".../sweet_test_types.txt",
    # },
}

INPUT_TTL_FOLDER = ("")

OUTPUT_TTL_FOLDER = ("")

# One row per rdfs:subClassOf relation matching a test pair.
MATCH_REPORT_CSV_PATH = ("")

# One row per removed node, broken down by test pair.
REMOVED_NODES_REPORT_CSV_PATH = ("")

# One summary row per ontology.
ONTOLOGY_SUMMARY_CSV_PATH = ("")

# RDFS comments may produce accidental matches.
INCLUDE_RDFS_COMMENT = False

# Print all removed nodes for each pair.
# This may produce extensive output for large branches.
PRINT_ALL_REMOVED_NODES = True

# If True, ontologies that cannot be parsed are not copied to the output.
SKIP_PARSE_ERRORS = True

OBOINOWL_HAS_EXACT_SYNONYM = URIRef(
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"
)


# =========================================================
# DATA MODELS
# =========================================================

@dataclass(frozen=True)
class MatchedPair:
    source_name: str
    test_parent: str
    test_child: str
    matched_parent_surface: str
    matched_child_surface: str
    parent_uri: URIRef
    child_uri: URIRef


@dataclass(frozen=True)
class RemovalTrace:
    depth: int
    reason: str
    caused_by: str
    predicate: str


# =========================================================
# NORMALIZATION AND TEST DATA LOADING
# =========================================================

def norm_text(value: str) -> str:
    """Lowercase, trim, and collapse whitespace."""
    value = value.strip().lower()
    return re.sub(r"\s+", " ", value)


def uri_localname(uri: URIRef) -> str:
    """Extract the local name from an IRI."""
    value = str(uri)

    if "#" in value:
        return value.rsplit("#", 1)[-1]

    return value.rstrip("/").rsplit("/", 1)[-1]


def load_taskc_pairs_multi_source() -> Dict[Tuple[str, str], Set[str]]:
    """Return {(parent_norm, child_norm): {source_1, source_2, ...}}."""
    pair_sources: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for source_name, paths in TASKC_SOURCES.items():
        pairs_path = Path(paths["pairs"])

        try:
            with pairs_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

        except FileNotFoundError:
            print(f"[WARN] Pairs file not found: {pairs_path}")
            continue

        except Exception as exc:
            print(f"[ERROR] Could not load {pairs_path}: {exc}")
            continue

        source_count = 0

        for item in data:
            parent = norm_text(item.get("parent", ""))
            child = norm_text(item.get("child", ""))

            if not parent or not child:
                continue

            pair_sources[(parent, child)].add(source_name)
            source_count += 1

        print(
            f"[TEST] Source={source_name} "
            f"valid pairs read={source_count}"
        )

    print(f"[TEST] Total unique pairs={len(pair_sources)}")
    return dict(pair_sources)


def debug_show_pairs(
    pair_sources: Dict[Tuple[str, str], Set[str]],
    max_examples: int = 5,
) -> None:
    if not pair_sources:
        print("[WARN] No test pairs were loaded.")
        return

    print("[TEST] Examples of loaded pairs:")

    for index, ((parent, child), sources) in enumerate(
        sorted(pair_sources.items())[:max_examples],
        start=1,
    ):
        print(
            f"  {index}. child='{child}' -> parent='{parent}' "
            f"sources={sorted(sources)}"
        )


# =========================================================
# SURFACE FORMS
# =========================================================

def collect_surface_forms(graph: Graph, node: URIRef) -> Set[str]:
    """Collect forms used to compare an RDF entity with test terms."""
    predicates = [
        RDFS.label,
        SKOS.prefLabel,
        SKOS.altLabel,
        OBOINOWL_HAS_EXACT_SYNONYM,
    ]

    if INCLUDE_RDFS_COMMENT:
        predicates.append(RDFS.comment)

    forms: Set[str] = set()

    for predicate in predicates:
        for obj in graph.objects(node, predicate):
            if not isinstance(obj, Literal):
                continue

            normalized = norm_text(str(obj))

            if normalized:
                forms.add(normalized)

    local_name = norm_text(uri_localname(node))

    if local_name:
        forms.add(local_name)

    return forms


def find_surface_pair_matches(
    child_forms: Set[str],
    parent_forms: Set[str],
    pair_sources: Dict[Tuple[str, str], Set[str]],
) -> List[Tuple[str, str, str]]:
    """Return all matches as (source_name, parent_surface, child_surface) tuples."""
    matches: Set[Tuple[str, str, str]] = set()

    for child_surface in child_forms:
        for parent_surface in parent_forms:
            sources = pair_sources.get((parent_surface, child_surface), set())

            for source_name in sources:
                matches.add(
                    (source_name, parent_surface, child_surface)
                )

    return sorted(matches)


# =========================================================
# TEST RELATION DETECTION
# =========================================================

def find_matched_test_pairs(
    graph: Graph,
    pair_sources: Dict[Tuple[str, str], Set[str]],
) -> List[MatchedPair]:
    """Find explicit child rdfs:subClassOf parent relations and check whether
    their surface forms match test pairs."""
    surface_cache: Dict[URIRef, Set[str]] = {}
    matches: Set[MatchedPair] = set()

    def forms(node: URIRef) -> Set[str]:
        if node not in surface_cache:
            surface_cache[node] = collect_surface_forms(graph, node)

        return surface_cache[node]

    for child, _, parent in graph.triples(
        (None, RDFS.subClassOf, None)
    ):
        if not isinstance(child, URIRef):
            continue

        if not isinstance(parent, URIRef):
            continue

        child_forms = forms(child)
        parent_forms = forms(parent)

        if not child_forms or not parent_forms:
            continue

        surface_matches = find_surface_pair_matches(
            child_forms=child_forms,
            parent_forms=parent_forms,
            pair_sources=pair_sources,
        )

        for source_name, parent_surface, child_surface in surface_matches:
            matches.add(
                MatchedPair(
                    source_name=source_name,
                    test_parent=parent_surface,
                    test_child=child_surface,
                    matched_parent_surface=parent_surface,
                    matched_child_surface=child_surface,
                    parent_uri=parent,
                    child_uri=child,
                )
            )

    return sorted(
        matches,
        key=lambda item: (
            item.source_name,
            item.test_parent,
            item.test_child,
            str(item.parent_uri),
            str(item.child_uri),
        ),
    )


# =========================================================
# DEPENDENCY INDEXES
# =========================================================

def build_dependency_indexes(
    graph: Graph,
):
    """Build two indexes:
    
    incoming_by_object[o]:
        (s, p) pairs for triples s p o.
    
    outgoing_by_subject[s]:
        (p, o) pairs for triples s p o."""
    incoming_by_object = defaultdict(list)
    outgoing_by_subject = defaultdict(list)

    for subject, predicate, obj in graph:
        outgoing_by_subject[subject].append((predicate, obj))

        if isinstance(obj, (URIRef, BNode)):
            incoming_by_object[obj].append((subject, predicate))

    return incoming_by_object, outgoing_by_subject


def node_to_text(node) -> str:
    if isinstance(node, BNode):
        return f"_:{node}"

    return str(node)


def node_kind(node) -> str:
    if isinstance(node, URIRef):
        return "URIRef"

    if isinstance(node, BNode):
        return "BNode"

    return type(node).__name__


def dependency_reason(predicate: URIRef) -> str:
    if predicate == RDFS.subClassOf:
        return "subclass_depends_on_removed_node"

    if predicate == OWL.equivalentClass:
        return "equivalent_class_depends_on_removed_node"

    if predicate == OWL.annotatedSource:
        return "reified_axiom_annotated_source"

    if predicate == OWL.annotatedTarget:
        return "reified_axiom_annotated_target"

    if predicate == OWL.someValuesFrom:
        return "restriction_some_values_from_removed_node"

    if predicate == OWL.allValuesFrom:
        return "restriction_all_values_from_removed_node"

    if predicate == OWL.onClass:
        return "restriction_on_class_removed_node"

    if predicate == RDF.first:
        return "rdf_list_contains_removed_node"

    return "subject_depends_on_removed_object"


# =========================================================
# RECURSIVE EXPANSION OF THE SUBGRAPH TO REMOVE
# =========================================================

def compute_removal_closure(
    matched_pair: MatchedPair,
    incoming_by_object,
    outgoing_by_subject,
) -> Dict[object, RemovalTrace]:
    """Start from the matched child and parent and expand removal in two
    controlled directions:
    
    1. Incoming dependencies:
           X predicate removed_node
    
       X is treated as dependent on the removed node and is also removed.
    
    2. Outgoing anonymous structures:
           removed_node predicate _:blank
    
       The blank node is treated as part of the removed node definition.
    
    Do not follow outgoing URIRef objects: these are often ancestors,
    properties, or shared entities, and following them could expand removal
    excessively throughout the ontology."""
    traces: Dict[object, RemovalTrace] = {}
    queue = deque()

    def add_node(
        node,
        depth: int,
        reason: str,
        caused_by: str = "",
        predicate: str = "",
    ) -> None:
        if not isinstance(node, (URIRef, BNode)):
            return

        if node in traces:
            return

        traces[node] = RemovalTrace(
            depth=depth,
            reason=reason,
            caused_by=caused_by,
            predicate=predicate,
        )
        queue.append(node)

    add_node(
        matched_pair.child_uri,
        depth=0,
        reason="matched_test_child_seed",
    )
    add_node(
        matched_pair.parent_uri,
        depth=0,
        reason="matched_test_parent_seed",
    )

    while queue:
        current = queue.popleft()
        current_depth = traces[current].depth

        # Every subject pointing to a removed node is treated as dependent on it.
        for subject, predicate in incoming_by_object.get(current, []):
            add_node(
                subject,
                depth=current_depth + 1,
                reason=dependency_reason(predicate),
                caused_by=node_to_text(current),
                predicate=str(predicate),
            )

        # Preserve the integrity of restrictions, lists, and other
        # anonymous RDF structures belonging to a definition.
        for predicate, obj in outgoing_by_subject.get(current, []):
            if not isinstance(obj, BNode):
                continue

            add_node(
                obj,
                depth=current_depth + 1,
                reason="blank_node_owned_by_removed_node",
                caused_by=node_to_text(current),
                predicate=str(predicate),
            )

    return traces


def collect_triples_to_remove(
    graph: Graph,
    removed_nodes: Set[object],
) -> Set[Tuple[object, object, object]]:
    """Collect every triple containing a removed node as:
    
    - subject: remove the complete node definition;
    - object: avoid dangling references;
    - predicate: cover this exceptional case for completeness."""
    triples_to_remove = set()

    for subject, predicate, obj in graph:
        if (
            subject in removed_nodes
            or predicate in removed_nodes
            or obj in removed_nodes
        ):
            triples_to_remove.add((subject, predicate, obj))

    return triples_to_remove


# =========================================================
# PRINTING AND REPORTS
# =========================================================

def print_pair_removal(
    ontology_relpath: Path,
    matched_pair: MatchedPair,
    traces: Dict[object, RemovalTrace],
) -> None:
    print()
    print(f"[MATCH] Ontology: {ontology_relpath}")
    print(
        "  Test pair: "
        f"child='{matched_pair.test_child}' -> "
        f"parent='{matched_pair.test_parent}' "
        f"[source={matched_pair.source_name}]"
    )
    print(
        "  RDF relation: "
        f"<{matched_pair.child_uri}> "
        f"rdfs:subClassOf "
        f"<{matched_pair.parent_uri}>"
    )
    print(f"  Associated nodes to remove: {len(traces)}")

    if not PRINT_ALL_REMOVED_NODES:
        return

    sorted_nodes = sorted(
        traces.items(),
        key=lambda item: (
            item[1].depth,
            node_to_text(item[0]),
        ),
    )

    for node, trace in sorted_nodes:
        details = [
            f"depth={trace.depth}",
            f"type={node_kind(node)}",
            f"node={node_to_text(node)}",
            f"reason={trace.reason}",
        ]

        if trace.caused_by:
            details.append(f"caused_by={trace.caused_by}")

        if trace.predicate:
            details.append(f"predicate={trace.predicate}")

        print("    - " + " | ".join(details))


def write_match_report_row(
    writer,
    ttl_path: Path,
    input_root: Path,
    matched_pair: MatchedPair,
    pair_node_count: int,
) -> None:
    writer.writerow(
        {
            "ontology_path": str(ttl_path),
            "ontology_relpath": str(ttl_path.relative_to(input_root)),
            "status": "MATCH",
            "source_name": matched_pair.source_name,
            "relation_type": "rdfs:subClassOf",
            "test_child": matched_pair.test_child,
            "test_parent": matched_pair.test_parent,
            "matched_child_surface": matched_pair.matched_child_surface,
            "matched_parent_surface": matched_pair.matched_parent_surface,
            "child_uri": str(matched_pair.child_uri),
            "parent_uri": str(matched_pair.parent_uri),
            "nodes_associated_with_pair": pair_node_count,
            "note": "TaskC taxonomy subgraph removed",
        }
    )


def write_removed_node_rows(
    writer,
    ttl_path: Path,
    input_root: Path,
    matched_pair: MatchedPair,
    traces: Dict[object, RemovalTrace],
) -> None:
    for node, trace in sorted(
        traces.items(),
        key=lambda item: (
            item[1].depth,
            node_to_text(item[0]),
        ),
    ):
        writer.writerow(
            {
                "ontology_path": str(ttl_path),
                "ontology_relpath": str(
                    ttl_path.relative_to(input_root)
                ),
                "source_name": matched_pair.source_name,
                "test_child": matched_pair.test_child,
                "test_parent": matched_pair.test_parent,
                "child_uri": str(matched_pair.child_uri),
                "parent_uri": str(matched_pair.parent_uri),
                "removed_node": node_to_text(node),
                "node_type": node_kind(node),
                "depth": trace.depth,
                "reason": trace.reason,
                "caused_by": trace.caused_by,
                "predicate": trace.predicate,
            }
        )


def write_parse_error_row(
    writer,
    ttl_path: Path,
    input_root: Path,
    error: Exception,
) -> None:
    writer.writerow(
        {
            "ontology_path": str(ttl_path),
            "ontology_relpath": str(ttl_path.relative_to(input_root)),
            "status": "PARSE_ERROR",
            "source_name": "",
            "relation_type": "",
            "test_child": "",
            "test_parent": "",
            "matched_child_surface": "",
            "matched_parent_surface": "",
            "child_uri": "",
            "parent_uri": "",
            "nodes_associated_with_pair": "",
            "note": str(error).replace("\n", " ")[:500],
        }
    )


# =========================================================
# SINGLE ONTOLOGY PROCESSING
# =========================================================

def process_ontology(
    ttl_path: Path,
    input_root: Path,
    output_root: Path,
    pair_sources: Dict[Tuple[str, str], Set[str]],
    match_writer,
    removed_nodes_writer,
):
    relpath = ttl_path.relative_to(input_root)
    out_path = output_root / relpath
    out_path.parent.mkdir(parents=True, exist_ok=True)

    graph = Graph()

    try:
        graph.parse(ttl_path.as_posix(), format="turtle")

    except Exception as exc:
        print(f"[PARSE_ERROR] {relpath}: {exc}")
        write_parse_error_row(
            writer=match_writer,
            ttl_path=ttl_path,
            input_root=input_root,
            error=exc,
        )

        if not SKIP_PARSE_ERRORS:
            out_path.write_bytes(ttl_path.read_bytes())

        return {
            "status": "PARSE_ERROR",
            "matches": 0,
            "unique_removed_nodes": 0,
            "removed_triples": 0,
            "remaining_triples": 0,
        }

    original_triples = len(graph)

    matched_pairs = find_matched_test_pairs(
        graph=graph,
        pair_sources=pair_sources,
    )

    if not matched_pairs:
        # Copy bytes unchanged to preserve the original formatting.
        out_path.write_bytes(ttl_path.read_bytes())

        return {
            "status": "UNCHANGED",
            "matches": 0,
            "unique_removed_nodes": 0,
            "removed_triples": 0,
            "remaining_triples": original_triples,
        }

    incoming_by_object, outgoing_by_subject = (
        build_dependency_indexes(graph)
    )

    all_removed_nodes: Set[object] = set()
    traces_by_pair = []

    # Compute the closure separately so that each removed node can be
    # attributed to the corresponding test pair.
    for matched_pair in matched_pairs:
        traces = compute_removal_closure(
            matched_pair=matched_pair,
            incoming_by_object=incoming_by_object,
            outgoing_by_subject=outgoing_by_subject,
        )

        traces_by_pair.append((matched_pair, traces))
        all_removed_nodes.update(traces.keys())

        print_pair_removal(
            ontology_relpath=relpath,
            matched_pair=matched_pair,
            traces=traces,
        )

        write_match_report_row(
            writer=match_writer,
            ttl_path=ttl_path,
            input_root=input_root,
            matched_pair=matched_pair,
            pair_node_count=len(traces),
        )

        write_removed_node_rows(
            writer=removed_nodes_writer,
            ttl_path=ttl_path,
            input_root=input_root,
            matched_pair=matched_pair,
            traces=traces,
        )

    triples_to_remove = collect_triples_to_remove(
        graph=graph,
        removed_nodes=all_removed_nodes,
    )

    for triple in triples_to_remove:
        graph.remove(triple)

    graph.serialize(
        destination=out_path.as_posix(),
        format="turtle",
    )

    remaining_triples = len(graph)

    print()
    print(f"[CLEANED] Ontology: {relpath}")
    print(f"  Matched pairs: {len(matched_pairs)}")
    print(f"  Unique nodes removed: {len(all_removed_nodes)}")
    print(f"  Original triples: {original_triples}")
    print(f"  Removed triples: {len(triples_to_remove)}")
    print(f"  Remaining triples: {remaining_triples}")
    print(f"  Output: {out_path}")

    return {
        "status": "CLEANED",
        "matches": len(matched_pairs),
        "unique_removed_nodes": len(all_removed_nodes),
        "removed_triples": len(triples_to_remove),
        "remaining_triples": remaining_triples,
    }


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print(
        "=== LOADING TASK C DATA "
        "(Taxonomy Discovery) ==="
    )

    pair_sources = load_taskc_pairs_multi_source()
    debug_show_pairs(pair_sources)

    if not pair_sources:
        raise RuntimeError(
            "No test pairs were loaded. Aborting."
        )

    input_root = Path(INPUT_TTL_FOLDER)
    output_root = Path(OUTPUT_TTL_FOLDER)

    if not input_root.exists():
        raise FileNotFoundError(
            f"INPUT_TTL_FOLDER does not exist: {input_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    ttl_files = sorted(input_root.rglob("*.ttl"))

    print()
    print("=== CONFIGURATION ===")
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Ontologies found: {len(ttl_files)}")
    print(f"INCLUDE_RDFS_COMMENT: {INCLUDE_RDFS_COMMENT}")
    print(f"PRINT_ALL_REMOVED_NODES: {PRINT_ALL_REMOVED_NODES}")
    print(f"SKIP_PARSE_ERRORS: {SKIP_PARSE_ERRORS}")

    match_fieldnames = [
        "ontology_path",
        "ontology_relpath",
        "status",
        "source_name",
        "relation_type",
        "test_child",
        "test_parent",
        "matched_child_surface",
        "matched_parent_surface",
        "child_uri",
        "parent_uri",
        "nodes_associated_with_pair",
        "note",
    ]

    removed_node_fieldnames = [
        "ontology_path",
        "ontology_relpath",
        "source_name",
        "test_child",
        "test_parent",
        "child_uri",
        "parent_uri",
        "removed_node",
        "node_type",
        "depth",
        "reason",
        "caused_by",
        "predicate",
    ]

    summary_fieldnames = [
        "ontology_path",
        "ontology_relpath",
        "status",
        "matches",
        "unique_removed_nodes",
        "removed_triples",
        "remaining_triples",
    ]

    match_report_path = Path(MATCH_REPORT_CSV_PATH)
    removed_nodes_report_path = Path(
        REMOVED_NODES_REPORT_CSV_PATH
    )
    summary_report_path = Path(ONTOLOGY_SUMMARY_CSV_PATH)

    match_report_path.parent.mkdir(parents=True, exist_ok=True)
    removed_nodes_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    totals = {
        "CLEANED": 0,
        "UNCHANGED": 0,
        "PARSE_ERROR": 0,
        "matches": 0,
        "unique_removed_nodes_sum": 0,
        "removed_triples": 0,
    }

    with (
        match_report_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as match_file,
        removed_nodes_report_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as removed_nodes_file,
        summary_report_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as summary_file,
    ):
        match_writer = csv.DictWriter(
            match_file,
            fieldnames=match_fieldnames,
        )
        removed_nodes_writer = csv.DictWriter(
            removed_nodes_file,
            fieldnames=removed_node_fieldnames,
        )
        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=summary_fieldnames,
        )

        match_writer.writeheader()
        removed_nodes_writer.writeheader()
        summary_writer.writeheader()

        for ttl_path in tqdm(
            ttl_files,
            desc="Removing Task C contamination",
            unit="ttl",
        ):
            result = process_ontology(
                ttl_path=ttl_path,
                input_root=input_root,
                output_root=output_root,
                pair_sources=pair_sources,
                match_writer=match_writer,
                removed_nodes_writer=removed_nodes_writer,
            )

            status = result["status"]
            totals[status] += 1
            totals["matches"] += result["matches"]
            totals["unique_removed_nodes_sum"] += (
                result["unique_removed_nodes"]
            )
            totals["removed_triples"] += (
                result["removed_triples"]
            )

            summary_writer.writerow(
                {
                    "ontology_path": str(ttl_path),
                    "ontology_relpath": str(
                        ttl_path.relative_to(input_root)
                    ),
                    **result,
                }
            )

    print()
    print("=== FINAL SUMMARY ===")
    print(f"Ontologies processed: {len(ttl_files)}")
    print(f"Ontologies cleaned: {totals['CLEANED']}")
    print(f"Unchanged ontologies: {totals['UNCHANGED']}")
    print(f"Parse errors: {totals['PARSE_ERROR']}")
    print(f"Total pair matches: {totals['matches']}")
    print(
        "Sum of unique removed nodes across ontologies: "
        f"{totals['unique_removed_nodes_sum']}"
    )
    print(
        f"Total triples removed: "
        f"{totals['removed_triples']}"
    )
    print()
    print(f"Match report: {match_report_path}")
    print(
        f"Detailed removed-node report: "
        f"{removed_nodes_report_path}"
    )
    print(f"Summary report: {summary_report_path}")
    print(f"Output ontologies: {output_root}")


if __name__ == "__main__":
    main()
