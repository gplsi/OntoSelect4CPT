    from pathlib import Path
    from collections import defaultdict
    import shutil
    import sys
    import csv
    from tqdm import tqdm

    from rdflib import Graph, URIRef, BNode, Literal
    from rdflib.namespace import RDF, OWL, RDFS


    # =========================================================
    # CONFIG
    # =========================================================
    INPUT_DIR = Path("")

    OUTPUT_DIR = Path("")

    THRESHOLD_MB = 10.0
    TARGET_FRAGMENT_MB = 10.0
    OVERWRITE_OUTPUT = False


    # =========================================================
    # UTILIDADES
    # =========================================================

    def ensure_output_dir(output_dir: Path, overwrite: bool = False):
        if output_dir.exists() and overwrite:
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)


    def file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 ** 2)


    def graph_with_namespaces_from(source_graph: Graph) -> Graph:
        g = Graph()

        for prefix, namespace in source_graph.namespaces():
            g.bind(prefix, namespace, override=True)

        return g


    def serialize_graph_ttl(graph: Graph, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ttl = graph.serialize(format="turtle")
        out_path.write_text(ttl, encoding="utf-8")


    def graph_size_bytes(graph: Graph) -> int:
        ttl = graph.serialize(format="turtle")
        return len(ttl.encode("utf-8"))


    def safe_fragment_ontology_iri(base_iri: URIRef, fragment_idx: int) -> URIRef:
        base = str(base_iri)

        if "#" in base:
            return URIRef(f"{base}_fragment_{fragment_idx:04d}")

        return URIRef(f"{base}#fragment_{fragment_idx:04d}")


    def get_ttl_files(root: Path):
        return sorted(root.rglob("*.ttl"))




    # =========================================================
    # DETECCIÓN DE SUJETO DE ONTOLOGÍA / METADATOS
    # =========================================================

    def find_ontology_subjects(g: Graph):
        ontology_subjects = set()

        for s in g.subjects(RDF.type, OWL.Ontology):
            ontology_subjects.add(s)

        return ontology_subjects


    def collect_ontology_metadata(g: Graph, ontology_subjects: set):
        metadata = set()
        original_ontology_iri = None

        for ont_subj in ontology_subjects:
            if isinstance(ont_subj, URIRef) and original_ontology_iri is None:
                original_ontology_iri = ont_subj

            for triple in g.triples((ont_subj, None, None)):
                metadata.add(triple)

        return metadata, original_ontology_iri


    # =========================================================
    # BLOQUES ESTRUCTURALES
    # =========================================================

    def build_subject_index(g: Graph):
        subj_index = defaultdict(list)

        for triple in g:
            subj_index[triple[0]].append(triple)

        return subj_index


    def collect_block_from_subject(root_subject, subj_index, visited_subjects):
        """
        Crea un bloque estructural:
        - toma todas las tripletas cuyo sujeto es root_subject
        - si alguna tripleta apunta a un BNode como objeto, sigue ese BNode
        """
        stack = [root_subject]
        block_triples = set()

        while stack:
            current = stack.pop()

            if current in visited_subjects:
                continue

            visited_subjects.add(current)

            for triple in subj_index.get(current, []):
                block_triples.add(triple)

                _, _, obj = triple

                if isinstance(obj, BNode) and obj not in visited_subjects:
                    stack.append(obj)

        return block_triples


    def build_structural_blocks(g: Graph, ontology_subjects: set):
        """
        Construye bloques estructurales basados en sujetos.

        Primero procesa sujetos nombrados no-ontología.
        Luego procesa BNodes restantes.
        """
        subj_index = build_subject_index(g)
        visited_subjects = set()
        blocks = []

        named_subjects = [
            s for s in subj_index.keys()
            if not isinstance(s, BNode) and s not in ontology_subjects
        ]

        for s in named_subjects:
            if s not in visited_subjects:
                block = collect_block_from_subject(s, subj_index, visited_subjects)
                if block:
                    blocks.append(block)

        for s in list(subj_index.keys()):
            if s in ontology_subjects:
                continue

            if s not in visited_subjects:
                block = collect_block_from_subject(s, subj_index, visited_subjects)
                if block:
                    blocks.append(block)

        return blocks


    # =========================================================
    # TAMAÑO DE BLOQUES
    # =========================================================

    def estimate_block_size_bytes(source_graph: Graph, block_triples: set) -> int:
        temp = graph_with_namespaces_from(source_graph)

        for triple in block_triples:
            temp.add(triple)

        return graph_size_bytes(temp)


    # =========================================================
    # CABECERA DE FRAGMENTO
    # =========================================================

    def add_fragment_ontology_header(
        fragment_graph: Graph,
        metadata_triples: set,
        original_ontology_iri,
        fragment_idx: int,
        source_file_name: str
    ):
        """
        Si hay owl:Ontology original, copiamos sus metadatos con nueva IRI.
        Si no, creamos una ontología sintética mínima.
        """
        if original_ontology_iri is not None:
            new_ont_iri = safe_fragment_ontology_iri(original_ontology_iri, fragment_idx)

            for s, p, o in metadata_triples:
                if s == original_ontology_iri:
                    fragment_graph.add((new_ont_iri, p, o))
                else:
                    fragment_graph.add((s, p, o))

            fragment_graph.add((new_ont_iri, RDF.type, OWL.Ontology))
            fragment_graph.add(
                (
                    new_ont_iri,
                    RDFS.label,
                    Literal(f"{source_file_name} fragment {fragment_idx:04d}")
                )
            )

        else:
            synthetic_iri = URIRef(
                f"http://example.org/fragmented/{source_file_name}#fragment_{fragment_idx:04d}"
            )

            fragment_graph.add((synthetic_iri, RDF.type, OWL.Ontology))
            fragment_graph.add(
                (
                    synthetic_iri,
                    RDFS.label,
                    Literal(f"{source_file_name} fragment {fragment_idx:04d}")
                )
            )


    # =========================================================
    # PARTICIONADO
    # =========================================================

    def partition_blocks_by_target_size(blocks, block_sizes, target_bytes):
        """
        Estrategia greedy:
        - añade bloques hasta alcanzar target_bytes
        - si un bloque excede target_bytes por sí mismo, va solo
        """
        fragments = []
        current_blocks = []
        current_size = 0

        for block, size in zip(blocks, block_sizes):
            if not current_blocks:
                current_blocks.append(block)
                current_size = size
                continue

            if current_size + size <= target_bytes:
                current_blocks.append(block)
                current_size += size
            else:
                fragments.append(current_blocks)
                current_blocks = [block]
                current_size = size

        if current_blocks:
            fragments.append(current_blocks)

        return fragments


    # =========================================================
    # PROCESADO DE ARCHIVOS
    # =========================================================

    def parse_ttl_or_raise(ttl_path: Path) -> Graph:
        """
        Parsea directamente el archivo TTL original con RDFLib.

        No elimina owl:Axiom ni ninguna otra declaración:
        la ontología completa se conserva antes de fragmentarla.
        """
        g = Graph()

        try:
            g.parse(ttl_path, format="turtle")
            return g

        except Exception as e:
            print(
                f"  [ERROR RDFLIB] No se pudo parsear {ttl_path.name}: {e}",
                file=sys.stderr
            )
            raise


    def copy_small_ontology(
        ttl_path: Path,
        input_root: Path,
        output_root: Path,
        summary_rows: list
    ):
        rel = ttl_path.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        g = parse_ttl_or_raise(ttl_path)

        out_graph = graph_with_namespaces_from(g)

        for triple in g:
            out_graph.add(triple)

        serialize_graph_ttl(out_graph, out_path)

        summary_rows.append({
            "source_file": str(ttl_path),
            "output_file": str(out_path),
            "fragmented": "no",
            "fragment_index": "",
            "source_size_mb": f"{file_size_mb(ttl_path):.6f}",
            "output_size_mb": f"{file_size_mb(out_path):.6f}",
            "triples": len(out_graph),
            "status": "ok",
            "error": ""
        })

        print(f"[SMALL] {ttl_path.name} -> {out_path}")


    def fragment_large_ontology(
        ttl_path: Path,
        input_root: Path,
        output_root: Path,
        threshold_mb: float,
        target_fragment_mb: float,
        summary_rows: list
    ):
        rel = ttl_path.relative_to(input_root)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        size_mb = file_size_mb(ttl_path)

        if size_mb <= threshold_mb:
            copy_small_ontology(
                ttl_path=ttl_path,
                input_root=input_root,
                output_root=output_root,
                summary_rows=summary_rows
            )
            return

        print(f"[FRAGMENT] {ttl_path.name} ({size_mb:.2f} MB)")

        # Se parsea directamente la ontología original.
        # No se elimina owl:Axiom ni ninguna otra estructura.
        g = parse_ttl_or_raise(ttl_path)

        ontology_subjects = find_ontology_subjects(g)
        metadata_triples, original_ontology_iri = collect_ontology_metadata(
            g,
            ontology_subjects
        )

        blocks = build_structural_blocks(g, ontology_subjects)

        if not blocks:
            print("  [WARN] No se detectaron bloques; se copia sin fragmentar.")

            copy_small_ontology(
                ttl_path=ttl_path,
                input_root=input_root,
                output_root=output_root,
                summary_rows=summary_rows
            )
            return

        print(f"  -> Bloques estructurales detectados: {len(blocks)}")

        block_sizes = [
            estimate_block_size_bytes(g, block)
            for block in blocks
        ]

        target_bytes = int(target_fragment_mb * 1024 * 1024)

        grouped_fragments = partition_blocks_by_target_size(
            blocks,
            block_sizes,
            target_bytes
        )

        print(f"  -> Fragmentos previstos: {len(grouped_fragments)}")

        base_name = ttl_path.stem
        suffix = ttl_path.suffix

        created = 0

        for idx, fragment_blocks in enumerate(grouped_fragments, start=1):
            frag_graph = graph_with_namespaces_from(g)

            add_fragment_ontology_header(
                fragment_graph=frag_graph,
                metadata_triples=metadata_triples,
                original_ontology_iri=original_ontology_iri,
                fragment_idx=idx,
                source_file_name=base_name
            )

            for block in fragment_blocks:
                for triple in block:
                    frag_graph.add(triple)

            if len(frag_graph) == 0:
                print(f"  [WARN] Fragmento {idx} vacío; se omite.")
                continue

            out_name = f"{base_name}_frag_{idx:04d}{suffix}"
            out_path = out_dir / out_name

            serialize_graph_ttl(frag_graph, out_path)

            created += 1

            summary_rows.append({
                "source_file": str(ttl_path),
                "output_file": str(out_path),
                "fragmented": "yes",
                "fragment_index": idx,
                "source_size_mb": f"{size_mb:.6f}",
                "output_size_mb": f"{file_size_mb(out_path):.6f}",
                "triples": len(frag_graph),
                "status": "ok",
                "error": ""
            })

        print(f"  -> {created} fragmentos creados en {out_dir}")

        if created == 0:
            summary_rows.append({
                "source_file": str(ttl_path),
                "output_file": "",
                "fragmented": "yes",
                "fragment_index": "",
                "source_size_mb": f"{size_mb:.6f}",
                "output_size_mb": "",
                "triples": "",
                "status": "no_fragments_created",
                "error": "No se creó ningún fragmento aunque había bloques"
            })


    # =========================================================
    # CSV RESUMEN
    # =========================================================

    def write_summary_csv(output_dir: Path, summary_rows: list):
        csv_path = output_dir / "fragmentation_summary.csv"

        fieldnames = [
            "source_file",
            "output_file",
            "fragmented",
            "fragment_index",
            "source_size_mb",
            "output_size_mb",
            "triples",
            "status",
            "error"
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in summary_rows:
                writer.writerow(row)

        print(f"[SUMMARY] CSV guardado en: {csv_path}")


    # =========================================================
    # MAIN
    # =========================================================

    def main():
        if str(INPUT_DIR).strip() == "":
            raise ValueError("Debes definir INPUT_DIR.")

        if str(OUTPUT_DIR).strip() == "":
            raise ValueError("Debes definir OUTPUT_DIR.")

        if not INPUT_DIR.exists():
            raise FileNotFoundError(f"No existe INPUT_DIR: {INPUT_DIR}")

        ensure_output_dir(OUTPUT_DIR, overwrite=OVERWRITE_OUTPUT)

        ttl_files = get_ttl_files(INPUT_DIR)

        print(f"Ontologías TTL encontradas: {len(ttl_files)}")
        print(f"INPUT_DIR: {INPUT_DIR}")
        print(f"OUTPUT_DIR: {OUTPUT_DIR}")
        print(f"THRESHOLD_MB: {THRESHOLD_MB}")
        print(f"TARGET_FRAGMENT_MB: {TARGET_FRAGMENT_MB}")

        summary_rows = []

        for ttl_path in tqdm(ttl_files, desc="Procesando ontologías", unit="archivo"):
            try:
                fragment_large_ontology(
                    ttl_path=ttl_path,
                    input_root=INPUT_DIR,
                    output_root=OUTPUT_DIR,
                    threshold_mb=THRESHOLD_MB,
                    target_fragment_mb=TARGET_FRAGMENT_MB,
                    summary_rows=summary_rows
                )

            except Exception as e:
                print(f"[ERROR] {ttl_path}: {e}", file=sys.stderr)

                summary_rows.append({
                    "source_file": str(ttl_path),
                    "output_file": "",
                    "fragmented": "",
                    "fragment_index": "",
                    "source_size_mb": f"{file_size_mb(ttl_path):.6f}" if ttl_path.exists() else "",
                    "output_size_mb": "",
                    "triples": "",
                    "status": "error",
                    "error": str(e)
                })

        write_summary_csv(OUTPUT_DIR, summary_rows)

        ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
        error_rows = [row for row in summary_rows if row.get("status") == "error"]


        total_ok_outputs = len(ok_rows)

        print("\nEstadísticas:")
        print(f"  Salidas creadas correctamente: {total_ok_outputs}")
        print(f"  Errores: {len(error_rows)}")

        if error_rows:
            print("\nArchivos con error:")
            for row in error_rows:
                print(f"  - {row['source_file']}: {row['error']}")

        print("\nProceso terminado.")


    if __name__ == "__main__":
        main()
