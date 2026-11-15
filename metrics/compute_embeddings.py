#!/usr/bin/env python3
"""
Extract ontology annotations and compute one embedding per TTL file.

How it works
------------
Read TTL files directly inside INPUT_FOLDER (subdirectories are not scanned).
Collect unique literals from the configured annotation predicates, excluding
URLs, empty strings, and date/datetime values. Split literals at periods and
group text into token-based chunks with the configured overlap. Encode each
chunk using MODEL_NAME and attention-masked mean pooling, then average the
chunk embeddings with weights proportional to their encoded token counts.
Use CUDA when available, otherwise CPU.

Write two filename-keyed JSON objects: ontology embeddings (or null for files
without usable chunks) and sorted annotation literals. Append processing
statistics to ontology_embedding_log_2.txt beside the embeddings JSON.
This script produces embeddings; it does not compute similarity scores.

Setup and execution
-------------------
1. Install dependencies:
       python3 -m pip install rdflib tqdm torch transformers numpy einops
2. Set INPUT_FOLDER, OUTPUT_FILE_EMB, and OUTPUT_FILE_LITS below.
   Create the output parent directories before running.
3. Run:
       python3 compute_ontology_embeddings.py
   Optionally override the configured output files:
       python3 compute_ontology_embeddings.py --output_emb EMBEDDINGS_JSON \
           --output_lits LITERALS_JSON

The model/tokenizer are downloaded on first use. Loading this model uses
trust_remote_code=True. Output JSON files are overwritten on each run.
"""

import os
import json
import re
import logging
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDFS, DC, DCTERMS, SKOS, FOAF, PROV, XSD
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# =========================
# CONFIGURATION
# =========================

# Directory containing the TTL ontologies (non-recursive).
INPUT_FOLDER = ""

# Output JSON files.
OUTPUT_FILE_EMB = ""
OUTPUT_FILE_LITS = ""

# Embedding model.
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Maximum model context length.
MODEL_MAX_TOKENS = 8192

# Chunking
CHUNK_TOKENS = 8192
CHUNK_OVERLAP = 2048 #Overlap used in the latest run on fragments from the 10 random datasets.

# =========================
# Optionally suppress noisy RDFLib warnings in the console.
# =========================
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

# =========================
# Regular expressions for filtering noise (URLs and dates).
# =========================
URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)

DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"([Tt ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?$"
)

# =========================
# Namespaces / Properties
# =========================
OBO = Namespace("http://www.geneontology.org/formats/oboInOwl#")
IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")
SCHEMA = Namespace("http://schema.org/")
VS = Namespace("http://www.w3.org/2003/06/sw-vocab-status/ns#")

PROPERTIES = [
    RDFS.label, RDFS.comment, RDFS.seeAlso, RDFS.isDefinedBy,
    SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel, SKOS.definition, SKOS.example,
    SKOS.note, SKOS.scopeNote, SKOS.editorialNote, SKOS.historyNote, SKOS.changeNote,
    DC.title, DC.description, DC.subject, DC.creator, DC.publisher, DC.source,
    DCTERMS.title, DCTERMS.description, DCTERMS.abstract, DCTERMS.subject,
    OBO.hasExactSynonym, OBO.hasBroadSynonym, OBO.hasNarrowSynonym,
    OBO.hasRelatedSynonym, OBO.hasDefinition, OBO.hasDbXref,
    IAO["0000115"], IAO["0000116"], IAO["0000117"], IAO["0000118"],
    SCHEMA.name, SCHEMA.description, SCHEMA.alternateName,
    FOAF.name, FOAF.page, FOAF.depiction,
    PROV.definition,
    VS.term_status, VS.usageNote
]

# =========================
# Utilities
# =========================
def get_all_ttl_files(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".ttl")
    ]


def is_noise_text(text: str) -> bool:
    """Filter common annotation noise: URLs, ISO dates/datetimes, and empty strings."""
    if text is None:
        return True

    t = text.strip()
    if not t:
        return True

    if URL_RE.match(t):
        return True

    if DATE_RE.match(t):
        return True

    return False


def literal_to_clean_str(o: Literal) -> str:
    """Safely convert an RDF literal to a string."""
    try:
        return str(o)
    except Exception:
        return o.n3()


def extract_literals_from_ttl(file_path):
    """Return a set of filtered annotation literal strings."""
    g = Graph()
    g.parse(file_path, format="turtle")

    literals = set()

    for _, p, o in g:
        if p not in PROPERTIES or o is None:
            continue

        if not isinstance(o, Literal):
            continue

        # Exclude literals explicitly typed as date or dateTime.
        if o.datatype in (XSD.dateTime, XSD.date):
            continue

        txt = literal_to_clean_str(o)

        # Exclude URLs and dates even when they are incorrectly typed.
        if is_noise_text(txt):
            continue

        literals.add(txt)

    return literals


def split_literals(literals):
    """Split long literals into smaller parts at periods."""
    parts = []
    for lit in literals:
        chunks = [p.strip() for p in lit.split(".") if p.strip()]
        parts.extend(chunks)
    return parts


def chunk_texts_tokenwise(texts, tokenizer, chunk_size, overlap):
    """Group text into token-based chunks and retain overlap between windows."""
    if not texts:
        return []

    chunks = []
    current_chunk = []
    current_ids = []
    for text in texts:
        input_ids = tokenizer(text, add_special_tokens=False).input_ids
        n = len(input_ids)
        if n == 0:
            continue

        # Split literals longer than the chunk size into separate token windows.
        if n > chunk_size:
            start = 0
            while start < n:
                end = min(start + chunk_size, n)
                chunk_ids = input_ids[start:end]
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                if end == n:
                    break
                start = max(0, end - overlap)
            continue

        # Close the current chunk if adding this literal exceeds chunk_size.
        if current_ids and (len(current_ids) + n > chunk_size):
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # Retain the last 'overlap' tokens for the next chunk.
            if overlap > 0 and len(current_ids) > overlap:
                # Reconstruct text from the overlap tokens.
                overlap_ids = current_ids[-overlap:]
                overlap_text = tokenizer.decode(overlap_ids, skip_special_tokens=True).strip()
                current_chunk = [overlap_text] if overlap_text else []
                current_ids = overlap_ids.copy()
            else:
                current_chunk = []
                current_ids = []

        current_chunk.append(text)
        current_ids.extend(input_ids)

    # Append the final chunk if any text remains.
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def mean_pool(last_hidden, attention_mask):
    """Apply attention-masked mean pooling to token representations."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_chunks(chunks, tokenizer, model, device):
    embeddings = []
    lengths = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=MODEL_MAX_TOKENS,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            pooled = mean_pool(output.last_hidden_state, inputs["attention_mask"])
            emb = pooled.squeeze(0).cpu().numpy()

        embeddings.append(emb)
        lengths.append(int(inputs["attention_mask"].shape[1]))

    return embeddings, lengths


# =========================
# Main pipeline
# =========================
def process_folder(folder):
    if not folder.strip():
        raise ValueError("Set INPUT_FOLDER before running.")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input directory not found: {folder}")

    ttl_files = get_all_ttl_files(folder)

    # Optional command-line overrides:
    #   --output_emb EMBEDDINGS_JSON
    #   --output_lits LITERALS_JSON
    output_emb = OUTPUT_FILE_EMB
    output_lits = OUTPUT_FILE_LITS
    import sys
    for i, arg in enumerate(sys.argv):
        if arg == "--output_emb" and i + 1 < len(sys.argv):
            output_emb = sys.argv[i + 1]
        if arg == "--output_lits" and i + 1 < len(sys.argv):
            output_lits = sys.argv[i + 1]


    if not output_emb.strip() or not output_lits.strip():
        raise ValueError(
            "Set OUTPUT_FILE_EMB and OUTPUT_FILE_LITS or provide "
            "--output_emb and --output_lits."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval().to(device)

    log_file = os.path.join(os.path.dirname(output_emb), "ontology_embedding_log_2.txt")
    with open(output_emb, "w", encoding="utf-8") as f_emb, \
         open(output_lits, "w", encoding="utf-8") as f_lits, \
         open(log_file, "a", encoding="utf-8") as f_log:
        f_emb.write("{\n")
        f_lits.write("{\n")
        first = True

        for ttl_file in tqdm(ttl_files, desc="Processing ontologies"):
            base_name = os.path.basename(ttl_file)

            # 1) Extract filtered annotation literals.
            literals = extract_literals_from_ttl(ttl_file)

            # Save literals as a sorted list for stable JSON output.
            literals_list = sorted(literals)

            # 2) Compute embeddings.
            split_lits = split_literals(literals)
            chunks = chunk_texts_tokenwise(
                split_lits,
                tokenizer,
                chunk_size=CHUNK_TOKENS,
                overlap=CHUNK_OVERLAP,
            )

            # Optional: limit processing to the first 1,000 chunks.
            # if len(chunks) > 1000:
            #     chunks = chunks[:1000]

            # Count all tokens in the split annotation literals.
            total_tokens = 0
            for lit in split_lits:
                total_tokens += len(tokenizer(lit, add_special_tokens=False).input_ids)

            # Report the chunk count, file size, and literal token count.
            file_size = os.path.getsize(ttl_file)
            log_line = (
                f"File: {base_name} | Size: {file_size} bytes | "
                f"Chunks: {len(chunks)} | Literal tokens: {total_tokens}"
            )
            print(log_line)
            f_log.write(log_line + "\n")
            f_log.flush()

            embeddings, lengths = embed_chunks(chunks, tokenizer, model, device)

            if embeddings:
                weights = np.array(lengths, dtype=np.float64)
                weights_sum = weights.sum()
                if weights_sum > 0:
                    weights /= weights_sum

                stacked = np.vstack(embeddings)
                ontology_embedding = (stacked * weights[:, None]).sum(axis=0)
                emb_value = ontology_embedding.tolist()
            else:
                emb_value = None

            # 3) Stream both JSON outputs to disk.
            if not first:
                f_emb.write(",\n")
                f_lits.write(",\n")
            else:
                first = False

            json.dump(base_name, f_emb)
            f_emb.write(": ")
            json.dump(emb_value, f_emb)

            json.dump(base_name, f_lits, ensure_ascii=False)
            f_lits.write(": ")
            json.dump(literals_list, f_lits, ensure_ascii=False)

            f_emb.flush()
            f_lits.flush()

        f_emb.write("\n}\n")
        f_lits.write("\n}\n")

    print(f"Embeddings saved to: {output_emb}")
    print(f"Literals saved to: {output_lits}")
    return None


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_folder(INPUT_FOLDER)
