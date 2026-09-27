"""Load literature band gaps from data/raw/literature/*.csv (offline; no network).

Two schemas are accepted and normalised to one record layout:

v1 (the curated set shipped with v1, a3bx3_literature_v1.csv)
    formula,A,B,X,Eg_PBE_eV,C11_GPa,C12_GPa,C44_GPa,year,doi,note
    The v1 file is documented as PBE-consistent, so functional = "PBE".
    It never recorded spin-orbit coupling, so soc = "unknown".

harvest (the week-3 literature harvest, dropped in later)
    formula,A,B,X,Eg_eV,functional,soc,gap_type,code,lattice_a_A,doi,doi_verified,
    arxiv_id,year,evidence,fetch_status,note

Nothing here decides what is trainable; curate.py does that from the `functional` column
and the config's include_soc flag. A row is never dropped silently: rows that cannot be
parsed are returned with `parse_error` set so curate can log them.
"""
from __future__ import annotations

import csv
from pathlib import Path

V1_COLUMNS = ["formula", "A", "B", "X", "Eg_PBE_eV", "C11_GPa", "C12_GPa", "C44_GPa",
              "year", "doi", "note"]
HARVEST_COLUMNS = ["formula", "A", "B", "X", "Eg_eV", "functional", "soc", "gap_type", "code",
                   "lattice_a_A", "doi", "doi_verified", "arxiv_id", "year", "evidence",
                   "fetch_status", "note"]

RECORD_FIELDS = ["formula", "A", "B", "X", "Eg_eV", "functional", "soc", "gap_type",
                 "lattice_a_A", "doi", "doi_verified", "arxiv_id", "year", "note",
                 "code", "evidence", "fetch_status", "source_file", "source_row", "schema",
                 "parse_error", "status", "source_id", "preprint"]

# fetch_status values meaning "the number was read from the source text"
READ_STATUSES = {"ok", "verified", "fetched", "full_text", "abstract_only"}
CORRECTIONS_FILE = "corrections.csv"
CORRECTION_COLUMNS = ["formula", "field", "old", "new", "action", "evidence", "doi"]


def detect_schema(header: list[str]) -> str:
    cols = set(header)
    if set(HARVEST_COLUMNS) <= cols:
        return "harvest"
    if set(V1_COLUMNS) <= cols:
        return "v1"
    missing_v1 = sorted(set(V1_COLUMNS) - cols)
    missing_h = sorted(set(HARVEST_COLUMNS) - cols)
    raise ValueError(f"unrecognised literature schema; missing v1 cols {missing_v1}, "
                     f"missing harvest cols {missing_h}")


def _float(v: str | None):
    v = (v or "").strip()
    if not v:
        return None
    return float(v)


def _norm_soc(v: str | None) -> str:
    v = (v or "").strip().lower()
    if v in {"yes", "y", "true", "1", "soc"}:
        return "yes"
    if v in {"no", "n", "false", "0"}:
        return "no"
    return "unknown"


def _norm_functional(v: str | None) -> str:
    v = (v or "").strip().upper().replace(" ", "")
    # Bare "GGA" is NOT mapped to PBE: it could be PW91. Such rows stay non-trainable.
    aliases = {"GGA-PBE": "PBE", "PBE-GGA": "PBE",
               "PBE-SOC": "PBE+SOC", "GGA-PBE+SOC": "PBE+SOC"}
    return aliases.get(v, v)


def load_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        schema = detect_schema(reader.fieldnames or [])
        rows = list(reader)
    out = []
    for i, r in enumerate(rows, start=2):  # row 1 is the header
        rec = {k: None for k in RECORD_FIELDS}
        rec.update(source_file=path.name, source_row=i, schema=schema)
        try:
            rec.update(formula=r["formula"].strip(), A=r["A"].strip(), B=r["B"].strip(),
                       X=r["X"].strip(), doi=(r.get("doi") or "").strip(),
                       note=(r.get("note") or "").strip())
            rec["year"] = int(r["year"]) if (r.get("year") or "").strip() else None
            if schema == "v1":
                rec["Eg_eV"] = _float(r["Eg_PBE_eV"])
                rec["functional"] = "PBE"
                rec["soc"] = "unknown"
                rec["gap_type"] = None
                rec["doi_verified"] = None
            else:
                rec["Eg_eV"] = _float(r["Eg_eV"])
                rec["functional"] = _norm_functional(r["functional"])
                rec["soc"] = _norm_soc(r["soc"])
                rec["gap_type"] = (r.get("gap_type") or "").strip() or None
                rec["lattice_a_A"] = _float(r.get("lattice_a_A"))
                dv = (r.get("doi_verified") or "").strip().lower()
                rec["doi_verified"] = True if dv in {"yes", "true", "1", "y"} else (
                    False if dv in {"no", "false", "0", "n"} else None)
                rec["arxiv_id"] = (r.get("arxiv_id") or "").strip() or None
                rec["code"] = (r.get("code") or "").strip() or None
                rec["evidence"] = (r.get("evidence") or "").strip() or None
                status = (r.get("fetch_status") or "").strip()
                rec["fetch_status"] = status or None
                if status and status.lower() not in READ_STATUSES:
                    rec["parse_error"] = f"fetch_status={status}"
            if rec["Eg_eV"] is None and not rec["parse_error"]:
                rec["parse_error"] = "no band gap value"
        except (KeyError, ValueError) as e:
            rec["parse_error"] = f"{type(e).__name__}: {e}"
        rec["status"] = "ok"
        set_source_id(rec)
        out.append(rec)
    return out


def set_source_id(rec: dict) -> None:
    """One id per source: the lower-cased DOI when it is a DOI, else arXiv:<id>, else the raw
    citation string. A row with only an arXiv id is a preprint."""
    doi = (rec.get("doi") or "").strip()
    if doi.lower().startswith("10."):
        rec["source_id"] = doi.lower()
    elif rec.get("arxiv_id"):
        rec["source_id"] = f"arXiv:{rec['arxiv_id']}"
    else:
        rec["source_id"] = doi or None
    rec["preprint"] = not doi.lower().startswith("10.") and bool(rec.get("arxiv_id"))


def load_corrections(literature_dir: Path) -> list[dict]:
    p = Path(literature_dir) / CORRECTIONS_FILE
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CORRECTION_COLUMNS:
            raise ValueError(f"{p.name}: columns must be {CORRECTION_COLUMNS}, got {reader.fieldnames}")
        return [{k: (v or "").strip() for k, v in r.items()} for r in reader]


def load_all(literature_dir: Path) -> list[dict]:
    files = sorted(p for p in Path(literature_dir).glob("*.csv") if p.name != CORRECTIONS_FILE)
    if not files:
        raise FileNotFoundError(f"no literature CSV in {literature_dir}")
    rows: list[dict] = []
    for p in files:
        rows.extend(load_file(p))
    return rows
