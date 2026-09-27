# Data card — A3BX3 band-gap dataset (v2, in progress)

Status as of 2026-09-27. Every count below comes from a file in this repo or a query run on
that date; nothing is estimated.

## What the dataset is

GGA-PBE band gaps (eV) of **A3BX3 inverse-perovskite pnictogen halides**,
A = Mg/Ca/Sr/Ba, B = P/As/Sb/Bi, X = F/Cl/Br/I. The label is the **PBE gap**, which
underestimates the real gap (typically by 30–50 % for semiconductors). A model trained on it
predicts PBE gaps, not measured ones.

| Split | Rows | File |
|---|---|---|
| Family training set | 23 unique formulas | `data/processed/family.csv` |
| Locked holdout (Sr3BiI3, Sr3BiBr3, Sr3BiCl3) | 3 | `data/processed/holdout_sr3bix3.csv` |

Coverage of the 23 training rows: A = Mg 4 · Ca 12 · Sr 4 · Ba 3; B = P 7 · As 5 · Sb 6 ·
Bi 5; X = F 1 · Cl 9 · Br 6 · I 7. Bi appears only with Ca and Mg, so no training row is a
Sr–Bi or Ba–Bi compound: the Sr3BiX3 holdout combines an A and a B that are each present but
never together. Ca carries half the set; Ba has 3 rows.

## Sources

| Source | Status 2026-09-27 | Rows in raw | Used for training? | Functional |
|---|---|---|---|---|
| Literature, curated in v1 (`data/raw/literature/a3bx3_literature_v1.csv`, byte-identical copy of `data/a3bx3_literature.csv`) | ✓ | 26 (23 + 3 holdout) | yes | PBE (file documented as PBE-consistent) |
| Literature harvest (week 3) | not yet delivered | 0 | will be, if `functional` = PBE (or PBE+SOC with `include_soc: true`) | per row |
| **Materials Project** (Jain et al. 2013, doi 10.1063/1.4812323) | **blocked: no `MP_API_KEY`**. `python -m a3bx3.fetch.mp` exits 2 with a message; nothing was fetched or faked | 0 | not decided (design open decision 3) | PBE / PBE+U |
| **JARVIS-DFT** (Choudhary et al. 2020, doi 10.1038/s41524-020-00440-1) | ✓ fetched keyless | 10 (`data/raw/jarvis_a3bx3.json`) | **no** — OptB88vdW ≠ PBE | OptB88vdW; TBmBJ for 5 rows |
| **OQMD** via OPTIMADE (Saal et al. 2013, doi 10.1007/s11837-013-0755-4) | ✓ HTTP 200 (see below) | 23 structures, 20 formulas (`data/raw/oqmd_a3bx3.json`) | **no** — comparison only | PBE (qmpy 1.6.0) |

JARVIS: dataset `dft_3d`, file `jdft_3d-9-24-2025.json` (93,902 entries), jarvis-tools
2026.6.12, v1 selection rule unchanged. It returns 10 rows where the committed v1 file
`data/a3bx3_jarvis.csv` has 9 (built from an older `dft_3d` snapshot); the new one is
Ca3AsI3 (JVASP-169990). Five of the ten are Cd/Zn A-sites, outside the alkaline-earth family.

### OQMD re-query (design week 1)

The design recorded a 502 from OQMD's OPTIMADE endpoint on the morning of 2026-09-27 and
called it "unverified, not zero". Re-queried the same day:

- `https://optimade.oqmd.org/...` — **DNS does not resolve** (that host name is wrong, which
  may explain the earlier failure). The provider's real base URL is `https://oqmd.org/optimade/v1`.
- `https://oqmd.org/optimade/v1/structures` with
  `chemical_formula_anonymous="A3B3C" AND nelements=3 AND elements HAS ANY "Mg","Ca","Sr","Ba"
  AND elements HAS ANY "P","As","Sb","Bi" AND elements HAS ANY "F","Cl","Br","I"` →
  **HTTP 200, 23 structures, 20 unique formulas, `more_data_available: false`**. Every
  returned structure satisfies all three `HAS ANY` clauses (checked), unlike the JARVIS
  endpoint.
- 11 of the 26 literature formulas also appear in OQMD (Pm-3m, lowest hull distance):
  `data/interim/cross_source_oqmd.csv`. OQMD minus literature ranges from −0.27 eV (Ca3PCl3)
  to +0.25 eV (Sr3BiBr3); 3 of 11 differ by more than the 0.2 eV conflict threshold.
- **OQMD contains Sr3BiBr3, a locked-holdout formula.** If OQMD is ever pooled, curate's
  holdout guard must drop it; the test suite fails if a holdout formula reaches a training fold.

## Curation rules (`src/a3bx3/curate.py`)

1. The formula must match its A/B/X columns as A3B1X3, or the row is excluded with a reason
   (`data/interim/excluded.csv`; currently empty).
2. Trainable = `functional == "PBE"`; `PBE+SOC` only when `configs/v2.yaml` sets
   `include_soc: true` (default off). HSE and bare "GGA" rows never enter the target.
3. One row per formula. Duplicates are resolved by: same functional as the target → most
   recent year → carries a structure (lattice constant) → first in file order. When two
   sources differ by more than 0.2 eV, every row goes to `data/interim/conflict_log.csv`
   with the chosen one marked. Values are never averaged. Current conflicts: **0** (the
   v1 file has no duplicate formulas); the rule is exercised by a test on a synthetic
   harvest file.
4. SOC: the v1 file never recorded spin–orbit coupling, so all 23 rows are `soc = unknown`.
   Results with and without SOC-unknown rows cannot be separated until the harvest fills
   the column.

## Provenance defects (known, not hidden)

`tests/test_v2_pipeline.py::test_every_data_row_has_a_doi_or_db_id_known_issues_listed`
requires the set of rows without a well-formed DOI to equal this list exactly
(`configs/v2.yaml: known_doi_issues`). No DOI has been invented to clear it.

| Formula(s) | `doi` field | Problem |
|---|---|---|
| Ca3BiI3, Ca3BiBr3, Ca3BiCl3, Sr3BiI3, Sr3BiBr3, Sr3BiCl3 | `NJC 2026,50,522 Islam` | a citation string (Islam et al., *New J. Chem.* 2026, 50, 522), not a DOI |
| Ca3AsCl3 | `10.1016/j.mtcomm.2024` | truncated Elsevier DOI: journal and year only, no article suffix |

Fix: look each one up (CrossRef) during the week-3 harvest and replace the string with the
verified DOI, then remove it from `known_doi_issues`.

## Integrity

- `data/raw/` is write-once. `data/raw/MANIFEST.sha256` holds a SHA-256 per raw file; the
  offline pipeline refuses to run on a mismatch, and `.gitattributes` stores these files
  byte-exact (no line-ending conversion).
- Fetchers treat an outage as a failure (non-zero exit), never as an empty result.

## Licence and attribution

Values are numbers extracted from published DFT studies (attributed per row by DOI) and
from the public JARVIS-DFT and OQMD databases (cite the DOIs above). Copyright in the
original publications remains with their publishers.
