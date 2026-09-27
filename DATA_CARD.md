# Data card — A3BX3 band-gap dataset (v2, in progress)

Status as of 2026-09-27. Every count below comes from a file in this repo or a query run on
that date; nothing is estimated. The pipeline files, not this card, are the record:
`data/interim/curate_summary.json`, `data/processed/family_*.csv`.

## What the dataset is

GGA-PBE band gaps (eV) of **A3BX3 inverse-perovskite pnictogen halides**,
A = Mg/Ca/Sr/Ba, B = P/As/Sb/Bi, X = F/Cl/Br/I. The label is the **PBE gap**, which
underestimates the real gap (typically by 30–50 % for semiconductors). A model trained on it
predicts PBE gaps, not measured ones.

| Dataset | Formulas | Label | File |
|---|---|---|---|
| **primary** | **39** | median of the no-SOC PBE values per formula (57 values) | `data/processed/family_primary.csv` |
| verified_only (sensitivity) | 31 | primary without preprint rows (45 values) | `family_verified_only.csv` |
| plus_unverified (sensitivity) | 42 | primary + the `source_unverified` v1 rows (61 values) | `family_plus_unverified.csv` |
| soc (separate target) | 21 | median of the PBE+SOC values (26 values) | `family_soc.csv` |
| Locked holdout | 3 (Sr3BiI3, Sr3BiBr3, Sr3BiCl3) | v1 reference rows, Islam et al. 2026 | `holdout_sr3bix3.csv` |

**Kill checkpoint K1** (design §9, unique primary PBE formulas ≥ 35): **39 → PASS.**
Caveat: 8 of the 39 come only from one preprint (see decision 3). Without them (the
verified_only set) the count is 31, which is below 35.

Primary coverage: A = Mg 10 · Ca 10 · Sr 7 · Ba 12; B = P 13 · As 8 · Sb 12 · Bi 6;
X = F 4 · Cl 11 · Br 12 · I 12. Gap range 0.106 eV (Mg3SbI3) to 3.184 eV (Mg3SbF3). v1's
range was 0.78–2.30 eV, so MAE values are not comparable across versions. Compare
skill against the mean baseline instead. Sr–Bi still has no training row. Ba–Bi now has
three rows (Ba3BiX3), all from the preprint.

## Sources

| Source | Status 2026-09-27 | Rows in raw | Used for training? | Functional |
|---|---|---|---|---|
| Literature, curated in v1 (`data/raw/literature/a3bx3_literature_v1.csv`, byte-identical copy of `data/a3bx3_literature.csv`, never edited) | ✓ | 26 (23 + 3 holdout) | yes, after `corrections.csv` | recorded as PBE |
| **Literature harvest 2026-09-27** (`data/raw/literature/literature_harvest_2026-09-27.csv`, copied unchanged) | ✓ | 159 rows / 37 sources; 53 PBE, 26 PBE+SOC | PBE and PBE+SOC rows only | per row, quoted `evidence` column |
| Corrections (`data/raw/literature/corrections.csv`) | ✓ | 19 corrections | applied by `curate.py` to v1 rows only | — |
| **Materials Project** (doi 10.1063/1.4812323) | **blocked: no `MP_API_KEY`**. `fetch.mp` exits 2; nothing fetched or faked | 0 | not decided | PBE / PBE+U |
| **JARVIS-DFT** (doi 10.1038/s41524-020-00440-1) | ✓ keyless | 10 (`jarvis_a3bx3.json`) | **no**: OptB88vdW ≠ PBE | OptB88vdW; TBmBJ |
| **OQMD** via OPTIMADE (doi 10.1007/s11837-013-0755-4) | ✓ HTTP 200 | 23 structures, 20 formulas (`oqmd_a3bx3.json`) | **no**: comparison only | PBE (qmpy 1.6.0) |

All raw files are listed in `data/raw/MANIFEST.sha256`.

**How the harvest's key claims were checked (2026-09-27):**

- **CrossRef:**
  - `10.1039/D4RA08680C` returns 404.
  - `10.1039/d4ra09093d` is the Mg3AB3 paper (RSC Adv 15, 5766).
  - `10.1039/d5nj02800k` is Islam et al., *New J. Chem.* 50, 522–536, "…Sr3BiX3 (X = I, Br, and Cl)…".
  - `10.1134/S1063782625603371` is a Ca3AsBr3 paper.
  - `10.1007/s11082-024-07968-2` is a Ca3SbCl3-only paper.
- **Europe PMC full text of PMC11840809, Table 4:**
  - Mg3BiBr3: GGA 1.071, HSE06 1.626.
  - Mg3BiI3: GGA 0.224, HSE06 0.867.
  - The methods state GGA-PBE, and the paper says SOC was omitted.
- **arXiv API:** 2604.01942 is Adhikari, Alam and Johari, "Lead-free antiperovskite derivatives Ba3MA3…", posted 2026-04-02.

## Coordinator decisions (2026-09-27) and how they are implemented

1. **Corrections go in `corrections.csv`, never in the v1 file.** Each correction must match
   exactly one v1 row, or curate stops (tested). Log: `data/interim/corrections_applied.csv`.

   | Formula | v1 value | Action | Why |
   |---|---|---|---|
   | Mg3BiI3, Mg3BiBr3 | 0.867, 1.626 "PBE" | relabel → **HSE06** (not trainable); DOI fixed to 10.1039/d4ra09093d | Table 4 of that paper; see "v1 label errors". Primary uses its PBE values 0.224 and 1.071 |
   | Sr3BiI3/Br3/Cl3 (holdout) | DOI "NJC 2026,50,522 Islam" | DOI fixed → 10.1039/d5nj02800k | CrossRef |
   | Ca3BiI3/Br3/Cl3 | 1.29/1.60/1.76, same citation | `source_unverified`, **sensitivity only** | the cited paper covers Sr3BiX3 only. Probable source 10.1016/j.mseb.2025.118871 is paywalled |
   | Ca3AsI3 | 1.58 | **excluded**, `source_mismatch` | its DOI is a Ca3AsBr3 paper, which gives Ca3AsBr3 = 1.58 |
   | Ca3SbF3/Br3/I3 | 2.24/1.67/1.35 | **excluded**, `source_unverified` | the DOI covers Ca3SbCl3 only |
   | Ca3SbCl3 | 1.83 | value corrected → **1.814**, kept in primary, flagged `functional_unconfirmed` | the DOI's abstract gives 1.814 eV "using GGA" (flavour not named) |
   | Sr3PCl3, Sr3PBr3 | 1.70, 1.55 | kept in primary, flagged `functional_unconfirmed` | the abstract says "GGA" only; full text not reachable |
   | Ca3AsCl3 | 1.742, truncated DOI | `source_unverified`, **sensitivity only** | no Ca3AsCl3 paper found; candidates paywalled |

   Ca3SbCl3, Sr3PCl3 and Sr3PBr3 are flagged the same way for consistency. All three are
   v1 rows whose readable source text says only "GGA". The harvest's own bare-"GGA" rows are
   not trainable. This is my call beyond the coordinator's list: keeping the v1 rows (v1
   recorded them as PBE) and flagging them is less destructive than dropping them.
2. **Label policy.**
   - The primary target is GGA-PBE **without SOC**.
   - `PBE+SOC` rows form the separate `soc` target and are never pooled with no-SOC rows
     (tested).
   - v1 rows never recorded SOC. They stay in primary with `soc = unknown`, flagged
     `soc_unknown`. 23 of 39 primary formulas carry that flag.
   - When a formula has several values, the label is their **median**.
   - The same number from the same source counts once. v1 and the harvest quote 9 identical
     values: `dedupe_log.csv`.
   - The spread of values is logged in `conflict_log.csv`. A spread above 0.2 eV is flagged
     `high_conflict` and kept. Primary has 3 such formulas:

   | Formula | Values (eV) | Spread |
   |---|---|---|
   | Ca3PCl3 | 1.978 / 2.035 / 2.206 | 0.23 |
   | Ba3SbBr3 | 0.973 / 1.191 | 0.22 |
   | Mg3BiBr3 | 0.859 / 1.071 | 0.21 |

   12 primary formulas have more than one value.
3. **Preprint rows are in primary, with a sensitivity run without them.** The preprint is
   arXiv 2604.01942 (VASP, SI Table S4).
   - 8 primary formulas come only from it: Ba3AsBr3, Ba3BiBr3, Ba3BiCl3, Ba3BiI3, Ba3PBr3,
     Ba3PCl3, Ba3PI3, Ba3SbCl3.
   - 4 more formulas have a preprint value in their median.
   - `verified_only` drops every preprint row.
4. **The holdout stays locked.**
   - Only the v1 reference rows are kept for Sr3BiX3.
   - Every other row for those formulas, from any source, is excluded (tested). None is in
     the harvest, but OQMD has Sr3BiBr3.
   - **Literature range for the target:**

   | Sr3BiI3 PBE value | Code | Source |
   |---|---|---|
   | **1.324 (holdout target)** | — | Islam 2026 |
   | 1.164, a = 6.845 Å | CASTEP | 10.1039/d5ra07289a, full text |
   | 1.30 | WIEN2k | 10.1088/1361-6641/ada17e, abstract |

   The target sits between two independent reports. The literature spread for the target
   is at least 0.16 eV, as large as the models' holdout errors.
5. See `reports/metrics_v2.md` for primary and the three sensitivity datasets.
6. K1: see above.

## v1 label errors found (v1 is public; its README numbers are left as a historical record)

- **Mg3BiI3 and Mg3BiBr3 carried HSE06 gaps labelled as PBE.** v1 stored 0.867 and
  1.626 eV. In the source (RSC Adv 2025, doi 10.1039/d4ra09093d, Table 4) those are the
  HSE06 values; the GGA-PBE values are 0.224 and 1.071 eV. The row's DOI (10.1039/D4RA08680C)
  does not resolve. The elastic constants in the same rows match that paper's Table 2 exactly,
  which identifies the source. Effect: two of v1's 23 training labels were too high by 0.64 and
  0.56 eV. The v1 metrics (LOO 0.151, LOBO 0.146 eV) were computed on those labels. v2 still
  reproduces them exactly on the uncorrected file (`v1_reproduction` in `metrics_v2.json`),
  which checks the code, not the data.
- **Source mismatches.**
  - The Ca3AsI3 row (1.58) cites a Ca3AsBr3 paper.
  - The Ca3SbF3/Br3/I3 rows cite a Ca3SbCl3-only paper.
  - The Ca3SbCl3 row (1.83) does not match its own source (1.814).
  - The Ca3BiX3 rows cite the Sr3BiX3 paper.
- **Unresolvable DOI.** Ca3AsCl3's DOI is truncated (`10.1016/j.mtcomm.2024`).
- **Functional stated only as "GGA"** in the readable text for Sr3PCl3, Sr3PBr3 and Ca3SbCl3.

In short: of v1's 23 training rows, 2 had the wrong functional, 5 had a source that does not
contain the value, 3 had a citation for a different compound, and 1 had a truncated DOI.
That makes 11 rows with a defect. The fixes are in `corrections.csv`; the v1 file is unchanged.

## Other literature evidence worth knowing

- **Ba3BiI3.** The preprint gives PBE 0.786 eV. v1's family model predicted
  0.74 ± 0.15 eV, against the monotonic A-site trend (~1.60) asserted earlier. The new value
  supports the family model, but it is one preprint value. The harvest skipped two other
  Ba3BiI3 reports (1.03 and 1.43 eV) because their functional was ambiguous.
- **Label noise floor.** Code-to-code spread for one formula is 0.1–0.2 eV (Ca3PBr3 0.13,
  Ca3PI3 0.11; Ca3PCl3 0.23 across QE, WIEN2k and QuantumATK). Treat this as irreducible
  label noise; see MODEL_CARD.

### OQMD re-query (design week 1)

The design recorded a 502 from OQMD's OPTIMADE endpoint on the morning of 2026-09-27 and
called it "unverified, not zero". I re-queried the same day:

- **`https://optimade.oqmd.org`:** the host name does not resolve in DNS. The real base
  URL is `https://oqmd.org/optimade/v1`.
- **The real endpoint:** HTTP 200, **23 structures, 20 unique formulas,
  `more_data_available: false`**. Every returned structure satisfies all three
  `HAS ANY` clauses.
- **Overlap with the literature:** 15 formulas, 14 primary plus the holdout Sr3BiBr3
  (`data/interim/cross_source_oqmd.csv`). OQMD minus literature ranges from −0.25 eV
  (Ca3PF3) to +0.25 eV (Sr3BiBr3).

JARVIS: dataset `dft_3d`, file `jdft_3d-9-24-2025.json` (93,902 entries), jarvis-tools
2026.6.12. It gives 10 family rows, 5 of them Cd/Zn A-sites.

## Provenance defects still open

`test_every_data_row_has_a_doi_or_db_id_known_issues_listed` requires the rows with neither
a well-formed DOI nor an arXiv id (after corrections) to equal this list exactly. It also
checks that none of them reaches primary. No DOI has been invented.

| Formula(s) | `doi` field | Status |
|---|---|---|
| Ca3BiI3, Ca3BiBr3, Ca3BiCl3 | `NJC 2026,50,522 Islam` | citation of a Sr3BiX3-only paper; sensitivity only |
| Ca3AsCl3 | `10.1016/j.mtcomm.2024` | truncated DOI; sensitivity only |

The preprint rows have `arxiv_id` 2604.01942 and no DOI. CrossRef cannot verify the DataCite
DOI 10.48550/arXiv.2604.01942.

## Integrity

- `data/raw/` is write-once. `data/raw/MANIFEST.sha256` holds a SHA-256 per raw file. The
  offline pipeline refuses to run on a mismatch. `.gitattributes` stores these files
  byte-exact.
- Fetchers treat an outage as a failure (non-zero exit), never as an empty result.

## Licence and attribution

Values are numbers extracted from published DFT studies, attributed per row by DOI or arXiv
id, and from the public JARVIS-DFT and OQMD databases (cite the DOIs above). Copyright in the
original publications remains with their publishers.
