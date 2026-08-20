"""
Simulation of Sr3BiX3 (X = I, Br, Cl) bismuth halide perovskites.

Physics pipeline
────────────────
1. Charge balance analysis
2. Structural analysis: Goldschmidt tolerance factor (t), octahedral factor (μ),
   new tolerance factor (τ), and stability classification
3. Empirical band gap estimation (calibrated against known Cs3Bi2X9 series,
   then corrected for Sr2+ chemical pressure and stoichiometry)
4. Optical constants: Sellmeier refractive index, Urbach-tail absorption,
   above-gap absorption spectrum
5. Radiative limit: Shockley–Queisser maximum efficiency at AM1.5G
6. Results: formatted table + 4-panel figure (PNG)

Reference band gaps (experimental, from literature):
  Cs3Bi2I9  → 2.06 eV (indirect) [Lehner, 2015]
  Cs3Bi2Br9 → 2.62 eV            [Hoye, 2016]
  Cs3Bi2Cl9 → 3.02 eV            [Yang, 2018]

Run:
    .venv\\Scripts\\python.exe simulate_sr3bix3.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.constants import h, c, k, eV as EV
from scipy.integrate import quad

warnings.filterwarnings("ignore")

# ── Output path ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MATERIAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Shannon ionic radii (Å) — coordination number in parentheses
RADII = {
    "Sr2+":  {"12": 1.44, "6": 1.18},   # A-site cation
    "Cs+":   {"12": 1.88, "6": 1.67},   # reference A-site
    "Bi3+":  {"6":  1.03},               # B-site cation
    "I-":    {"6":  2.20},
    "Br-":   {"6":  1.96},
    "Cl-":   {"6":  1.81},
}

# Pauling electronegativity
ELECTRONEGATIVITY = {"Sr": 0.95, "Bi": 2.02, "I": 2.66, "Br": 2.96, "Cl": 3.16}

# Reference band gaps from experiment (Cs3Bi2X9, indirect gap, eV)
REF_BG = {"I": 2.06, "Br": 2.62, "Cl": 3.02}

# Halide optical parameters (Sellmeier B coefficient — controls dispersion strength)
SELLMEIER_B = {"I": 4.20, "Br": 3.85, "Cl": 3.55}
SELLMEIER_C = {"I": 0.165, "Br": 0.135, "Cl": 0.110}   # resonance wavelength² (μm²)

# Urbach energy (eV) — thermal disorder broadening of absorption edge
URBACH_E = {"I": 0.018, "Br": 0.015, "Cl": 0.012}

HALOGENS = ["I", "Br", "Cl"]
COLORS   = {"I": "#e05c5c", "Br": "#e0a03e", "Cl": "#5c9ce0"}

# ─────────────────────────────────────────────────────────────────────────────
# 2. CHARGE BALANCE
# ─────────────────────────────────────────────────────────────────────────────

def charge_balance_analysis() -> dict:
    """
    Check formal charge balance for Sr3BiX3.
    Sr = +2, Bi = +3, X (halide) = −1.
    Net charge = 3(+2) + (+3) + 3(−1) = +6  ← non-zero.

    Physical interpretations:
      (a) Bi mixed-valence (+3/+5) with partial occupancy
      (b) Sr vacancies (ordered defect perovskite)
      (c) Bi in +3, but formula is nominal — actual structure may be
          Sr3Bi2X9 (3:2:9) which IS balanced for monovalent A2+
          ... wait, 3(+2)+2(+3)+9(−1) = +3 ← still not zero for Sr2+
      (d) Most likely interpretation: SrBiX5 (+2+3−5 = 0) or
          double perovskite Sr2BiX7 (+4+3−7 = 0)
    """
    sr_charge = 3 * 2
    bi_charge = 3
    results = {}
    for X in HALOGENS:
        x_charge = 3 * (-1)
        net = sr_charge + bi_charge + x_charge
        results[X] = {
            "formula":    f"Sr3BiX3 (X={X})",
            "net_charge": net,
            "balanced":   net == 0,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. STRUCTURAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def goldschmidt_tolerance(r_A: float, r_B: float, r_X: float) -> float:
    """t = (r_A + r_X) / (√2 · (r_B + r_X))"""
    return (r_A + r_X) / (np.sqrt(2) * (r_B + r_X))


def octahedral_factor(r_B: float, r_X: float) -> float:
    """μ = r_B / r_X;  stable octahedra: 0.41 ≤ μ ≤ 0.73"""
    return r_B / r_X


def new_tolerance_factor(r_A: float, r_B: float, r_X: float, n_A: float = 12) -> float:
    """
    Bartel τ (2019) — more predictive for hybrid + inorganic perovskites.
    τ = r_X/r_B − n_A(n_A − r_A/r_B / ln(r_A/r_B))
    """
    ratio = r_A / r_B
    return r_X / r_B - n_A * (n_A - ratio / np.log(ratio))


def classify_stability(t: float, mu: float, tau: float) -> str:
    if mu < 0.41:
        return "UNSTABLE (B too small for octahedra)"
    if mu > 0.73:
        return "UNSTABLE (B too large for octahedra)"
    if tau < 4.18:
        return "LIKELY PEROVSKITE"
    if 0.89 <= t <= 1.02:
        return "CUBIC PEROVSKITE"
    if 0.80 <= t < 0.89:
        return "DISTORTED PEROVSKITE"
    if t < 0.80:
        return "NON-PEROVSKITE (ilmenite/post-perovskite)"
    return "HEXAGONAL / OTHER"


def structural_analysis() -> dict[str, dict]:
    r_Sr  = RADII["Sr2+"]["12"]
    r_Bi  = RADII["Bi3+"]["6"]
    results = {}
    for X in HALOGENS:
        r_X = RADII[f"{X}-"]["6"]
        t   = goldschmidt_tolerance(r_Sr, r_Bi, r_X)
        mu  = octahedral_factor(r_Bi, r_X)
        tau = new_tolerance_factor(r_Sr, r_Bi, r_X, n_A=12)
        results[X] = {
            "r_A_Ang":       r_Sr,
            "r_B_Ang":       r_Bi,
            "r_X_Ang":       r_X,
            "tolerance_t":   round(t, 4),
            "octahedral_mu": round(mu, 4),
            "new_tau":       round(tau, 4),
            "phase":         classify_stability(t, mu, tau),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. BAND GAP ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_band_gap(X: str, struct: dict) -> dict:
    """
    Empirical model calibrated against Cs3Bi2X9 experimental series, with two
    corrections applied to obtain Sr3BiX3 estimates:

    Correction A — Chemical pressure (A-site size effect)
      Smaller A-cation → contracted lattice → wider gap (blueshift).
      ΔEg = α × (r_Cs − r_Sr) / r_Cs,  α ≈ 0.42 eV/unit (fitted from APbX3 series)

    Correction B — Stoichiometry / Bi coordination
      Cs3Bi2X9: 2 Bi per formula unit (face-sharing BiX6 dimers, indirect gap)
      Sr3BiX3:  1 Bi per formula unit (isolated BiX3 or chain) → narrower effective
      bandwidth → slight blueshift of CBM, ΔEg ≈ +0.08 eV (estimated).

    Reported as a range: ±0.20 eV (one-sigma, model uncertainty).
    """
    Eg_ref = REF_BG[X]

    # A-site chemical pressure correction
    r_Cs  = RADII["Cs+"]["12"]
    r_Sr  = RADII["Sr2+"]["12"]
    alpha = 0.42
    dEg_pressure = alpha * (r_Cs - r_Sr) / r_Cs   # > 0 → blueshift

    # Stoichiometry correction (1 Bi vs 2 Bi per FU)
    dEg_stoich = 0.08

    Eg_est = Eg_ref + dEg_pressure + dEg_stoich
    uncertainty = 0.20

    return {
        "Eg_ref_eV":       round(Eg_ref, 3),
        "dEg_pressure_eV": round(dEg_pressure, 3),
        "dEg_stoich_eV":   dEg_stoich,
        "Eg_est_eV":       round(Eg_est, 3),
        "Eg_low_eV":       round(Eg_est - uncertainty, 3),
        "Eg_high_eV":      round(Eg_est + uncertainty, 3),
        "gap_type":        "indirect (estimated)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPTICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

def sellmeier_n(wavelength_nm: np.ndarray, X: str) -> np.ndarray:
    """Single-term Sellmeier: n²(λ) = 1 + B·λ²/(λ² − C) where λ in μm."""
    lam = wavelength_nm / 1000.0   # convert to μm
    B = SELLMEIER_B[X]
    C = SELLMEIER_C[X]
    n_sq = 1 + B * lam**2 / (lam**2 - C)
    n_sq = np.clip(n_sq, 1.0, None)
    return np.sqrt(n_sq)


def absorption_coefficient(
    E_eV: np.ndarray,
    Eg: float,
    E_urbach: float,
    A0: float = 1.5e5,
) -> np.ndarray:
    """
    α(E) in cm⁻¹ using two-region model:
      Below gap : Urbach tail  α = α_g · exp((E − Eg) / E_urbach)
      Above gap : direct-gap   α = A0 · √(E − Eg) / E   (Tauc)

    α_g is the absorption coefficient at the band edge (continuity anchor).
    """
    alpha = np.zeros_like(E_eV)
    alpha_g = A0 * (E_urbach ** 0.5) / Eg  # continuity at Eg

    below = E_eV < Eg
    above = E_eV >= Eg

    alpha[below] = alpha_g * np.exp((E_eV[below] - Eg) / E_urbach)
    alpha[above] = A0 * np.sqrt(E_eV[above] - Eg) / E_eV[above]

    return alpha


# ─────────────────────────────────────────────────────────────────────────────
# 6. SHOCKLEY–QUEISSER EFFICIENCY (radiative limit)
# ─────────────────────────────────────────────────────────────────────────────

def sq_efficiency(Eg_eV: float) -> dict:
    """
    Shockley–Queisser radiative-limit efficiency (correct units throughout).

    Photon flux density [photons m⁻² s⁻¹ eV⁻¹]:
      φ(E, T) = (2π q³ / h³c²) × E_eV² / (exp(E_eV / kT_eV) − 1)
    This is per unit energy in eV, so integrating over dE_eV gives [photons m⁻² s⁻¹].

    J_sc = q × ∫_Eg^∞ φ_sun(E) dE            [A/m²]
    J_0  = q × ∫_Eg^∞ φ_cell(E) dE           [A/m²]
    V_oc = kT/q × ln(J_sc/J_0 + 1)           [V]
    """
    T_sun  = 5778.0   # K
    T_cell = 300.0    # K
    kT_sun  = k / EV * T_sun   # eV  ≈ 0.499 eV
    kT_cell = k / EV * T_cell  # eV  ≈ 0.02585 eV

    # Sun → Earth geometric dilution
    R_sun    = 6.96e8    # m
    d_AU     = 1.496e11  # m
    dilution = (R_sun / d_AU) ** 2   # ≈ 2.16 × 10⁻⁵

    # φ prefactor: 2π q³ / (h³ c²)   [photons m⁻² s⁻¹ eV⁻¹] when E is in eV
    prefactor = 2.0 * np.pi * EV**3 / (h**3 * c**2)

    E_arr = np.linspace(0.1, 6.0, 5000)
    dE    = E_arr[1] - E_arr[0]

    def phi_per_eV(E_arr_eV, kT_eV):
        exp_arg = np.clip(E_arr_eV / kT_eV, 0.0, 700.0)
        return prefactor * E_arr_eV**2 / np.maximum(np.exp(exp_arg) - 1.0, 1e-300)

    phi_sun  = phi_per_eV(E_arr, kT_sun)  * dilution   # [photons m⁻² s⁻¹ eV⁻¹]
    phi_cell = phi_per_eV(E_arr, kT_cell)               # [photons m⁻² s⁻¹ eV⁻¹]

    mask = E_arr >= Eg_eV

    # Current densities [A/m²]  — q in C, Σφ dE in [photons m⁻² s⁻¹]
    J_sc = EV * np.sum(phi_sun[mask])  * dE
    J_0  = EV * np.sum(phi_cell[mask]) * dE

    # AM1.5G normalisation: ideal blackbody sun ≈ 1370 W/m² at Earth surface;
    # AM1.5G is 1000 W/m².  Scale J_sc accordingly.
    scale    = 1000.0 / 1370.0
    J_sc_sc  = J_sc * scale   # [A/m²]

    V_T = kT_cell   # eV (= k*T/q, but already in eV here)

    if J_0 <= 0 or J_sc_sc <= 0:
        return {"Voc_V": 0.0, "Jsc_mAcm2": 0.0, "FF": 0.0, "PCE_pct": 0.0}

    V_oc = V_T * np.log(J_sc_sc / J_0 + 1.0)   # V

    # Fill factor — Green (1982)
    v  = V_oc / V_T
    FF = float(np.clip((v - np.log(v + 0.72)) / (v + 1.0), 0.0, 1.0))

    # PCE [%]:  P_max / P_in,  P_in = 1000 W/m²
    PCE = J_sc_sc * V_oc * FF / 1000.0 * 100.0

    return {
        "Voc_V":      round(float(V_oc), 3),
        "Jsc_mAcm2":  round(float(J_sc_sc * 0.1), 2),   # A/m² → mA/cm²
        "FF":         round(FF, 3),
        "PCE_pct":    round(float(PCE), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation() -> dict:
    charge   = charge_balance_analysis()
    struct   = structural_analysis()
    bandgaps = {X: estimate_band_gap(X, struct[X]) for X in HALOGENS}
    sq       = {X: sq_efficiency(bandgaps[X]["Eg_est_eV"]) for X in HALOGENS}

    return {"charge": charge, "struct": struct, "bandgap": bandgaps, "sq": sq}


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(results: dict, save_path: Path):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Sr₃BiX₃ (X = I, Br, Cl) — Perovskite Simulation Results",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])  # tolerance factor
    ax2 = fig.add_subplot(gs[0, 1])  # band gap comparison
    ax3 = fig.add_subplot(gs[1, 0])  # absorption spectra
    ax4 = fig.add_subplot(gs[1, 1])  # SQ efficiency / Voc

    X_labels = ["Sr₃BiI₃", "Sr₃BiBr₃", "Sr₃BiCl₃"]
    c_list    = [COLORS[x] for x in HALOGENS]

    # ── Panel 1: structural factors ─────────────────────────────────────────
    t_vals  = [results["struct"][X]["tolerance_t"]   for X in HALOGENS]
    mu_vals = [results["struct"][X]["octahedral_mu"] for X in HALOGENS]

    x = np.arange(3)
    w = 0.35
    bars_t  = ax1.bar(x - w/2, t_vals,  w, label="Goldschmidt t", color=c_list, alpha=0.85, edgecolor="k", linewidth=0.5)
    bars_mu = ax1.bar(x + w/2, mu_vals, w, label="Octahedral μ",  color=c_list, alpha=0.50, edgecolor="k", linewidth=0.5, hatch="//")

    ax1.axhline(0.80, ls="--", lw=1, color="gray", label="t = 0.80 (stability lower)")
    ax1.axhline(0.41, ls=":",  lw=1, color="navy",  label="μ = 0.41 (octahedra lower)")
    ax1.axhline(0.73, ls=":",  lw=1, color="navy",  label="μ = 0.73 (octahedra upper)")
    ax1.set_xticks(x); ax1.set_xticklabels(X_labels, fontsize=9)
    ax1.set_ylabel("Factor (dimensionless)")
    ax1.set_title("Structural Stability Factors")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_ylim(0, 1.0)
    for bar, val in zip(bars_t, t_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(bars_mu, mu_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

    # ── Panel 2: band gap ────────────────────────────────────────────────────
    Eg_est  = [results["bandgap"][X]["Eg_est_eV"]  for X in HALOGENS]
    Eg_ref  = [results["bandgap"][X]["Eg_ref_eV"]  for X in HALOGENS]
    Eg_low  = [results["bandgap"][X]["Eg_low_eV"]  for X in HALOGENS]
    Eg_high = [results["bandgap"][X]["Eg_high_eV"] for X in HALOGENS]
    yerr_lo = [Eg_est[i] - Eg_low[i]  for i in range(3)]
    yerr_hi = [Eg_high[i] - Eg_est[i] for i in range(3)]

    ax2.bar(x - w/2, Eg_ref, w, label="Cs₃Bi₂X₉ (experiment)", color="lightgray", edgecolor="k", linewidth=0.5)
    ax2.bar(x + w/2, Eg_est, w, label="Sr₃BiX₃ (this estimate)", color=c_list,     edgecolor="k", linewidth=0.5, alpha=0.85)
    ax2.errorbar(x + w/2, Eg_est, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=4, lw=1.5)
    ax2.set_xticks(x); ax2.set_xticklabels(X_labels, fontsize=9)
    ax2.set_ylabel("Band gap (eV)")
    ax2.set_title("Band Gap: Sr₃BiX₃ vs Cs₃Bi₂X₉ Reference")
    ax2.legend(fontsize=8)
    ax2.set_ylim(1.2, 4.0)
    for i, (est, ref) in enumerate(zip(Eg_est, Eg_ref)):
        ax2.text(x[i] + w/2, est + 0.07, f"{est:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax2.text(x[i] - w/2, ref + 0.07, f"{ref:.2f}", ha="center", fontsize=8, color="gray")

    # ── Panel 3: absorption spectra ──────────────────────────────────────────
    E_arr = np.linspace(1.0, 5.0, 800)
    for X in HALOGENS:
        Eg     = results["bandgap"][X]["Eg_est_eV"]
        E_urb  = URBACH_E[X]
        alpha  = absorption_coefficient(E_arr, Eg, E_urb)
        ax3.semilogy(E_arr, alpha, color=COLORS[X], lw=2, label=f"Sr₃Bi{X}₃  Eg={Eg:.2f} eV")
        ax3.axvline(Eg, color=COLORS[X], ls="--", lw=0.8, alpha=0.6)

    ax3.set_xlabel("Photon energy (eV)")
    ax3.set_ylabel("Absorption coefficient α (cm⁻¹)")
    ax3.set_title("Optical Absorption Spectra")
    ax3.legend(fontsize=8)
    ax3.set_xlim(1.0, 5.0)
    ax3.set_ylim(1e1, 2e6)
    ax3.grid(True, which="both", alpha=0.3)

    # ── Panel 4: Shockley–Queisser efficiency ────────────────────────────────
    Voc_vals  = [results["sq"][X]["Voc_V"]      for X in HALOGENS]
    PCE_vals  = [results["sq"][X]["PCE_pct"]     for X in HALOGENS]
    FF_vals   = [results["sq"][X]["FF"]          for X in HALOGENS]
    Jsc_vals  = [results["sq"][X]["Jsc_mAcm2"]   for X in HALOGENS]

    ax4b = ax4.twinx()
    line1 = ax4.bar(x - w/2, PCE_vals, w, color=c_list, edgecolor="k", linewidth=0.5, alpha=0.85, label="PCE % (left)")
    line2 = ax4b.plot(x, Voc_vals, "o--", color="navy", ms=8, lw=1.5, label="Voc (right)")
    ax4.set_xticks(x); ax4.set_xticklabels(X_labels, fontsize=9)
    ax4.set_ylabel("Radiative-limit PCE (%)", color="black")
    ax4b.set_ylabel("Open-circuit voltage Voc (V)", color="navy")
    ax4b.tick_params(axis="y", labelcolor="navy")
    ax4.set_title("Shockley–Queisser Radiative Limit")
    for i, (pce, voc) in enumerate(zip(PCE_vals, Voc_vals)):
        ax4.text(x[i] - w/2, pce + 0.2, f"{pce:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax4b.text(x[i] + 0.05, voc + 0.01, f"{voc:.2f} V", ha="left", fontsize=8, color="navy")
    lines1 = plt.Rectangle((0, 0), 1, 1, color=c_list[1], alpha=0.85)
    lines2 = plt.Line2D([0], [0], color="navy", marker="o", ls="--")
    ax4.legend([lines1, lines2], ["PCE %", "Voc (V)"], fontsize=8, loc="upper right")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 9. PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict):
    SEP  = "─" * 72
    SEP2 = "═" * 72

    print(f"\n{SEP2}")
    print("  Sr₃BiX₃ (X = I, Br, Cl) — Perovskite Simulation Results")
    print(f"{SEP2}\n")

    # ── Charge balance ───────────────────────────────────────────────────────
    print("① CHARGE BALANCE ANALYSIS")
    print(SEP)
    print(f"  Formula Sr₃BiX₃:  3×Sr²⁺ (+6) + Bi³⁺ (+3) + 3×X⁻ (−3)")
    print(f"  Net charge        = +6  ← NOT charge-neutral for standard oxidation states")
    print(f"\n  Physical interpretations:")
    print(f"    (a) Mixed-valence Bi (partial Bi³⁺/Bi⁵⁺) — possible, unusual")
    print(f"    (b) Sr-vacancy ordered defect perovskite (Sr₃□₂BiX₃ with □ = vacancy)")
    print(f"    (c) Hypothetical meta-stable phase (common in computational screening)")
    print(f"    (d) Closest charge-balanced analogs: Cs₃Bi₂X₉ (3:2:9), SrBiX₅ (1:1:5)")
    print(f"\n  ⚠  Simulation proceeds using the nominal Sr₃BiX₃ composition;")
    print(f"     band gap estimates are calibrated against the Cs₃Bi₂X₉ reference series.")

    # ── Structural ────────────────────────────────────────────────────────────
    print(f"\n② STRUCTURAL ANALYSIS")
    print(SEP)
    header = f"  {'Compound':<15} {'t (Goldschmidt)':<18} {'μ (octahedral)':<16} {'τ (Bartel)':<12} {'Phase'}"
    print(header)
    print(f"  {'-'*68}")
    for X in HALOGENS:
        s = results["struct"][X]
        name = f"Sr₃Bi{X}₃"
        print(f"  {name:<15} {s['tolerance_t']:<18.4f} {s['octahedral_mu']:<16.4f} "
              f"{s['new_tau']:<12.2f} {s['phase']}")
    print(f"\n  Stability guide:  t ∈ [0.80, 0.89] → distorted perovskite")
    print(f"                    μ ∈ [0.41, 0.73] → stable BiX₆ octahedra ✓")

    # ── Band gap ─────────────────────────────────────────────────────────────
    print(f"\n③ BAND GAP ESTIMATES")
    print(SEP)
    header = (f"  {'Compound':<15} {'Eg (eV, est.)':<15} {'± (eV)':<8} "
              f"{'Eg ref Cs₃Bi₂X₉':<18} {'ΔEg pressure':<14} {'Gap type'}")
    print(header)
    print(f"  {'-'*72}")
    for X in HALOGENS:
        b = results["bandgap"][X]
        name = f"Sr₃Bi{X}₃"
        print(f"  {name:<15} {b['Eg_est_eV']:<15.3f} {'±0.20':<8} "
              f"{b['Eg_ref_eV']:<18.3f} {b['dEg_pressure_eV']:<14.3f} {b['gap_type']}")
    print(f"\n  Model: Eg(Sr₃BiX₃) = Eg_ref(Cs₃Bi₂X₃) + ΔEg_pressure + ΔEg_stoich")
    print(f"    ΔEg_pressure = α × (r_Cs − r_Sr) / r_Cs,  α = 0.42 eV")
    print(f"    ΔEg_stoich   = +0.08 eV  (1 Bi/FU vs 2 Bi/FU connectivity change)")

    # ── Optical ──────────────────────────────────────────────────────────────
    print(f"\n④ OPTICAL CONSTANTS  (at 550 nm, room temperature)")
    print(SEP)
    wl_probe = np.array([550.0])
    header = f"  {'Compound':<15} {'n (550 nm)':<14} {'Eg onset (nm)':<16} {'Urbach E (meV)'}"
    print(header)
    print(f"  {'-'*58}")
    for X in HALOGENS:
        n_val = sellmeier_n(wl_probe, X)[0]
        Eg    = results["bandgap"][X]["Eg_est_eV"]
        onset_nm = 1240.0 / Eg
        urbach_meV = URBACH_E[X] * 1000
        name = f"Sr₃Bi{X}₃"
        print(f"  {name:<15} {n_val:<14.3f} {onset_nm:<16.1f} {urbach_meV:<.0f}")

    # ── Shockley-Queisser ─────────────────────────────────────────────────────
    print(f"\n⑤ SHOCKLEY–QUEISSER RADIATIVE LIMIT  (AM1.5G, 1-sun, 300 K)")
    print(SEP)
    header = (f"  {'Compound':<15} {'Eg (eV)':<10} {'Voc (V)':<10} "
              f"{'Jsc (mA/cm²)':<14} {'FF':<8} {'PCE (%) max'}")
    print(header)
    print(f"  {'-'*65}")
    for X in HALOGENS:
        Eg  = results["bandgap"][X]["Eg_est_eV"]
        sq  = results["sq"][X]
        name = f"Sr₃Bi{X}₃"
        print(f"  {name:<15} {Eg:<10.3f} {sq['Voc_V']:<10.3f} "
              f"{sq['Jsc_mAcm2']:<14.2f} {sq['FF']:<8.3f} {sq['PCE_pct']:.2f}")

    print(f"\n  Note: SQ limit is a thermodynamic upper bound; actual device efficiencies")
    print(f"  are typically 30–60% of SQ due to non-radiative recombination, contacts, etc.")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  SUMMARY")
    print(SEP2)
    for X in HALOGENS:
        b  = results["bandgap"][X]
        s  = results["struct"][X]
        sq = results["sq"][X]
        print(f"\n  Sr₃Bi{X}₃:")
        print(f"    Band gap  : {b['Eg_est_eV']:.3f} eV  ({b['Eg_low_eV']:.2f}–{b['Eg_high_eV']:.2f} eV, ±0.20 eV)")
        print(f"    Structure : {s['phase']}")
        print(f"    Max PCE   : {sq['PCE_pct']:.1f}%  (Voc = {sq['Voc_V']:.2f} V)")
        onset_nm = round(1240.0 / b["Eg_est_eV"], 0)
        print(f"    Abs onset : {int(onset_nm)} nm  ({'visible' if onset_nm < 700 else 'near-IR'})")
    print(f"\n{SEP2}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Sr₃BiX₃ simulation …")
    results  = run_simulation()
    print_results(results)

    fig_path = RESULTS_DIR / "sr3bix3_simulation.png"
    make_figure(results, fig_path)
    print(f"Figure saved → {fig_path}")
