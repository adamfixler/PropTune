# PropTune

A blade-element-momentum-theory (BEMT) propeller design and optimization tool built for a human-powered aircraft (HPA), specifically an English Channel crossing mission modeled on MIT's Daedalus project. It optimizes blade chord, twist, and RPM for maximum propulsive efficiency at a required cruise thrust, using [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox)'s gradient-based optimizer (`Opti`, backed by IPOPT) and [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) for differentiable airfoil aerodynamics. The optimized blade is then exported to XFLR5, STEP (CAD), and QBlade formats.

## Requirements

- Python 3.12 (developed against 3.12.10)
- [aerosandbox](https://pypi.org/project/aerosandbox/) (includes NeuralFoil and CasADi as dependencies)
- casadi
- matplotlib
- numpy (used indirectly via `aerosandbox.numpy`)

`requirements.txt` pins the exact versions this project has been developed and validated against. NeuralFoil is a trained model bundled inside `aerosandbox` — an untested upgrade could silently change its Cl/Cd predictions for the same inputs, with no error, so these versions are pinned deliberately rather than left open-ended. Bump them only intentionally, and re-check the optimizer's results afterward.

Install into a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

To update the pins later (e.g. after deliberately upgrading a package and re-validating), regenerate the file from what's actually installed:

```bash
pip freeze > requirements.txt
```

(this dumps every installed package, so on a shared/system Python you may want to prune it back down to just the packages this project actually imports — a fresh virtual environment avoids that problem entirely).

## Project files

| File | Purpose |
|---|---|
| `main.py` | Entry point. Sets up the design point, builds the optimization problem, solves it, and exports the result. **This is the script you run and edit.** |
| `BEM.py` | `BEMAnalysis` — the blade-element-momentum solver: builds a differentiable induction-factor rootfinder per station and integrates thrust/torque/power/efficiency across the blade. |
| `geometry.py` | `Propeller` geometry class, `.bem` file loader (`load_bem`), and exporters (`export_airfoil_dat`, `export_qblade_bld`, `to_wing`, `display_rotor`). |
| `BaselineAnalyze.py` | Standalone script: analyzes (does not optimize) a loaded `.bem` baseline at a fixed RPM/velocity, for sanity-checking the solver against a known real design. |
| `deadelus_MIL_baseline.bem`, `daedalus.bem` | Example `.bem` geometry files (real/reference Daedalus propeller data) usable as an optimization seed via `USE_BEM_FILE = True`. |
| `outputs/` | Generated exports (regenerated each run, created automatically if missing) — see [Outputs](#outputs). |
| `q blade/` | A QBlade project file, kept for manual cross-checking in QBlade's own solver. |

## Quick start

1. Open `main.py` and edit the **USER CONFIGURATION** block at the top (see [Configuration](#configuration) below) to match your design point.
2. Run it:

   ```bash
   python main.py
   ```

3. The optimizer prints its convergence log, then a summary (efficiency, RPM, thrust, torque, power) and a per-station table comparing the initial guess to the optimized chord/twist. A three-view wireframe of the optimized blade is displayed, and the result is exported to disk (see [Outputs](#outputs)).

To instead just *analyze* (not optimize) the bundled Daedalus baseline `.bem` file at a fixed operating point:

```bash
python BaselineAnalyze.py
```

## Configuration

Everything you're likely to want to change lives in the `USER CONFIGURATION` block at the top of `main.py`:

| Variable | Meaning |
|---|---|
| `USE_BEM_FILE` | `False`: seed the optimizer from generic Daedalus specs + basic flow physics. `True`: seed from an actual `.bem` file's station geometry. |
| `BEM_FILENAME` | `.bem` file to load when `USE_BEM_FILE = True`. |
| `airfoil` | Airfoil name passed to `asb.Airfoil` / NeuralFoil (default `"dae51"`). |
| `velocity` | Cruise airspeed, m/s. |
| `T_required` | Required cruise thrust, N — a hard equality constraint. |
| `N` | Number of radial stations. NeuralFoil builds a large symbolic graph per station, un-batched; 10–20 is the practical ceiling before IPOPT's Hessian evaluation runs out of memory. |
| `radius`, `hub_radius`, `n_blades`, `max_chord` | From-scratch seed geometry (only used when `USE_BEM_FILE = False`). `hub_radius` is an absolute value in meters, not a fraction. |
| `chord_min/max`, `twist_min/max`, `rpm_guess`, `rpm_min/max` | Optimizer design-variable bounds and initial guesses for chord (m), twist (deg), and RPM. `rpm_guess` doubles as the from-scratch twist seed's assumed operating RPM. |
| `alpha_min_deg`, `alpha_max_deg` | Per-station angle-of-attack constraint, deg. |
| `min_analysis_confidence` | Minimum NeuralFoil `analysis_confidence` allowed per station (keeps the optimizer inside NeuralFoil's trusted region). |
| `smoothness_weight`, `twist_scale` | Soft penalty weight and twist-vs-chord normalization for the anti-jaggedness objective term. |
| `max_chord_rate`, `max_twist_rate` | Hard per-segment rate-of-change bounds (chord in m/m of span, twist in deg/m of span). |
| `OUTPUT_DIR` | Folder all exports are written into (default `outputs/`, created automatically). |
| `airfoil_dat_filename`, `xflr5_xml_filename`, `step_filename`, `qblade_bld_filename`, `qblade_polar_filename`, `prop_csv_filename` | Output filenames, inside `OUTPUT_DIR`. |

## The BEM formulas

### Velocity triangle and inflow

At each radial station $r$, with axial and swirl induction factors $a$ and $a'$, rotor angular speed $\Omega$, and freestream velocity $V$:

$$V_{ax} = V(1+a) \qquad V_{tan} = \Omega r (1-a')$$

$$\phi = \text{atan2}(V_{ax},\, V_{tan}) \qquad W = \sqrt{V_{ax}^2 + V_{tan}^2}$$

$$\alpha = \beta - \phi \qquad Re = \frac{\rho\, W\, c}{\mu}$$

where $\beta$ is the local blade twist and $c$ the local chord. $C_l(\alpha, Re)$ and $C_d(\alpha, Re)$ come from NeuralFoil (`airfoil.get_aero_from_neuralfoil`), evaluated on the same differentiable graph the optimizer differentiates through.

### Prandtl tip and hub loss

Standard Prandtl tip-loss factor, plus a mirrored hub-loss factor for the root cutout, combined multiplicatively:

$$f_{tip} = \frac{B}{2}\cdot\frac{R - r}{r\sin\phi} \qquad F_{tip} = \frac{2}{\pi}\arccos\!\left(e^{-f_{tip}}\right)$$

$$f_{hub} = \frac{B}{2}\cdot\frac{r - r_{hub}}{r\sin\phi} \qquad F_{hub} = \frac{2}{\pi}\arccos\!\left(e^{-f_{hub}}\right)$$

$$F = F_{tip} \cdot F_{hub}$$

($B$ = blade count, $R$ = tip radius, $r_{hub}$ = hub cutout radius.) Each factor is $\to 1$ away from its edge and $\to 0$ at it, keeping $F$ well-behaved across the whole span.

### Induction-factor residuals

Blade solidity and normal/tangential aerodynamic coefficients:

$$\sigma = \frac{B c}{2\pi r} \qquad C_n = C_l\cos\phi - C_d\sin\phi \qquad C_t = C_l\sin\phi + C_d\cos\phi$$

$a$ and $a'$ at each station solve the coupled BEM momentum-balance residuals (via a Newton rootfinder, not left as free optimizer variables — see `BEM.py`'s docstring for why):

$$4Fa\sin^2\phi - (1+a)\,\sigma C_n = 0$$

$$4Fa'\sin\phi\cos\phi - (1-a')\,\sigma C_t = 0$$

### Section loads and integrated performance

$$q = \tfrac{1}{2}\rho W^2 \qquad L = qcC_l \qquad D = qcC_d$$

$$F_n = L\cos\phi - D\sin\phi \qquad F_t = L\sin\phi + D\cos\phi$$

Per-station thrust/torque contribution, using each station's own segment width $\Delta r_i$ (supports non-uniform spacing):

$$dT_i = F_n\, B\, \Delta r_i \qquad dQ_i = F_t\, r\, B\, \Delta r_i$$

$$T = \sum_i dT_i \qquad Q = \sum_i dQ_i \qquad P = Q\Omega \qquad \eta = \frac{TV}{P}$$

### Optimization problem

$$\min_{\text{chord},\,\text{twist},\,\text{rpm}} \Big[-\eta + w_s \cdot S\Big]$$

subject to

$$T = T_{required} \qquad \eta \le 1 \qquad \alpha_{min} \le \alpha_i \le \alpha_{max} \qquad \text{confidence}_i \ge c_{min}$$

$$\left|\frac{\Delta\text{chord}}{\Delta r}\right| \le \text{max chord rate} \qquad \left|\frac{\Delta\text{twist}}{\Delta r}\right| \le \text{max twist rate}$$

where the smoothness penalty $S$ is a Riemann-sum approximation of $\int\!\left[\left(\frac{d\,\text{chord}/dr}{\text{chord scale}}\right)^2 + \left(\frac{d\,\text{twist}/dr}{\text{twist scale}}\right)^2\right] dr$ along the span — penalizing jagged station-to-station steps without biasing the design toward any particular reference taper.

## Outputs

Each run of `main.py` (over)writes into `outputs/` (created automatically if it doesn't exist):

- **`outputs/dae51.dat`** — Selig-format airfoil coordinates. Import into XFLR5's airfoil database (under the exact same name, e.g. `DAE51`) *before* opening the plane XML below.
- **`outputs/optimized_prop_blade.xml`** — XFLR5 plane file (via AeroSandbox's native exporter). Note: XFLR5's LLT/VLM analysis treats this as a static wing in uniform flow, not a rotating blade — it won't reproduce this project's BEM results, it's just a geometry cross-check.
- **`outputs/optimized_prop_blade.step`** — Self-contained CAD solid (STEP), openable in Fusion 360, SolidWorks, FreeCAD, etc.
- **`outputs/optimized_prop.bld`** — QBlade blade definition (geometry only). QBlade also needs a full 360°-alpha `.plr` polar per station, which this project's NeuralFoil-based BEM does not compute (it deliberately stays in a narrow, well-attached alpha band). Generate a real polar inside QBlade (Direct Analysis + Viterna extrapolation) before this file is usable there — see `export_qblade_bld`'s docstring in `geometry.py` for the full workflow. The polar filename referenced inside this file is relative to `outputs/` itself.
- **`outputs/optimized_prop.csv`** — Optimized geometry + operating point, reloadable via `geometry.load_prop_csv` without re-running the optimizer (used by `verify_xfoil.py` and `propeller_performance_map.py`).
- **`outputs/propeller_performance_map.png`** — Off-design thrust/power/efficiency chart, written by `propeller_performance_map.py`.

## Notes and caveats

- Only `N ≈ 10–20` radial stations are practical — NeuralFoil's per-station symbolic graph makes higher resolutions memory-prohibitive for IPOPT's Hessian evaluation.
- Station centers are always placed strictly between `hub_radius` and `radius` (segment midpoints, never exact endpoints) — a station exactly at the hub or tip would zero out its loss factor and let the optimizer exploit the resulting degenerate residual.

## License

© 2026 adamfixler. All rights reserved. This repository is shared for reference only — no license is granted to use, modify, or redistribute this code.
