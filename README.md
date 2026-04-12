# Squeezed-Vacuum Bosonic Codes

Python simulation toolkit accompanying the paper:

> **Squeezed-vacuum bosonic codes**  
> N. Gutman, E. Blumenthal, S. Hacohen-Gourgy, A. Orda, and I. Kaminer  
> Technion — Israel Institute of Technology

We introduce a family of bosonic quantum error-correcting codes built as a rotation-symmetric superposition of squeezed vacuum states, offering simultaneous protection against photon loss and dephasing noise. The codes have interleaved Fock-space support $n \equiv 2k \pmod{2m}$ and code distance $d = m$ against single-photon loss.

---

## Overview

The "squeezed-vacuum codes" are parametrized by two numbers:
- **m** — the (even) number of squeezed-vacuum "legs"  
- **r** — the squeezing strength

The two logical codewords are:

$$|0_L\rangle \propto \sum_{j=0}^{m-1} S\!\left(r,\tfrac{\pi j}{m}\right)|0\rangle, \qquad |1_L\rangle \propto \sum_{j=0}^{m-1} (-1)^j S\!\left(r,\tfrac{\pi j}{m}\right)|0\rangle$$

Increasing $m$ improves loss tolerance at the cost of higher dephasing sensitivity — the same trade-off as cat codes, but built from a distinct non-Gaussian primitive (conditional squeezing instead of conditional displacement).

<p align="center">
  <img src="outputs/figures/Code States Grid.png" alt="Wigner functions of m-legged squeezed-vacuum codes (m=2,4,6,8). Top row: |0_L⟩; bottom row: |1_L⟩." width="85%">
</p>

---

## Key Features

- **Code family generation**: Squeezed-vacuum codes, cat codes, binomial codes, and GKP codes — all in a unified interface via [codes_built_in_superposition.py](src/codes_built_in_superposition.py).

- **Preparation protocols** (Algorithm 1 & 2 from the paper):  
  - *Probabilistic* — H–CR–H sequences with post-selection; generates $m = 2^k$-legged codes (**Algorithm 1**) or general even-$m$ codes (**Algorithm 2**)  
  - *Deterministic* — feed-forward $\bar{X}$ corrections after each measurement step  
  - *Arbitrary state preparation* — rotated-measurement-frame circuit absorbs target $(\alpha, \beta)$ into the final qubit rotation

  <p align="left">
    <img src="assets/fig00.png" alt="Preparation circuits for multi-legged squeezed-vacuum codes" width="45%">
  </p>

- **Comparison with cat and binomial codes**: The 2-legged squeezed code achieves code distance $d=2$ (vs $d=1$ for the 2-legged cat) using a single conditional-squeezing gate.

  <p align="left">
    <img src="assets/fig02.png" alt="2-legged cat vs squeezed code: circuits, Wigner functions, and Fock-space support" width="85%">
  </p>

- **Knill–Laflamme violation analysis**: Numerical benchmarking of the KL violation function over photon-loss and dephasing channels, comparing squeezed-vacuum, cat, binomial, and GKP codes.

- **Logical operations**: Native $\bar{Z}$ (phase-space rotation $R(\pi/m)$) and entangling $\overline{CZ}$ (cross-Kerr); logical $\bar{X}$ and $\bar{H}$ via gate teleportation.

- **Analytical expressions**: Closed-form Fock-basis amplitudes and mean photon number (Eq. 23–25 from the paper), implemented symbolically with SymPy.

- **Visualization**: Wigner function plots, Fock-number distributions, and phase-space representations.

---

## Physics Background

### Squeezed-Vacuum Primitive

A single squeezed-vacuum state elongated along direction $\theta$ is:

$$S(r, \theta)|0\rangle = \frac{1}{\sqrt{\cosh r}} \sum_{n=0}^{\infty} \frac{\sqrt{(2n)!}}{2^n n!} e^{i 2n\theta} \tanh^n r \, |2n\rangle$$

Note that squeezed vacuum occupies only **even** Fock numbers, which gives the code its sparse support structure.

### Code Structure and Distance

The $m$-legged code word $|\psi_k\rangle \propto \sum_{j=0}^{m-1} e^{i\frac{2\pi j}{m}k} S(r, \frac{\pi j}{m})|0\rangle$ is supported on photon numbers $n \equiv 2k \pmod{2m}$. The logical codewords $|0_L\rangle := |\psi_0\rangle$ and $|1_L\rangle := |\psi_{m/2}\rangle$ have interleaved Fock support separated by $\Delta n = m$, giving code distance $d = m$ against single-photon loss — **independent of $r$**.

### Knill–Laflamme Conditions

Code performance is measured via the KL violation:

$$V_\mathrm{KL}(\mathcal{N})_{i,j} = \sum_{a,b} \left|\delta_{ij} + (-1)^{\delta_{ij}} \langle i | K_a^\dagger K_b | j \rangle\right|$$

for loss and dephasing Kraus operators:

$$K_j^\text{loss}(\gamma) = \sqrt{\frac{\gamma^j}{j!}}(1-\gamma)^{n/2} a^j, \qquad K_j^\text{dephasing}(\gamma) = \sqrt{\frac{\gamma^j}{j!}} e^{-\frac{\gamma}{2}n^2} n^j$$

Exact satisfaction ($V_\mathrm{KL} = 0$) corresponds to a perfect code; smaller values indicate better performance.

---

## Repository Structure

```
squeezed-vacuum-codes/
├── src/
│   ├── codes_built_in_superposition.py  # m-legged code generators (squeeze, cat, binomial, GKP)
│   ├── preparation_protocols.py         # Probabilistic & deterministic preparation (Algorithms 1 & 2)
│   ├── cost_functions.py                # KL-violation cost functions
│   ├── kraus_maps.py                    # Kraus operators for loss and dephasing channels
│   ├── mean_photon_number.py            # Mean photon number computation
│   ├── analytical_expressions.py        # Symbolic Fock-basis derivations
│   ├── noise.py                         # Lindbladian noise simulation (qutip.mesolve)
│   ├── measurements.py                  # Qubit measurement and post-selection
│   ├── visualizations.py               # Wigner functions, Fock distributions
│   ├── bosonic_operators.py             # Ladder operators and utilities
│   ├── rotation.py                      # Phase-space rotation operators
│   ├── squeezing_direction.py           # Direction-to-phase mapping: φ(θ) = 2θ + π
│   ├── metrics.py                       # Cross-overlap matrices
│   ├── squeezing_code.py               # Low-level squeezing utilities
│   ├── code_paths.py                    # Output path management
│   ├── gkp/                            # GKP code utilities (Löwdin orthogonalization)
│   ├── quantum/                        # Quantum-information primitives
│   │   ├── density_matrices/
│   │   ├── metrics/
│   │   ├── quantum_information/
│   │   ├── qutip_support/
│   │   └── visualizations/
│   └── utils/                          # General utilities (caches, maths, visuals, …)
├── scripts/
│   ├── plot_code_states.py             # Wigner-function grid for a chosen code
│   ├── probability_of_preparation.py   # Success-probability analysis
│   ├── illustrate.py                   # Quick illustrative plots
│   ├── graphs_for_paper/               # Publication figures
│   │   ├── code_family.py             #   Wigner grid: m-legged code family
│   │   ├── numerics1.py               #   KL-violation vs noise / mean-photon-number
│   │   ├── overlap.py                 #   Codeword overlap vs squeezing r
│   │   ├── preparation_probabilities.py  #  Success probability vs n̄
│   │   └── arbitrary_states.py        #   Arbitrary logical state preparation
│   └── study/                          # Research / exploratory scripts
│       ├── logical_x_test.py
│       ├── study_measurement_free_code.py
│       ├── test_analytical_expressions.py
│       └── test_orthonormalization_of_codes.py
├── tests/                              # Unit tests (pytest)
├── outputs/
│   └── figures/                        # Generated figure outputs
├── globals.py                          # Global configuration (dataclass)
└── requirements.txt
```

---

## Installation

### Prerequisites
- Python 3.11+ (3.11.9 recommended)
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/AdQuanta-Quantum-Information/squeezed-vacuum-codes.git
cd squeezed-vacuum-codes

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **qutip** | 5.2.2 | Quantum states, operators, and dynamics |
| **numpy** | 2.3.4 | Numerical computation |
| **scipy** | 1.16.3 | Scientific computing |
| **sympy** | 1.14.0 | Symbolic Fock-basis derivations |
| **matplotlib** | 3.10.7 | Visualization |
| **mpmath** | 1.3.0 | High-precision arithmetic for Kraus series |
| **joblib** | 1.5.2 | Persistent disk caching |

---

## Key Scripts

### Generating paper figures

- **[scripts/graphs_for_paper/code_family.py](scripts/graphs_for_paper/code_family.py)** — Wigner-function grid of $m$-legged squeezed-vacuum codes ($m = 2, 4, 6, 8$), two logical codewords per column.

- **[scripts/graphs_for_paper/numerics1.py](scripts/graphs_for_paper/numerics1.py)** — KL-violation $V_\mathrm{KL}$ vs noise rate $\gamma$ (and vs mean photon number $\bar{n}$), comparing squeezed-vacuum, cat, binomial, and GKP codes.

- **[scripts/graphs_for_paper/preparation_probabilities.py](scripts/graphs_for_paper/preparation_probabilities.py)** — Success probability of Algorithms 1 & 2 as a function of $\bar{n}$, for various $m$.

- **[scripts/graphs_for_paper/overlap.py](scripts/graphs_for_paper/overlap.py)** — Codeword overlap between logical states as a function of squeezing strength $r$.

- **[scripts/graphs_for_paper/arbitrary_states.py](scripts/graphs_for_paper/arbitrary_states.py)** — Fidelity of the rotated-measurement-frame preparation of $|+_L\rangle$ vs $r$.

### Exploratory scripts

- **[scripts/plot_code_states.py](scripts/plot_code_states.py)** — Renders Wigner functions and Fock distributions for any supported code type and number of legs.

- **[scripts/study/study_measurement_free_code.py](scripts/study/study_measurement_free_code.py)** — Numerical exploration of deterministic preparation using conditional rotations.

- **[scripts/study/logical_x_test.py](scripts/study/logical_x_test.py)** — Validates the logical-$\bar{X}$ operator implementation and state transitions within the code space.

- **[scripts/study/test_analytical_expressions.py](scripts/study/test_analytical_expressions.py)** — Checks closed-form Fock amplitudes against direct numerical computation.

---

## Configuration

[globals.py](globals.py) exposes a frozen dataclass:

```python
from globals import Globals

Globals.PRECISE          # True  — use mpmath high-precision arithmetic
Globals.DEBUG            # False — enable assertion checks
Globals.CACHE_ON_DISK    # True  — persist joblib cache across runs
Globals.LaTeX_RENDERING  # True  — use LaTeX font rendering in plots
```

## Caching

Expensive computations (Kraus matrices, cost functions) are cached with `joblib`. The persistent cache lives in `joblib_cache/`. Clear it if you modify core functions:

```bash
# Windows
Remove-Item -Recurse -Force joblib_cache

# macOS / Linux
rm -rf joblib_cache/
```

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $r$ | Squeezing strength (nats) |
| $\theta$ | Squeezing direction angle |
| $m$ | Number of legs (even integer) |
| $\gamma$ | Noise-strength parameter |
| $\bar{n}$ | Mean photon number |
| $\alpha$ | Displacement amplitude (cat codes) |
| $\|0_L\rangle, \|1_L\rangle$ | Logical codewords |
| $a, a^\dagger$ | Bosonic ladder operators |
| $\hat{n} = a^\dagger a$ | Number operator |

---

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@article{gutman2025squeezed,
  title   = {Squeezed-vacuum bosonic codes},
  author  = {Gutman, N. and Blumenthal, E. and Hacohen-Gourgy, S. and Orda, A. and Kaminer, I.},
  year    = {2025},
  note    = {Technion -- Israel Institute of Technology}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

Questions or collaboration inquiries: open a GitHub issue or email [nirgutman212@campus.technion.ac.il](mailto:nirgutman212@campus.technion.ac.il).

## Acknowledgments

This work was supported by the Israel Science Foundation (ISF), Grant No. 385/23 and Grant No. 1315/24, and partially by the Flagship research project QUBIT of the Helen Diller Quantum Center at the Technion.

This code makes use of [QuTiP](https://qutip.org/) for quantum mechanics simulation.
