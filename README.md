# Squeezed Vacuum Quantum Error Correction Codes

Python tools for simulating multi-legged squeezed vacuum bosonic quantum error correction codes. Includes probabilistic and deterministic preparation protocols, Kraus operator analysis, and numerical validation of quantum error correction conditions.



## Overview

This codebase accompanies research on bosonic quantum error correction codes constructed from superpositions of squeezed vacuum states in multiple directions. The codes can protect quantum information encoded in continuous-variable (CV) systems against photon loss and dephasing errors.

![alt text](assets/fig00.png)

### Key Features

- **Multi-legged code generation**: Create m-legged bosonic codes (cat codes and squeezed vacuum codes) with arbitrary number of legs
- **Preparation protocols**: 
  - Probabilistic preparation via post-selection on ancilla qubits
  - Deterministic preparation using controlled squeezing and feed-forward
  - Measurement-free protocols using controlled rotations
- **Error analysis**: 
  - Kraus operator formalism for photon loss and dephasing channels
  - Numerical verification of Knill-Laflamme quantum error correction conditions
  - Cost function computation for comparing code performance
- **Logical operations**: Implementation of logical X, Z gates and state rotations
- **Analytical expressions**: Symbolic computation of code properties using SymPy
- **Visualization**: Wigner function plots, Fock distributions, and phase-space representations

## Physics Background

### Squeezed Vacuum States

![alt text](assets/fig01.png) 

The codes are built from squeezed vacuum states of the form:
```
|S(r,θ)⟩ = S(r,θ)|0⟩
```
where `S(r,θ) = exp[r(e^{2iθ}a² - e^{-2iθ}a†²)/2]` is the squeezing operator, `r` is the squeezing strength, and `θ` is the squeezing direction in phase space.


### Multi-legged Codes

An m-legged squeezed code encodes logical states as:
```
|0_L⟩ ∝ Σⱼ |S(r, 2πj/m)⟩
|1_L⟩ ∝ Σⱼ e^{i2πj/m} |S(r, 2πj/m)⟩
```

These codes can detect and correct specific errors while using only single-mode bosonic states.

### Error Correction

The codes satisfy the Knill-Laflamme conditions:
```
⟨ψₖ|E†ᵢEⱼ|ψₗ⟩ = δₖₗ cᵢⱼ
```
for photon loss (`Eⱼ = √(γʲ/j!) (1-γ)^{n/2} aʲ`) and dephasing (`Eⱼ = √(γʲ/j!) exp(-γn²/2) nʲ`) channels.



![alt text](assets/fig02.png)


## Repository Structure

```
squeezed-vacuum-codes/
├── src/
│   ├── squeezing_code.py           # Main SqueezingCode class
│   ├── codes_built_in_superposition.py  # m-legged code generators
│   ├── preparation_circuits.py     # Probabilistic preparation
│   ├── deterministic_preparation.py # Deterministic protocols
│   ├── kraus_maps.py               # Kraus operators for noise channels
│   ├── cost_functions.py           # Performance metrics (KL divergence, fidelity)
│   ├── mean_photon_number.py       # Resource analysis
│   ├── analytical_expressions.py   # Symbolic derivations
│   ├── noise.py                    # Noise simulation via Lindbladian evolution
│   ├── measurements.py             # Qubit measurement and post-selection
│   ├── visualizations.py           # Wigner functions and plotting
│   ├── bosonic_operators.py        # Ladder operators and utilities
│   └── quantum/                    # Quantum mechanics utilities
│       ├── density_matrices/
│       ├── metrics/
│       └── visualizations/
├── scripts/
│   ├── prove_QECC.py              # Verify error correction conditions
│   ├── probability_of_preparation.py # Success probability analysis
│   ├── study_measurement_free_code.py # Deterministic protocols
│   ├── plot_code_states.py        # Visualization scripts
│   ├── logical_x_test.py          # Logical operator verification
│   └── graphs_for_paper/          # Publication figures
│       ├── code_family.py
│       └── numerics1.py
├── tests/                          # Unit tests
├── globals.py                      # Configuration flags
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.11.9 (recommended)
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

### Dependencies

Core packages:
- **qutip** (5.2.2): Quantum mechanics simulation
- **numpy** (2.3.4): Numerical computations
- **scipy** (1.16.3): Scientific computing
- **sympy** (1.14.0): Symbolic mathematics
- **matplotlib** (3.10.7): Visualization
- **mpmath** (1.3.0): High-precision arithmetic for Kraus operators
- **joblib** (1.5.2): Caching and parallelization

## Usage Examples

### Creating a 2-legged Squeezed Code

```python
from src.squeezing_code import SqueezingCode

# Initialize code with 2 legs
code = SqueezingCode(
    squeezing_strength=2.0,  # r parameter
    num_moments=200,         # Fock space truncation
    num_legs=2,
    verbose=True
)

# Get logical codewords
logic_0, logic_1 = code.logical_states()

# Get logical operators
X = code.logical_X()
Z = code.logical_Z()
```

### Probabilistic Preparation

```python
from src.preparation_circuits import probabilistic_2_legged_code

# Prepare via post-selection
r = 0.5
branches = probabilistic_2_legged_code(r=r, num_moments=200)

# branches[0] contains state and probability for |0_L⟩
# branches[1] contains state and probability for |1_L⟩
prob_logical_1 = branches[1]['prob']
```

### Computing Cost Functions

```python
from src.cost_functions import compute_cost_on_logical_codewords
import numpy as np

# Compare codes with different numbers of legs
costs = compute_cost_on_logical_codewords(
    fixed_param_name="r",
    fixed_value=2.0,
    x_name="γ",
    x_vec=np.logspace(-6, -2, 20),
    num_moments=500,
    num_code_states=3,  # Compare m=2,4,6
    code="squeeze",
    measurement="overlap01",
    noise_method="kraus-KL-style"
)
```

### Verifying QECC Conditions

```python
from scripts.prove_QECC import (
    identify_error_model,
    define_encoding_scheme,
    demonstrate_error_detection,
    verify_qecc_conditions
)

# Define code
num_moments = 100
code_states = define_encoding_scheme("squeezed", num_moments)

# Check orthogonality
overlap = demonstrate_error_detection(*code_states)

# Verify Knill-Laflamme conditions
errors = identify_error_model()  # ['loss', 'dephasing']
conditions = verify_qecc_conditions(code_states, errors, num_moments)
```

### Analytical Probability Calculation

```python
from scripts.probability_of_preparation import prob_analytical_evaluated_at

# Analytical formula for preparation success
r_value = 1.0
prob_0 = prob_analytical_evaluated_at(L=0, r_val=r_value)
prob_1 = prob_analytical_evaluated_at(L=1, r_val=r_value)

print(f"P(|0_L⟩) = {prob_0:.4f}")
print(f"P(|1_L⟩) = {prob_1:.4f}")
```

### Visualization

```python
from src.visualizations import plot_light_states

# Create and visualize code states
from src.codes_built_in_superposition import simple_m_legged_code

psi_0, psi_1 = simple_m_legged_code(
    m=4,                # 4-legged code
    strength=2.0,       # squeezing strength
    num_moments=200,
    code_type="squeeze"
)

plot_light_states([psi_0, psi_1])
```

## Key Scripts

### Research Scripts

- **`scripts/probability_of_preparation.py`**: Analyzes success probability of probabilistic preparation as a function of squeezing strength. Compares analytical expressions with numerical simulations.

- **`scripts/prove_QECC.py`**: Systematically verifies that the codes satisfy quantum error correction conditions. Implements the 8-step verification process from theory.

- **`scripts/study_measurement_free_code.py`**: Explores deterministic preparation using conditional rotations without ancilla measurement.

- **`scripts/deterministic_preparation.py`**: Implements feed-forward protocols for deterministic code preparation.

- **`scripts/logical_x_test.py`**: Tests logical X operator implementation and validates state transitions.

### Analysis Scripts

- **`scripts/graphs_for_paper/code_family.py`**: Generates performance comparison plots for codes with different numbers of legs.

- **`scripts/graphs_for_paper/numerics1.py`**: Produces numerical analysis figures for publication.

## Configuration

Edit `globals.py` to control:

```python
PRECISE = True          # Use high-precision arithmetic (mpmath)
DEBUG = True            # Enable assertion checks
CACHE_ON_DISK = True    # Cache expensive computations
LaTeX_RENDERING = True  # Use LaTeX in plots
```

## Caching

The code uses `joblib` for caching expensive computations. Cache directories:
- **RAM cache**: Temporary, cleared on exit
- **Disk cache**: `joblib_cache/` (persistent across runs)

Clear cache if you modify core functions:
```bash
rm -rf joblib_cache/
```

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Key test files:
- `tests/test_kraus.py`: Validate Kraus operator completeness
- `tests/test_squeeze_behavior.py`: Check squeezing operations
- `tests/test_safe_factorial.py`: Numerical stability tests

## Performance Notes

- **Fock space truncation**: Use `num_moments=200-500` for most calculations. Higher values increase accuracy but slow computation.
- **Kraus operators**: Dephasing channel requires many more operators than loss. Adjust `KRAUS_COST_THRESHOLD` in `globals.py` if needed.
- **Precision**: Enable `PRECISE=True` for Kraus operators with very small γ to avoid numerical underflow.
- **Parallelization**: Some functions support parallel execution via joblib (see `src/utils/caches.py`).

## Mathematical Notation

- `r`: Squeezing strength (unitless)
- `θ`: Squeezing direction angle
- `m`: Number of legs in code
- `γ`: Noise strength parameter
- `α`: Displacement amplitude (for cat codes)
- `|0_L⟩, |1_L⟩`: Logical codewords
- `a, a†`: Bosonic ladder operators
- `n̂ = a†a`: Number operator
- `X, P`: Position and momentum operators

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{squeezed-vacuum-codes,
  author = {AdQuanta - Quantum Information},
  title = {Squeezed Vacuum Quantum Error Correction Codes},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AdQuanta-Quantum-Information/squeezed-vacuum-codes}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 AdQuanta - Quantum Information

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

## Acknowledgments

This work uses the QuTiP (Quantum Toolbox in Python) library for quantum mechanics simulations.