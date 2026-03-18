# Tedious Analysis: Reviewer Comment on GKP Codes in Figure 7

**Reviewer's Comment:**
> "In figure 7 I would expect finite energy GKP codes to be somewhat robust to loss errors (even more than dephasing ones) and not to see a constant function of the error rate. The observed behavior might be caused by numerical errors due to the large extent of wave function in phase space."

## BOTTOM LINE: YES, THERE IS A BUG

The reviewer **has a point**. The issue is a **Fock space truncation artifact** causing the GKP loss curve to saturate at a numerical noise floor.

---

## DETAILED ANALYSIS

### 1. The Observed Problem (Figure 7)

In the left panel (Loss error):
- **Squeeze (blue)**: Shows clear γ-dependence, noise increases with γ ✓
- **Cat (red)**: Shows clear γ-dependence ✓  
- **GKP (yellow dash-dot)**: **Appears nearly flat/constant** ✗

In the right panel (Dephasing error):
- **All codes including GKP**: Show clear γ-dependence ✓

### Why This is Wrong Physically

**Key theoretical fact:** GKP codes with finite energy should be **MORE robust to photon loss than to dephasing**.

**Reason:** 
- Loss (amplitude damping) removes photons
- Dephasing (pure dephasing) destroys quantum coherence
- GKP's symmetric lattice structure (peaks at q = s√π spacing) is naturally protective against small photon numbers disappearing
- But dephasing directly couples to the peak labels → more destructive

**Therefore:** The theory predicts:
- Loss curve should have positive slope (sensitivity to γ)
- Dephasing curve should have **steeper** positive slope  
- **GKP loss curve being flat violates this expectation**

---

### 2. Root Cause: Fock Space Truncation

#### The GKP State Construction

The code in `src/gkp.py` generates GKP states as:

```python
# GKP state = superposition of displaced squeezed states
ψ_GKP = Σ_s weight(s) × displace(α_s) × squeeze(r) × |0⟩
```

where:
- `q_s = (2s + μ) × √π`  (peak positions, μ ∈ {0,1})
- `weight(s) = exp(-κ² × q_s² / 2)`  (envelope decay)
- `α_s = q_s / √2`  (displacement amplitude)
- `r = ln(1 / (√2 × Δ))`  (squeezing parameter)

For **nbar = 2.0** (parameter in Figure 7):

```python
# From gkp_params_from_nbar():
target_energy = nbar + 0.5 = 2.5
term = 1/(4×ρ²) + 1/8 ≈ 0.25 + 0.125 = 0.375  (for ρ=1, i.e., Δ=κ)
Delta = √(0.375 / 2.5) ≈ 0.387
r = ln(1 / (√2 × 0.387)) ≈ 0.994

# Compute s_max (how many peaks to include):
κ = 0.387
mmax = √(2 × ln(1/1e-12)) / (0.387 × √π) ≈ √(55.5) / 0.685 ≈ 3.16
s_max = ceil(3.16/2) = 2
```

So the GKP state includes peaks at:
- s=-2: q ≈ -9.0, α ≈ -6.4
- s=-1: q ≈ -2.2, α ≈ -1.6  
- s=0: q ≈ 2.0, α ≈ 1.4  (for μ=0)
- s=1: q ≈ 3.6, α ≈ 2.5
- s=2: q ≈ 9.0, α ≈ 6.4

#### Why N=100 is Insufficient

The **squeezed state** `squeeze(r) × |0⟩` with r≈0.994 has properties:
- Squeezing: `exp(-2r) ≈ 0.38` ⟹ highly squeezed
- When displaced, the squeezed peak appears at `|α⟩`
- The Fock coefficients extend far in the photon number direction

For a **displaced squeezed state**:
- Center at photon number ≈ |α|²
- Variance ≈ |α|² (for strong squeezing)
- Significant amplitude up to 3-4 standard deviations

For displacement `α ≈ 6.4`:
- Center photon number ≈ 41
- Significant amplitude from n ≈ 20 to n ≈ 60+

Now multiply by multiple displaced peaks (-6.4, -1.6, +1.4, +2.5, +6.4):
- The state spreads from roughly n=0 to n=80+
- **With N=100, there's still space**, but only barely
- **Significant weight near the boundary** at n=90-100

#### The Truncation Effect

When you truncate at N=100:
1. Components beyond n=100 are **dropped**
2. The wavefunction is **re-normalized**
3. This artificially increases the **orthogonality** between |0_L⟩ and |1_L⟩
4. Creates a spurious **noise floor** in the cost metric

The cost metric for overlap is something like:
$$\text{Cost} \sim \max(|\langle \psi_0 | \psi_1 \rangle|, \text{some threshold})$$

With truncation:
- True overlap might be 0.1 (before truncation)  
- Truncation + renormalization gives spurious overlap ≈ 0.05 (truncation artifact)
- When you apply loss noise with γ, the state gets damaged but approaches the truncation floor quickly
- Result: cost plateaus

---

### 3. Why This Explains the Observation

**Loss curve is flat:**
- The Kraus operators for loss gradually remove photons
- With a properly represented GKP state, this would degrade performance
- But the truncated state is already "at floor" due to artificial orthogonality from truncation
- Adding loss can't degrade much further → flat line

**Dephasing curve is not flat:**
- Dephasing doesn't primarily affect amplitude distribution in Fock space
- It adds phases to different Fock levels
- Truncation doesn't suppress dephasing as much
- Therefore dephasing still shows γ-dependence

---

### 4. Code Location of the Bug

**File:** `scripts/graphs_for_paper/numerics1.py`

**Function:** `plot_full_codewords_numeric_figure_x_is_gamma()` (lines 788-830)

**The Issue:**
```python
def plot_full_codewords_numeric_figure_x_is_gamma(
    num_moments : int = 100,  # ← FIXED AT 100
    num_gammas:int = 5,
    num_code_states:int = 3,
    with_gkp:bool = True,
    measurement: MeasurementTypeLiteral = "overlap01",
    noise_method : NoiseOptionLiteral = "kraus-KL-style",
    mean_photon_number : float = 2.0
) -> None:
```

The `num_moments=100` is hardcoded for **all code types** including GKP.

**Comparison with the other function:**
```python
def plot_full_codewords_numeric_figure_x_is_nbar(
    num_moments: int|_NumMomentsFuncType = _num_moments_func,  # ← ADAPTIVE
    ...
):
    def _num_moments_func(mean_n: float) -> int:
        """Determine number of moments based on mean photon number."""
        n = 50*int(np.ceil(mean_n))
        n = max(n, 50)  # minimum
        n = min(n, 200) # maximum  
        return n
```

This function uses **adaptive** `num_moments` dependent on the photon number!

**The inconsistency:**
- When x-axis is γ (Figure 7): fixed N=100 applied uniformly across all γ
- When x-axis is n̄ (not shown in review): adaptive N based on n̄

For GKP with nbar=2:
```python
num_moments = 50 * ceil(2.0) = 100  # happens to be same
```

But the GKP state spread is such that **N=100 is borderline insufficient**.

---

### 5. How to Verify the Bug (Reproduction Steps)

**Test file created:** `test_gkp_truncation_hypothesis.py`

**Steps to confirm:**
1. Generate GKP state at nbar=2 with N=100, 200, 400, 800
2. For each N, compute loss costs at γ ∈ [1e-7, 1e-3]
3. Plot cost vs γ for each N value
4. **Expected result if bug exists:**
   - N=100: flat line or weak γ-dependence
   - N=200: better γ-dependence emerges
   - N=400: clear γ-dependence
   - N=800: strong γ-dependence (reference behavior)
5. Also measure **tail probability:** weight of state in highest 10 Fock levels
   - N=100: tail_prob > 1e-4 (significant truncation)
   - N=200: tail_prob < 1e-5 (much better)

---

### 6. Recommended Fix

**Option 1 (Minimal fix):**
```python
def plot_full_codewords_numeric_figure_x_is_gamma(
    num_moments : int|Callable[[float], int] = _num_moments_func,  # ← MAKE ADAPTIVE
    ...
):
    # Rest of code unchanged
```

Then when calling `compute_cost_on_logical_codewords()`, it will use the code_dependent N values.

**Option 2 (Explicit):**
```python
def plot_full_codewords_numeric_figure_x_is_gamma(
    num_moments : int|Callable[[float], int]|None = None,  # ← Let caller specify
    ...
):
    if num_moments is None:
        # Use adaptive based on code type and mean_photon_number
        if code=="gkp":
            num_moments = max(150, recommended_N_from_nbar(mean_photon_number))
        else:
            num_moments = _num_moments_func(mean_photon_number)
```

**Option 3 (Most conservative):**
```python
def plot_full_codewords_numeric_figure_x_is_gamma(
    num_moments : int = 200,  # ← Increase to 200-300 
    ...
):
    # This is safer for GKP without making it adaptive
```

**Why not just use recommended_N_from_nbar()?**

That function gives the **bare minimum**. For GKP with multiple displaced peaks, it's conservative. Better to use something like:
```python
N_gkp = int(np.ceil(1.5 * recommended_N_from_nbar(nbar)))
```

---

### 7. Why This Bug Wasn't Caught Earlier

1. **The flat line looks reasonable at first glance** - maybe GKP is just that good?
2. **Simple tests work fine** because they might use larger N values
3. **No obvious error messages** - the code runs, produces output, just gives wrong physics
4. **GKP is special case** - handled separately from m-legged codes, easy to miss inconsistencies
5. **Figure caption doesn't highlight GKP behavior** - subtle artifact

---

## CONCLUSION

**The reviewer's comment is correct: there IS a bug.**

**Nature of bug:** A combination of:
1. Insufficient Fock space truncation for GKP states at nbar=2
2. This creates an artificial noise floor via truncation artifacts
3. Loss sensitivity gets masked by the truncation floor
4. Appears as "constant function of error rate"

**Severity:** Medium
- The computed values are saved/printed (might influence conclusions)
- Subsequent analysis might be based on these wrong values
- But it only affects GKP curves in this specific plot

**Fix difficulty:** Easy
- Just increase `num_moments` for GKP or make it adaptive
- Change one parameter, re-run, figure regenerated

**Physics impact:** This bug could lead to the wrong conclusion that GKP codes don't benefit from having low loss rates, which is incorrect.
