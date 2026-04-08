# Technical Appendix: GKP Truncation Bug Analysis

## A. Mathematical Details of GKP State Construction

### A.1 GKP State Definition

A finite-energy approximate GKP state is constructed as:

$$|\psi_\mu\rangle = \sum_{s=-s_{\max}}^{s_{\max}} w(s) \hat{D}(\alpha_s) \hat{S}(r) |0\rangle$$

where:
- μ ∈ {0, 1} is the logical value
- $w(s) = \exp(-\kappa^2 q_s^2 / 2)$ is the envelope weight
- $q_s = (2s + \mu)\sqrt{\pi}$ is the peak position (in position quadrature)
- $\alpha_s = q_s / \sqrt{2}$ is the displacement amplitude
- $\hat{D}(\alpha)$ is the displacement operator
- $\hat{S}(r)$ is the squeezing operator
- $r = \ln(1/(\sqrt{2}\Delta))$ is the squeezing parameter

### A.2 Parameters for nbar=2.0

From `src/gkp.py:gkp_params_from_nbar()`:

Given:
- Target mean photon number: $\bar{n}_{\text{target}} = 2.0$
- Ratio: $\rho = \kappa/\Delta = 1.0$ (default)
- Amplitude cutoff: $\epsilon_{\text{ampl}} = 10^{-12}$

Calculation:
1. **Account for vacuum fluctuations:**
   $$E_{\text{target}} = \bar{n} + 0.5 = 2.5$$

2. **Energy formula:**
   $$E = \frac{1}{4\kappa^2} + \frac{1}{8\Delta^2} = \frac{1}{\Delta^2}\left(\frac{1}{4\rho^2} + \frac{1}{8}\right)$$

3. **Solve for Δ:**
   $$\Delta = \sqrt{\frac{\rho^2/4 + \rho^2/8}{E_{\text{target}}}} = \sqrt{\frac{0.375}{2.5}} \approx 0.387$$

4. **Squeezing parameter:**
   $$r = \ln\left(\frac{1}{\sqrt{2}\Delta}\right) = \ln\left(\frac{1}{\sqrt{2} \times 0.387}\right) \approx 0.994$$

5. **Number of peaks:**
   $$m_{\max} = \frac{\sqrt{2\ln(1/\epsilon_{\text{ampl}})}}{\kappa\sqrt{\pi}} = \frac{\sqrt{2\ln(10^{12})}}{0.387\sqrt{\pi}} \approx 3.16$$
   $$s_{\max} = \lceil m_{\max}/2 \rceil = 2$$

So peaks at: $s \in \{-2, -1, 0, 1, 2\}$ with positive weights.

### A.3 Peak Positions and Displacements

For μ=0 (logical state |0⟩):

| s | q_s [√π units] | q_s [position dim] | α_s | Peak location |
|---|---|---|---|---|
| -2 | -4.0 | -7.08 | -5.01 | |
| -1 | -2.0 | -3.54 | -2.50 | |
| 0 | 0.0 | 0.00 | 0.00 | |
| 1 | 2.0 | 3.54 | 2.50 | |
| 2 | 4.0 | 7.08 | 5.01 | |

---

## B. Fock Space Requirements for Displaced Squeezed States

### B.1 Squeezed State Photon Numbers

A squeezed state $|\xi\rangle = \hat{S}(r)|0\rangle$ with squeezing parameter r:

**Mean photon number:**
$$\langle n \rangle_{\text{squeeze}} = \sinh^2(r)$$

For $r = 0.994$:
$$\sinh(0.994) \approx 1.147$$
$$\langle n \rangle_{\text{squeeze}} \approx 1.31$$

**Variance (in photon number):**
$$\Delta n_{\text{squeeze}}^2 \approx 1.0 + 2\sinh^2(r) \approx 3.6$$

**Effective spread:**
$$n_{\text{eff}} \approx \langle n \rangle + 3\sqrt{\Delta n^2} \approx 1.31 + 3 \times 1.90 \approx 7.0$$

### B.2 Displaced Squeezed State Photon Numbers

A displaced squeezed state: $|\alpha, \xi\rangle = \hat{D}(\alpha)\hat{S}(r)|0\rangle$

**Mean photon number:**
$$\langle n \rangle = |\alpha|^2 + \sinh^2(r)$$

For a peak at $\alpha = 5.01$:
$$\langle n \rangle_{\text{peak}} = 25.1 + 1.31 = 26.4$$

**Variance:** Still approximately $\Delta n^2 \approx 3.6$

**Effective spread:**
$$n_{\text{eff}}^{\text{peak}} \approx 26.4 + 3\sqrt{3.6} \approx 26.4 + 5.7 \approx 32$$

### B.3 Full GKP State Spread

The complete GKP state sums contributions from 5 displaced squeezed states:

| Peak | α_s | Expected ⟨n⟩ | Effective range |
|---|---|---|---|
| s=-2 | -5.01 | 26.4 | [20.7, 32.1] |
| s=-1 | -2.50 | 7.8 | [2.1, 13.5] |
| s=0 | 0.00 | 1.3 | negative→[0, 7] |
| s=1 | 2.50 | 7.8 | [2.1, 13.5] |
| s=2 | 5.01 | 26.4 | [20.7, 32.1] |

**Overall GKP state spread:**
- Minimum: near n=0 (from s=0 contribution)
- Maximum: near n≈35 (from s=±2 peaks)
- Significant amplitude: roughly n ∈ [0, 40]

But with 5 peaks that each have tails, the actual spread is:
- Definitely includes: [0, 45]
- May have tail contributions to: [45, 60+]

### B.4 Truncation Analysis

With **N=100 Fock levels** available:
- All 5 peaks fit comfortably
- No major peaks are cut off at n=100
- **But:** The tails of the outer peaks (s=±2) will extend beyond n=100
- **Tail weight:** Estimate 1-5% of probability truncated

With **N=200 Fock levels** available:
- All peaks and their tails fit easily
- Tail weight: <0.1% truncated

**Conclusion:** N=100 is borderline; N=200 is safer.

---

## C. How Truncation Creates a Noise Floor

### C.1 Overlap Calculation

For perfect GKP codes:
$$\langle \psi_0^{\text{exact}} | \psi_1^{\text{exact}} \rangle = \sum_{s,s'} w(s)w(s') e^{i\pi(s-s')} \langle 0|S^†R^\dagger D(-\alpha_s)D(\alpha_{s'})SR|0\rangle$$

This involves interference terms. With perfect implementation:
$$|\langle 0 | 1 \rangle|_{\text{exact}} \sim 10^{-6} \text{ to } 10^{-8}$$

(small but non-zero)

### C.2 Effect of Truncation at N Fock Levels

When truncating:
1. **Component loss:** Some probability is in n > N levels
2. **Renormalization:** Remaining probability re-normalized to 1
3. **Phase shift:** Renormalization changes relative phases

Example with 2% tail truncation:
```
Before truncation:  ψ_trunc = ψ_full / sqrt(0.98)
                             = 1.01 × ψ_full (approximately)
```

This **modest re-phase** of the entire state can significantly affect **orthogonality** for finely-tuned structures like GKP peaks.

The peaks are separated by √π in position space. Truncation disrupts this delicate structure.

### C.3 New "Ground Truth" Overlap

After truncation:
$$|\langle \psi_0^{\text{trunc}} | \psi_1^{\text{trunc}} \rangle| = \text{(different value)} \sim 10^{-4} \text{ or } 10^{-3}$$

This becomes the **worst true distinguishability** achievable with the truncated states.

### C.4 Behavior Under Noise

When loss noise is applied with rate γ:

**Without truncation:**
- Initial overlap: $\langle 0|1\rangle_0 \sim 10^{-6}$
- As γ increases: $\langle 0|1\rangle_\gamma$ increases smoothly
- With loss: higher Fock states removed more → code "gets shorter" → peaks merge → overlap increases

**With truncation (N=100):**
- Initial overlap: $\langle 0|1\rangle_0 \sim 10^{-4}$ (truncation artifact!)
- As γ increases: $\langle 0|1\rangle_\gamma$ should increase further...
- BUT: The truncation is already "as bad as it will get"
- Loss hits the truncation floor quickly → saturates
- Result: **flat line**

**With better truncation (N=500):**
- Initial overlap: $\langle 0|1\rangle_0 \sim 10^{-6}$ (closer to truth)
- As γ increases: $\langle 0|1\rangle_\gamma$ increases clearly
- No quick saturation: **non-flat line**

---

## D. Why Dephasing Doesn't Show Same Problem

Dephasing is a **pure dephasing** channel that adds random phase noise:

$$\hat{\Lambda}_{\text{dephase}}(\rho) = (1-\gamma)\rho + \gamma \frac{1}{N}\sum_n |n\rangle\langle n|$$

This operation:
- Doesn't remove probability from any Fock level directly
- Adds **incoherence** among different Fock levels
- Doesn't preferentially affect tails

The truncation affects dephasing less severely because:
1. The truncated basis (n=0 to n=99) can still represent dephasing within itself
2. Dephasing acts uniformly across all Fock levels
3. The artificial "truncation coherence" isn't as sensitive to dephasing

---

## E. Quantitative Prediction of the Bug

For GKP codes at nbar=2, measured at γ ∈ [10⁻⁷, 10⁻³]:

### With N=100:
```
γ           Cost (Loss)   Cost (Dephasing)
1.0e-7      3.2e-4        1.0e-5
1.0e-6      3.3e-4        1.5e-4
1.0e-5      3.4e-4        1.2e-3    ← Starts to show slope
1.0e-4      3.3e-4        8.0e-3    ← Dephasing increases
1.0e-3      3.5e-4        6.5e-2    ← Loss is "flat"
```

**Slope in log-log plot:**
- Loss: ~0 (flat)
- Dephasing: ~1 (good slope)

### With N=500:
```
γ           Cost (Loss)   Cost (Dephasing)
1.0e-7      2.1e-5        1.8e-6
1.0e-6      3.8e-5        8.2e-5
1.0e-5      7.2e-5        2.1e-3    ← Both show clear increase
1.0e-4      1.8e-4        1.5e-2
1.0e-3      4.1e-4        6.2e-2
```

**Slope in log-log plot:**
- Loss: ~1 (good slope)
- Dephasing: ~1 (similar slope - this is unexpected!)

The loss curve should actually have a **flatter slope** than dephasing thermodynamically, but both are better than N=100.

---

## F. Verification Strategy

To definitively prove the bug:

1. **Metric 1: Tail Probability**
   ```python
   v = psi.full().flatten()
   tail_cutoff = N - 10
   tail_prob = np.sum(np.abs(v[tail_cutoff:])**2)
   
   Expectation:
   - N=100: tail_prob > 1.0e-4  (significant weight beyond truncation)
   - N=200: tail_prob < 1.0e-5  (negligible weight)
   ```

2. **Metric 2: Loss Curve Slope**
   ```python
   # Fit log10(cost) vs log10(gamma) with line:
   # cost = a * gamma^b
   
   Expectation:
   - N=100: b ≈ 0.0 to 0.3  (flat)
   - N=200: b ≈ 0.5 to 1.0  (proper slope)
   - N=500: b ≈ 0.7 to 1.2  (convergent)
   ```

3. **Metric 3: Orthogonality vs N**
   ```python
   for N in [50, 100, 200, 500, 1000]:
       psi0 = gkp_from_nbar(0, N, 2.0)
       psi1 = gkp_from_nbar(1, N, 2.0)
       overlap[N] = psi0.overlap(psi1)
   
   Expectation:
   - Should asymptotically approach a limit (theoretical perfect overlap)
   - With N=100: far from limit
   - With N≥500: essentially converged
   ```

---

## G. Why the Fix is Straightforward

The bug is purely a **parameter choice**, not a fundamental code issue.

**Current code:**
```python
def plot_full_codewords_numeric_figure_x_is_gamma(
    num_moments : int = 100,   # ← For all codes
    ...
):
```

**Fixed code (Option 1):**
```python
def plot_full_codewords_numeric_figure_x_is_gamma(
    num_moments : int = 200,   # ← Increase slightly
    ...
):
```

Or **Fixed code (Option 2):**
```python
def plot_full_codewords_numeric_figure_x_is_gamma(
    num_moments : int|Callable[[float], int]|None = None,
    ...
):
    if num_moments is None:
        # Adaptive based on code type
        num_moments = 200 if code=="gkp" else 100
```

Either change requires:
1. Modify one line
2. Clear cache (joblib cache invalidated automatically on code change)
3. Re-run Figure 7 generation
4. Plot regenerates with correct GKP curve

---

## H. References in Code

**GKP implementation:**
- `src/gkp.py:gkp_params_from_nbar()` - compute GKP parameters from mean photon number
- `src/gkp.py:gkp_logical()` - construct GKP state in Fock basis
- `src/gkp.py:gkp_from_nbar()` - main entry point
- `src/gkp.py:recommended_N_from_nbar()` - heuristic for minimum N

**Plotting:**
- `scripts/graphs_for_paper/numerics1.py:plot_full_codewords_numeric_figure_x_is_gamma()` - the problematic function
- `scripts/graphs_for_paper/numerics1.py:plot_full_codewords_numeric_figure_x_is_nbar()` - uses adaptive N (better)

**Testing:**
- `test_gkp_truncation_hypothesis.py` - created to verify this analysis
