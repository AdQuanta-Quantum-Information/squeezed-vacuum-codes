# Math Changes: Proof Version → Current Version

Comparing `_agent_data/proof2.pdf` + `proof2-SuppMat.pdf` (the published proof)
against the current LaTeX source in `sections/` and `supplementary_material/`.

Focus: state-forming equations and their downstream consequences.

---

## 1. Core Definition — Phase Sign in `|ψ_k⟩` (Eq. 6 → `eq:psi_k`)

### Proof version (Eq. 6):
$$|\psi_k\rangle \propto \sum_{j=0}^{m-1} e^{+i \frac{2\pi j}{m} k} \, S\!\left(r,\frac{\pi j}{m}\right) |\text{vacuum}\rangle$$

### Current version (`eq:psi_k`):
$$|\psi_k\rangle \propto \sum_{j=0}^{m-1} e^{-i \frac{2\pi j}{m} k} \, S\!\left(r,\frac{\pi j}{m}\right) |\text{vac}\rangle$$

**Change:** The sign of the phase exponent was flipped from `+i` to `−i`.

---

## 2. Roots-of-Unity Sum in the Supplementary Derivation (SuppMat Eq. 22)

Expanding the definition into the Fock basis introduces a sum over `j` that filters photon numbers.

### Proof SuppMat (Eq. 22), arising from `e^{+i 2πjk/m}`:
$$\sum_{j=0}^{m-1} e^{i\frac{2\pi j}{m}(n+k)} = m \cdot \delta_{n \equiv -k \,(\mathrm{mod}\; m)}$$

### Current version (eq. `fock_amp_derivation`), arising from `e^{-i 2πjk/m}`:
$$\sum_{j=0}^{m-1} e^{i\frac{2\pi j}{m}(n-k)} = m \cdot \delta_{n \equiv k \,(\mathrm{mod}\; m)}$$

**Change:** The Kronecker-delta condition changes from `n ≡ −k (mod m)` to `n ≡ k (mod m)`.

---

## 3. Introduction of `k*` in the Proof — Eliminated in Current Version

Because the proof version's delta selected `n ≡ −k (mod m)`, the intermediate derivation needed an auxiliary index:

### Proof SuppMat introduces `k*`:
> "so by defining `k*` to be the first non-negative value to obey `k* := (−k) mod m`, we get:"

The final Fock-basis expression in the proof (Eq. 23) is in terms of `k*`:
$$|\psi_k\rangle = \frac{\tilde{N}_k\, m}{\sqrt{\cosh r}} \sum_{\ell=0}^{\infty}
\frac{\sqrt{[2(\ell m + k^*)]!}}{2^{\ell m+k^*}\,(\ell m+k^*)!}\,
(\tanh r)^{\ell m+k^*}\,|2(\ell m+k^*)\rangle$$

### Current version — no `k*` needed:
The final expression is directly in terms of `k` (eq. `fock_amp_derivation2`):
$$|\psi_k\rangle = \frac{\tilde{N}_k\, m}{\sqrt{\cosh r}} \sum_{\ell=0}^{\infty}
\frac{\sqrt{[2(\ell m + k)]!}}{2^{\ell m+k}\,(\ell m+k)!}\,
(\tanh r)^{\ell m+k}\,|2(\ell m+k)\rangle$$

**Change:** The `k*` workaround is entirely removed. All occurrences of `k*` are replaced by `k`.

---

## 4. Normalization Constant `Ñ_k` (SuppMat Eq. 24 → `eq:norm-Nk`)

### Proof SuppMat (Eq. 24) — uses `k*`:
$$\tilde{N}_k = \frac{\sqrt{\cosh r}}{m}
\left[\sum_{\ell=0}^{\infty} \frac{[2(\ell m+k^*)]!}{4^{\ell m+k^*}\,((\ell m+k^*)!)^2}\,\tanh^{2(\ell m+k^*)} r\right]^{-1/2}$$

### Current version (`eq:norm-Nk`) — uses `k`:
$$\tilde{N}_k = \frac{\sqrt{\cosh r}}{m}
\left[\sum_{\ell=0}^{\infty} \frac{[2(\ell m+k)]!}{4^{\ell m+k}\,((\ell m+k)!)^2}\,\tanh^{2(\ell m+k)} r\right]^{-1/2}$$

**Change:** `k*` → `k` throughout.

---

## 5. Mean Photon Number Formula (SuppMat Eq. 25 → `eq:nk-mean`)

### Proof SuppMat (Eq. 25) — uses `k*`:
$$n_k = \frac{\sum_{\ell=0}^{\infty} 2(\ell m + k^*)\,c_{k\ell}}{\sum_{\ell=0}^{\infty} c_{k\ell}},
\quad c_{k\ell} := \frac{[2(\ell m+k^*)]!}{4^{\ell m+k^*}\,((\ell m+k^*)!)^2}\,(\tanh r)^{2(\ell m+k^*)}$$

### Current version (`eq:nk-mean`) — uses `k`:
$$n_k = \frac{\sum_{\ell=0}^{\infty} 2(\ell m + k)\,c_{k\ell}}{\sum_{\ell=0}^{\infty} c_{k\ell}},
\quad c_{k\ell} := \frac{[2(\ell m+k)]!}{4^{\ell m+k}\,((\ell m+k)!)^2}\,(\tanh r)^{2(\ell m+k)}$$

**Change:** `k*` → `k` throughout.

---

## 6. Stated Fock-Space Support (unchanged in wording, but now internally consistent)

Both versions state:
> "In Fock space, this state occupies photon numbers `n ≡ 2k (mod 2m)`."

- **Proof version:** This statement was *inconsistent* with the derivation for general `k ≠ 0, m/2`. The proof's derivation gives support at `n = 2k* (mod 2m) = 2(−k mod m) (mod 2m)`, which equals `2k (mod 2m)` only for `k = 0` and `k = m/2` (the two logical codewords actually used).
- **Current version:** The statement is now *fully consistent* with the derivation for all `k ∈ [0, m−1]`.

---

## 7. Logical Codewords `|0_L⟩` and `|1_L⟩` (Eq. 7 → `eq:family_def`) — UNCHANGED

$$|0_L\rangle \propto \sum_{j=0}^{m-1} S\!\left(r,\frac{\pi j}{m}\right)|\text{vacuum}\rangle$$
$$|1_L\rangle \propto \sum_{j=0}^{m-1} (-1)^j\, S\!\left(r,\frac{\pi j}{m}\right)|\text{vacuum}\rangle$$

These equations are **identical** in both versions. This is expected: both `k=0` and `k=m/2` yield the same phase factor whether you use `e^{+i 2πjk/m}` or `e^{-i 2πjk/m}`, since `e^{±iπj} = (−1)^j`.

---

## 8. All Other Math Blocks — UNCHANGED

The following equations are identical in both versions:

| Equation | Content |
|---|---|
| Eq. (1) / `eq:sqz-ham-general` | Squeezing Hamiltonian H_S(κ) |
| Eq. (2) / `eq:sqz-unitary-phi` | Squeezing unitary S(ζ) |
| Eq. (3) / `eq:theta-to-phi` | Phase relation φ(θ) = 2θ + π |
| Eq. (4) / `eq:S_r_theta` | Direction-labeled S(r,θ) |
| Eq. (5) / `eq:conditional-squeezing` | CS(r;θ₀,θ₁) conditional squeezing |
| Eq. (8) / `eq:probability_of_preparation` | prob(L\|r) preparation probability |
| Eq. (9) / `eq:conditional-rotation` | CR(θ) conditional rotation |
| Eq. (10) / `eq:logical-x` | X̄ logical Pauli-X |
| Eq. (11) / `eq:cr_phase` | CR_phased(θ,φ) |
| Eq. (12) / `eq:cr_phase_decomposition` | Decomposition of CR_phased |
| Eq. (13) / `eq:logical-Zm` | Z̄ = R(π/m) logical Pauli-Z |
| Eq. (14) / `eq:CROT_mamb` | CZ entangling gate |
| Eq. (15–18) / noise/KL eqs. | Kraus operators, KL violation |
| SuppMat Eq. (20) / `eq:squeezed_vacuum_fock_rep` | Squeezed vacuum in Fock basis |

---

## Summary of Changes

| Location | What changed |
|---|---|
| Main body Eq. 6 (`eq:psi_k`) | Phase `e^{+i 2πjk/m}` → `e^{−i 2πjk/m}` |
| SuppMat Eq. 22 (roots-of-unity sum) | `δ_{n≡−k(mod m)}` → `δ_{n≡k(mod m)}` |
| SuppMat Eq. 23 (Fock expansion) | `k*` replaced by `k` everywhere |
| SuppMat Eq. 24 (normalization `Ñ_k`) | `k*` replaced by `k` everywhere |
| SuppMat Eq. 25 (mean photon number `n_k`) | `k*` replaced by `k` everywhere |
| Proof text in SuppMat | Definition of `k* := (−k) mod m` removed |

---

## Assessment: Is the New Version Correct?

**Yes, the current version is mathematically correct and is an improvement over the proof.**

### Why the proof had a flaw

In the proof, the phase convention `e^{+i 2πjk/m}` caused the roots-of-unity sum to select photon numbers satisfying `n ≡ −k (mod m)`. This means the Fock-space support of `|ψ_k⟩` was at `n = 2k* (mod 2m)` where `k* = (−k) mod m`, **not** at `n = 2k (mod 2m)` as stated in the paper text. The inconsistency was harmless for the published results because only `k = 0` and `k = m/2` were used as codewords — and for these two values, `k = k*` (since `−0 mod m = 0` and `−(m/2) mod m = m/2`). However, the general family of `|ψ_k⟩` states was incorrectly described for `k ≠ 0, m/2`.

### Why the current version is correct

Changing to `e^{−i 2πjk/m}` makes the roots-of-unity filter select `n ≡ k (mod m)`, placing Fock support at `n = 2k (mod 2m)` — exactly as stated. The `k*` detour is eliminated, the normalization and mean photon number formulas simplify, and the derivation is now fully internally consistent for all `k`.

### Additional checks

1. **Codewords `|0_L⟩`, `|1_L⟩` are unaffected.** The two logical states used in the code are given by `k=0` and `k=m/2`. For `k=0` the phase is 1 regardless of sign; for `k=m/2` both `e^{+iπj}` and `e^{−iπj}` equal `(−1)^j`. So no physical result changes.
2. **Distance property is unaffected.** The separation `Δn = m` between the two codeword supports (n=0,2m,4m,... vs n=m,3m,5m,...) is the same in both versions.
3. **Probability formula (Eq. 8)** and all gate definitions are unchanged.
4. **Normalization formula structure** is correct in the current version: `Ñ_k` depends only on `m, k, r`, and is well-defined for all `k ≥ 0`.
