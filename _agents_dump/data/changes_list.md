Here is the conservative summary of changes made/requested since the proofing stage began.

## Main-text proof corrections

1. **Fixed a broken equation reference in Section 3.2.**
   The proof had “Equation )” when referring to the conditional rotation (CR(\theta)). This was corrected to refer properly to Eq./Equation (9). The broken form appears in the proof. 

2. **Fixed broken Algorithm references in Section 3.2.**
   The proof had production errors such as:

   * “protocol from1”
   * “summarized in2”
   * “Both Algorithms1 and 2,share”

   These were corrected to references to Algorithm 1 and Algorithm 2 with proper spacing. 

3. **Flagged/fixed subsection numbering under Logical Readout.**
   The proof used awkward numbering such as 4.4.0.1 and 4.4.0.2 for “Logical Z” and “Logical X.” The requested correction was to make these normal unnumbered or properly numbered headings.

4. **Fixed the Figure 5 caption spacing.**
   The proof had (R_z(\varphi))inserted with no space before “inserted.” The requested correction was to insert the missing space.

5. **Corrected the Introduction’s Supporting Information pointer.**
   The proof still said “appendices A to C,” but the final proof/SI structure uses Appendix SA–SD and includes Figure 9 in Appendix SD. The requested correction was to refer to the Supporting Information appendices consistently.

## Mathematical consistency correction

6. **Fixed the sign convention in Eq. (6).**
   Eq. (6) was changed from a positive phase convention to

   [
   e^{-i2\pi jk/m},
   ]

   so that the main-text statement (n\equiv 2k \pmod{2m}) is consistent with the derivation. In the latest version I saw, Eq. (6) indeed has the minus sign. 

7. **Updated the Supporting Information derivation to match the new convention.**
   The roots-of-unity filter was changed from the old (n+k) / (-k) convention to the new (n-k) / (k) convention. Consequently, the formulas now use (\ell m+k) directly rather than (k^\ast=(-k)\bmod m). This affects the derivation and notation only; it does not change the logical codewords, figures, numerics, or conclusions. 

## Supporting Information formatting changes

8. **Made the SI into a standalone file.**
   A title block was added to the SI, including the article title, author list, and affiliations. This is justified because the publisher moved the SI into a separate online file. 

9. **Changed the SI terminology to match the journal.**
   You decided to use **“Supporting Information”** rather than “Supplementary Material,” matching the journal’s website/proof terminology.

10. **Changed SI section and equation labels to match the editor’s format.**
    The SI sections/equations were adjusted to follow the proof convention, e.g. Appendix SA–SD and Equation S(24)/S(25), rather than the original merged-manuscript A–D / Eq. 20–27 style.

11. **Handled the SI-only reference as Ref. [73].**
    You decided not to restart numbering locally in the SI. Instead, the single SI-only reference, Weedbrook et al., remains Ref. [73], formatted in the journal style. This is the least disruptive option because the editor had already moved this reference after the SI.

## Reference-list corrections

12. **Requested correction of malformed/incomplete references.**
    These include:

* Ref. 62: title typo and published-reference details.
* Ref. 63: missing arXiv information.
* Ref. 68: malformed punctuation after “R. Srinivas.”
* Ref. 71: missing journal/year/details.
* Ref. 72: missing arXiv or published-reference details.

13. **Decided not to fight Ref. 46.**
    The proof’s “M. Gutierrez Galan” likely reflects the correct two-part family name, so this does not need correction.

14. **Decided not to push Ref. 33 further.**
    Since the GitHub link works in your PDF, you are not insisting on a DOI or expanded bibliographic entry at this stage.

## What this is *not*


* No figures were changed.
* No numerical data were changed.
* No conclusions were changed.
* No new scientific claims were added.
* The main mathematical change is a sign-convention consistency correction between Eq. (6), the SI derivation, and the stated Fock-space support.

---

## Code changes implementing the mathematical corrections (items 6 & 7)

The following source-code edits were applied to bring the implementation into full alignment with the corrected math convention (`e^{-i2πjk/m}`, `k` in place of `k*`).

### `src/codes_built_in_superposition.py`

| Location | Old | New |
|---|---|---|
| `simple_m_legged_state`, leg phase factor | `exp(+1j * phase)` | `exp(-1j * phase)` |

The comment was updated accordingly: `# e^{-i2πjk/m}`.

### `src/mean_photon_number.py`

| Function | Old | New |
|---|---|---|
| `_analytic_mean_photon_number_for_squeezed_k_state` | `k_star = (-k) % m` then `n_expr = l*m + k_star` | `n_expr = l*m + k` (k_star line removed) |
| `_numerical_exact_summation_mean_photon_number_for_squeezed_codeword` | `k_star = (-k) % m` then `n = l*m + k_star` | `n = l*m + k` (k_star line removed) |

### `src/analytical_expressions.py`

| Location | Old | New |
|---|---|---|
| `get_normalization_factor` | `k_star = _get_first_nonnegative_k_mod_m(k, m)` then `n = l*m + k_star` | `n = l*m + k` (k_star line removed) |
| `_get_first_nonnegative_k_mod_m` | entire helper function present | **deleted** (no longer needed anywhere) |
| `squeezed_superposition_state`, Formula 1 | `k_star = _get_first_nonnegative_k_mod_m(k_val, m_val)` then `n_val = l*m_val + k_star` | `n_val = l*m_val + k_val` (k_star line removed) |
| `squeezed_superposition_state`, Formula 2 (debug cross-check) | `leg_phase = sp.exp(+i * 2 * π * j * k_val / m_val)` | `leg_phase = sp.exp(-i * 2 * π * j * k_val / m_val)` |

### Why these changes are safe for all existing results

For the two logical codewords actually used (`k = 0` and `k = m/2`), the old and new conventions are numerically identical:
- `k = 0`: phase factor is 1 regardless of sign.
- `k = m/2`: `e^{±iπj} = (−1)^j` — same result either way.

The changes only affect the general `|ψ_k⟩` family for `k ≠ 0, m/2`, which was never exposed as a physical output in the published results.

### Verification

The full test suite was run after the changes:

```
pytest tests/ -x -q
```

**Result: 13 passed, 2 warnings in 193 s.**

The 2 warnings are pre-existing and unrelated to these changes:
- `PytestReturnNotNoneWarning` in `test_kraus.py` (test returns a value instead of using `assert`).
- `RuntimeWarning: overflow encountered in exp` in `src/utils/maths.py` (large-argument factorial edge case).
