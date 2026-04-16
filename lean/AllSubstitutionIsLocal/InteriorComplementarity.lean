import AllSubstitutionIsLocal.FacetActivation

/-!
# Interior Complementarity

The central results on information interactions in Bayesian decision problems.
The Jensen gap `E[f(X)] - f(E[X])` equals the expected Bregman divergence under
the Bayesian martingale property.

## Main statements

- `bregman_decomposition` : `jensenGap(g - h) = jensenGap(g) - jensenGap(h)`.
- `jensenGap_linear` : linear functions have zero Jensen gap (martingale property).
- `interior_complementarity` : if substitute force vanishes and complement force ≥ 0,
  channels are complements.
- `substitution_requires_boundary_crossing` : substitutes imply strictly positive substitute force.
- `interior_complementarity_geometric` : paper's Theorem 1 (geometric form).
- `substitution_requires_boundary_crossing_geometric` : paper's Theorem 2 (geometric form).

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [Nonempty S]
variable {O : Type*} [Fintype O] [Nonempty O]
variable {A : Type*} [Fintype A] [Nonempty A]

/-- The value of information: `E[V(b_post)] - V(b)`. -/
noncomputable def VoI (V : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) : ℝ :=
  ∑ o : O, marginalProb b k o * V (posterior b k o (hmarg o)) - V b

lemma VoI_eq_jensenGap (V : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    VoI V b k hmarg = jensenGap V b k hmarg := rfl

/-! ## Algebraic Core -/

/-- The Jensen gap of a difference splits: `jensenGap(g - h) = jensenGap(g) - jensenGap(h)`.
Decomposes the information interaction into complement and substitute forces. -/
theorem bregman_decomposition
    (g h : Belief S → ℝ) (k : ObsKernel S O) (b : Belief S)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    jensenGap (fun b' => g b' - h b') b k hmarg =
    jensenGap g b k hmarg - jensenGap h b k hmarg := by
  simp only [jensenGap, mul_sub, Finset.sum_sub_distrib]; ring

/-- Adding a constant to `f` does not change the Jensen gap. -/
theorem jensenGap_add_const (f : Belief S → ℝ) (c : ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    jensenGap (fun b' => f b' + c) b k hmarg = jensenGap f b k hmarg := by
  simp only [jensenGap, mul_add, Finset.sum_add_distrib]
  linarith [show ∑ o : O, marginalProb b k o * c = c by
    rw [← Finset.sum_mul, marginalProb_sum, one_mul]]

/-- Linear functions have zero Jensen gap (the Bayesian martingale property). -/
theorem jensenGap_linear (w : S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    jensenGap (fun b' => ∑ s, w s * b'.val s) b k hmarg = 0 := by
  simp only [jensenGap, posterior, marginalProb, div_eq_mul_inv]
  have cancel : ∀ o : O, (∑ s, b.val s * k.prob s o) *
      (∑ s, w s * (b.val s * k.prob s o * (∑ s', b.val s' * k.prob s' o)⁻¹)) =
      ∑ s, w s * (b.val s * k.prob s o) := by
    intro o; rw [Finset.mul_sum]; congr 1; ext s
    have : (∑ s', b.val s' * k.prob s' o) ≠ 0 := ne_of_gt (hmarg o)
    field_simp
  simp_rw [cancel, Finset.sum_comm (s := univ (α := O)),
    ← mul_assoc, ← Finset.mul_sum, k.sum_one, mul_one, sub_self]

/-! ## Main Theorems -/

/-- **Interior complementarity**: if the substitute force vanishes and the complement
force is non-negative, channels are complements (`jensenGap(g - h) ≥ 0`). -/
theorem interior_complementarity
    (g h : Belief S → ℝ) (k : ObsKernel S O) (b : Belief S)
    (hmarg : ∀ o, marginalProb b k o > 0)
    (hg_nonneg : jensenGap g b k hmarg ≥ 0)
    (hh_zero : jensenGap h b k hmarg = 0) :
    jensenGap (fun b' => g b' - h b') b k hmarg ≥ 0 := by
  rw [bregman_decomposition]; linarith

/-- **Substitution requires boundary-crossing**: if channels are substitutes
(`jensenGap(g - h) < 0`) and the complement force is non-negative, then the
substitute force is strictly positive (`jensenGap(h) > 0`). -/
theorem substitution_requires_boundary_crossing
    (g h : Belief S → ℝ) (k : ObsKernel S O) (b : Belief S)
    (hmarg : ∀ o, marginalProb b k o > 0)
    (hg_nonneg : jensenGap g b k hmarg ≥ 0)
    (hsub : jensenGap (fun b' => g b' - h b') b k hmarg < 0) :
    jensenGap h b k hmarg > 0 := by
  rw [bregman_decomposition] at hsub; linarith

/-! ## Geometric Wrappers (paper's Theorems 1 and 2) -/

/-- **Interior complementarity (geometric, Theorem 1)**: if `aStar` remains optimal at every
reachable posterior, and `g` has non-negative Jensen gap, then `jensenGap(g - V) ≥ 0`. -/
theorem interior_complementarity_geometric
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (aStar : A)
    (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (hopt_post : ∀ (o : O) (a : A),
      ∑ s, r a s * (posterior b k o (hmarg o)).val s
      ≤ ∑ s, r aStar s * (posterior b k o (hmarg o)).val s)
    (g : Belief S → ℝ) (hg_nonneg : jensenGap g b k hmarg ≥ 0) :
    jensenGap (fun b' => g b' - terminalValue r b'.val) b k hmarg ≥ 0 :=
  interior_complementarity g (fun b' => terminalValue r b'.val) k b hmarg hg_nonneg
    (jensenGap_terminalValue_zero_of_posteriors_optimal r b k hmarg aStar hopt_b hopt_post)

/-- **Substitution requires boundary-crossing (geometric, Theorem 2)**: if the combined
Jensen gap is strictly negative and `g` has non-negative Jensen gap, some posterior
strictly prefers a rival action — a decision boundary is crossed. -/
theorem substitution_requires_boundary_crossing_geometric
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (aStar : A)
    (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (g : Belief S → ℝ) (hg_nonneg : jensenGap g b k hmarg ≥ 0)
    (hsub : jensenGap (fun b' => g b' - terminalValue r b'.val) b k hmarg < 0) :
    ∃ o : O, ∃ a : A,
      ∑ s, r a s * (posterior b k o (hmarg o)).val s
      > ∑ s, r aStar s * (posterior b k o (hmarg o)).val s :=
  boundary_crossing_of_strict_jensenGap_terminalValue r b k hmarg aStar hopt_b
    (substitution_requires_boundary_crossing g (fun b' => terminalValue r b'.val)
      k b hmarg hg_nonneg hsub)
