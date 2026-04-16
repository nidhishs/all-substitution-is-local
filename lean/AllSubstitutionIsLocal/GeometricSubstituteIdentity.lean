import AllSubstitutionIsLocal.FacetActivation

/-!
# Geometric Substitute Identity

The quantitative identity expressing the expected Bregman divergence of the PWLC value
function as a facet-by-facet sum of reward-gap times perpendicular distance.

Let `R_a = {b' : a ∈ argmax_{a'} r_{a'} · b'}` denote the closed decision region for action
`a`. Fix `b ∈ Δ^{K-1}` and a tie-breaking convention selecting `a*(b) ∈ argmax_a r_a · b`.
For any channel with signal space `O` and posteriors `b^o`, let `a'_o` be the
posterior-optimal action. Define

  d_perp(b^o) := max(0, (r_{a'_o} - r_{a*(b)}) · b^o / ‖r_{a'_o} - r_{a*(b)}‖)
                 if a'_o ≠ a*(b), else 0.

Then

  E[D_h(b^, b)] = Σ_o P(o | b) · ‖r_{a'_o} - r_{a*(b)}‖ · d_perp(b^o),

where `D_h` is the Bregman divergence of the PWLC value `h = V(b) = max_a r_a · b`.

## Main statements

- `geometric_substitute_identity` : the identity stated above.
- `interior_complementarity_from_identity` : corollary — if all posteriors lie in a common
  decision region, the substitute force vanishes.
- `boundary_crossing_from_identity` : corollary — if the substitute force is strictly
  positive, some posterior lies in a different region.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/

open Finset BigOperators

set_option linter.unusedSectionVars false
set_option linter.unusedFintypeInType false

variable {S : Type*} [Fintype S] [Nonempty S]
variable {O : Type*} [Fintype O] [Nonempty O]
variable {A : Type*} [Fintype A] [Nonempty A]

/-! ## Geometric quantities -/

/-- The Euclidean norm of the reward gap `r_{a'} - r_{aStar}` over the state space. -/
noncomputable def rewardGapNorm (r : A → S → ℝ) (aStar a' : A) : ℝ :=
  Real.sqrt (∑ s : S, (r a' s - r aStar s) ^ 2)

lemma rewardGapNorm_nonneg (r : A → S → ℝ) (aStar a' : A) :
    0 ≤ rewardGapNorm r aStar a' := Real.sqrt_nonneg _

/-- If the reward-gap norm is zero, the reward vectors coincide pointwise on `S`. -/
lemma rewardGap_eq_zero_of_norm_zero (r : A → S → ℝ) (aStar a' : A)
    (h : rewardGapNorm r aStar a' = 0) : ∀ s, r a' s = r aStar s := by
  unfold rewardGapNorm at h
  have hsqnn : ∀ s : S, 0 ≤ (r a' s - r aStar s) ^ 2 := fun s => sq_nonneg _
  have hsum_zero : ∑ s : S, (r a' s - r aStar s) ^ 2 = 0 := by
    have hsum_nn : 0 ≤ ∑ s : S, (r a' s - r aStar s) ^ 2 :=
      Finset.sum_nonneg (fun s _ => hsqnn s)
    have := (Real.sqrt_eq_zero hsum_nn).mp h
    exact this
  intro s
  have hsq : (r a' s - r aStar s) ^ 2 = 0 := by
    have := Finset.sum_eq_zero_iff_of_nonneg (s := Finset.univ)
      (f := fun s => (r a' s - r aStar s) ^ 2)
      (fun s _ => hsqnn s) |>.mp hsum_zero s (Finset.mem_univ s)
    exact this
  have : r a' s - r aStar s = 0 := by
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hsq
  linarith

/-- The perpendicular distance of a point `p` past the `aStar`-vs-`a'` facet, truncated
at zero. Defined as `max(0, ⟨r_{a'} - r_{aStar}, p⟩ / ‖r_{a'} - r_{aStar}‖)`. -/
noncomputable def dPerp (r : A → S → ℝ) (aStar a' : A) (p : S → ℝ) : ℝ :=
  max 0 ((∑ s, (r a' s - r aStar s) * p s) / rewardGapNorm r aStar a')

lemma dPerp_nonneg (r : A → S → ℝ) (aStar a' : A) (p : S → ℝ) :
    0 ≤ dPerp r aStar a' p := le_max_left _ _

/-! ## The per-observation identity -/

/-- The dot-product `⟨r_{a'} - r_{aStar}, p⟩` equals `regret r aStar p` whenever `a'` is the
argmax at `p`. -/
lemma rewardGap_dot_eq_regret (r : A → S → ℝ) (aStar a' : A) (p : S → ℝ)
    (hopt : ∑ s, r a' s * p s = terminalValue r p) :
    ∑ s, (r a' s - r aStar s) * p s = regret r aStar p := by
  unfold regret
  simp_rw [sub_mul, Finset.sum_sub_distrib]
  linarith [hopt]

/-- **Per-observation facet identity.** The product `‖r_{a'} - r_{aStar}‖ · d_perp(p)`
equals the regret `regret r aStar p`, whenever `a'` is an argmax at `p`. -/
lemma rewardGapNorm_mul_dPerp_eq_regret
    (r : A → S → ℝ) (aStar a' : A) (p : S → ℝ)
    (hopt : ∑ s, r a' s * p s = terminalValue r p) :
    rewardGapNorm r aStar a' * dPerp r aStar a' p = regret r aStar p := by
  have hdot : ∑ s, (r a' s - r aStar s) * p s = regret r aStar p :=
    rewardGap_dot_eq_regret r aStar a' p hopt
  have hdot_nn : 0 ≤ ∑ s, (r a' s - r aStar s) * p s := by
    rw [hdot]; exact regret_nonneg r aStar p
  set n := rewardGapNorm r aStar a'
  by_cases hn : n = 0
  · -- Norm zero implies reward vectors agree, hence regret = 0
    have hreq : ∀ s, r a' s = r aStar s := rewardGap_eq_zero_of_norm_zero r aStar a' hn
    have hdot_zero : ∑ s, (r a' s - r aStar s) * p s = 0 := by
      apply Finset.sum_eq_zero
      intro s _
      rw [hreq s]; ring
    have hregret_zero : regret r aStar p = 0 := by rw [← hdot]; exact hdot_zero
    rw [hn]
    simp [hregret_zero]
  · -- Norm positive: max collapses to the numerator divided by norm
    have hn_pos : 0 < n := lt_of_le_of_ne (rewardGapNorm_nonneg r aStar a') (Ne.symm hn)
    have hnum_nn : 0 ≤ (∑ s, (r a' s - r aStar s) * p s) / n :=
      div_nonneg hdot_nn (le_of_lt hn_pos)
    have : dPerp r aStar a' p = (∑ s, (r a' s - r aStar s) * p s) / n := by
      unfold dPerp
      exact max_eq_right hnum_nn
    rw [this, mul_div_cancel₀ _ hn, hdot]

/-! ## Main theorem -/

/-- **Geometric Substitute Identity.** The Jensen gap of the PWLC value function equals a
facet-by-facet sum: each observation `o` contributes its marginal probability times the
product of the reward-gap norm and the perpendicular distance from the posterior past the
corresponding facet. -/
theorem geometric_substitute_identity
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0)
    (aStar : A) (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (aPost : O → A)
    (hopt_post : ∀ o, ∑ s, r (aPost o) s * (posterior b k o (hmarg o)).val s =
      terminalValue r (posterior b k o (hmarg o)).val) :
    jensenGap (fun b' => terminalValue r b'.val) b k hmarg =
    ∑ o : O, marginalProb b k o *
      (rewardGapNorm r aStar (aPost o) *
        dPerp r aStar (aPost o) (posterior b k o (hmarg o)).val) := by
  rw [jensenGap_eq_weighted_regret r b k hmarg aStar hopt_b]
  refine Finset.sum_congr rfl (fun o _ => ?_)
  rw [rewardGapNorm_mul_dPerp_eq_regret r aStar (aPost o)
    (posterior b k o (hmarg o)).val (hopt_post o)]

/-! ## Corollaries -/

/-- **Interior Complementarity (corollary).** If every posterior's argmax agrees with
`aStar`, then the Jensen gap of the PWLC value function is zero. -/
theorem interior_complementarity_from_identity
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0)
    (aStar : A) (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (hopt_post : ∀ o, ∑ s, r aStar s * (posterior b k o (hmarg o)).val s =
      terminalValue r (posterior b k o (hmarg o)).val) :
    jensenGap (fun b' => terminalValue r b'.val) b k hmarg = 0 := by
  rw [geometric_substitute_identity r b k hmarg aStar hopt_b
      (aPost := fun _ => aStar) hopt_post]
  apply Finset.sum_eq_zero
  intro o _
  have hzero : rewardGapNorm r aStar aStar = 0 := by
    unfold rewardGapNorm
    simp
  rw [hzero]; ring

/-- **Boundary-Crossing Converse (corollary).** If the Jensen gap of the PWLC value is
strictly positive, some posterior has an argmax differing from `aStar` (strictly: beats
`aStar` in dot product). -/
theorem boundary_crossing_from_identity
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0)
    (aStar : A) (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (hpos : jensenGap (fun b' => terminalValue r b'.val) b k hmarg > 0) :
    ∃ o : O, ∃ a : A,
      ∑ s, r a s * (posterior b k o (hmarg o)).val s >
      ∑ s, r aStar s * (posterior b k o (hmarg o)).val s :=
  (facet_activation_bridge r b k hmarg aStar hopt_b).mp hpos

/-! ## Relation to existing geometric wrappers

The pre-existing theorems `interior_complementarity_geometric` and
`substitution_requires_boundary_crossing_geometric` (in `InteriorComplementarity.lean`) are
left unchanged: they are corollaries via `jensenGap_terminalValue_zero_of_posteriors_optimal`
and `facet_activation_bridge`, which themselves are now one-line consequences of the
identity above. See `interior_complementarity_from_identity` and
`boundary_crossing_from_identity` for the direct identity-based forms. -/
