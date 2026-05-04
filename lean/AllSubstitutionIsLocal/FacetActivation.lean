import AllSubstitutionIsLocal.JensenGap
import AllSubstitutionIsLocal.Regret

/-!
# Facet-Activation Bridge Lemma

Connects "VoI > 0" (value-theoretic) to "some posterior crosses a decision boundary"
(geometric). This makes the decision-value interpretation a proved consequence: positive VoI is
equivalent to the human signal crossing a reward-induced decision boundary.

## Main statements

- `jensenGap_eq_weighted_regret` : VoI = weighted sum of regrets at posteriors.
- `regret_pos_iff_rival_action` : regret > 0 ↔ some rival action outperforms `aStar`.
- `facet_activation_bridge` : VoI > 0 ↔ some posterior prefers a rival action.
- `jensenGap_terminalValue_zero_of_posteriors_optimal` : Theorem 1 direction.
- `boundary_crossing_of_strict_jensenGap_terminalValue` : Theorem 2 direction.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [Nonempty S]
variable {O : Type*} [Fintype O] [Nonempty O]
variable {A : Type*} [Fintype A] [Nonempty A]

/-- The Jensen gap of the terminal value equals the weighted sum of regrets at posteriors. -/
lemma jensenGap_eq_weighted_regret
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (aStar : A)
    (hopt : ∑ s, r aStar s * b.val s = terminalValue r b.val) :
    jensenGap (fun b' => terminalValue r b'.val) b k hmarg =
    ∑ o : O, marginalProb b k o * regret r aStar (posterior b k o (hmarg o)).val := by
  simp only [jensenGap, regret]; rw [← hopt]
  have linear_expand : ∑ s, r aStar s * b.val s =
      ∑ o : O, marginalProb b k o *
        (∑ s, r aStar s * (posterior b k o (hmarg o)).val s) := by
    conv_lhs => arg 2; ext s; rw [prior_eq_weighted_posterior b k hmarg s]
    simp_rw [Finset.mul_sum, Finset.sum_comm (s := univ (α := O)), mul_left_comm]
  rw [linear_expand]; simp_rw [mul_sub, Finset.sum_sub_distrib]

private theorem strict_jensenGap_pwlc
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (aStar : A)
    (hopt : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (o0 : O) (hregret : regret r aStar (posterior b k o0 (hmarg o0)).val > 0) :
    jensenGap (fun b' => terminalValue r b'.val) b k hmarg > 0 := by
  rw [jensenGap_eq_weighted_regret r b k hmarg aStar hopt]
  have hnn : ∀ o ∈ Finset.univ,
      0 ≤ marginalProb b k o * regret r aStar (posterior b k o (hmarg o)).val :=
    fun o _ => mul_nonneg (le_of_lt (hmarg o)) (regret_nonneg r aStar _)
  calc (0 : ℝ)
      < marginalProb b k o0 * regret r aStar (posterior b k o0 (hmarg o0)).val :=
        mul_pos (hmarg o0) hregret
    _ ≤ ∑ o : O, marginalProb b k o * regret r aStar (posterior b k o (hmarg o)).val :=
        Finset.single_le_sum hnn (Finset.mem_univ o0)

/-- The rival-action characterization of positive regret:
    `regret r aStar p > 0` iff some rival action outperforms `aStar` at `p`.
    Geometrically: `p` lies across a decision boundary (facet activation). -/
lemma regret_pos_iff_rival_action (r : A → S → ℝ) (aStar : A) (p : S → ℝ) :
    regret r aStar p > 0 ↔ ∃ a : A, ∑ s, r a s * p s > ∑ s, r aStar s * p s := by
  unfold regret terminalValue
  constructor
  · intro h
    obtain ⟨a, _, ha⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty
      (fun a => ∑ s : S, r a s * p s)
    exact ⟨a, by linarith⟩
  · intro ⟨a, ha⟩
    linarith [Finset.le_sup' (fun a' => ∑ s : S, r a' s * p s) (Finset.mem_univ a)]

/-- **Facet-Activation Bridge**: VoI > 0 ↔ some posterior prefers a rival action over `aStar`.
    Positive decision value is equivalent to the human-updated posterior
    crossing a reward-induced decision facet. -/
theorem facet_activation_bridge
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (aStar : A)
    (hopt : ∑ s, r aStar s * b.val s = terminalValue r b.val) :
    jensenGap (fun b' => terminalValue r b'.val) b k hmarg > 0 ↔
    ∃ o : O, ∃ a : A,
      ∑ s, r a s * (posterior b k o (hmarg o)).val s >
      ∑ s, r aStar s * (posterior b k o (hmarg o)).val s := by
  constructor
  · intro hvoi
    rw [jensenGap_eq_weighted_regret r b k hmarg aStar hopt] at hvoi
    have hnn : ∀ o : O,
        0 ≤ marginalProb b k o * regret r aStar (posterior b k o (hmarg o)).val :=
      fun o => mul_nonneg (le_of_lt (hmarg o)) (regret_nonneg r aStar _)
    obtain ⟨o, _, ho⟩ : ∃ o ∈ Finset.univ,
        0 < marginalProb b k o * regret r aStar (posterior b k o (hmarg o)).val := by
      by_contra hcon; push Not at hcon
      linarith [Finset.sum_eq_zero (s := Finset.univ)
        (fun o _ => le_antisymm (hcon o (Finset.mem_univ o)) (hnn o))]
    have hregret : regret r aStar (posterior b k o (hmarg o)).val > 0 := by
      rcases (mul_pos_iff.mp ho) with ⟨_, h⟩ | ⟨h, _⟩
      · exact h
      · linarith [hmarg o]
    exact ⟨o, (regret_pos_iff_rival_action r aStar _).mp hregret⟩
  · intro ⟨o, a, ha⟩
    exact strict_jensenGap_pwlc r b k hmarg aStar hopt o
      ((regret_pos_iff_rival_action r aStar _).mpr ⟨a, ha⟩)

/-- **Theorem 1 direction**: if `aStar` remains optimal at every reachable posterior,
    the Jensen gap of the terminal value vanishes. -/
theorem jensenGap_terminalValue_zero_of_posteriors_optimal
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (aStar : A)
    (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (hopt_post : ∀ (o : O) (a : A),
      ∑ s, r a s * (posterior b k o (hmarg o)).val s
      ≤ ∑ s, r aStar s * (posterior b k o (hmarg o)).val s) :
    jensenGap (fun b' => terminalValue r b'.val) b k hmarg = 0 := by
  set V := jensenGap (fun b' => terminalValue r b'.val) b k hmarg
  have hge : 0 ≤ V := by
    rw [show V = _ from jensenGap_eq_weighted_regret r b k hmarg aStar hopt_b]
    exact Finset.sum_nonneg fun o _ =>
      mul_nonneg (le_of_lt (hmarg o)) (regret_nonneg r aStar _)
  have hle : V ≤ 0 := by
    by_contra hlt; push Not at hlt
    obtain ⟨o, a, ha⟩ := (facet_activation_bridge r b k hmarg aStar hopt_b).mp hlt
    exact absurd ha (not_lt.mpr (hopt_post o a))
  linarith

/-- **Theorem 2 direction**: if the Jensen gap of the terminal value is strictly positive,
    some posterior strictly prefers a rival action over `aStar`. -/
theorem boundary_crossing_of_strict_jensenGap_terminalValue
    (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (aStar : A)
    (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (hpos : jensenGap (fun b' => terminalValue r b'.val) b k hmarg > 0) :
    ∃ o : O, ∃ a : A,
      ∑ s, r a s * (posterior b k o (hmarg o)).val s
      > ∑ s, r aStar s * (posterior b k o (hmarg o)).val s :=
  (facet_activation_bridge r b k hmarg aStar hopt_b).mp hpos
