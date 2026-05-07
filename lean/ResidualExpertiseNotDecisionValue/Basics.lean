import Mathlib

/-!
# Bayesian Primitives

Shared definitions used across all modules: `Belief`, `ObsKernel`, `marginalProb`,
`posterior`, and the two core identities `marginalProb_sum` and
`prior_eq_weighted_posterior`.
-/

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [Nonempty S]
variable {O : Type*} [Fintype O] [Nonempty O]

/-- A probability distribution over a finite state space `S`. -/
structure Belief (S : Type*) [Fintype S] where
  val : S → ℝ
  nonneg : ∀ s, 0 ≤ val s
  sum_one : ∑ s : S, val s = 1

/-- A likelihood kernel mapping states to observation probabilities. -/
structure ObsKernel (S O : Type*) [Fintype S] [Fintype O] where
  prob : S → O → ℝ
  nonneg : ∀ s o, 0 ≤ prob s o
  sum_one : ∀ s, ∑ o : O, prob s o = 1

/-- The marginal probability of observation `o`: `∑_s b(s) * P(o | s)`. -/
noncomputable def marginalProb (b : Belief S) (k : ObsKernel S O) (o : O) : ℝ :=
  ∑ s : S, b.val s * k.prob s o

/-- The Bayesian posterior after observing `o`: `b_post(s) = b(s) P(o|s) / P(o)`.
    When `marginalProb b k o = 0`, returns the prior `b` as a junk value. -/
noncomputable def posterior (b : Belief S) (k : ObsKernel S O) (o : O) : Belief S :=
  if h : marginalProb b k o > 0 then
    { val := fun s => b.val s * k.prob s o / marginalProb b k o
      nonneg := fun s => div_nonneg (mul_nonneg (b.nonneg s) (k.nonneg s o)) (le_of_lt h)
      sum_one := by
        simp only [div_eq_mul_inv]; rw [← Finset.sum_mul]
        rw [show ∑ s : S, b.val s * k.prob s o = marginalProb b k o from rfl]
        exact mul_inv_cancel₀ (ne_of_gt h) }
  else b

/-- Marginal probabilities are non-negative. -/
lemma marginalProb_nonneg (b : Belief S) (k : ObsKernel S O) (o : O) :
    0 ≤ marginalProb b k o :=
  Finset.sum_nonneg fun s _ => mul_nonneg (b.nonneg s) (k.nonneg s o)

/-- When the marginal is positive, the posterior has the Bayes formula. -/
lemma posterior_val_eq (b : Belief S) (k : ObsKernel S O) (o : O)
    (h : marginalProb b k o > 0) (s : S) :
    (posterior b k o).val s = b.val s * k.prob s o / marginalProb b k o := by
  simp [posterior, h]

/-- Marginal probabilities sum to one. -/
lemma marginalProb_sum (b : Belief S) (k : ObsKernel S O) :
    ∑ o : O, marginalProb b k o = 1 := by
  simp only [marginalProb, Finset.sum_comm (s := univ (α := O))]
  simp_rw [← Finset.mul_sum, k.sum_one, mul_one]; exact b.sum_one

/-- The prior is a convex combination of posteriors: `b(s) = ∑_o m(o) * b_post(o, s)`.
    Holds unconditionally — zero-mass observations contribute zero to the sum. -/
lemma prior_eq_weighted_posterior (b : Belief S) (k : ObsKernel S O) (s : S) :
    b.val s = ∑ o : O, marginalProb b k o * (posterior b k o).val s := by
  have key : ∀ o : O, marginalProb b k o * (posterior b k o).val s =
      b.val s * k.prob s o := by
    intro o
    by_cases h : marginalProb b k o > 0
    · rw [posterior_val_eq b k o h s]
      field_simp [ne_of_gt h]
    · have heq : marginalProb b k o = 0 :=
        le_antisymm (not_lt.mp h) (marginalProb_nonneg b k o)
      -- term b.val s * k.prob s o = 0 because it's a nonneg term in the zero sum
      have hterm : b.val s * k.prob s o = 0 := by
        apply le_antisymm _ (mul_nonneg (b.nonneg s) (k.nonneg s o))
        have hnn : ∀ s' ∈ Finset.univ, (0 : ℝ) ≤ b.val s' * k.prob s' o :=
          fun s' _ => mul_nonneg (b.nonneg s') (k.nonneg s' o)
        have : b.val s * k.prob s o ≤ ∑ s' : S, b.val s' * k.prob s' o :=
          Finset.single_le_sum hnn (Finset.mem_univ s)
        linarith [show ∑ s' : S, b.val s' * k.prob s' o = 0 from by simpa [marginalProb] using heq]
      -- posterior is junk value b, LHS = 0 * b.val s = 0
      have hpost : (posterior b k o).val s = b.val s := by
        simp [posterior, h]
      rw [hpost, heq, zero_mul, hterm]
  simp_rw [key, ← Finset.mul_sum, k.sum_one, mul_one]
