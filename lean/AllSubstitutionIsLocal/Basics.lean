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

/-- The Bayesian posterior after observing `o`: `b_post(s) = b(s) P(o|s) / P(o)`. -/
noncomputable def posterior (b : Belief S) (k : ObsKernel S O) (o : O)
    (h : marginalProb b k o > 0) : Belief S where
  val s := b.val s * k.prob s o / marginalProb b k o
  nonneg s := div_nonneg (mul_nonneg (b.nonneg s) (k.nonneg s o)) (le_of_lt h)
  sum_one := by
    simp only [div_eq_mul_inv]; rw [← Finset.sum_mul]
    rw [show ∑ s : S, b.val s * k.prob s o = marginalProb b k o from rfl]
    exact mul_inv_cancel₀ (ne_of_gt h)

/-- Marginal probabilities sum to one. -/
lemma marginalProb_sum (b : Belief S) (k : ObsKernel S O) :
    ∑ o : O, marginalProb b k o = 1 := by
  simp only [marginalProb, Finset.sum_comm (s := univ (α := O))]
  simp_rw [← Finset.mul_sum, k.sum_one, mul_one]; exact b.sum_one

/-- The prior is a convex combination of posteriors: `b(s) = ∑_o m(o) * b_post(o, s)`. -/
lemma prior_eq_weighted_posterior (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (s : S) :
    b.val s = ∑ o : O, marginalProb b k o * (posterior b k o (hmarg o)).val s := by
  simp only [posterior, marginalProb]
  have hmarg' : ∀ o, (∑ s : S, b.val s * k.prob s o) ≠ 0 :=
    fun o => ne_of_gt (hmarg o)
  conv_rhs => arg 2; ext o; rw [mul_div_cancel₀ _ (hmarg' o)]
  rw [← Finset.mul_sum, k.sum_one, mul_one]
