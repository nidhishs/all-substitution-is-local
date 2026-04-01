/-!
# Piecewise-Linearity of the Interaction

This file proves that the expected posterior value `E[V(b_post)]` can be written
as a sum of maxima of linear functions of the belief `b`, from which
piecewise-linearity of VoI and ΔVoI follows.

The key step is the **unnormalized cancellation identity**: for each observation `o`,
the normalizing denominator in the posterior cancels against the marginal probability,
leaving the right-hand side manifestly linear in `b` for each fixed action `a`.

## Main definitions

- `Belief` : a probability distribution over a finite state space.
- `ObsKernel` : a likelihood kernel mapping states to observation probabilities.
- `terminalValue` : the optimal expected reward `V(b) = max_a r_a * b`.

## Main statements

- `unnorm_posterior_value` : the cancellation identity: `P(o|b) * V(b^o) = sup_a ∑_s r(a,s) P(o|s) b(s)`.
- `expected_value_as_sup_linear` : `E[V(b_post)]` equals a sum of `sup'` of linear
  functions of `b`.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/
import Mathlib

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [DecidableEq S] [Nonempty S]
variable {O : Type*} [Fintype O] [DecidableEq O] [Nonempty O]
variable {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]

/-! ## Bayesian Primitives -/

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

/-- The optimal expected reward at belief `b`: `V(b) = max_a ∑_s r(a,s) * b(s)`. -/
noncomputable def terminalValue (r : A → S → ℝ) (b : S → ℝ) : ℝ :=
  Finset.sup' Finset.univ Finset.univ_nonempty (fun a => ∑ s, r a s * b s)

/-! ## Cancellation Identity -/

/-- The unnormalized posterior value cancellation:
    `P(o|b) * V(b^o) = sup_a ∑_s r(a,s) * P(o|s) * b(s)`.
    The normalizing denominator in the posterior cancels against the marginal probability,
    leaving each summand linear in `b` for fixed action `a`. -/
theorem unnorm_posterior_value (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (o : O) (hmarg : marginalProb b k o > 0) :
    marginalProb b k o * terminalValue r (posterior b k o hmarg).val =
    Finset.sup' Finset.univ Finset.univ_nonempty
      (fun a => ∑ s, r a s * k.prob s o * b.val s) := by
  unfold terminalValue
  rw [Finset.comp_sup'_eq_sup'_comp _ (marginalProb b k o * ·)
    (fun x y => by simp [mul_max_of_nonneg x y (le_of_lt hmarg)])]
  congr 1
  ext a
  simp only [Function.comp, posterior]
  rw [Finset.mul_sum]
  congr 1
  ext s
  field_simp

/-! ## Expected Value as Sum of Sup of Linear Functions -/

/-- `E[V(b_post)] = ∑_o sup_a ∑_s r(a,s) * P(o|s) * b(s)`:
    the expected posterior value is a sum of maxima of linear functions of `b`. -/
theorem expected_value_as_sup_linear (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    ∑ o : O, marginalProb b k o * terminalValue r (posterior b k o (hmarg o)).val =
    ∑ o : O, Finset.sup' Finset.univ Finset.univ_nonempty
      (fun a => ∑ s, r a s * k.prob s o * b.val s) := by
  congr 1
  ext o
  exact unnorm_posterior_value r b k o (hmarg o)
