import AllSubstitutionIsLocal.Basics
import AllSubstitutionIsLocal.Regret

/-!
# Piecewise-Linearity of the Interaction

`E[V(b_post)]` is a sum of maxima of linear functions of `b`, establishing
piecewise-linearity of VoI and ΔVoI. The key step is the unnormalized cancellation
identity: the Bayesian normalizer cancels against the marginal, leaving each summand
linear in `b` for fixed `a`.

## Main statements

- `unnorm_posterior_value` : `P(o|b) * V(b^o) = sup_a ∑_s r(a,s) P(o|s) b(s)`.
- `expected_value_as_sup_linear` : `E[V(b_post)]` is a sum of sup of linear functions of `b`.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [Nonempty S]
variable {O : Type*} [Fintype O] [Nonempty O]
variable {A : Type*} [Fintype A] [Nonempty A]

/-- The unnormalized posterior value cancellation:
    `P(o|b) * V(b^o) = sup_a ∑_s r(a,s) * P(o|s) * b(s)`.
    The normalizing denominator in the posterior cancels against the marginal,
    leaving each summand linear in `b` for fixed action `a`. -/
theorem unnorm_posterior_value (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (o : O) (hmarg : marginalProb b k o > 0) :
    marginalProb b k o * terminalValue r (posterior b k o hmarg).val =
    Finset.sup' Finset.univ Finset.univ_nonempty
      (fun a => ∑ s, r a s * k.prob s o * b.val s) := by
  unfold terminalValue
  rw [Finset.comp_sup'_eq_sup'_comp _ (marginalProb b k o * ·)
    (fun x y => by simp [mul_max_of_nonneg x y (le_of_lt hmarg)])]
  congr 1; ext a
  simp only [Function.comp, posterior]
  rw [Finset.mul_sum]; congr 1; ext s; field_simp

/-- `E[V(b_post)] = ∑_o sup_a ∑_s r(a,s) * P(o|s) * b(s)`:
    the expected posterior value is a sum of maxima of linear functions of `b`. -/
theorem expected_value_as_sup_linear (r : A → S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    ∑ o : O, marginalProb b k o * terminalValue r (posterior b k o (hmarg o)).val =
    ∑ o : O, Finset.sup' Finset.univ Finset.univ_nonempty
      (fun a => ∑ s, r a s * k.prob s o * b.val s) := by
  congr 1; ext o; exact unnorm_posterior_value r b k o (hmarg o)
