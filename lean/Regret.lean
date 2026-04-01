/-!
# Regret Interpretation of the Substitute Force

The Bregman divergence `D_h(b', b)` for `h = V = max_a r_a * b` equals the regret of using
action `a*(b)` at belief `b'`:

  `D_h(b', b) = V(b') - r_{a*(b)} * b' = Regret(a*(b), b')`

This gives an economic interpretation of the tug-of-war:

  `ΔVoI = E[D_g(b_post, b)] - E[Regret(a*(b), b_post)]`

Substitution occurs when expected regret exceeds expected information-value curvature.

## Main definitions

- `terminalValue` : the optimal expected reward `V(b) = max_a r_a * b`.
- `regret` : `Regret(a, b) = V(b) - r_a * b`.

## Main statements

- `regret_nonneg` : regret is non-negative.
- `regret_eq_zero_iff` : regret is zero iff the action is optimal.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/
import Mathlib

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [DecidableEq S] [Nonempty S]
variable {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]

/-- The optimal expected reward at belief `b`: `V(b) = max_a ∑_s r(a,s) * b(s)`. -/
noncomputable def terminalValue (r : A → S → ℝ) (b : S → ℝ) : ℝ :=
  Finset.sup' Finset.univ Finset.univ_nonempty (fun a => ∑ s, r a s * b s)

/-- The regret of committing to action `a` at belief `b`: `V(b) - r_a * b`. -/
noncomputable def regret (r : A → S → ℝ) (a : A) (b : S → ℝ) : ℝ :=
  terminalValue r b - ∑ s, r a s * b s

/-- Regret is non-negative: `V(b) ≥ r_a * b` for all `a`. -/
lemma regret_nonneg (r : A → S → ℝ) (a : A) (b : S → ℝ) :
    0 ≤ regret r a b := by
  unfold regret terminalValue
  simp only [sub_nonneg]
  exact Finset.le_sup' (fun a' => ∑ s, r a' s * b s) (Finset.mem_univ a)

/-- Regret is zero iff `a` is optimal at `b`. -/
lemma regret_eq_zero_iff (r : A → S → ℝ) (a : A) (b : S → ℝ) :
    regret r a b = 0 ↔ ∑ s, r a s * b s = terminalValue r b := by
  unfold regret; constructor <;> intro h <;> linarith
