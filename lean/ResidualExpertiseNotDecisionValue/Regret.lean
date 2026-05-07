import Mathlib

/-!
# Regret and Terminal Value

The PWLC terminal value `V(b) = max_a r_a · b` and regret `Regret(a, b) = V(b) - r_a · b`.
-/

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [Nonempty S]
variable {A : Type*} [Fintype A] [Nonempty A]

/-- The optimal expected reward at belief `b`: `V(b) = max_a ∑_s r(a,s) * b(s)`. -/
noncomputable def terminalValue (r : A → S → ℝ) (b : S → ℝ) : ℝ :=
  Finset.sup' Finset.univ Finset.univ_nonempty (fun a => ∑ s, r a s * b s)

/-- The regret of committing to action `a` at belief `b`: `V(b) - r_a · b`. -/
noncomputable def regret (r : A → S → ℝ) (a : A) (b : S → ℝ) : ℝ :=
  terminalValue r b - ∑ s, r a s * b s

/-- Regret is non-negative: `V(b) ≥ r_a · b` for all `a`. -/
lemma regret_nonneg (r : A → S → ℝ) (a : A) (b : S → ℝ) : 0 ≤ regret r a b := by
  unfold regret terminalValue; simp only [sub_nonneg]
  exact Finset.le_sup' (fun a' => ∑ s, r a' s * b s) (Finset.mem_univ a)

/-- Regret is zero iff `a` is optimal at `b`. -/
lemma regret_eq_zero_iff (r : A → S → ℝ) (a : A) (b : S → ℝ) :
    regret r a b = 0 ↔ ∑ s, r a s * b s = terminalValue r b := by
  unfold regret; constructor <;> intro h <;> linarith
