/-!
# Jensen's Inequality for Bayesian Belief Updates

This file proves Jensen's inequality in the setting of Bayesian posterior
lotteries: if `f` is convex, then `E[f(b_post)] ≥ f(b)`, or equivalently,
the Jensen gap is non-negative.

The proof chain is:
1. The prior is a convex combination of posteriors (`prior_eq_weighted_posterior`).
2. Convexity gives Jensen's inequality in this setting (`jensenGap_nonneg`).
3. A bridge to Mathlib's `ConvexOn` on the probability simplex
   (`convexOn_simplex_implies_beliefConvex`) connects to the standard library.

## Main definitions

- `Belief` : a probability distribution over a finite state space.
- `ObsKernel` : a likelihood kernel mapping states to observation probabilities.
- `BeliefConvexOn` : Jensen's inequality holds for all convex combinations of
  beliefs indexed by `I`.
- `probSimplex` : the probability simplex `{p : S → ℝ | p ≥ 0, ∑ p = 1}`.

## Main statements

- `prior_eq_weighted_posterior` : the prior is a convex combination of posteriors.
- `jensenGap_nonneg` : if `f` is belief-convex, the Jensen gap is non-negative.
- `convexOn_simplex_implies_beliefConvex` : Mathlib `ConvexOn` on the simplex
  implies `BeliefConvexOn`.
- `jensenGap_nonneg_of_convexOn` : end-to-end result combining the above.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/
import Mathlib

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [DecidableEq S] [Nonempty S]
variable {O : Type*} [Fintype O] [DecidableEq O] [Nonempty O]

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

/-- The Jensen gap of `f` under the posterior lottery: `E[f(b_post)] - f(b)`. -/
noncomputable def jensenGap (f : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) : ℝ :=
  ∑ o : O, marginalProb b k o * f (posterior b k o (hmarg o)) - f b

/-- Marginal probabilities sum to one. -/
lemma marginalProb_sum (b : Belief S) (k : ObsKernel S O) :
    ∑ o : O, marginalProb b k o = 1 := by
  simp only [marginalProb, Finset.sum_comm (s := univ (α := O))]
  simp_rw [← Finset.mul_sum, k.sum_one, mul_one]; exact b.sum_one

/-! ## Bayesian Identity -/

/-- The prior is a convex combination of posteriors: `b(s) = ∑_o m(o) * b_post(o, s)`. -/
lemma prior_eq_weighted_posterior (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (s : S) :
    b.val s = ∑ o : O, marginalProb b k o * (posterior b k o (hmarg o)).val s := by
  simp only [posterior, marginalProb]
  have hmarg' : ∀ o, (∑ s : S, b.val s * k.prob s o) ≠ 0 :=
    fun o => ne_of_gt (hmarg o)
  conv_rhs => arg 2; ext o; rw [mul_div_cancel₀ _ (hmarg' o)]
  rw [← Finset.mul_sum, k.sum_one, mul_one]

/-! ## Belief Convexity -/

/-- `f` is belief-convex over index type `I` if Jensen's inequality holds for all `I`-indexed
convex combinations of beliefs. -/
def BeliefConvexOn (I : Type*) [Fintype I] {S : Type*} [Fintype S]
    (f : Belief S → ℝ) : Prop :=
  ∀ (w : I → ℝ) (bs : I → Belief S) (b₀ : Belief S),
    (∀ i, 0 ≤ w i) → (∑ i : I, w i = 1) →
    (∀ s, b₀.val s = ∑ i : I, w i * (bs i).val s) →
    f b₀ ≤ ∑ i : I, w i * f (bs i)

/-! ## Jensen's Inequality -/

/-- If `f` is belief-convex, the Jensen gap is non-negative. -/
theorem jensenGap_nonneg (f : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0)
    (hconv : BeliefConvexOn O f) :
    0 ≤ jensenGap f b k hmarg := by
  unfold jensenGap
  have key : f b ≤ ∑ o : O, marginalProb b k o * f (posterior b k o (hmarg o)) :=
    hconv (fun o => marginalProb b k o) (fun o => posterior b k o (hmarg o)) b
      (fun o => le_of_lt (hmarg o))
      (marginalProb_sum b k)
      (fun s => prior_eq_weighted_posterior b k hmarg s)
  linarith

/-! ## Bridge to Mathlib's ConvexOn -/

/-- The probability simplex `{p : S → ℝ | p ≥ 0, ∑ p = 1}`. -/
def probSimplex (S : Type*) [Fintype S] : Set (S → ℝ) :=
  { p | (∀ s, 0 ≤ p s) ∧ ∑ s : S, p s = 1 }

/-- The probability simplex is convex. -/
lemma probSimplex_convex : Convex ℝ (probSimplex S) := by
  intro x hx y hy a b ha hb hab
  simp only [probSimplex, Set.mem_setOf_eq] at *
  refine ⟨fun s => add_nonneg (mul_nonneg ha (hx.1 s)) (mul_nonneg hb (hy.1 s)), ?_⟩
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      hx.2, hy.2, mul_one, mul_one, hab]

/-- A belief's value function lies in the probability simplex. -/
lemma belief_val_mem_simplex (b : Belief S) : b.val ∈ probSimplex S :=
  ⟨b.nonneg, b.sum_one⟩

/-- Mathlib `ConvexOn` on the probability simplex implies `BeliefConvexOn`. -/
theorem convexOn_simplex_implies_beliefConvex {I : Type*} [Fintype I]
    {g : (S → ℝ) → ℝ} (hg : ConvexOn ℝ (probSimplex S) g)
    {f : Belief S → ℝ} (hfg : ∀ b, f b = g b.val) :
    BeliefConvexOn I f := by
  intro w bs b₀ hw hws hval
  rw [hfg b₀]; simp_rw [hfg]
  have hval_eq : b₀.val = ∑ i : I, w i • (bs i).val := by
    ext s; simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]; exact hval s
  rw [hval_eq]
  have hmsl := hg.map_sum_le (t := univ) (fun i _ => hw i)
    (by simp [hws]) (fun i _ => belief_val_mem_simplex (bs i))
  simp only [smul_eq_mul] at hmsl
  exact hmsl

/-- Convexity on the simplex implies non-negative Jensen gap. Combines
`convexOn_simplex_implies_beliefConvex` and `jensenGap_nonneg`. -/
theorem jensenGap_nonneg_of_convexOn
    {g : (S → ℝ) → ℝ} (hg : ConvexOn ℝ (probSimplex S) g)
    {f : Belief S → ℝ} (hfg : ∀ b, f b = g b.val)
    (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    0 ≤ jensenGap f b k hmarg :=
  jensenGap_nonneg f b k hmarg (convexOn_simplex_implies_beliefConvex hg hfg)
