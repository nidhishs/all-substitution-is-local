import AllSubstitutionIsLocal.Basics

/-!
# Jensen Gap and Belief Convexity

The Jensen gap `E[f(b_post)] - f(b)` for Bayesian posterior lotteries,
Jensen's inequality via `BeliefConvexOn`, and a bridge to Mathlib's `ConvexOn`.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [Nonempty S]
variable {O : Type*} [Fintype O] [Nonempty O]

/-- The Jensen gap of `f` under the posterior lottery: `E[f(b_post)] - f(b)`. -/
noncomputable def jensenGap (f : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) : ℝ :=
  ∑ o : O, marginalProb b k o * f (posterior b k o (hmarg o)) - f b

/-- `f` is belief-convex over index type `I` if Jensen's inequality holds for all
`I`-indexed convex combinations of beliefs. -/
def BeliefConvexOn (I : Type*) [Fintype I] {S : Type*} [Fintype S]
    (f : Belief S → ℝ) : Prop :=
  ∀ (w : I → ℝ) (bs : I → Belief S) (b₀ : Belief S),
    (∀ i, 0 ≤ w i) → (∑ i : I, w i = 1) →
    (∀ s, b₀.val s = ∑ i : I, w i * (bs i).val s) →
    f b₀ ≤ ∑ i : I, w i * f (bs i)

/-- If `f` is belief-convex, the Jensen gap is non-negative. -/
theorem jensenGap_nonneg (f : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) (hconv : BeliefConvexOn O f) :
    0 ≤ jensenGap f b k hmarg := by
  unfold jensenGap
  linarith [hconv (fun o => marginalProb b k o) (fun o => posterior b k o (hmarg o)) b
    (fun o => le_of_lt (hmarg o)) (marginalProb_sum b k)
    (fun s => prior_eq_weighted_posterior b k hmarg s)]

/-- The probability simplex `{p : S → ℝ | p ≥ 0, ∑ p = 1}`. -/
def probSimplex (S : Type*) [Fintype S] : Set (S → ℝ) :=
  { p | (∀ s, 0 ≤ p s) ∧ ∑ s : S, p s = 1 }

lemma probSimplex_convex : Convex ℝ (probSimplex S) := by
  intro x hx y hy a b ha hb hab
  simp only [probSimplex, Set.mem_setOf_eq] at *
  refine ⟨fun s => add_nonneg (mul_nonneg ha (hx.1 s)) (mul_nonneg hb (hy.1 s)), ?_⟩
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      hx.2, hy.2, mul_one, mul_one, hab]

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

/-- Convexity on the simplex implies non-negative Jensen gap. -/
theorem jensenGap_nonneg_of_convexOn
    {g : (S → ℝ) → ℝ} (hg : ConvexOn ℝ (probSimplex S) g)
    {f : Belief S → ℝ} (hfg : ∀ b, f b = g b.val)
    (b : Belief S) (k : ObsKernel S O) (hmarg : ∀ o, marginalProb b k o > 0) :
    0 ≤ jensenGap f b k hmarg :=
  jensenGap_nonneg f b k hmarg (convexOn_simplex_implies_beliefConvex hg hfg)
