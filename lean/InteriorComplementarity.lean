/-!
# Interior Complementarity

This file formalizes the main results on information interactions in Bayesian
decision problems. The central object is the Jensen gap `E[f(X)] - f(E[X])`,
which equals the expected Bregman divergence under the Bayesian martingale
property, avoiding the need for explicit subgradient bookkeeping.

## Main definitions

- `Belief` : a probability distribution over a finite state space.
- `ObsKernel` : a likelihood kernel (channel) mapping states to observation
  probabilities.
- `marginalProb` : the marginal probability of an observation.
- `posterior` : the Bayesian posterior belief after observing an outcome.
- `VoI` : the value of information of a channel.
- `jensenGap` : the Jensen gap `E[f(b_post)] - f(b)`.

## Main statements

- `bregman_decomposition` : the interaction `jensenGap(g - h)` splits as
  `jensenGap(g) - jensenGap(h)`.
- `jensenGap_add_const` : adding a constant to `f` does not change the
  Jensen gap.
- `jensenGap_linear` : linear functions have zero Jensen gap (the Bayesian
  martingale property).
- `interior_complementarity` : if the substitute force vanishes and the
  complement force is non-negative, then channels are complements.
- `substitution_requires_boundary_crossing` : if channels are substitutes and
  the complement force is non-negative, then the substitute force is strictly
  positive.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
-/
import Mathlib

open Finset BigOperators

set_option linter.unusedSectionVars false

variable {S : Type*} [Fintype S] [DecidableEq S] [Nonempty S]
variable {O : Type*} [Fintype O] [DecidableEq O] [Nonempty O]

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
    simp only [div_eq_mul_inv]
    rw [← Finset.sum_mul]
    rw [show ∑ s : S, b.val s * k.prob s o = marginalProb b k o from rfl]
    exact mul_inv_cancel₀ (ne_of_gt h)

/-- The value of information: `E[V(b_post)] - V(b)`. -/
noncomputable def VoI (V : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) : ℝ :=
  ∑ o : O, marginalProb b k o * V (posterior b k o (hmarg o)) - V b

/-! ## Jensen Gap -/

/-- The Jensen gap of `f` under the posterior lottery: `E[f(b_post)] - f(b)`. Under the
Bayesian martingale property this equals the expected Bregman divergence of `f`, avoiding
explicit subgradient bookkeeping. -/
noncomputable def jensenGap (f : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) : ℝ :=
  ∑ o : O, marginalProb b k o * f (posterior b k o (hmarg o)) - f b

/-- The value of information is the Jensen gap of `V`. -/
lemma VoI_eq_jensenGap (V : Belief S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    VoI V b k hmarg = jensenGap V b k hmarg := rfl

/-- Marginal probabilities sum to one. -/
lemma marginalProb_sum (b : Belief S) (k : ObsKernel S O) :
    ∑ o : O, marginalProb b k o = 1 := by
  simp only [marginalProb, Finset.sum_comm (s := univ (α := O))]
  simp_rw [← Finset.mul_sum, k.sum_one, mul_one]; exact b.sum_one

/-! ## Bregman Decomposition -/

/-- The Jensen gap of a difference splits: `jensenGap(g - h) = jensenGap(g) - jensenGap(h)`.
This decomposes the information interaction into a complement force and a substitute force. -/
theorem bregman_decomposition
    (g h : Belief S → ℝ) (k : ObsKernel S O) (b : Belief S)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    jensenGap (fun b' => g b' - h b') b k hmarg =
    jensenGap g b k hmarg - jensenGap h b k hmarg := by
  simp only [jensenGap, mul_sub, Finset.sum_sub_distrib]; ring

/-- Adding a constant to `f` does not change the Jensen gap. -/
theorem jensenGap_add_const (f : Belief S → ℝ) (c : ℝ)
    (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    jensenGap (fun b' => f b' + c) b k hmarg = jensenGap f b k hmarg := by
  simp only [jensenGap, mul_add, Finset.sum_add_distrib]
  have : ∑ o : O, marginalProb b k o * c = c := by
    rw [← Finset.sum_mul, marginalProb_sum, one_mul]
  linarith

/-- Linear functions have zero Jensen gap. This is the Bayesian martingale property:
`E[b_post(s)] = b(s)`. -/
theorem jensenGap_linear (w : S → ℝ) (b : Belief S) (k : ObsKernel S O)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    jensenGap (fun b' => ∑ s, w s * b'.val s) b k hmarg = 0 := by
  simp only [jensenGap, posterior, marginalProb, div_eq_mul_inv]
  have cancel : ∀ o : O, (∑ s, b.val s * k.prob s o) *
      (∑ s, w s * (b.val s * k.prob s o * (∑ s', b.val s' * k.prob s' o)⁻¹)) =
      ∑ s, w s * (b.val s * k.prob s o) := by
    intro o; rw [Finset.mul_sum]; congr 1; ext s
    have : (∑ s', b.val s' * k.prob s' o) ≠ 0 := ne_of_gt (hmarg o)
    field_simp
  simp_rw [cancel, Finset.sum_comm (s := univ (α := O)),
    ← mul_assoc, ← Finset.mul_sum, k.sum_one, mul_one, sub_self]

/-! ## Main Theorems -/

/-- **Interior complementarity**: if the substitute force vanishes and the complement force is
non-negative, then channels are complements (`jensenGap(g - h) ≥ 0`). -/
theorem interior_complementarity
    (g h : Belief S → ℝ) (k : ObsKernel S O) (b : Belief S)
    (hmarg : ∀ o, marginalProb b k o > 0)
    (hg_nonneg : jensenGap g b k hmarg ≥ 0)
    (hh_zero : jensenGap h b k hmarg = 0) :
    jensenGap (fun b' => g b' - h b') b k hmarg ≥ 0 := by
  rw [bregman_decomposition]; linarith

/-- **Substitution requires boundary-crossing**: if channels are substitutes
(`jensenGap(g - h) < 0`) and the complement force is non-negative, then the substitute force
is strictly positive (`jensenGap(h) > 0`).

The substitute force `E[h(b_post)] - h(b)` is positive iff `h` is nonlinear on the posterior
support, i.e., posteriors span multiple decision regions. Combined with
`interior_complementarity`, this localizes substitution to beliefs near decision boundaries.
The converse gap is non-empty: boundary-crossing does not imply substitution when the complement
force dominates. -/
theorem substitution_requires_boundary_crossing
    (g h : Belief S → ℝ) (k : ObsKernel S O) (b : Belief S)
    (hmarg : ∀ o, marginalProb b k o > 0)
    (hg_nonneg : jensenGap g b k hmarg ≥ 0)
    (hsub : jensenGap (fun b' => g b' - h b') b k hmarg < 0) :
    jensenGap h b k hmarg > 0 := by
  rw [bregman_decomposition] at hsub; linarith
