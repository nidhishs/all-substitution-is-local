import AllSubstitutionIsLocal.GeometricSubstituteIdentity
import AllSubstitutionIsLocal.PiecewiseLinear

/-!
# Geometric Complement Identity

The parallel identity for the *complement force* `E[D_g]`, matching the structural form
of `geometric_substitute_identity` in `GeometricSubstituteIdentity.lean`.

## Setup

Let `i` be the outer channel (posteriors `b̂_{i, o_i}`) and `j` be the inner channel
(kernel `k_j`). The complement force is the Jensen gap of `g(b') := E_{o_j}[V(post_j(o_j | b'))]`
under channel `i`'s lottery.

Using `unnorm_posterior_value`, each `o_j`-summand of `g` is a PWLC function in `b'` with
*twisted rewards*

  `r̃_{a, o_j}(s) := r_a(s) · k_j(s, o_j)`.

Concretely, `P(o_j | b') · V(post_j(o_j | b')) = max_a Σ_s r̃_{a, o_j}(s) · b'(s) =
terminalValue (r̃_{·, o_j}) b'.val`.

Summing over `o_j` gives the `g` used in the paper's Proposition 1.

## Identity

```
E[D_g(b̂_i, b)] = Σ_{o_i} P(o_i | b, i) · Σ_{o_j}
                 ‖r̃_{a_j*(b̂_{i,o_i}), o_j} − r̃_{a_j*(b), o_j}‖
                 · d_⊥( b̂_{i,o_i}; F̃_{a_j*(b), a_j*(b̂_{i,o_i}), o_j} )
```

The proof reuses `geometric_substitute_identity` per inner observation `o_j` with rewards
`r̃_{·, o_j}`, sums, and exchanges the two finite sums.

## Combined sign(ΔVoI) characterization

Subtracting the substitute identity (with untwisted rewards) from the complement identity
gives a per-outer-posterior local comparison:

```
ΔVoI(j|i,b) = Σ_{o_i} P(o_i | b, i) · [
                Σ_{o_j} regret_j(a_j*(b), b̂_{i,o_i})
                       − regret(a*(b), b̂_{i,o_i}) ]
```

where `regret_j` is the regret in the twisted reward geometry for observation `o_j`.
This is the quantitative local characterization of `sign(ΔVoI)` that round-1 reviewers
asked for.

## References

- Shah, Mandal, Azhar, *All Substitution Is Local*.
- Paper proof and numerical verification in `lean/scratchpad.md`.
-/

open Finset BigOperators

set_option linter.unusedSectionVars false
set_option linter.unusedFintypeInType false

variable {S : Type*} [Fintype S] [Nonempty S]
variable {OI : Type*} [Fintype OI] [Nonempty OI]
variable {OJ : Type*} [Fintype OJ] [Nonempty OJ]
variable {A : Type*} [Fintype A] [Nonempty A]

/-! ## Twisted rewards and the `g` function -/

/-- Twisted reward matrix for the inner channel at observation `o_j`:
    `r̃_{a, o_j}(s) := r_a(s) · k_j(s, o_j)`. -/
noncomputable def twistedReward (r : A → S → ℝ) (k : ObsKernel S OJ) (o : OJ) :
    A → S → ℝ := fun a s => r a s * k.prob s o

/-- Per-inner-observation summand of `g`: the terminal value under twisted rewards. -/
noncomputable def gSummand (r : A → S → ℝ) (k : ObsKernel S OJ) (o : OJ)
    (b' : Belief S) : ℝ := terminalValue (twistedReward r k o) b'.val

/-- The "look-ahead" value `g(b') = E_{o_j ∼ P(·|b', j)}[V(post_j(o_j | b'))]`,
    expressed via twisted rewards. By `unnorm_posterior_value`, for any belief `b'` with
    positive marginals `P(o_j | b')`, we have `g(b') = Σ_{o_j} gSummand r k o_j b'`. -/
noncomputable def gValue (r : A → S → ℝ) (k : ObsKernel S OJ) (b' : Belief S) : ℝ :=
  ∑ o : OJ, gSummand r k o b'

/-- **Bridge to classical g.** For any belief `b'` with positive channel-j marginals, the
twisted-reward sum `gValue r k b'` equals the classical look-ahead value
`E_{o_j ∼ P(·|b', j)}[V(post_j(o_j | b'))]`.
This makes `gValue` the right object of study for the complement-force identity. -/
lemma gValue_eq_expected_post_value (r : A → S → ℝ) (k : ObsKernel S OJ)
    (b' : Belief S) (hmarg : ∀ o : OJ, marginalProb b' k o > 0) :
    gValue r k b' =
    ∑ o : OJ, marginalProb b' k o * terminalValue r (posterior b' k o (hmarg o)).val := by
  unfold gValue gSummand
  refine Finset.sum_congr rfl (fun o _ => ?_)
  -- Rewrite the RHS (classical form) using unnorm_posterior_value.
  rw [unnorm_posterior_value r b' k o (hmarg o)]
  -- LHS is `terminalValue (twistedReward r k o) b'.val`; expand it to the same `sup'`.
  unfold terminalValue twistedReward
  refine congr_arg _ ?_
  ext a
  refine Finset.sum_congr rfl (fun s _ => ?_)
  ring

/-! ## Jensen gap is additive in its function argument -/

/-- The Jensen gap is additive over finite sums of functions. -/
lemma jensenGap_sum {I : Type*} [Fintype I]
    (f : I → Belief S → ℝ) (b : Belief S) (k : ObsKernel S OI)
    (hmarg : ∀ o, marginalProb b k o > 0) :
    jensenGap (fun b' => ∑ i : I, f i b') b k hmarg =
    ∑ i : I, jensenGap (fun b' => f i b') b k hmarg := by
  unfold jensenGap
  rw [Finset.sum_sub_distrib]
  congr 1
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun o _ => ?_)
  rw [Finset.mul_sum]

/-! ## Main theorem -/

/-- **Geometric Complement Identity.** The Jensen gap of `g` under channel `i`'s posterior
lottery equals an expected double sum of reward-gap-weighted perpendicular crossing depths
in the twisted reward geometry of channel `j`. -/
theorem geometric_complement_identity
    (r : A → S → ℝ) (b : Belief S)
    (kI : ObsKernel S OI) (kJ : ObsKernel S OJ)
    (hmargI : ∀ o, marginalProb b kI o > 0)
    (aStarJ : OJ → A)
    (hoptJ_b : ∀ oJ, ∑ s, twistedReward r kJ oJ (aStarJ oJ) s * b.val s =
      terminalValue (twistedReward r kJ oJ) b.val)
    (aPostJ : OI → OJ → A)
    (hoptJ_post : ∀ oI oJ,
      ∑ s, twistedReward r kJ oJ (aPostJ oI oJ) s * (posterior b kI oI (hmargI oI)).val s =
      terminalValue (twistedReward r kJ oJ) (posterior b kI oI (hmargI oI)).val) :
    jensenGap (gValue r kJ) b kI hmargI =
    ∑ oI : OI, marginalProb b kI oI *
      ∑ oJ : OJ,
        rewardGapNorm (twistedReward r kJ oJ) (aStarJ oJ) (aPostJ oI oJ)
        * dPerp (twistedReward r kJ oJ) (aStarJ oJ) (aPostJ oI oJ)
            (posterior b kI oI (hmargI oI)).val := by
  -- Pull the outer sum through jensenGap, since g = Σ_{oJ} gSummand.
  have h_gval : gValue r kJ = fun b' => ∑ oJ : OJ, gSummand r kJ oJ b' := rfl
  rw [h_gval, jensenGap_sum]
  -- Apply `geometric_substitute_identity` per oJ with twisted rewards.
  have per_oJ : ∀ oJ : OJ,
      jensenGap (fun b' => gSummand r kJ oJ b') b kI hmargI =
      ∑ oI : OI, marginalProb b kI oI *
        (rewardGapNorm (twistedReward r kJ oJ) (aStarJ oJ) (aPostJ oI oJ)
          * dPerp (twistedReward r kJ oJ) (aStarJ oJ) (aPostJ oI oJ)
              (posterior b kI oI (hmargI oI)).val) := by
    intro oJ
    have := geometric_substitute_identity (twistedReward r kJ oJ) b kI hmargI
      (aStarJ oJ) (hoptJ_b oJ) (fun oI => aPostJ oI oJ)
      (fun oI => hoptJ_post oI oJ)
    -- jensenGap of the summand is exactly terminalValue-Jensen-gap for twisted rewards.
    exact this
  rw [show (∑ oJ : OJ, jensenGap (fun b' => gSummand r kJ oJ b') b kI hmargI) =
      ∑ oJ : OJ, ∑ oI : OI, marginalProb b kI oI *
        (rewardGapNorm (twistedReward r kJ oJ) (aStarJ oJ) (aPostJ oI oJ)
          * dPerp (twistedReward r kJ oJ) (aStarJ oJ) (aPostJ oI oJ)
              (posterior b kI oI (hmargI oI)).val)
      from Finset.sum_congr rfl (fun oJ _ => per_oJ oJ)]
  -- Exchange the two sums and distribute marginalProb.
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun oI _ => ?_)
  rw [Finset.mul_sum]

/-! ## Combined sign(ΔVoI) identity -/

/-- **Combined ΔVoI identity.** The difference between complement and substitute
Jensen gaps equals a per-outer-posterior sum of (twisted-regret minus untwisted-regret).
This is the local geometric characterization of `sign(ΔVoI)`. -/
theorem delta_voi_per_posterior_identity
    (r : A → S → ℝ) (b : Belief S)
    (kI : ObsKernel S OI) (kJ : ObsKernel S OJ)
    (hmargI : ∀ o, marginalProb b kI o > 0)
    (aStar : A) (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (aStarJ : OJ → A)
    (hoptJ_b : ∀ oJ, ∑ s, twistedReward r kJ oJ (aStarJ oJ) s * b.val s =
      terminalValue (twistedReward r kJ oJ) b.val) :
    jensenGap (gValue r kJ) b kI hmargI
    - jensenGap (fun b' => terminalValue r b'.val) b kI hmargI =
    ∑ oI : OI, marginalProb b kI oI *
      ((∑ oJ : OJ, regret (twistedReward r kJ oJ) (aStarJ oJ)
            (posterior b kI oI (hmargI oI)).val)
       - regret r aStar (posterior b kI oI (hmargI oI)).val) := by
  -- Rewrite both Jensen gaps as weighted regrets.
  rw [jensenGap_eq_weighted_regret r b kI hmargI aStar hopt_b]
  -- For the g-side: use jensenGap_sum + jensenGap_eq_weighted_regret per oJ.
  have h_gval : gValue r kJ = fun b' => ∑ oJ : OJ, gSummand r kJ oJ b' := rfl
  rw [h_gval, jensenGap_sum]
  have per_oJ : ∀ oJ : OJ,
      jensenGap (fun b' => gSummand r kJ oJ b') b kI hmargI =
      ∑ oI : OI, marginalProb b kI oI *
        regret (twistedReward r kJ oJ) (aStarJ oJ)
          (posterior b kI oI (hmargI oI)).val := by
    intro oJ
    exact jensenGap_eq_weighted_regret (twistedReward r kJ oJ) b kI hmargI
      (aStarJ oJ) (hoptJ_b oJ)
  rw [show (∑ oJ : OJ, jensenGap (fun b' => gSummand r kJ oJ b') b kI hmargI) =
      ∑ oJ : OJ, ∑ oI : OI, marginalProb b kI oI *
        regret (twistedReward r kJ oJ) (aStarJ oJ)
          (posterior b kI oI (hmargI oI)).val
      from Finset.sum_congr rfl (fun oJ _ => per_oJ oJ)]
  rw [Finset.sum_comm]
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl (fun oI _ => ?_)
  rw [← Finset.mul_sum, ← mul_sub]

/-! ## Corollaries -/

/-- **Per-posterior ΔVoI sign characterization.** If the bracketed local quantity
`Σ_{o_j} regret_j(a_j*(b), b̂_{i,o_i}) − regret(a*(b), b̂_{i,o_i})` is non-negative at
every outer posterior, then `ΔVoI ≥ 0` (channels are complements).
This is the round-2-requested local geometric characterization of `sign(ΔVoI)`. -/
theorem delta_voi_nonneg_of_local_complement
    (r : A → S → ℝ) (b : Belief S)
    (kI : ObsKernel S OI) (kJ : ObsKernel S OJ)
    (hmargI : ∀ o, marginalProb b kI o > 0)
    (aStar : A) (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (aStarJ : OJ → A)
    (hoptJ_b : ∀ oJ, ∑ s, twistedReward r kJ oJ (aStarJ oJ) s * b.val s =
      terminalValue (twistedReward r kJ oJ) b.val)
    (hlocal : ∀ oI : OI,
      regret r aStar (posterior b kI oI (hmargI oI)).val ≤
      ∑ oJ : OJ, regret (twistedReward r kJ oJ) (aStarJ oJ)
        (posterior b kI oI (hmargI oI)).val) :
    jensenGap (gValue r kJ) b kI hmargI
    - jensenGap (fun b' => terminalValue r b'.val) b kI hmargI ≥ 0 := by
  rw [delta_voi_per_posterior_identity r b kI kJ hmargI aStar hopt_b aStarJ hoptJ_b]
  apply Finset.sum_nonneg
  intro oI _
  exact mul_nonneg (le_of_lt (hmargI oI)) (by linarith [hlocal oI])

/-- **Boundary-crossing converse (complement side).** If `ΔVoI < 0` (substitutes), there
exists an outer posterior where the untwisted regret *strictly exceeds* the sum of
twisted-j regrets. Geometrically: channel j's combined decision-relevance at that posterior
is strictly less than the direct decision error. -/
theorem local_substitute_witness_of_delta_voi_neg
    (r : A → S → ℝ) (b : Belief S)
    (kI : ObsKernel S OI) (kJ : ObsKernel S OJ)
    (hmargI : ∀ o, marginalProb b kI o > 0)
    (aStar : A) (hopt_b : ∑ s, r aStar s * b.val s = terminalValue r b.val)
    (aStarJ : OJ → A)
    (hoptJ_b : ∀ oJ, ∑ s, twistedReward r kJ oJ (aStarJ oJ) s * b.val s =
      terminalValue (twistedReward r kJ oJ) b.val)
    (hneg : jensenGap (gValue r kJ) b kI hmargI
            - jensenGap (fun b' => terminalValue r b'.val) b kI hmargI < 0) :
    ∃ oI : OI,
      (∑ oJ : OJ, regret (twistedReward r kJ oJ) (aStarJ oJ)
        (posterior b kI oI (hmargI oI)).val)
      < regret r aStar (posterior b kI oI (hmargI oI)).val := by
  by_contra hall
  push Not at hall
  have := delta_voi_nonneg_of_local_complement r b kI kJ hmargI aStar hopt_b aStarJ hoptJ_b
    (fun oI => hall oI)
  linarith
