import Mathlib

/-!
# Measure-Theoretic Beliefs

Generalisation of `Basics.lean` to infinite state spaces via measure theory.

## Overview

`Basics.lean` formalises Bayesian reasoning over *finite* types using plain functions.
This file lifts the core structures to *arbitrary measurable spaces* using:

- `ProbabilityMeasure` (Mathlib) in place of `Belief`
- `ProbabilityTheory.Kernel` with `IsMarkovKernel` in place of `ObsKernel`
- `Measure.compProd` / `Measure.snd` for the marginal
- `Measure.condKernel` (Mathlib's regular conditional distribution) for the posterior
- `Measure.disintegrate` for the key identity

## Motivation

In program synthesis and generative modelling settings:
1. The state space `S` (programs, latents) may be infinite
2. Beliefs arise from generative models, not explicit probability tables
3. Observations may have zero probability under concentrated beliefs

The finite proofs break because:
- `∑ s : S` does not typecheck for infinite `S`
- The `∀ o, marginalProb > 0` assumption fails when beliefs concentrate

## Assumptions beyond the finite case

1. `[StandardBorelSpace S]` — required for `Measure.condKernel` to exist (ensures
   the regular conditional distribution is well-defined)
2. `[Nonempty S]` — required for `Measure.condKernel`
3. `[SFinite μ]` — for `Measure.compProd` (follows from `IsProbabilityMeasure`)
4. `[IsSFiniteKernel κ]` — for `Measure.compProd` (follows from `IsMarkovKernel`)
5. Positivity condition weakens: `∀ o, marginalProb > 0` becomes a pointwise condition
   in `bayesFormula`, and disappears entirely in the disintegration theorem

## Key design decision: swapping the joint

Mathlib's `Measure.condKernel : Measure (α × Ω) → Kernel α Ω` gives a kernel from
the *first* component to the second. For Bayesian inference we want `O → Measure S`
(posterior over states given observations). So we form the joint over `(O × S)` by
swapping `Prod.swap`, and apply `condKernel` to that.

## Mathlib coverage

| Component               | Mathlib lemma                        | Status                          |
|-------------------------|--------------------------------------|---------------------------------|
| Joint distribution      | `Measure.compProd`                   | ✓ available                     |
| Marginal on O           | `Measure.snd_compProd`               | ✓ available                     |
| Marginal on S           | `Measure.fst_compProd`               | ✓ available                     |
| Posterior kernel        | `Measure.condKernel`                 | ✓ needs `StandardBorelSpace S`  |
| Disintegration          | `Measure.disintegrate`               | ✓ available                     |
| Bayes formula pointwise | `Measure.condKernel_apply_of_ne_zero`| ✓ needs `MeasurableSingletonClass O` |
-/

open ProbabilityTheory MeasureTheory

set_option linter.unusedSectionVars false

variable {S O : Type*} [MeasurableSpace S] [MeasurableSpace O]

/-! ## §1  Beliefs as probability measures -/

/-- A **measure-theoretic belief** over `S` is a Borel probability measure `μ`.
    Replaces `Basics.Belief` (which uses `S → ℝ` summing to 1) when `S` may be infinite.

    The `ProbabilityMeasure S` type (Mathlib) is `{μ : Measure S // IsProbabilityMeasure μ}`. -/
structure MeasureBelief (S : Type*) [MeasurableSpace S] where
  μ : ProbabilityMeasure S

/-- Access the underlying `Measure`. -/
noncomputable def MeasureBelief.toMeasure (b : MeasureBelief S) : Measure S :=
  b.μ.toMeasure

/-- Coerce a `MeasureBelief` to its underlying `Measure` for use in Mathlib lemmas. -/
noncomputable instance instCoeToMeasure : Coe (MeasureBelief S) (Measure S) :=
  ⟨MeasureBelief.toMeasure⟩

/-- A belief carries an `IsProbabilityMeasure` instance. -/
instance instIsProbMeasureBelief (b : MeasureBelief S) :
    IsProbabilityMeasure (b : Measure S) := b.μ.prop

/-- A probability measure is σ-finite (needed for `Measure.compProd`). -/
instance instSFiniteBelief (b : MeasureBelief S) : SFinite (b : Measure S) := inferInstance

/-! ## §2  Markov kernels -/

/-- A **Markov kernel** `S → O`: for each `s : S`, a probability measure on `O`.
    Replaces `Basics.ObsKernel` (which uses `S → O → ℝ` summing to 1).

    Wraps `ProbabilityTheory.Kernel S O` with the `IsMarkovKernel` constraint,
    which asserts that `κ s` is a probability measure for every `s`. -/
structure MeasureKernel (S O : Type*) [MeasurableSpace S] [MeasurableSpace O] where
  k   : Kernel S O
  isM : IsMarkovKernel k

/-- Register the Markov property as a typeclass instance. -/
attribute [instance] MeasureKernel.isM

/-- A Markov kernel is s-finite (needed for `Measure.compProd`). -/
instance instIsSFiniteKernel (mk : MeasureKernel S O) : IsSFiniteKernel mk.k := inferInstance

/-! ## §3  Joint distribution -/

/-- The **joint distribution** `μ ⊗ₘ κ : Measure (S × O)`.
    Finite analogue: the table `b(s) * k(s, o)`.

    `Measure.compProd μ κ` integrates the kernel `κ` against `μ`:
    for a measurable set `A ⊆ S × O`,
    `(μ ⊗ₘ κ)(A) = ∫_s κ(s, {o | (s, o) ∈ A}) dμ(s)`. -/
noncomputable def jointMeasure (b : MeasureBelief S) (mk : MeasureKernel S O) :
    Measure (S × O) :=
  (b : Measure S) ⊗ₘ mk.k

/-- The joint is a probability measure (product of a probability measure and a Markov kernel). -/
instance instIsProbJoint (b : MeasureBelief S) (mk : MeasureKernel S O) :
    IsProbabilityMeasure (jointMeasure b mk) := by
  unfold jointMeasure; exact inferInstance

/-! ## §4  Marginal on observations -/

/-- The **observation marginal**: `(μ ⊗ₘ κ).snd`.
    Finite analogue: `marginalProb b k o = ∑_s b(s) * k(s, o)`.

    Key identity (Mathlib): `Measure.snd_compProd : (μ ⊗ₘ κ).snd = ↑κ ∘ₘ μ`
    i.e., the marginal equals the "bind" `μ.bind κ` (integrate `κ(s, ·)` over `μ`). -/
noncomputable def marginalMeasure (b : MeasureBelief S) (mk : MeasureKernel S O) :
    ProbabilityMeasure O :=
  ⟨(jointMeasure b mk).snd, by
    constructor
    simp [jointMeasure, Measure.snd_compProd, measure_univ]⟩

/-- The marginal `(μ ⊗ₘ κ).snd` equals the monadic bind `μ.bind κ`. -/
lemma marginalMeasure_eq_bind (b : MeasureBelief S) (mk : MeasureKernel S O) :
    (marginalMeasure b mk).toMeasure = (b : Measure S).bind (fun s => mk.k s) := by
  simp [marginalMeasure, ProbabilityMeasure.toMeasure, jointMeasure, Measure.snd_compProd]

/-! ## §5  Prior marginal identity -/

/-- The **prior** `μ` equals the `fst` marginal of the joint `μ ⊗ₘ κ`.
    Used in the disintegration proof to connect swappedJoint back to the prior. -/
lemma prior_eq_fst_marginal (b : MeasureBelief S) (mk : MeasureKernel S O) :
    (jointMeasure b mk).fst = (b : Measure S) :=
  Measure.fst_compProd (b : Measure S) mk.k

/-! ## §6  Swapped joint and posterior kernel -/

/-- The **swapped joint** `(μ ⊗ₘ κ).map Prod.swap : Measure (O × S)`.

    Design rationale: `Measure.condKernel : Measure (α × Ω) → Kernel α Ω` gives a kernel
    from the *first* component to the second. To get `O → Measure S` (the Bayesian posterior),
    we need the joint over `(O × S)`, which we obtain by swapping the standard `(S × O)` joint. -/
noncomputable def swappedJoint (b : MeasureBelief S) (mk : MeasureKernel S O) :
    Measure (O × S) :=
  (jointMeasure b mk).map Prod.swap

/-- The swapped joint is a probability measure (`Prod.swap` is a bijection). -/
instance instIsProbSwapped (b : MeasureBelief S) (mk : MeasureKernel S O) :
    IsProbabilityMeasure (swappedJoint b mk) := by
  constructor
  simp [swappedJoint, Measure.map_apply measurable_swap MeasurableSet.univ, measure_univ]

/-- The **posterior kernel**: `O → Measure S`.

    This is Mathlib's `Measure.condKernel` applied to the swapped joint.
    For `o : O`, `posteriorKernel b mk o` is the conditional distribution of `s` given `o`.

    REQUIRES: `[StandardBorelSpace S] [Nonempty S]`

    These conditions ensure `Measure.condKernel` exists as a measurable kernel.
    `StandardBorelSpace` is satisfied by `ℝ`, `ℕ`, any countable type, any compact metric space.

    Comparison to finite case:
    - Finite: `posterior b k o h : Belief S` — a function `S → ℝ`, requires `h : marginalProb > 0`
    - Infinite: `posteriorKernel b mk : Kernel O S` — defined everywhere, even at zero-probability `o` -/
noncomputable def posteriorKernel (b : MeasureBelief S) (mk : MeasureKernel S O)
    [StandardBorelSpace S] [Nonempty S] : Kernel O S :=
  (swappedJoint b mk).condKernel

/-- The posterior kernel is a Markov kernel (since `swappedJoint` is a probability measure). -/
instance instIsMarkovPosterior (b : MeasureBelief S) (mk : MeasureKernel S O)
    [StandardBorelSpace S] [Nonempty S] :
    IsMarkovKernel (posteriorKernel b mk) := by
  unfold posteriorKernel; exact inferInstance

/-! ## §7  Key lemma: fst of swappedJoint = observation marginal -/

/-- The `fst` marginal of the swapped joint equals the observation marginal.

    Proof strategy: `fst ∘ Prod.swap = snd` (elementwise), so
    `map fst (map Prod.swap ρ) = map (fst ∘ Prod.swap) ρ = map snd ρ`.

    This lemma connects the two representations:
    - `swappedJoint.fst` (what `Measure.disintegrate` uses internally)
    - `marginalMeasure` (our named definition)

    Without this lemma, `disintegration` cannot rewrite `swappedJoint.fst`
    in terms of `marginalMeasure`. -/
lemma swappedJoint_fst_eq_marginal (b : MeasureBelief S) (mk : MeasureKernel S O) :
    (swappedJoint b mk).fst = (marginalMeasure b mk).toMeasure := by
  simp only [swappedJoint, marginalMeasure, ProbabilityMeasure.toMeasure,
    Subtype.coe_mk, jointMeasure]
  rw [Measure.fst, Measure.map_map measurable_fst measurable_swap,
      show Prod.fst ∘ @Prod.swap S O = Prod.snd from funext fun _ => rfl,
      ← Measure.snd]

/-! ## §8  Disintegration — the measure-theoretic `prior_eq_weighted_posterior` -/

/-- **Disintegration theorem**: the observation marginal, averaged against the posterior
    kernel, reconstructs the swapped joint.

    `marginalO ⊗ₘ posteriorKernel = swappedJoint`

    Unpacking: for every measurable `A ⊆ O × S`,
    `∫_O posteriorKernel(o, Aₒ) d(marginalO)(o) = swappedJoint(A)`
    where `Aₒ = {s | (o, s) ∈ A}` is the slice.

    Finite version (`Basics.lean`, `prior_eq_weighted_posterior`):
    `b(s) = ∑_o m(o) * b_post(o)(s)`

    The finite version is a *section* of this theorem: take `A = O × {s}` (or integrate
    against a test function `1_s`). The measure version is strictly stronger.

    REQUIRES: `[StandardBorelSpace S] [Nonempty S]`

    Proof: apply `swappedJoint_fst_eq_marginal` to rewrite `marginalO` as `swappedJoint.fst`,
    then invoke Mathlib's `Measure.disintegrate`. -/
theorem disintegration (b : MeasureBelief S) (mk : MeasureKernel S O)
    [StandardBorelSpace S] [Nonempty S] :
    (marginalMeasure b mk).toMeasure ⊗ₘ posteriorKernel b mk = swappedJoint b mk := by
  rw [← swappedJoint_fst_eq_marginal]
  -- After rewriting, goal becomes the literal statement of Measure.disintegrate
  -- `show` is needed so Lean can find the IsCondKernel instance by unfolding posteriorKernel
  show (swappedJoint b mk).fst ⊗ₘ (swappedJoint b mk).condKernel = swappedJoint b mk
  exact Measure.disintegrate _ _

/-! ## §9  Bayes formula pointwise -/

/-- **Bayes formula at a point** (raw form).

    When `o` has nonzero marginal probability (in the sense of ENNReal measure):
    `posteriorKernel(o)(A) = marginalO({o})⁻¹ * swappedJoint({o} ×ˢ A)`

    This is the exact measure-theoretic Bayes formula:
    `P(s ∈ A | o) = P(o)⁻¹ * P(o, s ∈ A)`.

    Comparison to finite version (`Basics.lean`):
    - Finite: requires `hmarg : marginalProb b k o > 0` to define `posterior`
    - Infinite: `condKernel o` is defined for all `o`; the formula only holds at `o`
      where the marginal is nonzero, but the kernel is still *defined* everywhere

    REQUIRES:
    - `[StandardBorelSpace S] [Nonempty S]` (for `posteriorKernel`)
    - `[MeasurableSingletonClass O]` (for `{o}` to be a measurable set)
    - `(swappedJoint b mk).fst {o} ≠ 0` (positivity at `o`, as an ENNReal) -/
theorem bayesFormula (b : MeasureBelief S) (mk : MeasureKernel S O)
    [StandardBorelSpace S] [Nonempty S] [MeasurableSingletonClass O]
    (o : O) (ho : (swappedJoint b mk).fst {o} ≠ 0) (A : Set S) :
    posteriorKernel b mk o A =
      ((swappedJoint b mk).fst {o})⁻¹ * swappedJoint b mk ({o} ×ˢ A) :=
  Measure.condKernel_apply_of_ne_zero ho A

/-- **Bayes formula at a point** (cleaner form, using `marginalMeasure` directly).

    Equivalent to `bayesFormula`, with `swappedJoint.fst {o}` replaced by
    `marginalMeasure {o}` via `swappedJoint_fst_eq_marginal`.

    This matches the finite version more closely: the denominator is explicitly
    the observation marginal probability `m(o)`. -/
theorem bayesFormula' (b : MeasureBelief S) (mk : MeasureKernel S O)
    [StandardBorelSpace S] [Nonempty S] [MeasurableSingletonClass O]
    (o : O) (ho : (marginalMeasure b mk).toMeasure {o} ≠ 0) (A : Set S) :
    posteriorKernel b mk o A =
      ((marginalMeasure b mk).toMeasure {o})⁻¹ * swappedJoint b mk ({o} ×ˢ A) := by
  rw [← swappedJoint_fst_eq_marginal] at ho ⊢
  exact Measure.condKernel_apply_of_ne_zero ho A

/-! ## §10  Gap analysis: what additional lemmas are needed

### GAP 1: `prior_as_mixture` — the prior as a mixture of posteriors

`disintegration` gives us `marginalO ⊗ₘ posteriorKernel = swappedJoint` over `O × S`.

The finite identity is: `b(s) = ∑_o m(o) * b_post(o)(s)` (an equality of reals).

The measure-theoretic version would be:
```
(b : Measure S) = (swappedJoint b mk).snd
```
i.e., the prior is the `snd` marginal of the swapped joint.

Proof sketch:
  `(swappedJoint b mk).snd`
  `= (map Prod.swap (μ ⊗ₘ κ)).snd`        (def of swappedJoint)
  `= map (snd ∘ Prod.swap) (μ ⊗ₘ κ)`      (Measure.map_map)
  `= map Prod.fst (μ ⊗ₘ κ)`               (snd ∘ Prod.swap = fst)
  `= (μ ⊗ₘ κ).fst`                         (def of fst)
  `= μ`                                     (Measure.fst_compProd)

All steps use Mathlib lemmas. The gap is just that we haven't stated and proved this
as a named lemma here.

### GAP 2: Jensen gap theorem in the measure-theoretic setting

`JensenGap.lean` proves `jensenGap_nonneg` for belief-convex `f` over finite `O`.
The measure-theoretic version would require:
  - A notion of measure-theoretic belief-convexity (Bochner integral version)
  - `∫ f(posteriorKernel o) d(marginalO)(o) ≥ f(b.μ)`

This requires the Bochner integral Jensen inequality, which Mathlib has as
`MeasureTheory.inner_le_iff` and `MeasureTheory.snorm_indicator_le` but the
specific form for convexity of functionals on `ProbabilityMeasure` is not
packaged as a ready-to-use lemma.

### GAP 3: StandardBorelSpace requirement

`StandardBorelSpace S` is needed for `Measure.condKernel` to exist.
For program synthesis where `S` is a type of programs (a countable set, or
a Cantor space, or a complete separable metric space of syntax trees), this
is satisfied. But for general `MeasurableSpace S`, the posterior kernel
may not exist (the regular conditional distribution can fail to exist
without separability).

If `S` is countable with the discrete σ-algebra, `StandardBorelSpace` holds and
the posterior kernel is exactly `(marginalMeasure {o})⁻¹ * joint({o} × ·)`.
-/
