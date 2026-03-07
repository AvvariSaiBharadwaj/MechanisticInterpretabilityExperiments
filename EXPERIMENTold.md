# Causal Steering Experiment Plan

## Objective

Extend the Beyond-Components GP experiment into a scalar-based causal steering study in GPT-2 Small.

The goal is to demonstrate that a small number of gender-sensitive singular directions behave like controllable logit receptors: by intervening on their scalar activation coefficients, we can produce precise and directional changes in pronoun logits such as ` he` and ` she`.

## Why this experiment

The current repository already supports:

- singular-value decomposition of OV, QK, and MLP operators
- direction-level masking
- activation-based intervention for Gender Pronoun tasks
- scalar swapping based on empirically observed activation ranges

This makes the repository a strong base for a causal-steering experiment rather than only a circuit-discovery or ablation experiment.

## Current starting point in the repository

The closest existing implementation is:

- `experiments/ablation/intervention.py`

That file currently:

- uses a fixed set of gender-relevant OV singular directions
- computes the scalar coefficient `V' · u_i` for each direction
- replaces that scalar with an opposite-gender empirical mean
- converts the scalar change into a residual-stream intervention using `sigma_i * v_i`
- adds the total intervention at the final residual stream before `ln_final`
- measures changes in pronoun-related logits and prediction flips

In other words, the repository already contains a discrete scalar-swap intervention. Our work will generalize this into an explicit causal-steering framework with continuous scalar control and better experiment traceability.

## Working interpretation of key concepts

### Causal steering

Causal steering means intervening on an internal model variable and measuring the downstream effect on model outputs while keeping the prompt fixed.

In this project, the internal variable is the scalar coefficient of an activation along a discovered singular direction.

### Scalar-based intervention

For a selected direction, we do not replace an entire head or full activation vector. Instead, we manipulate only a scalar coefficient.

If `u_i` is the left singular vector defining a context-sensitive receptor and `v_i` is the corresponding output direction, then the relevant scalar is:

`alpha_i = V' · u_i`

where `V'` is the augmented attention context at the intervention site.

We then change `alpha_i` to:

- an opposite-gender mean
- an additive offset
- a sigma-scaled target
- or a value from a sweep over a range

The downstream residual change is:

`delta_resid = (alpha_i_new - alpha_i_old) * sigma_i * v_i`

### Logit receptor

For this experiment, a logit receptor is an OV singular direction whose scalar activation produces a predictable change in downstream pronoun logits.

Operationally, a useful receptor direction should satisfy most of the following:

- it shows gender-separable scalar statistics on GP data
- it produces directional changes in `he` vs `she` logit difference under intervention
- it has relatively targeted effect compared with unrelated logits
- it yields monotonic or near-monotonic response under scalar sweeps

## Main hypothesis

A small number of OV singular directions in GPT-2 Small act as controllable gender-sensitive logit receptors.

If we intervene directly on their scalar activations, then:

- increasing masculine-associated coefficients should systematically increase the ` he` logit relative to ` she`
- increasing feminine-associated coefficients should systematically increase the ` she` logit relative to ` he`
- the sign and magnitude of steering should be predictable from the sign and magnitude of the scalar intervention

## Secondary hypotheses

- a continuous scalar sweep will reveal dose-response behavior rather than only binary flips
- not all discovered directions are equally causal; some will behave as stronger control axes than others
- steering multiple compatible directions together will produce stronger shifts than steering one direction alone
- plural or non-gender comparison directions should show weaker or different effects on ` he`/` she` logits

## Task target

We will focus first on:

- model: GPT-2 Small
- task: GP (Gender Pronoun)
- target tokens: ` he`, ` she`
- contrast token for specificity checks: ` they`

## Proposed intervention site

### Primary intervention site

The primary intervention site will be:

- the final residual stream at the last token position
- specifically the residual before `ln_final`

This matches the current implementation in `experiments/ablation/intervention.py` and has the following advantages:

- it preserves compatibility with the current OV-direction derivation
- it allows additive composition across several directions
- it directly exposes downstream changes in unembedding logits
- it avoids reimplementing full in-network hook-based edits for the first version

### Rationale

Each OV singular direction contributes a write vector to the residual stream. Summing interventions in residual space gives a clean and interpretable view of how direction-level changes alter output logits.

### Deferred intervention sites

These are possible follow-up variants, but not first priority:

- intervention at head output before residual addition
- intervention at intermediate residual stream layers
- intervention on MLP directions
- direct manipulation in normalized residual space after `ln_final`

## Direction definition

### Primary direction family

We will use OV singular directions from the pretrained GPT-2 Small attention heads as discovered by the repository.

For each selected direction we use:

- `u_i`: left singular vector over the augmented OV input space
- `sigma_i`: singular value
- `v_i`: right singular vector over residual output space

### Operational meaning

- `u_i` determines how strongly the current context activates the direction
- `sigma_i` scales the strength of the write
- `v_i` defines where in residual space the head writes

### Initial candidate directions

The repository already contains empirically measured GP directions in `DIRECTION_RANGES` inside `experiments/ablation/intervention.py`, including:

- `L9.H7.SV1`
- `L11.H8.SV6`
- `L10.H9.SV0`
- `L11.H8.SV9`
- `L9.H7.SV0` as a comparison direction

We will begin with these directions because they already have gender-conditioned scalar statistics.

## Steering modes to support

We want to support more than one intervention rule.

### Mode 1: Mean swap

Replace the current scalar with the opposite-gender mean.

This reproduces the current range-swap logic and serves as a baseline steering mode.

### Mode 2: Additive delta

Set:

`alpha_new = alpha_old + delta`

This is the cleanest form of continuous steering and should be the default causal-steering mode.

### Mode 3: Sigma-scaled additive steering

Set:

`alpha_new = alpha_old + k * std_group`

where `k` is a sweep parameter and `std_group` is based on empirical scalar statistics.

### Mode 4: Direct target value sweep

Set the scalar directly to one of a sequence of values across a defined range, such as:

- from `she_mean - 2 * she_std` to `he_mean + 2 * he_std`

This is useful for dose-response plots.

## Experimental targets

### Primary target metric

The main metric will be the pronoun logit contrast:

- for masculine prompts: `logit( he) - logit( she)`
- for feminine prompts: `logit( she) - logit( he)`

### Secondary metrics

- raw ` he` logit
- raw ` she` logit
- raw ` they` logit
- probability of ` he`
- probability of ` she`
- top-1 prediction flip rate
- exact match against labels
- effect size relative to baseline

### Specificity metrics

To support the claim of precise control, we should also track:

- movement in ` they` logit
- optionally average absolute change in a small set of unrelated common-token logits
- whether steering causes broad degeneration or primarily target-specific shifts

## Experimental structure

### Phase 1: Reproduce and formalize existing scalar swap behavior

Purpose:

- confirm the current code path as the baseline intervention method
- save detailed outputs in a consistent experiment format

### Phase 2: Continuous scalar steering per direction

For each selected direction:

- test a scalar sweep over several values
- plot or save the response of ` he` and ` she` logit differences
- measure monotonicity and flip behavior

### Phase 3: Multi-direction steering

Test grouped interventions:

- masculine directions only
- feminine directions only
- all gender directions together
- comparison direction only

### Phase 4: Compare steering rules

Compare:

- mean swap
- additive delta
- sigma-scaled steering
- direct target-value sweeps

## Design choices finalized for first implementation

### Choice: intervention site

Final choice for first implementation:

- final residual stream at last token before `ln_final`

### Choice: direction family

Final choice for first implementation:

- OV singular directions only

### Choice: initial direction source

Final choice for first implementation:

- start from the hard-coded candidate directions already present in `DIRECTION_RANGES`

### Choice: first steering modes to implement

Final choice for first implementation:

- mean swap
- additive delta
- direct scalar sweep

### Choice: task scope

Final choice for first implementation:

- GP only

### Choice: model scope

Final choice for first implementation:

- GPT-2 Small only

## Open design questions

These are not blockers, but we should keep them explicit.

### Question 1

Should the scalar sweep be centered around:

- the sample’s current scalar
- the group mean
- or a global fixed range per direction

Tentative answer:

- support both current-scalar-centered additive sweeps and empirical-range sweeps

### Question 2

Should we intervene only on examples whose gold label matches the targeted gender?

Tentative answer:

- yes for the first version, because it keeps interpretation clean

### Question 3

Should we define receptor relevance using empirical separation only, or also by explicit alignment with unembedding contrast vectors?

Tentative answer:

- first use empirical separation and causal effect
- later add optional analysis against unembedding contrast such as `W_U[:, he] - W_U[:, she]`

### Question 4

Should interventions be applied independently per direction or jointly?

Tentative answer:

- both, but independent single-direction sweeps should come first for interpretability

## Expected code changes

### New documentation artifact

- `EXPERIMENT.md`

Purpose:

- traceability of experiment goals and implementation choices
- running record of assumptions and finalized decisions

### Likely code changes

- extend `experiments/ablation/intervention.py` into a more general steering utility, or
- create a new ablation script dedicated to causal steering experiments

### Functions likely to be added or generalized

- function to compute per-direction scalar activations on GP batches
- function to build interventions for different steering modes
- function to run scalar sweeps and collect detailed metrics
- function to serialize detailed experiment outputs for later plotting

## Proposed file strategy

Preferred approach:

- keep existing `experiments/ablation/intervention.py` as the legacy range-swap baseline
- create a new experiment file for scalar causal steering so the original study remains intact

Tentative new file name:

- `experiments/ablation/causal_steering.py`

This isolates the new experiment while still reusing the model-loading and direction machinery.

## Implementation criteria for success

We will consider the first implementation successful if it can:

- load GPT-2 Small and the trained GP circuit
- evaluate baseline GP pronoun metrics
- run steering on at least one selected singular direction
- support at least one continuous scalar sweep
- save structured results to disk
- show directional changes in pronoun logit contrast

## Risks and caveats

- the hard-coded candidate directions may not transfer perfectly across checkpoints
- final-residual intervention is clean but may blur where within the network the effect originated
- strong steering may induce normalization artifacts after `ln_final`
- monotonicity may break for large steering magnitudes because of nonlinear downstream effects

## What we will do next

1. Map the existing intervention code into reusable helper pieces.
2. Implement a new causal-steering script for GP in GPT-2 Small.
3. Support continuous scalar steering modes.
4. Save detailed results for traceability and plotting.
5. Update this document if design choices change during implementation.

## Change log

### 2026-03-07

- Created the initial experiment plan.
- Confirmed that the repository already contains a scalar-swap OV intervention baseline.
- Fixed the first implementation scope to GPT-2 Small, GP task, OV singular directions, and final-residual steering before `ln_final`.
- Chose to document the experiment before implementation for traceability.
