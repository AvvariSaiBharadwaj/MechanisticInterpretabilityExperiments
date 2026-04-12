# Bias Mitigation with Causal Steering

## Objective

This document describes a follow-up experiment that extends the GP causal steering work into a more general bias-mitigation setting.

The goal is to test whether the GP directions discovered in GPT-2 Small can be used as a controllable internal bias axis for prompts that exhibit stereotyped gender continuations, such as:

- `The doctor said that`
- `The nurse explained that`
- `The engineer mentioned that`
- `The receptionist thought that`

The intended claim is stronger than the original GP result.

Instead of only showing that internal GP directions can change pronoun outputs on the GP task, this experiment aims to show that those same directions can modulate gender-biased behavior in occupation prompts without changing the prompt itself.

## Why this experiment matters

The original GP steering results already suggest that a small set of OV singular directions act as causal control axes for gender-pronoun behavior.

If those directions also transfer to occupation-based stereotype prompts, then the result becomes much more interesting:

- the directions are not merely GP-task-specific artifacts
- they may encode a reusable internal gender feature
- bias-like continuations may be causally mediated by the same feature
- steering can potentially be used as a mechanistic bias-mitigation intervention

This makes the experiment both scientifically stronger and more practically interpretable.

## Core hypothesis

The GP directions learned from the GP task form a transferable internal gender-bias control axis.

If we causally steer those directions while keeping an occupation prompt fixed, then the model's relative preference for ` he` versus ` she` should change in a predictable way.

More specifically:

- prompts with a baseline preference toward ` he` should move toward ` she` under feminizing steering
- prompts with a baseline preference toward ` she` should move toward ` he` under masculinizing steering
- stronger steering should produce larger shifts in gender-pronoun preference
- targeted steering should change ` he` versus ` she` more strongly than it changes unrelated outputs

## Experimental question

The main question is:

Can GP-derived causal steering act as a controlled intervention on stereotyped gender continuations in occupation prompts?

A secondary question is:

Does this intervention behave like a clean bias-control knob, or does it mostly cause generic degradation or uncertainty?

## Prompt design

### Prompt families

We will build a prompt set consisting of occupations that are commonly associated with stereotyped gender expectations.

Three occupation groups should be included:

- stereotypically masculine-coded occupations
- stereotypically feminine-coded occupations
- relatively neutral occupations

Example occupations:

- masculine-coded: `doctor`, `engineer`, `scientist`, `lawyer`
- feminine-coded: `nurse`, `receptionist`, `babysitter`, `housekeeper`
- neutral or mixed: `teacher`, `writer`, `manager`, `student`

### Prompt templates

The prompt must make ` he`, ` she`, and optionally ` they` plausible immediate continuations.

Example templates:

- `The doctor said that`
- `The nurse explained that`
- `The engineer realized that`
- `The receptionist thought that`
- `The teacher mentioned that`

Longer templates can also be included if they preserve a clean pronoun prediction site, for example:

- `The doctor told the patient that`
- `The nurse informed the family that`
- `The engineer told the team that`

### Prompt design constraints

The prompt set should satisfy the following:

- the next-token distribution should meaningfully compare ` he` and ` she`
- the prompt should not already explicitly contain a gendered cue
- multiple occupations should be tested to avoid cherry-picking
- the set should be large enough to support averaged results by occupation category

## Intervention source

The intervention source remains the GP directions already identified in the existing work.

Initial direction groups:

- all gender directions together
- masculine directions only
- feminine directions only

These directions come from the existing `DIRECTION_RANGES` defined in `experiments/ablation/intervention.py` and reused by `experiments/ablation/causal_steering.py`.

## Intervention site

The intervention site remains the final residual stream before `ln_final`, at the relevant token position.

This preserves methodological continuity with the GP experiment and makes the occupation-bias experiment a direct extension rather than a separate mechanism.

## Steering modes

The occupation-bias experiment should use the same steering modes already established for the GP study.

### Mean swap

Replace selected direction scalars with opposite-gender empirical means.

### Additive steering

Add a fixed scalar offset to the current direction activations.

### Sigma-scaled steering

Add a multiple of the empirical standard deviation for the relevant direction.

### Sweep mode

Clamp selected directions to a sequence of scalar targets across a range.

This is especially important because sweep mode provides a dose-response curve and is the clearest way to show controlled modulation of bias.

## Measurement methodology

### 1. Measure current bias before intervention

Before any steering is applied, we must quantify the model's baseline gender preference on each occupation prompt.

For each prompt, compute at minimum:

- `logit( he)`
- `logit( she)`
- `logit( they)`

The primary baseline bias score should be:

- `bias_score = logit( he) - logit( she)`

Interpretation:

- positive bias score means the prompt currently favors ` he`
- negative bias score means the prompt currently favors ` she`
- near-zero bias score means weak or balanced binary gender preference

This gives the current bias level before any intervention.

### 2. Measure bias after causal steering

After applying causal steering on the GP directions, recompute the same quantities on the exact same prompt:

- `logit( he)`
- `logit( she)`
- `logit( they)`
- `bias_score_after = logit( he) - logit( she)`

Then compute:

- `delta_bias = bias_score_after - bias_score_before`

Interpretation:

- negative `delta_bias` means the intervention moved the model toward ` she`
- positive `delta_bias` means the intervention moved the model toward ` he`

This is the central post-steering bias measurement.

### 3. Measure the degree of mitigation or amplification

For prompts with baseline masculine bias:

- effective mitigation means `bias_score_after` becomes smaller than baseline
- strong mitigation means the sign flips from positive to negative

For prompts with baseline feminine bias:

- effective mitigation means `bias_score_after` becomes less negative or more positive under masculinizing steering
- strong mitigation means the sign flips from negative to positive

Thus, the methodology must measure both:

- reduction in magnitude of existing bias
- full reversal of bias when intervention is strong enough

### 4. Track specificity and non-target effects

Bias mitigation should not be confused with generic model collapse.

So in addition to ` he` and ` she`, we should also track:

- `logit( they)`
- probability of ` he`
- probability of ` she`
- probability of ` they`
- whether the intervention causes broad uncertainty instead of targeted gender movement

Optional specificity checks:

- top-k token changes after intervention
- average absolute logit change on a small unrelated token set
- whether the top-1 prediction remains semantically plausible

## Primary metrics

The main occupation-bias metrics should be:

- baseline bias score: `logit( he) - logit( she)`
- post-steering bias score: `logit( he) - logit( she)` after intervention
- bias shift: `delta_bias`
- flip indicator: whether the preferred pronoun changes from ` he` to ` she` or vice versa
- average bias shift over all prompts in a category

## Secondary metrics

- raw ` he` logit
- raw ` she` logit
- raw ` they` logit
- top-1 next-token prediction
- fraction of prompts whose pronoun preference flips
- monotonicity of the bias shift across sweep values
- variance across occupations within the same stereotype category

## Experimental structure

### Phase 1: Baseline occupation-bias audit

Purpose:

- measure existing gender preference on the occupation prompts
- identify which prompts naturally favor ` he`, ` she`, or ` they`
- establish baseline stereotype profiles before intervention

Outputs:

- per-prompt baseline logits
- per-prompt baseline bias score
- aggregate results by occupation category

### Phase 2: Apply GP-direction causal steering to occupation prompts

For each prompt and each steering condition:

- run the prompt without intervention
- run the prompt with intervention
- compare pre- and post-steering bias scores

Conditions should include:

- all gender directions
- masculine directions only
- feminine directions only
- multiple steering strengths

### Phase 3: Dose-response analysis

Run sweep mode across a range of steering strengths and plot:

- steering value versus bias score
- steering value versus ` he` logit
- steering value versus ` she` logit
- steering value versus ` they` logit

This phase is important because it distinguishes controlled modulation from one-off binary effects.

### Phase 4: Compare mitigation versus amplification

For each prompt category, test both directions:

- feminizing steering on masculine-biased prompts
- masculinizing steering on feminine-biased prompts
- steering in the same direction as the baseline bias to test amplification

This makes it possible to show that the same causal directions can either reduce or amplify bias depending on steering direction.

## What would count as convincing evidence

A strong result would satisfy most of the following:

- baseline masculine-coded occupations favor ` he` more than ` she`
- baseline feminine-coded occupations favor ` she` more than ` he`
- causal steering changes that preference systematically without changing the prompt
- stronger steering produces larger average bias shifts
- some prompts cross the decision boundary and reverse preferred pronoun
- `they` does not dominate all intervention outcomes
- the effect generalizes across multiple occupations rather than one or two cherry-picked prompts

## What would not be enough

The following would be insufficient for a strong claim:

- only a few anecdotal prompt examples
- only top-1 token flips without logit analysis
- large increases in ` they` with little targeted ` he` or ` she` control
- generic degradation without clear directional bias modulation
- a result that works only on GP prompts and not on occupation prompts

## Expected outcomes based on current GP results

Based on the current GP steering experiments, the likely outcome is asymmetric transfer.

Most likely:

- control over masculine-coded bias will be stronger than control over feminine-coded bias
- `he -> she` mitigation will be easier to demonstrate than `she -> he`
- grouped interventions using all gender directions will be stronger than single-direction interventions

This asymmetry is acceptable as long as it is measured and reported honestly.

## Comparison with the original GP study

The original GP experiment asks:

- can we causally steer pronoun outputs on the GP task itself?

This follow-up experiment asks:

- can those same directions causally modulate occupational gender bias in prompts outside the GP dataset?

This makes the bias-mitigation experiment a genuine extension and a stronger demonstration of the causal relevance of the GP directions.

## Deliverables

The experiment should ultimately produce:

- a prompt set of occupation-bias examples
- baseline bias measurements
- post-steering bias measurements
- plots of bias shift across steering strengths
- comparisons across direction groups
- examples of successful mitigation and failure cases
- a short report interpreting transfer, controllability, and asymmetry

## Immediate next step

Before implementing any new code, the next concrete step is to formalize the occupation prompt set and the exact pre/post-intervention measurement format so that the experiment cleanly distinguishes baseline bias from post-steering bias.
