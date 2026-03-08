# Combined Summary: GP Causal Steering, Smart Ablation, and Occupation-Bias Transfer

## Overview

This document summarizes the main findings so far across two connected experimental threads:

- causal steering and smart ablation on the GP task
- transfer of GP-derived directions to occupation-bias prompts

The central question across both settings is whether a small set of GP-related OV singular directions in GPT-2 Small acts like a real causal control mechanism for gendered pronoun behavior.

## Main takeaway

The results so far strongly suggest that these directions are not just correlated with model behavior. They have real causal influence.

On the GP task, steering and ablation can substantially change pronoun behavior, especially on `he` prompts. On occupation prompts such as `doctor` and `nurse`, these same directions also shift the model's pronoun preference, which suggests that the GP directions transfer beyond the original GP dataset.

At the same time, the control is not fully symmetric or fully understood yet. Some direction groups show stronger and cleaner control than others, and the effect is often stronger in one direction than the other.

## Part 1: GP task results

### What was tested

On the GP task, two intervention families were compared:

- continuous causal steering using scalar sweeps on GP-related OV directions
- smart ablation using opposite-gender range swaps, with optional sigma amplification

### Baseline GP performance

- **HE prompts**: accuracy `84.00%`, target logit diff `2.525`
- **SHE prompts**: accuracy `70.51%`, target logit diff `2.840`

### Key GP findings

#### 1. Causal steering changes GP outputs in a controlled way

The steering sweep shows a dose-response effect.

- on `he` prompts, the best sweep reduces accuracy from `84.00%` to `32.67%`
- on `he` prompts, target logit diff drops from `2.525` to `0.228`
- on `she` prompts, the strongest sweep only reduces accuracy to `53.21%`
- on `she` prompts, target logit diff remains positive at `1.480`

This means the steering directions clearly matter, but the control is asymmetric.

#### 2. Smart ablation is even stronger than smooth steering

Smart ablation produces stronger endpoint effects than the causal sweep.

- for `he`, smart ablation drives accuracy to `10.67%`
- for `he`, target logit diff becomes negative at `-0.768`
- for `he`, there are `52` flips to `she`

For `she`, the effect exists but is weaker:

- smart ablation reaches `46.79%` accuracy
- target logit diff remains positive at `1.487`
- only `4` flips to `he`

#### 3. Sigma amplification can fully cross the decision boundary

At `sigma x 2.0`, `he` accuracy goes to `0.00%` and flip-to-`she` reaches `100.00%` in the tracked setting.

This is strong evidence that the intervention is not a minor perturbation. It can fully override the model's original GP output in at least one direction.

#### 4. Direction-selective runs matter

The selective causal comparisons show:

- `he` swap masculine only: smart ablation accuracy `45.33%`, closest causal accuracy `46.00%`
- `she` swap feminine only: smart ablation accuracy `57.05%`, closest causal accuracy `58.97%`

This suggests that direction grouping changes the strength and shape of the effect.

### GP conclusion

The GP experiments support a strong causal claim:

> these directions causally control gender-pronoun behavior in GPT-2 Small

But the control is not symmetric. The current directions influence `he -> she` more strongly than `she -> he`.

## Part 2: Occupation-bias transfer results

### What was tested

A new prompt set of `36` occupation prompts was created across three categories:

- `masculine_coded`
- `feminine_coded`
- `neutral`

The question was whether the same GP-derived directions would shift `logit(he) - logit(she)` on these prompts.

The baseline mean bias score across the prompt set was:

- `0.532`

A positive value means the model prefers `he` over `she` on average.

### All-gender direction group

Condition summary:

- `sweep:-1.0` -> steered bias `0.210`, delta bias `-0.321`, flip rate `27.78%`
- `sweep:0.0` -> steered bias `-0.097`, delta bias `-0.629`, flip rate `33.33%`
- `sweep:1.0` -> steered bias `-0.404`, delta bias `-0.935`, flip rate `41.67%`

Interpretation:

- all tested settings move the model toward `she`
- stronger tested sweep values produce larger feminizing shifts
- the shift is substantial enough to flip preferred binary pronoun on many prompts

### Masculine-only direction group

Condition summary:

- `sweep:-1.0` -> steered bias `-0.768`, delta bias `-1.300`, flip rate `44.44%`
- `sweep:0.0` -> steered bias `-0.188`, delta bias `-0.720`, flip rate `25.00%`
- `sweep:1.0` -> steered bias `0.392`, delta bias `-0.140`, flip rate `2.78%`

Interpretation:

- this group still mostly pushes the model toward `she`
- the strongest feminizing setting is `sweep:-1.0`
- by `sweep:1.0`, the effect becomes much weaker, though still slightly feminizing overall

### Feminine-only direction group

Condition summary:

- `sweep:-1.0` -> steered bias `0.614`, delta bias `0.083`, flip rate `19.44%`
- `sweep:0.0` -> steered bias `-0.273`, delta bias `-0.805`, flip rate `33.33%`
- `sweep:1.0` -> steered bias `-1.159`, delta bias `-1.691`, flip rate `61.11%`

Interpretation:

- this is the only tested group that shows a genuinely positive mean `delta_bias` at one setting
- `sweep:-1.0` slightly increases masculine preference overall
- `sweep:1.0` strongly pushes toward `she`
- this suggests the feminine direction group may provide the clearest bidirectional control structure on occupation prompts

## Cross-experiment synthesis

Putting the GP and occupation results together gives the following picture.

### Finding 1: The GP directions are genuinely causal

On GP, they strongly change pronoun outputs.

On occupations, they also shift pronoun preference without changing the prompt itself.

That means these directions are not merely descriptive features. They are usable control handles.

### Finding 2: The control transfers beyond the original GP dataset

The occupation-bias results are important because they show transfer.

The directions discovered on GP do not only matter for GP prompts. They also affect natural stereotype-like prompts such as `doctor`, `engineer`, `nurse`, and `receptionist`.

### Finding 3: The control geometry depends on the direction group

The direction subsets do not behave identically.

- `all_gender` produced broad feminizing shifts
- `masculine` also mostly produced feminizing shifts over the tested range
- `feminine` showed the clearest evidence of both feminizing and slightly masculinizing control depending on sweep value

This means the internal feature is not a simple one-dimensional dial in its current implementation. Different direction subsets seem to contribute differently to the final behavior.

### Finding 4: The current steering setup is still coarse

The present steering method uses shared sweep values across multiple directions.

That likely underfits the true geometry of the internal feature. A more calibrated per-direction sweep may recover cleaner and more symmetric control.

## Layman explanation

Here is the plain-English version.

Think of the model as having a few internal knobs related to gendered pronoun behavior.

We found some of those knobs on a pronoun task. Then we tried turning them.

### What happened on the pronoun task

When we turned those knobs, the model changed its answer.

In many cases, especially for cases where it originally wanted to say `he`, we could push it away from `he` and toward `she`. With the stronger intervention, we could do this very reliably.

So this tells us that these internal knobs are not just passive indicators. They actively influence what the model says.

### What happened on job prompts like doctor and nurse

Then we asked a harder question:

If the model has a stereotype like `doctor -> he` or `nurse -> she`, can we change that by turning the same knobs?

The answer is: **yes, to a meaningful extent**.

Turning those internal knobs often changed whether the model leaned more toward `he` or more toward `she` on occupation prompts.

That means the same internal mechanism we found on the GP task seems to also affect biased behavior in more natural prompts.

### Why the results are not perfectly clean yet

The knobs do not behave perfectly symmetrically yet.

Some settings mostly push the model toward `she`. Some direction groups behave differently from others. One group gives a small push toward `he` at one setting, while others do not.

So we are not yet at the stage where we can say:

> we can freely set the model to whatever gendered output we want in every case

But we **can** say:

> we have found internal control directions that measurably and causally change gendered behavior, and these effects transfer to occupation-bias prompts

That is already a strong result.

## What the results do and do not prove

### What the results do support

- GP-related OV directions have real causal influence on pronoun outputs
- those directions transfer to occupation prompts
- causal steering can mitigate or amplify gender preference on occupation prompts
- the effect depends strongly on the chosen direction group and steering value

### What the results do not yet support

- full symmetric control in all directions
- a claim that all occupational bias is captured by these directions alone
- a claim that the current steering setup is optimal or complete

## Recommended next steps

- run denser sweeps for each direction group
- add prompt-level examples of the strongest successful and failed interventions
- build a single comparison plot across `all_gender`, `masculine`, and `feminine`
- try per-direction calibrated sweep targets instead of one shared scalar value
- separate mitigation from generic uncertainty by analyzing `they` and unrelated token movement in more detail

## Pointers to detailed artifacts

- **GP comparison report**: `comparison_report/report.md`
- **Occupation all-gender report**: `bias_causal_steering/report_all_gender/report.md`
- **Occupation masculine report**: `bias_causal_steering/report_masculine/report.md`
- **Occupation feminine report**: `bias_causal_steering/report_feminine/report.md`
