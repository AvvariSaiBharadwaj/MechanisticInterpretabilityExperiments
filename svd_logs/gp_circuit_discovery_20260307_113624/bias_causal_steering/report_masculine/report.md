# Occupation Bias Causal Steering Report

## Objective

This report evaluates whether GP-derived causal steering directions transfer to occupation prompts and modulate stereotyped pronoun preference in a controlled way.

## Experimental Artifacts

- **Results JSON**: `../bias_causal_steering_sweep_masculine.json`
- **Prompt source**: `occupation_bias_prompts.json`
- **Mode**: `sweep`
- **Direction group**: `masculine`
- **Number of prompts**: `36`

## High-level findings

- **Most feminizing condition**: `sweep:-1.0` with mean `delta_bias = -1.300`
- **Most masculinizing condition**: none of the tested conditions produced positive mean `delta_bias`; the closest condition was `sweep:-1.0` with mean `delta_bias = -1.300`

## Bias score curves

![Bias score curves](figures/bias_score_curves.png)

## Delta bias curves

![Delta bias curves](figures/delta_bias_curves.png)

## Flip rates

![Flip rates](figures/flip_rates.png)

## Condition summary

| Condition | Mean baseline bias | Mean steered bias | Mean delta bias | Flip rate |
| --- | ---: | ---: | ---: | ---: |
| sweep:-1.0 | 0.532 | -0.768 | -1.300 | 44.44 |
| sweep:0.0 | 0.532 | -0.188 | -0.720 | 25.00 |
| sweep:1.0 | 0.532 | 0.392 | -0.140 | 2.78 |

## Interpretation

- **Negative delta bias** means steering moved the model toward `she`.
- **Positive delta bias** means steering moved the model toward `he`.
- **Flip rate** measures how often the preferred binary pronoun changed between baseline and steered evaluations.

## Next steps

- **Compare direction groups** by running the evaluator for `masculine`, `feminine`, and `all_gender` separately.
- **Extend prompt coverage** with more occupations and more template variants.
- **Add prompt-level case studies** for the strongest successful and failed mitigation examples.
