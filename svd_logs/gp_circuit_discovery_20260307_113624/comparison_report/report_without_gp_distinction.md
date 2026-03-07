# Paper-Style Report: Causal Steering and Smart Ablation on GP

## Abstract

We evaluate two intervention families on the GP task in GPT-2 Small: continuous scalar causal steering over selected OV singular directions and discrete smart range-based ablations that swap directions into empirically measured opposite-gender activation ranges. Across both methods, the dominant empirical pattern is asymmetric control: interventions are substantially stronger on `he` prompts than on `she` prompts. Causal steering provides a smooth dose-response curve, while smart ablation yields stronger endpoint effects and can fully reverse predictions under sigma amplification. The combined evidence supports the claim that the selected OV directions exert real downstream causal control over gender-pronoun behavior, while also indicating that the currently selected directions capture masculine evidence more completely than feminine evidence.

## Experimental Artifacts

- **Run directory**: `svd_logs/gp_circuit_discovery_20260307_113624`
- **Smart ablation input**: `smart_ablation/smart_ablation_results.json`
- **Primary causal steering input**: `causal_steering/causal_steering_sweep.json`
- **Additional causal group inputs available**: `all_gender`

## Task Setting

We compare interventions on two target subsets: `he` prompts and `she` prompts. The principal metrics are subset accuracy, target logit difference (`he - she` on `he` prompts and `she - he` on `she` prompts), `they` logit, and direct prediction flips. Smart ablation is evaluated as a discrete endpoint intervention, while causal steering is evaluated as a sweep over forced scalar values applied at the final residual stream before `ln_final`.

## Baseline Performance

- **HE prompts**: accuracy `84.00%`, target logit diff `2.525`, they logit `11.691`
- **SHE prompts**: accuracy `70.51%`, target logit diff `2.840`, they logit `11.886`

## Results

### 1. Continuous causal steering exhibits a dose-response effect

- **Strongest causal control is on `he` prompts**: the sweep reduces accuracy from `84.00%` baseline to `32.67%` at sweep value `1.492`, while shrinking target logit diff from `2.525` to `0.228`.
- **`she` control is weaker and asymmetric**: the lowest `she` sweep accuracy is `53.21%` at sweep value `-1.035`, with target logit diff still positive at `1.480`.

![Causal sweep curves](figures/causal_sweep_curves.png)

### 2. Smart ablation produces stronger endpoint effects

- **Discrete smart ablation is much stronger on `he` than the smooth sweep**: `exp1_masc_swap_all` drives accuracy to `10.67%` and target logit diff to `-0.768`, with `52` flips to `she`.
- **For `she`, discrete smart ablation is still moderate rather than catastrophic**: `exp2_fem_swap_all` reaches `46.79%` and target logit diff `1.487`, with `4` flips to `he`.

![Smart vs causal comparison](figures/smart_vs_causal_bars.png)

### 3. Direction-selective interventions matter

- **Selective direction swaps matter**: on `he`, swapping only masculine directions (`exp3`) leaves accuracy at `45.33%`, much higher than swapping all gender directions; on `she`, swapping only feminine directions (`exp4`) leaves accuracy at `57.05%`.
- **Closest causal match for masculine-only comparison**: the best available `all_gender` causal run reaches accuracy `47.33%` at sweep `-0.087` with target logit diff `0.806`.
- **Closest causal match for feminine-only comparison**: the best available `all_gender` causal run reaches accuracy `53.85%` at sweep `1.176` with target logit diff `2.286`.
- **Caveat**: dedicated causal sweep files for `masculine and feminine` were not found, so the script currently falls back to the all-gender sweep for missing groups.

![Direction-group causal comparison](figures/direction_group_sweeps.png)

### 4. Sigma amplification reveals a sharp nonlinear regime

- **Sigma amplification saturates quickly**: at `sigma x 2.0`, `he` accuracy is already `0.00%` and flip-to-`she` reaches `100.00%`, indicating the range-swap intervention can fully cross the decision boundary when amplified.

![Sigma amplification](figures/sigma_amplification.png)

## Quantitative Comparison

| Experiment | Gender | Causal group used | Smart acc | Closest causal acc | Smart diff | Closest causal diff | Smart they | Closest causal they |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| he swap all directions | he | all_gender | 10.67 | 32.67 | -0.768 | 0.228 | 7.300 | 7.672 |
| she swap all directions | she | all_gender | 46.79 | 53.21 | 1.487 | 1.480 | 7.503 | 7.256 |
| he swap masculine only | he | all_gender | 45.33 | 47.33 | 0.770 | 0.806 | 7.222 | 7.281 |
| she swap feminine only | she | all_gender | 57.05 | 53.85 | 2.232 | 2.286 | 7.423 | 7.804 |

## Discussion

- **Causal steering** gives a clean continuous dose-response curve. It is best for showing monotonicity and controllability.
- **Smart ablation** gives stronger endpoint effects because it swaps to opposite-gender empirical ranges and can be amplified with singular values. It is best for demonstrating maximal reversibility of the behavior.
- **The asymmetry is consistent across both methods**: masculine-side control is stronger than feminine-side control.
- **They logits drop sharply under both interventions** relative to baseline, which suggests the intervention is not merely shifting toward uncertainty; it is actively redistributing pronoun evidence.

## Limitations

- **Shared-scalar steering is coarse**: all selected directions are clamped to the same scalar target, which may underfit direction-specific calibration.
- **Asymmetric capture remains unresolved**: the current direction set appears to capture masculine evidence more completely than feminine evidence.
- **Direction-group comparison may still be partial** when dedicated masculine-only or feminine-only sweep files are unavailable.

## Conclusion

The two intervention families are complementary. Causal steering demonstrates smooth controllability and exposes the shape of the underlying response curve, while smart ablation demonstrates that opposite-gender range swaps can drive much larger endpoint changes, including full reversal under amplification. Taken together, the current evidence strongly supports a causal role for the selected OV singular directions in gender-pronoun behavior, with a persistent asymmetry that should guide the next round of mechanistic analysis.

## Next steps

- **Calibrate per-direction sweep values** instead of forcing one shared scalar across all directions.
- **Run and save dedicated causal sweeps for `masculine` and `feminine` groups** so the selective comparisons stop relying on fallback behavior.
- **Add a token-level decomposition** of logit changes to separate targeted `he <-> she` transfer from any residual uncertainty effects.