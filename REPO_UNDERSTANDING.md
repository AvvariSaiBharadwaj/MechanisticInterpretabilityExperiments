I’ll inspect the repo’s main docs and key model file, then summarize the project’s purpose, architecture, and data flow in plain language.
I’ve got the high-level idea from the README and core model. I’m quickly checking the training/config entry points so I can explain how it is actually used end-to-end, not just what it claims to do.
# What This Codebase Is

This repo is a **mechanistic interpretability project for transformers**, specifically GPT-2 small.

Its core claim is:

- **Standard transformer parts are too coarse**
- Instead of treating an attention head or MLP as one indivisible unit, this code **decomposes each component into singular directions** using SVD
- Then it **learns masks over those singular directions**
- The goal is to identify a **small, sparse, functionally meaningful circuit** that preserves the model’s behavior on a task

So the project is basically:

- **Take a pretrained transformer**
- **Factor its computation into low-rank directions**
- **Learn which directions matter**
- **Use interventions/patching to test whether those directions are causally important**

---

# The Main Idea in Plain English

A normal mechanistic interpretability workflow might say:

- “This head is important”
- “This MLP is important”

This repo asks a finer-grained question:

- **Which sub-directions inside that head or MLP are important?**

It does that by:

- constructing effective matrices for attention and MLP computation
- running **SVD** on them
- representing each component as singular vectors + singular values
- learning a soft mask over the singular values

That means each head/MLP is no longer just “on” or “off”.
Instead, the code can say:

- this direction matters
- this other direction does not
- a single head may actually contain multiple overlapping subfunctions

That is the “Beyond Components” idea.

---

# What the Core Class Does

## [MaskedTransformerCircuit](cci:2://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/models/masked_transformer_circuit.py:42:0-1753:19)

The file [src/models/masked_transformer_circuit.py](cci:7://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/models/masked_transformer_circuit.py:0:0-0:0) is the heart of the repo.

Conceptually, it wraps a pretrained `HookedTransformer` and creates a **masked surrogate model**.

### It stores:

- the original transformer
- cached SVD decompositions
- learnable masks for:
  - **QK** directions
  - **OV** directions
  - **MLP input** directions
  - **MLP output** directions

### Why those pieces?

- **QK** governs how attention scores are computed
- **OV** governs what information a head writes into the residual stream
- **MLP in/out** govern how feedforward layers transform representations

So the model is trying to learn a sparse subset of the transformer’s internal linear directions.

---

# How the Math Is Represented

## Attention: QK and OV

For each attention head, the code builds two effective matrices.

### `QK`
This captures the **query-key interaction** that determines attention scores.

The code combines:

- `W_Q`
- `W_K`
- `b_Q`
- `b_K`

into an **augmented matrix** with bias folded in.

That is why you see shapes like `d_model + 1`:
the extra dimension is a trick to include biases in the linear algebra.

### `OV`
This captures how attended information gets written back to the residual stream.

The code combines:

- `W_V`
- `W_O`
- `b_V`
- `b_O`

into another augmented effective matrix.

Then for both QK and OV it computes:

```python
U, S, Vh = torch.linalg.svd(...)
```

and learns masks over `S`.

So the effective computation becomes:

- original: `U @ diag(S) @ Vh`
- masked: `U @ diag(S * mask) @ Vh`

That is the core mechanism.

---

## MLP Layers

The same idea is extended to MLPs:

- `W_in` + `b_in`
- `W_out` + `b_out`

Again these are augmented to include bias terms, decomposed with SVD, and reconstructed with masked singular values.

So the repo treats **attention and MLPs symmetrically** as decomposable operators.

---

# What the Masks Mean

The masks are learnable parameters stored as logits, then passed through:

```python
torch.sigmoid(x)
```

So each mask value is between 0 and 1.

Interpretation:

- **near 1** = keep this singular direction
- **near 0** = suppress this singular direction

The config uses an L1-style sparsity penalty, so training pressures the model to preserve behavior with **as few active directions as possible**.

This is why the project talks about “discovering a circuit”:
the circuit is the subset of singular directions that are sufficient for the task.

---

# How Forward Pass Works

## Not a standard HuggingFace forward

The core class does **not** just call the transformer normally.

Instead, [forward_pass_through_model(...)](cci:1://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/models/masked_transformer_circuit.py:617:4-744:21) manually reconstructs a transformer forward pass:

- get token + positional embeddings
- loop over layers
- apply layer norm
- compute masked attention
- add residual
- apply second layer norm
- compute masked MLP
- add residual
- final unembedding to logits

So this is effectively a **custom forward implementation** that replaces original weights with masked, SVD-reconstructed weights.

That is important: the code isn’t just analyzing the model offline; it is actually **running a modified version of the model**.

---

# Why Activation Patching Is Involved

This code is not only doing sparse masking.
It also does **activation patching/intervention**.

From [experiments/train.py](cci:7://file:///Users/s.avvari/IIITH/precog/Beyond-Components/experiments/train.py:0:0-0:0), the flow is:

- run a **corrupted** input through the original model
- cache activations from that corrupted run
- extract specific corrupted activations
- run the **clean** input through the masked model
- optionally patch in corrupted activations at specific places

This is a causal testing setup.

## Intuition

If a certain direction/component really carries a function, then replacing or swapping that activation should produce a predictable change in output.

For example in the GP task:

- swap gender-related directional activations
- observe whether logits move from `he` toward `she`, or vice versa

So the code is trying to move from:

- “this direction correlates with the task”

to:

- **“this direction causally contributes to the task”**

---

# What Tasks It Studies

The repo supports 3 standard interpretability tasks.

## GP
**Gender Pronoun**

The model predicts pronouns in contexts where gender-sensitive behavior matters.

This is the task emphasized in the README intervention example.

## IOI
**Indirect Object Identification**

A classic circuit interpretability benchmark:
determine the correct indirect object in a sentence.

## GT
**Greater Than**

A numerical/comparison style task often used to probe transformer structure.

These tasks are useful because there is already prior interpretability literature around them, so this repo can compare its finer-grained findings against known “important heads”.

---

# What Training Is Actually Optimizing

From the training code and config, the masked model is trained to balance two things:

- **faithfulness to the full pretrained model**
- **sparsity of the learned mask**

The main fidelity metric appears to be **KL divergence** between:

- full-model logits
- masked-model logits

especially at the final relevant token.

So the objective is roughly:

- keep masked model behavior close to original model
- while minimizing the number/strength of active singular directions

That gives a sparse, behavior-preserving subcircuit.

---

# End-to-End Pipeline

## 1. Load pretrained GPT-2 small
Configured in YAML, e.g. `gpt2-small`.

## 2. Build [MaskedTransformerCircuit](cci:2://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/models/masked_transformer_circuit.py:42:0-1753:19)
This initializes masks and computes/caches SVD decompositions.

## 3. Load task data
From [src/data/data_loader.py](cci:7://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/data/data_loader.py:0:0-0:0) for GP, IOI, GT.

Each example typically has:

- a **clean** input
- a **corrupted** input
- target labels / contrast labels

## 4. Run corrupted input through original model
This creates cached activations for patching/intervention.

## 5. Run clean input through masked model
But using:

- masked singular directions
- optionally corrupted activations at chosen points

## 6. Compare masked vs full model
Metrics include:

- KL divergence
- accuracy
- exact match
- logit difference

## 7. Track sparsity
The code reports things like:

- relative sparsity
- full sparsity
- number of active components

## 8. Visualize and ablate
After training, intervention scripts test whether the discovered directions actually drive behavior.

---

# What the Project Is Trying to Prove Scientifically

The scientific thesis seems to be:

- important transformer computations are **not neatly localized** to whole heads/whole MLPs
- instead they are distributed across **low-rank singular subspaces**
- some heads known from prior work may actually contain **multiple overlapping functions**
- therefore, the right unit of interpretability may be **singular directions**, not architectural components

So this is both:

- a **method**
- and an **argument about how transformer computation is organized**

---

# Important Design Choices I Noticed

## Bias is folded into augmented matrices
That extra `+1` dimension is deliberate.
It lets the code treat affine transformations as linear ones in homogeneous coordinates.

## SVD is cached on disk
This matters because decomposing every layer/head repeatedly would be expensive.

## Masks can be selective
The constructor allows training only certain mask types:

- `qk`
- `ov`
- `mlp_in`
- `mlp_out`

So experiments can isolate different parts of the computation.

## Manual forward pass
This gives the authors precise control over where masked weights and patched activations enter.

## Fidelity is relative to pretrained behavior
The goal is generally not “maximize task accuracy from scratch”.
It is more like:

- preserve the original model’s task behavior
- with a minimal internal subcircuit

---

# What This Is Not

It is **not**:

- training a new language model
- fine-tuning GPT-2 in the usual sense
- pruning neurons/heads in the standard binary way
- doing generic compression for speed

It is specifically **interpretable structured masking over singular directions**.

---

# My Mental Model of the Repo

If I had to summarize the repo in one sentence:

> It is a framework for turning a pretrained transformer into a sparsely masked, SVD-factorized version so you can identify and test the minimal singular directions responsible for specific behaviors.

Or even shorter:

> **Mechanistic interpretability at sub-head / sub-MLP resolution.**

---

# The Most Important File Roles

- **[src/models/masked_transformer_circuit.py](cci:7://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/models/masked_transformer_circuit.py:0:0-0:0)**
  - Implements the main model logic
  - SVD decomposition
  - singular-value masking
  - custom forward pass
  - patching hooks

- **[experiments/train.py](cci:7://file:///Users/s.avvari/IIITH/precog/Beyond-Components/experiments/train.py:0:0-0:0)**
  - Orchestrates training
  - loads data/model/config
  - computes corrupted activations
  - trains masks
  - evaluates sparsity vs fidelity

- **`configs/*.yaml`**
  - Define task-specific settings and optimization hyperparameters

- **`experiments/ablation/...`**
  - Test whether discovered directions causally matter

- **[src/data/data_loader.py](cci:7://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/data/data_loader.py:0:0-0:0)**
  - Loads benchmark tasks

- **[src/utils/visualization.py](cci:7://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/utils/visualization.py:0:0-0:0)**
  - Produces heatmaps and plots of masks / training dynamics

---

# One Slightly More Critical Reading

The README is clear about the intended story, and the code aligns with it.

The key technical bet is that:

- the effective linear maps inside attention/MLP
- when decomposed via SVD
- yield singular directions that correspond to meaningful functional subcomputations

That is a strong but reasonable interpretability hypothesis.
The whole repo is built to test it empirically.

---

# Bottom Line

## In plain English

This code tries to answer:

- not just **which head matters**
- but **which direction inside that head matters**

It does that by:

- decomposing transformer computations with SVD
- learning sparse masks over singular values
- rebuilding a masked transformer forward pass
- validating discovered directions with causal interventions

## Status

- **I’ve understood the main purpose and architecture**
- **The repo is an interpretability research codebase, not an application product**

If you want, I can next do either of these:

- **walk you through [masked_transformer_circuit.py](cci:7://file:///Users/s.avvari/IIITH/precog/Beyond-Components/src/models/masked_transformer_circuit.py:0:0-0:0) function-by-function**
- **explain the training loop mathematically**
- **trace one concrete example, like the GP intervention path end-to-end**