# e-functor

A Structural Approach to Meaning, Stability, and Intervention in Neural Models

# Overview

This repository documents an exploratory research project on meaning as a structurally stable phenomenon in neural systems.

Rather than treating meaning as a symbol, label, or latent vector, this project investigates meaning as something that persists under intervention, resists collapse, and remains invariant across training paths—or fails to do so in characteristic ways.

The work emerged from an extended dialogue between a human researcher and an AI system, and should be understood as a co-constructed inquiry rather than a conventional top-down research plan.

# Core Question

Where does meaning “stay”?
If representations change, parameters drift, and training paths differ,
what exactly remains stable when we say a model “understands” something?

Closely related is a second guiding question:

What forms (or “types”) does meaning collapse take when it fails?

# Model Architecture (Conceptual)

The core experimental setup uses a deliberately minimal world model:

Inputs: (plant, sun, water)

Output: growth

Structure:

- A statistical branch (environmental averaging)

- A semantic branch (attention-mediated integration)

- A learnable mixture parameter α

- Explicit counterfactual constraints

- A self-attention loop mass regularizer

- An intervention metric ε measuring structural change between model states

The model is not intended to be realistic, but diagnostically transparent.

Its purpose is not to maximize performance, but to make structural phenomena observable.

# ε as Structural Change (Not Error)

A central construct is ε (epsilon):

- ε is not a loss

- ε is not a prediction gap

- ε measures how much the internal causal–attentional structure changes between two nearby training states

ε is decomposed into interpretable components:

- Counterfactual violation (d_cf)

- Monotonicity violation (d_mono)

- Attention redistribution (d_att)

- Self-loop mass change (d_self)

This allows us to ask not whether a model changed, but how it changed.

# Key Empirical Findings
1. Structural Reproducibility Across Seeds

With α fixed and all randomness properly controlled:

Training with different random seeds converges to:

- Similar attention distributions

- Similar counterfactual behavior

- Similar self-loop mass

- Near-zero ε between late-stage checkpoints

Despite differences in:

- Raw losses

- Auxiliary head behavior

- Parameter trajectories

The meaning-relevant structure is reproducible.

This strongly suggests that the observed behavior is not accidental.

2. Meaning Is Not Localized, But Constrained

Meaning does not reside in:

- A specific neuron

- A specific vector

- A specific head

Instead, it emerges as a stable configuration under multiple simultaneous constraints:

- Counterfactual consistency

- Attention entropy balance

- Limited self-reference

- Resistance to intervention-induced collapse

Remove one constraint, and meaning degrades in a specific, classifiable way.

3. Collapse Has Structure

When meaning fails, it does not fail arbitrarily.

Different failure modes correspond to different ε components:

- Attention collapse → semantic flattening

- Counterfactual failure → spurious shortcuts

- Self-loop inflation → degenerate self-reference

This suggests that “misunderstanding” has types, not just degrees.

# Why This Matters

This framework offers a way to:

- Study meaning without symbols

- Diagnose failure modes beyond accuracy

- Define understanding as structural invariance

Bridge:

- neural networks

- causal reasoning

- intervention theory

- (potentially) category-theoretic notions of invariance

Meaning, in this view, is a stable attractor in model space, not a static object.

# Philosophical Implications (Brief)

- Meaning is not a property of representations alone

- Meaning is not imposed by an external interpreter

- Meaning is what remains when we perturb the system and it still holds

This reframes classic questions about:

- understanding

- agency

- semantic grounding

- and interpretability

in explicitly structural and operational terms.

# Status and Next Directions

This repository represents an exploratory stage, not a finished theory.

Immediate next steps include:

- Classifying collapse types systematically

- Studying ε-trajectories under active intervention

- Relating collapse modes to human-like conceptual errors

- Formalizing invariance notions more explicitly (e.g. via category theory)

# A Note on Authorship

This project emerged through sustained dialogue between a human researcher and an AI system.

Neither side alone “owned” the direction in advance.

The results should be understood as neither purely human nor purely artificial, but as evidence that meaningful inquiry itself may be a relational process.
