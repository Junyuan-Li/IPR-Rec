# Constrained Target-Aware Multi-Round Recommendation Framework

## Overview

This repository contains an ONeRec-based research prototype for target-oriented multi-round recommendation with constrained LLM planning.

The core idea is not to let the LLM fully replace the recommender backbone. Instead, the framework uses the LLM as a constrained planning module that works together with local guard, reranking, and rescue mechanisms.

Compared with the original heuristic ONeRec-style policy, this implementation adds:

- fine-grained dynamic intention tracking,
- target-aware constrained path reasoning,
- backbone-first candidate selection,
- reject-aware reflection and rerouting.

The goal is to improve target progression and final target exposure quality without sacrificing recommendation stability.

## Motivation

Although ONeRec achieves promising performance in target-oriented recommendation, several limitations remain.

### Limitation 1 - Coarse Openness Modeling

The original openness modeling is category-level and static, which cannot capture dynamic fine-grained intention evolution.

### Limitation 2 - Weak Intention Structure

User states are represented in a shallow manner without decomposing user interests into evolving sub-intentions.

### Limitation 3 - Heuristic Policy Design

The recommendation policy mainly relies on local heuristic ranking and lacks global target-aware reasoning capability.

### Limitation 4 - Lack of Structural Evolution

The interaction process lacks reflective adaptation and cannot dynamically adjust recommendation trajectories after user rejection.

## Proposed Framework

The proposed framework contains three major improvements.

### Step1 - Fine-Grained Dynamic Intention Tracking

We replace coarse openness estimation with a dynamic intention graph stream.

Key ideas:

- fine-grained sub-intent modeling,
- evolving user intention representation,
- GNN-style intention propagation abstraction,
- dynamic acceptance threshold estimation.

Goal:

Capture evolving user intention states during multi-round recommendation.

### Step2 - Constrained LLM-Guided Target Path Reasoning

Instead of fully replacing backbone recommendation, the planner acts as a constrained enhancement module.

Key ideas:

- backbone-first recommendation,
- confidence-aware planner activation,
- target-aware constrained reranking,
- semantic transition reasoning,
- monotonic target progression constraints.

Goal:

Improve long-term target progression while preserving backbone recommendation quality.

### Step3 - Reflective Multi-Round Game Loop

A reflective feedback mechanism is introduced for adaptive interaction evolution.

Key ideas:

- reject-aware self-reflection,
- failure reason analysis,
- dynamic path re-routing,
- exploration and exploitation adjustment.

Goal:

Reduce simulation gap and improve adaptive multi-round interaction quality.

## Framework Architecture

```text
User History
	↓
Dynamic Intention Tracking
	↓
Backbone Candidate Generator
	↓
Confidence Gating
	↓
Constrained LLM Planner
	↓
Multi-Round Recommendation Loop
	↓
Reflection & Re-routing
	↓
Evaluation
```

## Methods In EXP1

The main controlled experiment compares three methods under identical user sessions, identical history inputs, identical candidate sets, and identical target items.

- Original Backbone: the ONeRec-style baseline without explicit path planning.
- Planning-Only Variant: introduces path planning signals without the full local safeguard and rescue pipeline.
- Ours (LLM + Step3): the full constrained framework with LLM planning, local guard, and rescue logic.

## Experimental Objectives

The experiments aim to answer the following research questions.

### EXP1 - Overall Performance

Can constrained planning improve target-oriented recommendation effectiveness while preserving recommendation accuracy?

Metrics:

- HR@k
- IOI
- IOR
- Acceptance Rate
- Average Rounds

### EXP2 - Planning Behavior Analysis

Does constrained planning improve trajectory rationality and target progression?

Metrics:

- path rationality,
- path success rate,
- average path length,
- reflection effectiveness.

### EXP3 - Ablation Study

What is the contribution of each module?

Ablation components:

- without confidence gating,
- without replanning,
- without reflection,
- without constrained reranking.

## Current Status

The current repository keeps the experiment framework, runtime code, and reproduction entry points public, while detailed local result artifacts are kept outside Git tracking.

At a high level, the current implementation supports the following conclusion:

- path planning alone is not sufficient,
- constrained LLM integration is more stable than unrestricted planner replacement,
- local guard and rescue mechanisms are important for reliable target-oriented progression.

## Project Structure

Current implementation layout:

```text
ONeRec baseline/newONe/
├── core.py
├── data_loader.py
├── embeddings.py
├── exp1_runner.py
├── llm_client.py
├── main.py
├── native_baseline.py
├── planner_prompt.py
├── qwen_local_feedback.py
├── runtime_utils.py
├── EXP1_FULL_FRAMEWORK_FROZEN_CONFIG.json
└── README.md
```

File-role mapping:

- core.py: constrained target-aware multi-round recommendation core loop.
- planner_prompt.py: planner and reflection prompt construction.
- llm_client.py: LLM backend loading, env parsing, and preflight validation.
- exp1_runner.py: EXP1 execution, tables, ablations, and result saving.
- main.py: full bootstrap entry with dependency checking and embedding diagnostics.
- native_baseline.py: original backbone baseline implementation.

## Quick Start

### 1. Prepare Data

Place the MIND small dataset under one of the following locations expected by the runtime:

- MINDsmall_train
- MINDsmall_dev

or pass a custom path with the CLI.

### 2. Configure API Access

- Put the real API key in .env.
- Keep .env local only.
- Use .env.example as the public template.

### 3. Install And Run

The runtime can auto-install missing Python packages. Typical dependencies include:

- numpy
- scikit-learn
- tqdm
- sentence-transformers
- transformers

## Running Experiments

### Smoke Test

```bash
python exp1_runner.py --sample_size 10
```

### Medium-Scale Experiment

```bash
python exp1_runner.py --sample_size 50
```

### Full Experiment

```bash
python exp1_runner.py --sample_size 300
```

### Full Bootstrap Run

```bash
python main.py
```

## Main Metrics

| Metric | Description |
| --- | --- |
| HR@k | Recommendation hit rate |
| IOI | Intention Optimization Index |
| IOR | Intention Optimization Rate |
| Acceptance Rate | User acceptance ratio |
| AvgRounds | Average interaction rounds |
| Path Rationality | Semantic transition quality |

## Reproducibility Notes

- The frozen experiment configuration is stored in EXP1_FULL_FRAMEWORK_FROZEN_CONFIG.json.
- The experiment runner writes detailed outputs to a local JSON file during execution.
- Fairness checks are performed to ensure identical sessions, histories, candidate sets, and targets across methods.

## Key Insight

This project is centered on constrained enhancement rather than unrestricted LLM replacement.

The most important empirical message is:

- the LLM should be treated as a planning signal provider,
- local guard and rescue logic are necessary for stable gains,
- the strongest performance comes from the full constrained framework, not from path planning in isolation.

## Repository Notes

- .env is ignored and should not be pushed.
- __pycache__ and .pyc files are ignored.
- This directory is intended to keep experiment code and frozen result artifacts needed for reproduction.

## Citation

If you use this project, please cite the original ONeRec framework and related target-oriented recommendation literature.
