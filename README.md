# Constrained Target-Aware Multi-Round Recommendation Framework

## Overview

This project is built upon the multi-round target-oriented recommendation framework ONeRec and proposes a constrained planning enhancement framework for improving target-oriented recommendation progression and interaction quality.

Unlike the original heuristic policy in ONeRec, the proposed framework introduces:

- fine-grained dynamic intention tracking,
- constrained LLM-guided target-aware path reasoning,
- reflective multi-round interaction evolution.

The framework aims to preserve the ranking capability of the original backbone while improving:

- long-term target progression,
- trajectory rationality,
- user acceptability,
- adaptive multi-round interaction.

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

## Current Experimental Findings

Preliminary experiments show that unconstrained free-form planning may damage recommendation ranking performance.

After introducing:

- confidence gating,
- constrained reranking,
- backbone-first recommendation,
- reject-aware replanning,

the proposed framework begins to achieve:

- stronger long-term target progression,
- higher interaction acceptability,
- improved target exposure quality,
- competitive recommendation accuracy.

This suggests that LLM planning should act as a constrained enhancement mechanism rather than fully replacing backbone ranking.

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
├── results_exp1.json
└── README.md
```

File-role mapping:

- core.py: constrained target-aware multi-round recommendation core loop.
- planner_prompt.py: planner and reflection prompt construction.
- llm_client.py: LLM backend loading, env parsing, and preflight validation.
- exp1_runner.py: EXP1 execution, tables, ablations, and result saving.
- main.py: full bootstrap entry with dependency checking and embedding diagnostics.
- native_baseline.py: original backbone baseline implementation.

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

## Key Insight

The project experimentally shows that:

- fully unconstrained LLM planning may weaken recommendation ranking,
- constrained planning can preserve backbone effectiveness,
- reflective rerouting improves long-term target-oriented interaction quality.

The framework therefore focuses on constrained target-aware recommendation enhancement rather than unrestricted planner replacement.

## Environment Notes

- Put the real API key in .env and keep it untracked.
- Use .env.example as the public template.
- The experiment runner writes outputs to results_exp1.json in the current directory.

## Citation

If you use this project, please cite the original ONeRec framework and related target-oriented recommendation literature.
