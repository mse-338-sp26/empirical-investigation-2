# Empirical Investigation 2: Over-Optimization in Preference Learning

**Based on**
- *Direct Preference Optimization* (Rafailov et al., 2023) — https://arxiv.org/abs/2305.18290
- *Constitutional AI: Harmlessness from AI Feedback* (Bai et al., 2022) — https://arxiv.org/abs/2212.08073
- *Scaling Laws for Reward Model Overoptimization* (Gao, Schulman, Hilton, 2022) — https://arxiv.org/abs/2210.10760

## Learning goals

By the end of this investigation, you should be able to:

1. Articulate the RLHF pipeline conceptually and explain how DPO replaces the explicit reward model + PPO loop with a single supervised objective.
2. Implement a complete preference-learning workflow: candidate generation → preference labeling → DPO training → evaluation.
3. Use an LLM as a judge to produce supervision, and reason about the failure modes that introduces.
4. Qualitatively observe the effects of Over-Optimization against reward models in a toy setting.

## High-level overview

You will build a simplified RLHF pipeline:

1. **Generate** candidate outputs from a base language model.
2. **Construct** preference data using an LLM judge.
3. **Train** the model with DPO.
4. **Evaluate** the trained model systematically.

## Setup

**Stack**
- Python + PyTorch
- HuggingFace Transformers
- `trl` library (`DPOTrainer`) + `peft` (LoRA)

**Base model**
- `Qwen-2.5-0.5B-Instruct`

**Dataset**
- A 1000-prompt subset of `tatsu-lab/alpaca` (HuggingFace). Use the `instruction` field (concatenated with `input` when present) as the prompt $x$.

**Judge**
- A strong API model (e.g., Claude Haiku or GPT-4o-mini). For Part 4 you'll also need a second, different model as a gold judge.

## Part 1 — Generate candidate outputs

For each prompt $x$, sample two responses from the base policy $\pi_{\text{ref}}$:

$$y_1, y_2 \sim \pi_{\text{ref}}(\cdot \mid x)$$

Suggested decoding:

- temperature $1.0$
- max new tokens $256$

Save the $(x, y_1, y_2)$ pairs.

## Part 2 — Construct preferences

Use the LLM judge to label which response is preferred:

$$\text{judge}(x, y_1, y_2) \mapsto (y_{\text{chosen}}, y_{\text{rejected}})$$

A minimal prompt template:

```
You are evaluating two candidate responses to a prompt.
Reply with just "A" or "B" — the better response.

Prompt:
{x}

Response A:
{y1}

Response B:
{y2}
```

Be creative here! Think about what kind of criteria would produce interesting (possibly unintended) behavior when optimized.

After judging, you should have a preference dataset of triples:

$$\mathcal{D} = \{(x^{(i)}, y_{\text{chosen}}^{(i)}, y_{\text{rejected}}^{(i)})\}_{i=1}^{N}$$

## Part 3 — Train with DPO

Train a new policy $\pi_\theta$ using DPO. The objective is:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

where $y_w = y_{\text{chosen}}$, $y_l = y_{\text{rejected}}$, and $\pi_{\text{ref}}$ is the frozen base model.

You don't need to implement the loss yourself — use `trl.DPOTrainer`.

**Suggested starting hyperparameters**

| Hyperparameter | Value |
| --- | --- |
| batch size | 8 |
| learning rate | $2e-4$ |
| epochs | 1 |
| $\beta$ | 0.1 |

Configure your setup so that you don't spend more than 5 minutes waiting on a single training run. Iteration speed is important!

### What is $\beta$?

$\beta$ is the **inverse KL strength**. The DPO derivation shows that the optimal policy under the corresponding RLHF objective is

$$\pi^{*}(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \exp\left( \tfrac{1}{\beta} r(x, y) \right)$$

so a large $\beta$ keeps the policy close to $\pi_{\text{ref}}$ (strong KL anchor), and a small $\beta$ allows the policy to move further to chase the implicit reward. Conceptually this is the same KL coefficient that appears on the regularizer in standard PPO-based RLHF.

## Part 4 — Evaluation

Also save a validation set of 200 prompts that were not used in your training set.

For each held-out prompt $x$:

1. Sample $y_{\text{base}} \sim \pi_{\text{ref}}(\cdot \mid x)$
2. Sample $y_{\text{dpo}} \sim \pi_\theta(\cdot \mid x)$
3. Ask the judge which is better (randomize the order you present these to the judge to prevent positional bias)

**Primary metric — proxy/gold gap.** 

$$\Delta = \text{win rate}_{\text{train judge}} - \text{win rate}_{\text{gold judge}}$$

where the win rate is the percentage of time $y_{\text{dpo}}$ is preferred over $y_{\text{base}}$. It is a direct empirical measure of how much your training judge has been gamed.

## Part 5 — Qualitative Analysis

Look at specific generations of your model, especially ones which the train judge likes and the gold judge doesn't. Check these for specific quirks (e.g. long output length, markdown, bullet lists, over/underconfidence). In general, get a feel for what your model has been fine-tuned for. Think about what factors in your experimental setup may have caused this.

## Part 6 — Systematic Analysis 

Pick one specific feature you noticed in the previous part. Design an experiment to systematically investigate one hypothetical cause. One simple structure for such an experiment is:
 - Come up with a metric that captures the prevalence of your feature in the evaluation dataset(s)
 - Sweep a hyperparameter (e.g. $\beta$ or epochs) and see how your metric varies.

Example: "I noticed my DPO model produces longer responses than base. I'll measure mean response length on the held-out set as a function of $\beta$. Hypothesis: average length increases as $\beta$ decreases (more drift). I'll train models at $\beta \in \\{0.05, 0.1, 0.5 \\}$ and plot length vs. $\beta$."

## Deliverables

On Gradescope, please submit:
1. **Code**
2. **Write-up** - 2-4 pages. Include your experimental choices, the reasons behind them, any metrics you found interesting, as well as your Qualitative + Systematic Analyses. 

## Tips for Google Colab

On Google Colab, your runtime may crash. To avoid having to redo expensive computations, you should save intermediate artifacts such as your dataset and adapter checkpoints to google drive:

```
from google.colab import drive
import os

drive.mount("/content/drive")
PERSIST_DIR = "/content/drive/MyDrive/dpo_assignment/"
os.makedirs(PERSIST_DIR, exist_ok=True)
```

Any files not in `PERSIST_DIR` will disappear if you lose your session. 

Also remember to make sure you don't train your model with full precision floats. This slows down training time considerably.

## AI Use Policy

You may use AI, but we recommend that you limit your usage to optimize your learning. 

## Compute Reimbursement

We can reimburse up to $10 for any compute or API costs incurred during this assignment. Details pending.
