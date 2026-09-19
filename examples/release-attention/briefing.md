# Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
**arXiv:** [1706.03762v7](https://arxiv.org/abs/1706.03762v7) · **Published:** 2017-06-12
**Local model:** qwen3.5:4b · **Session:** `0d03c77529e0`

## Why this paper matters

The authors propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1); [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1)

## Problem

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1); [p. 2, 2 Background](https://arxiv.org/pdf/1706.03762v7#page=2)

## Method

- The best performing models also connect the encoder and decoder through an attention mechanism. Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1); [p. 2, 2 Background](https://arxiv.org/pdf/1706.03762v7#page=2)
- The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. Most competitive neural sequence transduction models have an encoder-decoder structure. [p. 3, 3 Model Architecture](https://arxiv.org/pdf/1706.03762v7#page=3); [p. 2, 3 Model Architecture](https://arxiv.org/pdf/1706.03762v7#page=2)
- Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences. [p. 7, 4 Why Self-Attention](https://arxiv.org/pdf/1706.03762v7#page=7)

## Key results and claims

- Source wording (automatic review requested inspection): “Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.” “On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.” [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1); [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1)
- We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1)

## Limitations


No explicit limitations were identified in the selected evidence; this does not establish that the paper has none.

## Suggested questions

- How does the Transformer handle input and output modalities other than text?
- What are the specific computational costs of using multi-headed attention compared to separable convolutions?
- How does the model perform when generating sequences in a non-auto-regressive manner?

## Q1: How many GPUs were used, and how long was the big model trained?

*answered · 9.6s*

The big model was trained for 3.5 days on 8 P100 GPUs. [p. 8, 6.1 Machine Translation](https://arxiv.org/pdf/1706.03762v7#page=8)

The big models were trained for 300,000 steps (3.5 days). [p. 7, 5.2 Hardware and Schedule](https://arxiv.org/pdf/1706.03762v7#page=7)


## Q2: What was the total training cost in US dollars?

*insufficient evidence · 2.5s*

I could not find enough evidence in the retrieved passages to answer this question.

## Q3: How many warmup steps were used for the learning rate?

*answered · 6.0s*

We used warmup_steps = 4000. [p. 7, 5.3 Optimizer](https://arxiv.org/pdf/1706.03762v7#page=7)


## Evidence ledger

Quotations below were checked against extracted PDF text. This verifies provenance, not semantic entailment.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.

**`dd6a4c88f12ba735` · page 2 · 2 Background**

> To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> The best performing models also connect the encoder and decoder through an attention mechanism.

**`fb192913431d74fa` · page 2 · 2 Background**

> Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

**`ca349beccf97bb9f` · page 3 · 3 Model Architecture**

> The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

**`7be5f4ca916f91a9` · page 2 · 3 Model Architecture**

> Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35].

**`ae0f3bf0ce432534` · page 7 · 4 Why Self-Attention**

> Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. ∗Equal contribution.

**`b37abe7f7725f407` · page 8 · 6.1 Machine Translation**

> Training took 3.5 days on 8 P100 GPUs.

**`abf6f22b8f92a712` · page 7 · 5.2 Hardware and Schedule**

> The big models were trained for 300,000 steps (3.5 days).

**`a3d64752b5cce068` · page 7 · 5.3 Optimizer**

> We used warmup_steps = 4000.

## Extraction and retrieval notes

- Automatic support review replaced 1 briefing paraphrase(s) with visibly labeled source wording. Inspect those passages; the reviewer is fallible.
