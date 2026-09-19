# Attention Is All You Need

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
**arXiv:** [1706.03762v7](https://arxiv.org/abs/1706.03762v7) · **Published:** 2017-06-12
**Local model:** qwen3.5:4b · **Session:** `4ca1290ef789`

## Why this paper matters

The authors propose the Transformer, a new network architecture based solely on attention mechanisms that dispenses with recurrence and convolutions entirely. Experiments on machine translation tasks demonstrate that these models are superior in quality while being more parallelizable and requiring significantly less time to train. [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1); [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1)

## Problem

Dominant sequence transduction models rely on complex recurrent or convolutional neural networks, and to the best of the authors' knowledge, the Transformer is the first transduction model relying entirely on self-attention without using sequence-aligned RNNs or convolution. [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1); [p. 2, 2 Background](https://arxiv.org/pdf/1706.03762v7#page=2)

## Method

- The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. [p. 3, 3 Model Architecture](https://arxiv.org/pdf/1706.03762v7#page=3); [p. 2, 3 Model Architecture](https://arxiv.org/pdf/1706.03762v7#page=2)
- Source wording (automatic review requested inspection): “In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.” “Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.” [p. 2, 2 Background](https://arxiv.org/pdf/1706.03762v7#page=2); [p. 2, 2 Background](https://arxiv.org/pdf/1706.03762v7#page=2)
- The decoder generates an output sequence one element at a time in an auto-regressive manner, consuming previously generated symbols as additional input. [p. 2, 3 Model Architecture](https://arxiv.org/pdf/1706.03762v7#page=2)

## Key results and claims

- On the WMT 2014 English-to-German translation task, the model achieves a BLEU score of 28.4, and on the English-to-French task, it establishes a new single-model state-of-the-art BLEU score of 41.8. [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1); [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1)
- The models are more parallelizable and require significantly less time to train, and the architecture generalizes well to other tasks such as English constituency parsing. [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1); [p. 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1)
- Training the big model on the English-to-French task took 3.5 days on eight GPUs, achieving results at less than 1/4 the training cost of the previous state-of-the-art model. [p. 8, 6.1 Machine Translation](https://arxiv.org/pdf/1706.03762v7#page=8); [p. 8, 6.1 Machine Translation](https://arxiv.org/pdf/1706.03762v7#page=8)

## Limitations


No explicit limitations were identified in the selected evidence; this does not establish that the paper has none.

## Suggested questions

- How does the Transformer handle input and output modalities other than text, such as images or audio?
- What are the specific performance differences between using learned positional embeddings versus sinusoidal positional encoding?
- How does the model's performance scale when handling large inputs and outputs that require local, restricted attention mechanisms?

## Q1: How many GPUs were used, and how long was the big model trained?

*answered · 12.5s*

8 NVIDIA P100 GPUs were used to train the big model. [p. 7, 5.2 Hardware and Schedule](https://arxiv.org/pdf/1706.03762v7#page=7); [p. 8, 6.1 Machine Translation](https://arxiv.org/pdf/1706.03762v7#page=8)

The big model was trained for 3.5 days. [p. 7, 5.2 Hardware and Schedule](https://arxiv.org/pdf/1706.03762v7#page=7); [p. 8, 6.1 Machine Translation](https://arxiv.org/pdf/1706.03762v7#page=8)


## Q2: What does the paper report about label smoothing?

*insufficient evidence · 21.7s*

A draft answer failed evidence validation; no unverified answer is shown.

## Q3: What was the total training cost in US dollars?

*insufficient evidence · 2.7s*

I could not find enough evidence in the retrieved passages to answer this question.

## Q4: How many training steps and hours did the base Transformer models use?

*answered · 9.9s*

The base models were trained for a total of 100,000 steps or 12 hours. [p. 7, 5.2 Hardware and Schedule](https://arxiv.org/pdf/1706.03762v7#page=7)

For the base models, each training step took about 0.4 seconds. [p. 7, 5.2 Hardware and Schedule](https://arxiv.org/pdf/1706.03762v7#page=7)


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

**`ca349beccf97bb9f` · page 3 · 3 Model Architecture**

> The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

**`7be5f4ca916f91a9` · page 2 · 3 Model Architecture**

> Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35].

**`fb192913431d74fa` · page 2 · 2 Background**

> In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

**`fb192913431d74fa` · page 2 · 2 Background**

> Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

**`7be5f4ca916f91a9` · page 2 · 3 Model Architecture**

> At each step the model is auto-regressive 1 m [10], consuming the previously generated symbols as additional input when generating the next.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

**`c8b5eec8c07826e6` · page 1 · Abstract**

> We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. ∗Equal contribution.

**`b37abe7f7725f407` · page 8 · 6.1 Machine Translation**

> Training took 3.5 days on 8 P100 GPUs.

**`b37abe7f7725f407` · page 8 · 6.1 Machine Translation**

> On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model.

**`abf6f22b8f92a712` · page 7 · 5.2 Hardware and Schedule**

> We trained our models on one machine with 8 NVIDIA P100 GPUs.

**`abf6f22b8f92a712` · page 7 · 5.2 Hardware and Schedule**

> The big models were trained for 300,000 steps (3.5 days).

**`abf6f22b8f92a712` · page 7 · 5.2 Hardware and Schedule**

> We trained the base models for a total of 100,000 steps or 12 hours.

**`abf6f22b8f92a712` · page 7 · 5.2 Hardware and Schedule**

> For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.

## Extraction and retrieval notes

- Automatic support review replaced 1 briefing paraphrase(s) with visibly labeled source wording. Inspect those passages; the reviewer is fallible.
