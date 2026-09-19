# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
**arXiv:** [1810.04805v2](https://arxiv.org/abs/1810.04805v2) · **Published:** 2018-10-11
**Local model:** qwen3.5:4b · **Session:** `7016d3da0645`

## Why this paper matters

BERT is a new language representation model that stands for Bidirectional Encoder Representations from Transformers and is conceptually simple yet empirically powerful. [p. 1, Abstract](https://arxiv.org/pdf/1810.04805v2#page=1); [p. 1, Abstract](https://arxiv.org/pdf/1810.04805v2#page=1)

## Problem

Unlike recent models that rely on unidirectional language models or shallow concatenations, BERT is designed to pretrain deep bidirectional representations by jointly conditioning on both left and right context in all layers. [p. 1, Abstract](https://arxiv.org/pdf/1810.04805v2#page=1); [p. 2, 1 Introduction](https://arxiv.org/pdf/1810.04805v2#page=2)

## Method

- The model uses a masked language model objective to fuse left and right context, and also employs a next sentence prediction task to jointly pretrain text-pair representations. [p. 2, 1 Introduction](https://arxiv.org/pdf/1810.04805v2#page=2); [p. 2, 1 Introduction](https://arxiv.org/pdf/1810.04805v2#page=2)
- BERT demonstrates that pre-trained representations reduce the need for many heavily engineered task-specific architectures and advances the state of the art for eleven NLP tasks. [p. 2, 1 Introduction](https://arxiv.org/pdf/1810.04805v2#page=2); [p. 2, 1 Introduction](https://arxiv.org/pdf/1810.04805v2#page=2)
- The approach involves adding a simple classification layer to the pre-trained model and jointly fine-tuning all parameters on the downstream task. [p. 9, 5.3 Feature-based Approach with BERT](https://arxiv.org/pdf/1810.04805v2#page=9)

## Key results and claims

- The pre-trained BERT model obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5%, MultiNLI accuracy to 86.7%, SQuAD v1.1 Test F1 to 93.2, and SQuAD v2.0 Test F1 to 83.1. [p. 1, Abstract](https://arxiv.org/pdf/1810.04805v2#page=1)
- As a result, the model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks without substantial task-specific architecture modifications. [p. 1, Abstract](https://arxiv.org/pdf/1810.04805v2#page=1)
- Source wording (automatic review requested inspection): “BERT is the first finetuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures. • BERT advances the state of the art for eleven NLP tasks.” [p. 2, 1 Introduction](https://arxiv.org/pdf/1810.04805v2#page=2)

## Limitations


No explicit limitations were identified in the selected evidence; this does not establish that the paper has none.

## Suggested questions

- How does the performance of BERT compare to models using feature-based approaches on tasks requiring fine-grained token-level output?
- What are the computational trade-offs of using deep bidirectional architectures versus shallow unidirectional models for low-resource tasks?
- Can the next sentence prediction task be effectively combined with other pre-training objectives to further improve representation quality?

## Q1: What does BERT stand for?

*answered · 4.6s*

BERT stands for Bidirectional Encoder Representations from Transformers. [p. 1, Abstract](https://arxiv.org/pdf/1810.04805v2#page=1)


## Q2: What was the total training cost in US dollars?

*insufficient evidence · 2.5s*

I could not find enough evidence in the retrieved passages to answer this question.

## Evidence ledger

Quotations below were checked against extracted PDF text. This verifies provenance, not semantic entailment.

**`e1796cabe1b618c0` · page 1 · Abstract**

> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.

**`e1796cabe1b618c0` · page 1 · Abstract**

> BERT is conceptually simple and empirically powerful.

**`e1796cabe1b618c0` · page 1 · Abstract**

> Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

**`0866f1cf1cf65f28` · page 2 · 1 Introduction**

> Unlike Radford et al. (2018), which uses unidirectional language models for pre-training, BERT uses masked language models to enable pretrained deep bidirectional representations.

**`0866f1cf1cf65f28` · page 2 · 1 Introduction**

> Unlike left-toright language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pretrain a deep bidirectional Transformer.

**`0866f1cf1cf65f28` · page 2 · 1 Introduction**

> In addition to the masked language model, we also use a “next sentence prediction” task that jointly pretrains text-pair representations.

**`0866f1cf1cf65f28` · page 2 · 1 Introduction**

> This is also in contrast to Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs. • We show that pre-trained representations reduce the need for many heavily-engineered taskspecific architectures.

**`0866f1cf1cf65f28` · page 2 · 1 Introduction**

> BERT is the first finetuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures. • BERT advances the state of the art for eleven NLP tasks.

**`b41e4d43317ecd13` · page 9 · 5.3 Feature-based Approach with BERT**

> All of the BERT results presented so far have used the fine-tuning approach, where a simple classification layer is added to the pre-trained model, and all parameters are jointly fine-tuned on a downstream task.

**`e1796cabe1b618c0` · page 1 · Abstract**

> It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

**`e1796cabe1b618c0` · page 1 · Abstract**

> As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications.

## Extraction and retrieval notes

- Page 1: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 2: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 3: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 4: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 5: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 6: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 8: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 9: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 10: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 11: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 12: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 13: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 14: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 15: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Page 16: two-column reading order inferred. Complex tables or spanning figures may need manual inspection.
- Automatic support review replaced 1 briefing paraphrase(s) with visibly labeled source wording. Inspect those passages; the reviewer is fallible.
