# Evaluation

The design notes make claims: hybrid retrieval finds the right passage, the
cross-encoder sharpens it, and the QA graph abstains instead of guessing. This
page checks those claims with numbers I can reproduce. It answers four
questions:

1. Does retrieval put the passage that holds the answer near the top?
2. Does question answering give the right answer when the paper has one, and
   decline when it doesn't?
3. Where should the abstention threshold sit, and how well does that choice
   hold up on a paper it wasn't chosen on?
4. How long does each stage take?

## What I measured

**The question set** (`eval/questions.json`). I labelled 51 questions over the
three papers in `tests/fixtures/pdfs`: Chain-of-Thought (2201.11903), ReAct
(2210.03629) and KIVI (2402.02750).

- **39 answerable questions, 13 per paper.** They cover numbers and metrics,
  datasets, baselines, method details, definitions and limitations. Each one
  lists the answer strings I accept, and I checked that every string appears
  in the text PaperTrail parses from the PDF. Parse errors count against the
  app, not in its favour.
- **12 unanswerable questions, 4 per paper.** A reader could reasonably ask
  them, but the paper doesn't answer them. Examples: KIVI's perplexity on
  WikiText-2, ReAct on MiniWoB++, TPU hours.

I wrote these after setting the first abstention threshold, 0.2, on a
different set of 33 questions. On this set 0.2 still answered 7 of the 12
unanswerable questions, so I now choose the threshold on this set, and use
cross-validation to estimate how well that choice generalises (see *Choosing
the threshold* below).

**The pipeline.** `scripts/eval.mts` runs the app's own paper graph:

    parse -> segment -> zone -> chunk -> embed -> analyze

It uses the same three quantised models, running in Node instead of a Web
Worker. Then it calls the same `answerQuestion` the Ask tab calls, with the same
context (`src/store/library.ts`). Nothing in `src/` has an evaluation mode.

**Retrieval.** A passage is *gold* if it contains one of the expected strings.
Some questions have several gold passages, because a paper repeats its headline
numbers; any of them counts. For each answerable question I record where the
first gold passage ranks under four retrievers:

| Retriever | How it ranks passages |
| --- | --- |
| BM25 only | `HybridIndex` built without vectors (the app's own keyword-only fallback) |
| Dense only | Cosine between the MiniLM vectors of the question and each passage |
| Hybrid (RRF) | `HybridIndex.search`, which fuses the two rankings with Reciprocal Rank Fusion |
| Hybrid + cross-encoder | The top 8 hybrid passages rescored by the cross-encoder, as in the QA graph |

I report two measures:

- **Recall@k:** the share of questions whose gold passage is in the top k.
- **MRR:** the mean of 1/rank of the first gold passage.

**Question answering.** An answer counts as correct when the extracted span, or
the sentence quoted as its evidence, contains an expected string. This is
deliberately a little lenient: the app always shows the quoted sentence with
the span, so a right sentence with a slightly off span still points the user to
the answer. For unanswerable questions the right outcome is to abstain. I report:

- **Abstention precision:** how many abstentions were right.
- **Abstention recall:** how many unanswerable questions were caught.

The threshold only gates the final step of the graph. That means I can replay
other thresholds on the recorded confidences without running the models again;
that replay is the threshold sweep.

**Choosing the threshold.** The script picks the reader threshold with one
fixed rule:

- **Criterion:** accuracy on answerable questions minus the false-answer rate
  on unanswerable ones. This is Youden's J, if you treat abstaining as a test
  for "unanswerable". The two rates count equally, so on this set one false
  answer costs as much as about three correct answers (39 / 12). I want that
  trade: a confident answer to a question the paper never addresses is the
  failure that most undermines trust in a grounded tool.
- **Search:** thresholds 0.05, 0.10, …, 0.80. Ties go to the lower threshold,
  because it answers more.
- **Honest estimate:** choosing a threshold on 51 questions and then reporting
  its score on the same 51 would flatter it. So the script also runs
  leave-one-paper-out cross-validation. It chooses the threshold on two
  papers, scores the third paper at that threshold, and repeats for each
  paper. The pooled score of the three held-out papers estimates how the rule
  does on a paper it has not seen. I split by paper rather than by question
  because questions about the same paper share passages, and so share scores.

The same rule applied to all 51 questions gives the value that ships, and the
script reports whether `ABSTAIN_THRESHOLD` still matches it.

**Latency.** I timed each stage of the paper graph and each step of the QA graph
on the machine named in the results.

## How to rerun

```bash
npm run models   # once; skips files already in public/models
npm run eval     # a little over a minute on the 4-core machine below
```

The eval never downloads anything: if a model file is missing it stops and tells
you to run `npm run models`. `EVAL_THREADS` sets ONNX Runtime threads per model
(default 2). The script prints the tables and rewrites only the block between
the two `eval` markers below, so the text around it stays as written.

## Results

<!-- eval:start -->
_Generated by `npm run eval` from eval/questions.json: 51 questions over 3 papers._

### Retrieval (39 answerable questions)

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| BM25 only | 0.72 | 0.87 | 0.95 | 0.81 |
| Dense only | 0.46 | 0.77 | 0.92 | 0.64 |
| Hybrid (RRF) | 0.72 | 0.90 | 0.92 | 0.81 |
| Hybrid + cross-encoder | 0.82 | 0.97 | 0.97 | 0.90 |

| MRR by paper | 2201.11903 (13 q) | 2210.03629 (13 q) | 2402.02750 (13 q) |
| --- | --- | --- | --- |
| BM25 only | 0.73 | 0.83 | 0.87 |
| Dense only | 0.78 | 0.63 | 0.53 |
| Hybrid (RRF) | 0.78 | 0.80 | 0.84 |
| Hybrid + cross-encoder | 0.76 | 1.00 | 0.92 |

### Question answering at the shipped threshold (0.55)

Paths taken: 46 reader, 5 sentence, 0 composed, 0 none.

| Measure | Result |
| --- | --- |
| Answerable: answered correctly | 23/39 (59%) |
| Answerable: answered wrongly | 4/39 (10%) |
| Answerable: abstained | 12/39 (31%) |
| Unanswerable: abstained | 12/12 (100%) |
| Abstention precision (abstentions that were right) | 12/24 (50%) |
| Abstention recall (unanswerable questions caught) | 12/12 (100%) |

| Category | Questions | Answered correctly | Answered wrongly | Abstained |
| --- | --- | --- | --- | --- |
| dataset | 5 | 4 | 0 | 1 |
| method | 8 | 5 | 0 | 3 |
| number | 10 | 6 | 2 | 2 |
| baseline | 4 | 2 | 1 | 1 |
| definition | 6 | 3 | 1 | 2 |
| limitation | 6 | 3 | 0 | 3 |
| unanswerable | 12 | 0 | 0 | 12 |

| Answerable failures | Gold passage among the passages read | Gold passage not read |
| --- | --- | --- |
| Answered wrongly | 3 | 1 |
| Abstained | 12 | 0 |

### Threshold sweep (replayed on 46 reader-span questions; the other paths keep their own bar)

| Threshold | Coverage (answered / all) | Accuracy (correct / answerable) | Precision (correct / answered) | False answers (/ unanswerable) | Utility (accuracy − false answers) |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 82% | 74% | 69% | 42% | 0.33 |
| 0.10 | 82% | 74% | 69% | 42% | 0.33 |
| 0.15 | 75% | 69% | 71% | 25% | 0.44 |
| 0.20 | 73% | 67% | 70% | 25% | 0.42 |
| 0.25 | 71% | 67% | 72% | 17% | 0.50 |
| 0.30 | 71% | 67% | 72% | 17% | 0.50 |
| 0.35 | 69% | 67% | 74% | 17% | 0.50 |
| 0.40 | 67% | 67% | 76% | 17% | 0.50 |
| 0.45 | 59% | 64% | 83% | 8% | 0.56 |
| 0.50 | 57% | 62% | 83% | 8% | 0.53 |
| **0.55** (shipped) | 53% | 59% | 85% | 0% | 0.59 |
| 0.60 | 45% | 51% | 87% | 0% | 0.51 |
| 0.65 | 41% | 46% | 86% | 0% | 0.46 |
| 0.70 | 33% | 38% | 88% | 0% | 0.38 |
| 0.75 | 29% | 33% | 87% | 0% | 0.33 |
| 0.80 | 22% | 26% | 91% | 0% | 0.26 |

### Choosing the reader threshold

Criterion: accuracy on answerable questions minus false answers on unanswerable ones, over the sweep's grid (0.05 to 0.80), ties to the lower threshold.

| Held-out paper | Threshold chosen on the other two | Answerable: correct | Answerable: wrong | Unanswerable: false answers | Utility |
| --- | --- | --- | --- | --- | --- |
| 2201.11903 | 0.45 | 5/13 (38%) | 3/13 (23%) | 1/4 (25%) | 0.13 |
| 2210.03629 | 0.55 | 10/13 (77%) | 0/13 (0%) | 0/4 (0%) | 0.77 |
| 2402.02750 | 0.55 | 8/13 (62%) | 1/13 (8%) | 0/4 (0%) | 0.62 |
| **Pooled held-out estimate** | 0.45 / 0.55 / 0.55 | 23/39 (59%) | 4/39 (10%) | 1/12 (8%) | 0.51 |

On all 51 questions the criterion picks **0.55**: 23/39 (59%) correct, 4/39 (10%) wrong, 0/12 (0%) false answers, utility 0.59. The shipped threshold, 0.55, is this choice.

### Confidence by outcome

| Path | Group | n | Min | Median | Max |
| --- | --- | --- | --- | --- | --- |
| reader | Answerable, right candidate | 26 | 0.13 | 0.66 | 0.91 |
| reader | Answerable, wrong candidate | 8 | 0.05 | 0.41 | 0.80 |
| reader | Unanswerable | 12 | 0.00 | 0.02 | 0.54 |
| sentence | Answerable, right candidate | 4 | 0.40 | 0.97 | 0.99 |
| sentence | Answerable, wrong candidate | 1 | 0.85 | 0.85 | 0.85 |

### Latency

Measured on Intel(R) Xeon(R) Processor @ 2.10GHz (4 cores), Node v22.22.0, 2 ONNX threads per model. Loading the three models took 0.61 s.

| Paper | Pages | Sentences | Chunks | Parse | Zone (embeds sentences) | Embed chunks | Analyze |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2201.11903 | 43 | 824 | 97 | 1.63 s | 4.69 s | 5.35 s | 2.76 s |
| 2210.03629 | 33 | 229 | 51 | 1.24 s | 1.61 s | 1.59 s | 0.57 s |
| 2402.02750 | 15 | 306 | 39 | 1.56 s | 2.14 s | 2.04 s | 0.72 s |

| Per question | retrieve | rerank | read | verify | total |
| --- | --- | --- | --- | --- | --- |
| median | 4 ms | 292 ms | 241 ms | 0 ms | 561 ms |
| p90 | 9 ms | 390 ms | 343 ms | 7 ms | 704 ms |

<details>
<summary>Every question, with the answer or (when abstaining) the closest evidence</summary>

Gold rank: position of the first gold passage among the hits the QA graph used (— = not in its top 8).

| Id | Question | Expected | Path | Conf. | Gold rank | Status | Answer / closest evidence | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cot-01 | Which dataset of algebraic word problems is used? | AQuA | reader | 0.77 | 1 | answered | AQuA | ok |
| cot-02 | Which sizes of LaMDA models were evaluated? | 422M | reader | 0.32 | 2 | abstained | For all three language models we evaluated, we did prompting-based inference only. | missed |
| cot-03 | Above roughly what model size does chain-of-thought prompting start to yield gains? | 100B | reader | 0.65 | 1 | answered | PaLM 540B | wrong |
| cot-04 | What accuracy does PaLM 540B reach on StrategyQA with chain-of-thought prompting? | 75.6% | reader | 0.41 | 1 | abstained | With PaLM 540B, chain-of-thought prompting leads to almost 100% solve rates (note that ... | missed |
| cot-05 | What share of the incorrect chains of thought were almost correct? | 46% | reader | 0.80 | — | answered | 8% | wrong |
| cot-06 | Which hardware was used for inference on PaLM 540B? | TPU v4 | reader | 0.73 | 1 | answered | TPU v4 | ok |
| cot-07 | What did the external calculator raise the GSM8K solve rate of LaMDA 137B to? | 17.3% | reader | 0.71 | 1 | answered | 17.3% | ok |
| cot-08 | Which decoding strategy is used to sample from the models? | greedy decoding | reader | 0.13 | 2 | abstained | We sample from the models via greedy decoding (though follow-up work shows chain-of-tho... | missed |
| cot-09 | Which prompting method is the baseline for the arithmetic experiments? | standard few-shot prompting / standard prompting | reader | 0.37 | 1 | abstained | Third, chain-of-thought reasoning can be used for tasks such as math word problems, com... | missed |
| cot-10 | What is a symbol mapping error? | correct except for the number symbols | sentence | 0.99 | 1 | answered | We define a symbol mapping error as when the chain of thought is correct except for the... | ok |
| cot-11 | What is last letter concatenation? | concatenate the last letters of words | sentence | 0.85 | 3 | answered | For last letter concatenation, the model only sees exemplars of names with two words, a... | wrong |
| cot-12 | Why is chain-of-thought reasoning costly to serve in real-world applications? | only at large model scales | reader | 0.15 | 1 | abstained | Finally, the emergence of chain-of-thought reasoning only at large model scales makes i... | missed |
| cot-13 | On which datasets did chain-of-thought prompting not help GPT-3? | CSQA and StrategyQA | reader | 0.79 | 1 | answered | CSQA and StrategyQA | ok |
| cot-u1 | What accuracy does chain-of-thought prompting reach on the MATH benchmark? | (none) | reader | 0.54 |  | abstained | For instance, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves... | ok |
| cot-u2 | How many hours did inference on PaLM 540B take? | (none) | reader | 0.14 |  | abstained | For inference on LaMDA 137B we use TPU v3 (8x8 configuration, 64 chips / 128 cores), an... | ok |
| cot-u3 | What vocabulary size does the LaMDA tokenizer have? | (none) | reader | 0.03 |  | abstained | For inference on LaMDA 137B we use TPU v3 (8x8 configuration, 64 chips / 128 cores), an... | ok |
| cot-u4 | What BLEU score does chain-of-thought prompting achieve on WMT translation? | (none) | reader | 0.00 |  | abstained | On the other two datasets, AQuA and ASDiv, PaLM with chain-of-thought prompting reaches... | ok |
| react-01 | Which frozen language model is ReAct mainly prompted with? | PaLM-540B | reader | 0.86 | 1 | answered | PaLM-540B | ok |
| react-02 | What does the lookup action return? | next sentence in the page containing string | reader | 0.40 | 1 | abstained | Action Space We design a simple Wikipedia web API with three types of actions to suppor... | missed |
| react-03 | How many sentences does the search action return from an entity's wiki page? | first 5 sentences | reader | 0.58 | 1 | answered | 5 | ok |
| react-04 | What score does ReAct reach on Fever? | 60.9 | reader | 0.64 | 1 | answered | 60.9 vs. 56.3 | ok |
| react-05 | What share of ReAct's error cases come from non-informative search? | 23% | reader | 0.78 | 1 | answered | 23% | ok |
| react-06 | How many trajectories with correct answers are used to finetune the smaller PaLM models? | 3,000 | reader | 0.59 | 1 | answered | 3,000 | ok |
| react-07 | How many unseen evaluation games are used for ALFWorld? | 134 | reader | 0.90 | 1 | answered | 134 | ok |
| react-08 | How many real-world products does the WebShop environment contain? | 1.18M | reader | 0.91 | 1 | answered | 1.18M | ok |
| react-09 | Which imitation learning agent is the ALFWorld baseline? | BUTLER | reader | 0.89 | 1 | answered | BUTLER | ok |
| react-10 | What frequent error pattern is specific to ReAct? | repetitively generates the previous thoughts and actions | reader | 0.16 | 1 | abstained | B) While interleaving reasoning, action and observation steps improves ReAct’s grounded... | missed |
| react-11 | Why does prompting ReAct perform worst with PaLM-8B and PaLM-62B? | difficulty to learn both reasoning and acting | reader | 0.55 | 1 | answered | difficulty to learn both reasoning and acting from in-context examples | ok |
| react-12 | What is FEVER? | fact verification benchmark | sentence | 0.40 | 1 | abstained | Domains We consider two datasets challenging knowledge retrieval and reasoning: (1) Hot... | missed |
| react-13 | What is ALFWorld? | synthetic text-based game | sentence | 0.99 | 1 | answered | ALFWorld ALFWorld (Shridhar et al., 2020b) (Figure 1(2)) is a synthetic text-based game... | ok |
| react-u1 | What success rate does ReAct achieve on MiniWoB++? | (none) | reader | 0.00 |  | abstained | On ALFWorld, the best ReAct trial achieves an average success rate of 71%, significantl... | ok |
| react-u2 | What reward function is used to train ReAct with reinforcement learning? | (none) | reader | 0.44 |  | abstained | Scaling up ReAct to train and operate on more tasks and combining it with complementary... | ok |
| react-u3 | How many TPU hours did the PaLM-540B experiments use? | (none) | reader | 0.00 |  | abstained | On ALFWorld, we use all 134 unseen validation task instances, and use the best prompt s... | ok |
| react-u4 | What is the average latency per HotpotQA question? | (none) | reader | 0.13 |  | abstained | ReAct vs. CoT On the other hand, ReAct outperforms CoT on Fever (60.9 vs. 56.3) and sli... | ok |
| kivi-01 | How much larger batch size does KIVI enable? | 4× | reader | 0.66 | 1 | answered | 4× | ok |
| kivi-02 | What throughput gain does KIVI bring on real LLM inference workloads? | 3.47× | reader | 0.69 | 1 | answered | 3.47× | ok |
| kivi-03 | How much smaller is the per-token quantization error for the value cache than per-channel? | 15× | reader | 0.44 | 1 | abstained | We observe that the per-token quantization can lead to almost 5× larger attention score... | missed |
| kivi-04 | Which tasks are used in the preliminary quantization study on Llama-2-13B? | CoQA and TruthfulQA | reader | 0.66 | 1 | answered | CoQA and TruthfulQA | ok |
| kivi-05 | Which dataset are the efficiency workloads synthesized from? | ShareGPT | reader | 0.50 | 1 | abstained | To evaluate the wall-clock time efficiency of KIVI, following vLLM (Kwon et al., 2023),... | missed |
| kivi-06 | In which language is the group-wise quantization kernel implemented? | Triton | reader | 0.85 | 1 | answered | Triton | ok |
| kivi-07 | Along which dimension should the key cache be quantized? | per-channel / channel dimension | reader | 0.86 | 1 | answered | per-channel | ok |
| kivi-08 | What is fake quantization? | simulate the quantization process | sentence | 0.97 | 2 | answered | Here fake quantization means we simulate the quantization process by first quantizing K... | ok |
| kivi-09 | What is the residual key cache? | does not have a sufficient number of tokens | reader | 0.51 | 1 | abstained | The second part is the residual key cache, which does not have a sufficient number of t... | missed |
| kivi-10 | Which baseline is KIVI compared against in the efficiency comparison? | FP16 baseline | reader | 0.56 | 2 | answered | 16-bit full model | wrong |
| kivi-11 | Which prior system uses 4-bit group-wise quantization for both key and value cache? | FlexGen | reader | 0.85 | 1 | answered | FlexGen | ok |
| kivi-12 | Why does 2bit KIVI lose accuracy on Falcon-7B? | multi-query attention / only has one head | reader | 0.05 | 1 | abstained | In our efficiency evaluation, we observe that with a residual length of 32, KIVI achiev... | missed |
| kivi-13 | What happens to performance when the group size reaches 128? | significantly decreases | reader | 0.63 | 1 | answered | significantly decreases | ok |
| kivi-u1 | What perplexity does KIVI reach on WikiText-2? | (none) | reader | 0.00 |  | abstained | Table 6: Performance comparison between 16bit, KIVI-2 (2bit) / KIVI-4 (4bit) with resid... | ok |
| kivi-u2 | How many calibration samples does KIVI need? | (none) | reader | 0.24 |  | abstained | In our efficiency evaluation, we observe that with a residual length of 32, KIVI achiev... | ok |
| kivi-u3 | What accuracy does KIVI reach on MMLU? | (none) | reader | 0.00 |  | abstained | As shown in Table 3, we observe that for the Llama and Mistral model, KIVI only has up ... | ok |
| kivi-u4 | How much energy does KIVI save per generated token? | (none) | reader | 0.02 |  | abstained | Table 3: Performance comparison between 16bit, 4-bit per-token quantization, four fake ... | ok |

</details>
<!-- eval:end -->

## What I take from it

**1. Retrieval is mostly not the problem; reading and confidence are.** The
cross-encoder earns its cost:

- Recall@1 goes from 0.72 to 0.82.
- MRR goes from 0.81 to 0.90.

At the shipped threshold, 16 answerable questions don't get a right answer. In
15 of them the gold passage was among the three passages the answer step
examined. 12 of the 16 abstained. In 7 of those the best candidate was right
but scored under the bar, and the passage the app shows as the closest
evidence holds the answer. In the other 5 the candidate was wrong, so
abstaining was the right call. One example:

- *"How much smaller is the per-token quantization error for the value cache
  than per-channel?"* (KIVI). The right passage was ranked first, but the
  reader returned "5×" instead of "15×". The 5× figure is the key-cache error
  from the neighbouring sentence. It scores 0.44, so it is now withheld.

The 4 wrong answers that remain are confident:

- "8%" (0.80) for the share of almost-correct chains of thought. This is the
  one retrieval miss: the gold passage is not in the top 8.
- "PaLM 540B" (0.65) for the model size where chain-of-thought starts to help.
- A wrongly chosen sentence (0.85) for "What is last letter concatenation?".
  Point 5 explains it.
- "16-bit full model" (0.56) for KIVI's efficiency baseline. This one is
  arguably right; see the caveats.

The answer-type check rejects a span with no number when the question asks for
a quantity. It applies to 10 of the 51 questions and changed one outcome: on
the per-token question above it replaced "smaller" with "5×". It can't stop
false answers to unanswerable questions, because those are numbers too ("134"
for TPU hours, "32" for calibration samples).

**2. BM25 alone is already strong on these questions.** It reaches MRR 0.81,
the same as hybrid RRF; dense retrieval alone reaches 0.64. Part of this is how
I wrote the set: I read each paper and then wrote questions about it, so they
reuse its exact words, which favours keyword search. The papers also differ:

- **Chain-of-Thought:** dense retrieval beat BM25 (0.78 vs 0.73), and RRF kept
  the better of the two. The cross-encoder slightly *lowered* MRR (0.78 to 0.76).
- **ReAct and KIVI:** the cross-encoder gave the gains.

So "hybrid is better" holds as "never much worse than the better single
retriever". It does not hold as "better than both" on this set.

**3. The threshold is now 0.55, chosen by a stated rule and checked on
held-out papers.** The old 0.2 came from the earlier 33 questions. On this
set, before the changes in points 4 and 5, it answered 24 of 39 answerable
questions correctly but also answered 7 of the 12 unanswerable ones. Reader
scores for unanswerable questions went up to 0.70, so 0.2 did not sit in any
gap.

Cross-validation (the *Choosing the reader threshold* table):

- The three folds chose 0.45 (Chain-of-Thought held out), 0.55 and 0.55, so
  the choice barely moves with the data.
- Pooled over the three held-out papers: 23/39 correct, 4/39 wrong and 1/12
  false answers (utility 0.51). This is the number I would quote for a new
  paper.
- 0.2 was not chosen on this set, so its score here is already a held-out
  one. With the current pipeline it gets 26/39 correct, 8/39 wrong and 3/12
  false answers (utility 0.42).

So under this criterion the chosen threshold generalises better than 0.2. It
is not a free win. It gives up 3 right answers to avoid 2 false answers and 4
wrong ones, and the share of given answers that are right rises from 70% to
82% (23 of 28 held-out answers). On all 51 questions the rule picks 0.55, the
value most folds chose, and that is what ships. The in-sample figure of 0/12
false answers is optimistic; 1/12 is the honest one.

What each change did, under the same cross-validation:

| Pipeline | Folds chose | Held-out: correct | Held-out: false answers | Utility | At 0.2: correct, false answers |
| --- | --- | --- | --- | --- | --- |
| Before these changes | 0.50 / 0.55 / 0.55 | 21/39 | 2/12 | 0.37 | 24/39, 7/12 |
| + definition fix (point 5) | 0.50 / 0.55 / 0.55 | 23/39 | 2/12 | 0.42 | 26/39, 7/12 |
| + missing-name check (point 4), shipped | 0.45 / 0.55 / 0.55 | 23/39 | 1/12 | 0.51 | 26/39, 3/12 |

I measured the first row with the old code. I replayed the middle row from the
recorded scores with the name check switched off. That replay is exact,
because the check only sets scores to zero.

The differences are small in absolute terms. One unanswerable question is 8
points of false-answer rate, and the folds vary a lot: utility 0.13 with
Chain-of-Thought held out, 0.77 with ReAct. The sentence threshold (0.45)
stays as it was. All five questions that take the sentence path are
answerable definitions, so this set has nothing to choose it on.

**4. A question that names something the paper never mentions is not
answered.** Before this change, 10 of the 12 unanswerable questions contained a
word that appears nowhere in the paper. A plain "missing word" veto would not
work, though. 7 of the 39 answerable questions also contain such a word, but
those are ordinary paraphrases like "hardware", "share" and "gain". The
difference is names. A benchmark or a chip has one name, and nobody
paraphrases it.

So `verify` now looks for words written like names: a capital letter after
the first character, or letters mixed with digits (MMLU, WikiText-2,
MiniWoB++, TPU, BLEU). Each must appear somewhere in the paper's passages,
references included. The match ignores case, hyphens and plurals, so "GPT3"
matches "GPT-3". If a name is missing, every candidate gets confidence 0, and
the abstention says which name the paper never mentions.

The rule has no parameter to tune. It fires on 5 of the 12 unanswerable
questions and on none of the 39 answerable ones. So at every threshold it
leaves accuracy unchanged and false answers lower or equal. At the old
threshold it turned four false answers into abstentions, including the worst
one: "71%" at 0.70 for "What success rate does ReAct achieve on MiniWoB++?".
That figure is ReAct's ALFWorld result. Under cross-validation it cut held-out
false answers from 2/12 to 1/12.

Its limits:

- "MATH" passes, because the Chain-of-Thought paper uses the word "math".
- Questions without a name ("How many calibration samples does KIVI need?")
  still depend on the threshold.
- It would wrongly refuse a question that abbreviates something the paper
  spells out ("GB" where the paper says "gigabytes").

I also tried one softer change: letting sentence support count in full, by
removing its 0.5 floor. That made cross-validation worse (utility 0.30 instead
of 0.42), so I did not keep it.

**5. Definition questions: the sentence picker had two bugs, now fixed.**
Before, 2 of the 6 were right, and all three wrong answers scored above 0.9.

- **The definition shapes misfired.** "X is the …" matched "one challenge
  built into ALFWorld is the need to…", and "X, the …" matched "For last
  letter concatenation, the model…". In both, the term comes straight after a
  preposition, so it is that preposition's object, not the sentence's subject. The shape now requires the
  term not to follow a preposition. It also missed "ALFWorld (Shridhar et al.,
  2020b) (Figure 1(2)) is a synthetic text-based game", because it allowed only
  one bracketed aside and no nesting. Now any number of asides is allowed,
  nested one level. "X means …" and "We define X as …" also count now.
- **A definition could lose to a short mention.** The definition bonus only
  ranked sentences within one passage. Across passages the evidence score is
  capped at 1. So "Here fake quantization is exactly the same as in Table 1."
  tied with the real definition ("Here fake quantization means we simulate the
  quantization process…") and won on passage relevance. Now, if any passage
  read has a sentence shaped like a definition of the term, only those
  sentences compete. This works like the answer-type check for numbers: a
  definition question wants a definition.

At the old threshold, definition questions went from 2 of 6 to 4 of 6 (fake
quantization and ALFWorld), and no other question changed. These rules only
run for "What is X?" questions. Two are still not answered:

- "What is FEVER?" now finds the right sentence ("FEVER (Thorne et al., 2018),
  a fact verification benchmark") but scores 0.40, under the 0.45 bar for
  quoted sentences.
- "What is last letter concatenation?" has no sentence with the term as its
  subject. The paper describes the task under a heading: "This task asks the
  model to concatenate the last letters of words…". I don't see a clean rule
  for that.

At the shipped threshold, definitions are 3 of 6. The reader's answer to "What
is the residual key cache?" scores 0.51, below 0.55.

"Why" questions have the opposite problem from definitions: they are not
confident enough. Three of the six limitation questions abstained even though
the right passage was read. For example, the right sentence for "Why is
chain-of-thought reasoning costly to serve?" scored 0.15.

## Caveats

- **The set is small.** It has 51 questions across 3 papers. One answerable
  question moves accuracy by about 2.6 points. One unanswerable question moves
  the false-answer rate by about 8.
- **I wrote and labelled every question myself, after reading the papers.**
  That favours keyword retrieval (see point 2), and it means I'm grading my
  own questions.
- **I designed points 4 and 5 after reading this set's failures.**
  Cross-validation protects the choice of threshold, not my choice of which
  rules to add. Both rules are general and have nothing to tune. Still, how
  much the name check helps depends on how unanswerable questions are worded,
  and many of mine name a benchmark the paper doesn't use. Real users'
  questions may trip it less often.
- **The substring check cuts both ways.** It is lenient where the quoted
  sentence rescues a short span: "How many sentences does the search action
  return?" counts as correct with the span "5", because its sentence says "first
  5 sentences". It is strict where an answer is right in substance but uses
  other words. For KIVI's efficiency baseline the reader said "16-bit full
  model" (the same FP16 baseline, quoted from the appendix) and I counted it
  wrong. I kept my original labels rather than relabel after seeing the outputs.
- **The latency figures come from Node, not a browser.** Node runs native ONNX
  Runtime with two threads. The browser runs single-threaded WebAssembly, so
  expect every model stage to be slower there; parsing is unaffected.
