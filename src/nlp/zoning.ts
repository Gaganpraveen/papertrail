/**
 * Argumentative zoning (after Teufel's Argumentative Zoning): labels each
 * sentence with its rhetorical role — background, problem, contribution,
 * method, result, limitation, future work — or "other".
 *
 * Three independent signals are blended per zone:
 *  1. Cue phrases (0.45): a weighted regex lexicon of the stock phrases
 *     authors use for each move ("we propose", "outperforms", "a limitation
 *     of", "in future work"), plus citation density for background and
 *     extracted metrics for results. Weights add up and are squashed with
 *     1 - exp(-sum), so one strong cue gives ~0.6 and two give ~0.85.
 *  2. Prototype similarity (0.35, only with an embedding model): the cosine
 *     between the sentence and hand-written prototype sentences of each
 *     zone (max over prototypes), which catches paraphrases with no cue.
 *  3. Section prior (0.20): where the sentence sits. A sentence in
 *     "Experiments" is a priori a result, in "Related Work" background, at
 *     the end of the introduction a contribution.
 * Without embeddings the remaining weights are rescaled to sum to one.
 *
 * "other" competes as a class with a fixed baseline score, so a sentence is
 * labelled "other" exactly when no zone has convincing evidence. Scores go
 * through a softmax; confidence is the margin between the top two
 * probabilities.
 */
import type { EmbedFn, SectionKind, Sentence, Zone, ZonedSentence } from "@/types";
import { extractMetrics } from "./numbers";
import { cosine } from "./vector";

type ContentZone = Exclude<Zone, "other">;
type ZoneScores = Record<Zone, number>;

const CONTENT_ZONES: ContentZone[] = ["background", "problem", "contribution", "method", "result", "limitation", "future"];
const ALL_ZONES: Zone[] = [...CONTENT_ZONES, "other"];

// ──────────────────────────────── Prototypes ────────────────────────────────

/**
 * Prototype sentences per zone. They are deliberately topic-neutral so the
 * embedding similarity reflects rhetorical role rather than subject matter.
 */
export const ZONE_PROTOTYPES: Record<ContentZone, string[]> = {
  background: [
    "Deep neural networks have achieved remarkable success in many domains over the past decade.",
    "Prior work has extensively studied this problem from several perspectives.",
    "Many approaches have been proposed to improve the efficiency of these models.",
    "Recent advances in large-scale pretraining have transformed the field.",
    "This technique is widely used in practice and has become a standard component of modern systems.",
    "Earlier studies introduced the idea of learning representations from unlabeled data.",
    "Researchers have long been interested in understanding how such systems work.",
  ],
  problem: [
    "However, existing methods suffer from high computational cost and poor scalability.",
    "Despite this progress, the problem remains challenging and largely unsolved.",
    "This becomes a major bottleneck that limits practical deployment.",
    "Current approaches fail to handle these cases and degrade significantly.",
    "It is still unclear why these models behave this way.",
    "Collecting such data is expensive, which makes these methods hard to apply.",
    "A key challenge is that the required resources grow quickly with the input size.",
  ],
  contribution: [
    "In this paper, we propose a novel method that addresses these limitations.",
    "We introduce a simple yet effective framework for this problem.",
    "Our main contributions are summarized as follows.",
    "To the best of our knowledge, this is the first work to systematically study this question.",
    "We present a new benchmark and release our code and data publicly.",
    "This work presents an algorithm that is both efficient and easy to implement.",
    "We develop a theoretical analysis that explains this behavior.",
  ],
  method: [
    "We train the model with a learning rate of 0.001 and a batch size of 64.",
    "The output is computed as a weighted sum of the input features.",
    "Our architecture consists of three main components.",
    "Let x denote the input and y the corresponding label.",
    "The loss function is defined as the negative log-likelihood of the observed data.",
    "At each iteration, the algorithm samples a candidate and updates the parameters.",
    "We first split the data into small groups and then process each group independently.",
    "Specifically, we apply this operation to every layer of the network.",
  ],
  result: [
    "Our method outperforms all baselines by a large margin.",
    "Experimental results show that the proposed approach reduces cost by more than half.",
    "The model achieves state-of-the-art performance on all benchmarks.",
    "We observe that performance improves consistently as the model size increases.",
    "Table 2 shows that our approach matches the strongest baseline while being faster.",
    "Accuracy improves by 4.1 points compared with the previous best result.",
    "These results demonstrate that the method generalizes well to unseen settings.",
  ],
  limitation: [
    "A limitation of our approach is that it requires additional computation.",
    "Our evaluation is limited to a few datasets and may not generalize to other settings.",
    "We do not evaluate the method at larger scales due to resource constraints.",
    "The method fails in some cases and its performance depends on careful tuning.",
    "Our study only considers a small number of tasks, which may bias the conclusions.",
    "A full analysis of this issue is beyond the scope of this paper.",
  ],
  future: [
    "In future work, we plan to extend the method to other domains.",
    "We leave the exploration of larger models for future research.",
    "It would be interesting to combine our approach with other techniques.",
    "An exciting direction is to apply these ideas to new problems.",
    "Further research is needed to understand the theoretical properties of this method.",
    "We hope this work will inspire future studies in this area.",
  ],
};

// ─────────────────────────────── Cue lexicon ────────────────────────────────

interface Cue {
  re: RegExp;
  weight: number;
}

/** Weights: ~1.0 near-unambiguous stock phrase, ~0.6 typical, ~0.3 weak hint. */
const CUES: Record<Zone, Cue[]> = {
  background: [
    { re: /\b(?:has|have) been (?:widely |extensively |successfully )?(?:proposed|explored|investigated|studied|used|applied|adopted|developed)\b/i, weight: 0.8 },
    { re: /\b(?:has|have) (?:recently )?(?:become|shown|demonstrated|achieved|attracted|emerged|gained|received|revolutioni[sz]ed|transformed)\b/i, weight: 0.6 },
    { re: /\b(?:recently|in recent years|over the (?:past|last) (?:few )?(?:years|decades?))\b/i, weight: 0.5 },
    { re: /\b(?:prior|previous|existing|earlier|related|recent|pioneering|seminal) (?:work|works|studies|research|efforts|approaches|methods|literature)\b/i, weight: 0.6 },
    { re: /\b(?:widely|commonly|extensively|increasingly|routinely) (?:used|adopted|studied|applied|deployed|employed|recogni[sz]ed)\b/i, weight: 0.7 },
    { re: /\b(?:is|are) (?:a|an) (?:popular|common|standard|fundamental|well-known|widely used|central|classic|prominent)\b/i, weight: 0.6 },
    { re: /\b(?:typically|traditionally|conventionally|historically)\b/i, weight: 0.4 },
    { re: /\b(?:a (?:growing|large|substantial|rich) body of|a (?:long )?line of (?:work|research)|many (?:works|studies|researchers))\b/i, weight: 0.7 },
    { re: /\b(?:proposed|introduced|developed|studied|pioneered) by\b/i, weight: 0.5 },
    { re: /\b(?:for example|for instance),? [A-Z][\w-]+ (?:et al\.|\(\d{4}\)|\[\d+\])/u, weight: 0.4 },
  ],
  problem: [
    { re: /\b(?:remains?|remained|is still|are still|is yet|are yet) (?:an? )?(?:largely )?(?:challenging|difficult|unclear|unsolved|open|elusive|unexplored|underexplored|limited|poorly understood|an open (?:problem|question))\b/i, weight: 1.0 },
    { re: /\b(?:suffers?|suffered|suffering) from\b/i, weight: 0.9 },
    { re: /\bbottlenecks?\b/i, weight: 0.8 },
    { re: /\b(?:it is (?:unclear|not clear|difficult|hard|challenging|non-trivial)|little is known|poorly understood|not well understood)\b/i, weight: 0.9 },
    { re: /\b(?:costly|expensive|prohibitive(?:ly)?|inefficient|impractical|intractable|memory-intensive|computationally (?:expensive|intensive|demanding|costly)|quadratic(?:ally)?)\b/i, weight: 0.6 },
    { re: /\b(?:lack(?:s|ing)?|little attention|few (?:works|studies)|under-?explored|overlooked|neglected|a (?:critical |key )?gap)\b/i, weight: 0.6 },
    { re: /\b(?:challenges?|challenging|difficult(?:y|ies)?|obstacles?|hurdles?)\b/i, weight: 0.45 },
    { re: /\b(?:struggles?|struggling|fail(?:s|ed)? to|unable to|inability|cannot|can not)\b/i, weight: 0.5 },
    { re: /\b(?:limited by|hinder(?:s|ed)?|hamper(?:s|ed)?|imped(?:e|es|ed)|prevents?|restricts?)\b/i, weight: 0.6 },
    { re: /\b(?:prone to|vulnerable to|susceptible to|degrades?|deteriorates?)\b/i, weight: 0.5 },
    { re: /\b(?:require|requires|required|demand|demands) (?:a )?(?:large|massive|substantial|significant|extensive|expensive|huge|enormous)\b/i, weight: 0.6 },
    { re: /\bhowever\b|\bdespite\b|\balthough\b|\bnevertheless\b/i, weight: 0.35 },
    { re: /\b(?:problem|issue|drawback|weakness|shortcoming|trade-?off)s?\b/i, weight: 0.3 },
  ],
  contribution: [
    {
      // "we present additional results" reports results, it does not claim a contribution.
      re: /\b(?:we|this (?:paper|work|study|article)|in this (?:paper|work),? we|here,? we) (?:first )?(?:propose|introduce|present|develop|design|devise|formulate|put forward|contribute|offer)\b(?! (?:the |our |some |more |additional |further |detailed |full |extra )?(?:results|details|experiments|examples|ablations?|analys[ie]s|hyper-?parameters|statistics|samples|comparisons)\b)/i,
      weight: 1.0,
    },
    { re: /\b(?:our|the) (?:main |key |primary |major )?contributions?\b/i, weight: 1.0 },
    { re: /\bin this (?:paper|work|study|article|report)\b/i, weight: 0.6 },
    { re: /\bto (?:the best of )?our knowledge\b/i, weight: 0.8 },
    { re: /\b(?:the )?first (?:to|work|method|approach|study|attempt|systematic|framework)\b/i, weight: 0.6 },
    { re: /\b(?:novel|new) (?:method|approach|framework|algorithm|technique|architecture|model|paradigm|benchmark|dataset|objective|perspective|family)\b/i, weight: 0.6 },
    { re: /\b(?:we|our) (?:release|open-source|publicly release)\b|\bmake (?:\w+ ){0,4}(?:publicly |freely )?available\b|\b(?:code|models?|data(?:set)?|weights)(?: and \w+)? (?:is|are) (?:publicly |freely )?available\b/i, weight: 0.7 },
    { re: /\b(?:called|dubbed|named|termed) [A-Z]/u, weight: 0.6 },
    { re: /\b(?:we|this paper|this work) (?:show|demonstrate|establish|prove|reveal) that\b/i, weight: 0.4 },
    { re: /\bto (?:address|tackle|overcome|solve|mitigate|bridge|fill) (?:this|these|the|that)\b/i, weight: 0.5 },
    { re: /\b(?:we|our work) (?:take|make) (?:a|the) (?:first )?step\b/i, weight: 0.6 },
  ],
  method: [
    { re: /\bwe (?:then |first |also |further |additionally )?(?:use|employ|adopt|apply|utili[sz]e|leverage|follow|train|fine-tune|finetune|pre-?train|optimi[sz]e|compute|calculate|sample|construct|build|implement|initiali[sz]e|set|define|denote|parameteri[sz]e|encode|concatenate|feed|project|freeze|minimi[sz]e|maximi[sz]e|replace|instantiate|split|partition|quantize|group|perform|select|treat|modify|augment|generate|prompt|collect|filter|run|keep|store|cache)\b/i, weight: 0.8 },
    { re: /\b(?:is|are|was|were) (?:then )?(?:defined|computed|calculated|obtained|formulated|parameteri[sz]ed|implemented|initiali[sz]ed|trained|optimi[sz]ed|updated|sampled|estimated|represented|modell?ed|denoted|constructed|fed|passed|concatenated|projected|normali[sz]ed|quantized|stored|split|grouped|encoded) (?:as|by|using|via|with|through|from|in|into|to|per|along|for)\b/i, weight: 0.8 },
    { re: /\b(?:consists?|composed|comprises?|comprising) of\b/i, weight: 0.6 },
    { re: /\blet \S{1,12} (?:be|denote)\b|\bwhere \S{1,8} (?:is|are|denotes?|represents?|indicates?)\b/i, weight: 0.8 },
    { re: /\b(?:learning rate|batch size|hyper-?parameters?|epochs?|optimi[sz]er|warm-?up|weight decay|dropout|AdamW?|SGD|(?:is|are) set to|we set)\b/i, weight: 0.7 },
    { re: /\b(?:algorithm|procedure|pipeline|architecture|objective|loss function|formulation|update rule|module|encoder|decoder|layer|step)\b/i, weight: 0.25 },
    { re: /\b(?:first|then|next|finally|subsequently),? we\b/i, weight: 0.4 },
    { re: /\b(?:for each|for every|at each|at every|given an?|given the)\b/i, weight: 0.35 },
    { re: /\b(?:specifically|concretely|formally),/i, weight: 0.35 },
    { re: /\bto (?:this end|achieve this|do so|implement this)\b/i, weight: 0.4 },
    { re: /\b(?:we|the model) (?:is|are) (?:trained|fine-tuned|evaluated) on\b/i, weight: 0.4 },
  ],
  result: [
    { re: /\b(?:outperform(?:s|ed|ing)?|surpass(?:es|ed|ing)?|exceed(?:s|ed|ing)?|beat(?:s|ing)?)\b/i, weight: 1.0 },
    { re: /\b(?:achiev(?:es|ed|ing)?|attain(?:s|ed|ing)?|obtain(?:s|ed|ing)?|reach(?:es|ed|ing)?|deliver(?:s|ed)?) (?:an? |the )?(?:new )?(?:state-of-the-art|sota|competitive|comparable|strong|superior|better|higher|lower|significant|substantial|similar|best|near-|up to|\d)/i, weight: 1.0 },
    { re: /\bimprov(?:es|ed|ing|ement|ements)\b[^.]{0,60}?\bby\b/i, weight: 0.9 },
    { re: /\b(?:results?|experiments?|evaluations?|analys[ie]s|findings|ablations?|studies) (?:clearly |consistently )?(?:show|shows|demonstrate|demonstrates|indicate|indicates|suggest|suggests|reveal|reveals|confirm|confirms|verify|validate|highlight)\b/i, weight: 1.0 },
    { re: /\bwe (?:observe|find|found|see|notice|note|conclude) (?:that|a|an|the|significant|consistent)\b/i, weight: 0.7 },
    { re: /\b(?:significantly|substantially|consistently|considerably|dramatically|markedly|noticeably) (?:better|worse|higher|lower|faster|slower|improv\w*|outperform\w*|reduc\w*|increas\w*|boost\w*)/i, weight: 0.8 },
    { re: /\b(?:state-of-the-art|SOTA)\b/i, weight: 0.4 },
    { re: /\b(?:Table|Tab\.|Figure|Fig\.)\s*\d+[a-z]?\s+(?:shows|show|presents|reports|summari[sz]es|illustrates|demonstrates|compares|lists)\b|\bas (?:shown|reported|illustrated|seen) in (?:Table|Tab\.|Figure|Fig\.)/i, weight: 0.6 },
    { re: /\b(?:compared (?:to|with)|relative to|over the baselines?|than (?:the )?baselines?|against (?:the |strong )?baselines?)\b/i, weight: 0.5 },
    { re: /\b(?:leads? to|results? in|yields?|brings?) (?:an? )?(?:\d|significant|substantial|large|consistent|better|higher|lower|improved|notable|marginal|small)/i, weight: 0.7 },
    { re: /\b(?:comparable|on par|competitive) (?:with|to)\b|\bmatch(?:es|ing)? the performance\b/i, weight: 0.7 },
    { re: /\b(?:speed-?ups?|throughput|accuracy|perplexity|BLEU|ROUGE|F1|win rate|pass@\d)\b/i, weight: 0.25 },
    { re: /\b(?:generali[sz]es?|transfers?|scales?) (?:well|better|favorably|across|to)\b|\brobust (?:to|across)\b/i, weight: 0.4 },
  ],
  limitation: [
    { re: /\b(?:a|one|the|main|key|major|primary|another|an important|a notable) (?:limitation|weakness|drawback|shortcoming|caveat)s?\b/i, weight: 1.0 },
    { re: /\blimitations?\b/i, weight: 0.6 },
    { re: /\bwe (?:do|did|could|can|have) not\b|\bwe (?:cannot|can't|don't|didn't|haven't)\b/i, weight: 0.8 },
    { re: /\b(?:is|are|was|were|remains?) (?:currently |still )?(?:limited|restricted|confined) to\b/i, weight: 0.9 },
    { re: /\b(?:beyond|outside) (?:the )?scope\b/i, weight: 1.0 },
    { re: /\b(?:only|solely|exclusively) (?:evaluate|evaluated|test|tested|consider|considered|focus|focused|study|studied|experiment(?:ed)?)\b/i, weight: 0.7 },
    { re: /\bmay not (?:generali[sz]e|hold|transfer|scale|apply|extend|be (?:applicable|optimal|representative|sufficient))\b/i, weight: 0.9 },
    { re: /\b(?:our|this) (?:method|approach|work|study|analysis|model|framework|evaluation|system|technique)\b[^.]{0,60}\b(?:does not|do not|is not|cannot|can not|fails? to|struggles? (?:to|with)|limited to|only (?:works|applies|supports|handles)|requires|relies on|depends on|assumes)\b/i, weight: 0.45 },
    { re: /\b(?:we acknowledge|it should be noted|admittedly|a caveat|we caution|should be interpreted with caution)\b/i, weight: 0.7 },
    { re: /\bnot (?:yet )?(?:been )?(?:evaluated|tested|explored|investigated|considered|addressed|studied|verified|validated)\b/i, weight: 0.7 },
    { re: /\b(?:additional|extra|non-negligible|significant) (?:overhead|cost|computation|latency|memory)\b/i, weight: 0.4 },
    { re: /\b(?:sensitive to|depends? heavily on|relies? heavily on|assumes? (?:that|access))\b/i, weight: 0.4 },
  ],
  future: [
    { re: /\bfuture (?:work|research|directions?|studies|investigations?|versions?|efforts)\b/i, weight: 1.0 },
    { re: /\bwe (?:plan|hope|intend|aim|would like|are planning) to\b/i, weight: 0.9 },
    { re: /\bleave\b[^.]{0,60}\b(?:for|to) (?:future|further)\b|\b(?:is|are) left (?:for|to) future\b/i, weight: 1.0 },
    { re: /\bin the (?:near )?future\b/i, weight: 0.9 },
    { re: /\b(?:it (?:would|will|could) be|would be) (?:interesting|valuable|worthwhile|useful|promising|beneficial|important)\b/i, weight: 0.9 },
    { re: /\b(?:promising|exciting|interesting|important|open|natural) (?:future )?(?:directions?|avenues?|questions?|research directions?|next steps?|extensions?)\b/i, weight: 0.8 },
    { re: /\b(?:could|might|may) (?:also |potentially |further )?be (?:extended|applied|explored|combined|generali[sz]ed|improved|investigated|adapted)\b/i, weight: 0.5 },
    { re: /\b(?:further|additional|more) (?:research|investigation|work|study|exploration|studies|analysis) (?:is|are|will be) (?:needed|required|warranted|necessary)\b/i, weight: 0.8 },
    { re: /\bwe hope (?:that )?(?:this|our)\b|\bnext steps?\b|\bfollow-up work\b/i, weight: 0.6 },
    { re: /\bremains? (?:an )?open (?:question|problem)\b/i, weight: 0.4 },
  ],
  other: [
    { re: /\b(?:the )?(?:rest|remainder) of (?:this|the) (?:paper|article|work|section)\b/i, weight: 1.0 },
    { re: /\b(?:is|are) (?:organi[sz]ed|structured) as follows\b/i, weight: 1.0 },
    { re: /\bwe (?:thank|are grateful|gratefully)\b|\backnowledg/i, weight: 1.0 },
    { re: /\b(?:Section|Sec\.|Appendix)\s*[A-Z\d][\d.]* (?:describes|presents|discusses|introduces|reviews|concludes|provides|details|contains)\b/i, weight: 0.8 },
    { re: /^(?:Figure|Fig\.|Table)\s*[A-Z]?\d+[a-z]?\s*[.:|]/i, weight: 0.4 },
  ],
};

/** Own-work phrasing that turns a deficiency cue into a limitation ("our method fails to ..."). */
const SELF_CRITIQUE_RE =
  /\b(?:our|the proposed) (?:method|approach|model|framework|system|work|study|analysis|evaluation|experiments|results|technique|algorithm)\b|\bthis (?:work|paper|study)\b|\bwe (?:do|did|could|can|have) not\b|\bwe (?:cannot|can't|don't|didn't)\b/i;
const CITATION_RE = /\[\d+(?:\s*[,–-]\s*\d+)*\]|\bet al\.|\([A-Z][\p{L}'-]+(?:\s(?:and|&)\s[A-Z][\p{L}'-]+)?,?\s\d{4}[a-z]?[;)]/gu;
/** Each citation adds this much background evidence, up to the cap. */
const CITATION_WEIGHT = 0.3;
const CITATION_CAP = 0.9;
const METRIC_WEIGHT = 0.35;
const METRIC_CAP = 0.7;

function emptyScores(): ZoneScores {
  return { background: 0, problem: 0, contribution: 0, method: 0, result: 0, limitation: 0, future: 0, other: 0 };
}

function squash(evidence: number): number {
  return 1 - Math.exp(-evidence);
}

/** Cue-phrase evidence per zone, each in [0, 1). */
export function cueScores(text: string): Record<Zone, number> {
  const raw = emptyScores();
  for (const zone of ALL_ZONES) {
    for (const cue of CUES[zone]) if (cue.re.test(text)) raw[zone] += cue.weight;
  }
  const citations = text.match(CITATION_RE)?.length ?? 0;
  raw.background += Math.min(CITATION_CAP, citations * CITATION_WEIGHT);
  const metrics = extractMetrics(text).length;
  raw.result += Math.min(METRIC_CAP, metrics * METRIC_WEIGHT);
  if (raw.problem > 0 && SELF_CRITIQUE_RE.test(text)) {
    // The same deficiency words describe a limitation when the subject is the authors' own work.
    raw.limitation += raw.problem * 0.6;
    raw.problem *= 0.4;
  }
  const scores = emptyScores();
  for (const zone of ALL_ZONES) scores[zone] = squash(raw[zone]);
  return scores;
}

// ─────────────────────────────── Section priors ─────────────────────────────

type Prior = Record<ContentZone, number>;
const prior = (
  background: number,
  problem: number,
  contribution: number,
  method: number,
  result: number,
  limitation: number,
  future: number,
): Prior => ({ background, problem, contribution, method, result, limitation, future });

/**
 * How plausible each zone is in each kind of section (hand-set, 0..1).
 * Columns: background, problem, contribution, method, result, limitation, future.
 */
const SECTION_PRIORS: Record<SectionKind, Prior> = {
  abstract: prior(0.4, 0.45, 0.65, 0.35, 0.55, 0.05, 0.05),
  introduction: prior(0.55, 0.6, 0.6, 0.2, 0.25, 0.05, 0.05),
  related: prior(0.9, 0.3, 0.1, 0.15, 0.05, 0.05, 0.05),
  background: prior(0.75, 0.25, 0.1, 0.45, 0.05, 0.05, 0.05),
  method: prior(0.15, 0.1, 0.3, 0.9, 0.1, 0.05, 0.05),
  experiments: prior(0.1, 0.05, 0.1, 0.55, 0.72, 0.1, 0.05),
  results: prior(0.05, 0.05, 0.1, 0.2, 0.9, 0.15, 0.05),
  discussion: prior(0.2, 0.2, 0.2, 0.1, 0.5, 0.5, 0.4),
  limitations: prior(0.05, 0.2, 0.05, 0.05, 0.05, 0.95, 0.5),
  conclusion: prior(0.1, 0.1, 0.7, 0.1, 0.5, 0.35, 0.7),
  appendix: prior(0.1, 0.05, 0.05, 0.5, 0.4, 0.1, 0.05),
  references: prior(0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
  acknowledgments: prior(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
  other: prior(0.2, 0.15, 0.15, 0.3, 0.25, 0.1, 0.1),
};

/**
 * Position inside the section shifts the prior: introductions open with
 * background and close with contributions; abstracts close with results;
 * conclusions close with future work.
 */
function positionPrior(kind: SectionKind, position: number): Partial<Prior> {
  const early = position < 0.25;
  const late = position > 0.7;
  if (kind === "introduction") return early ? { background: 0.15, problem: 0.05 } : late ? { contribution: 0.2 } : {};
  if (kind === "abstract") return early ? { background: 0.15, problem: 0.1 } : late ? { result: 0.15 } : {};
  if (kind === "conclusion" || kind === "discussion") return late ? { future: 0.15, limitation: 0.05 } : {};
  return {};
}

function sectionPrior(kind: SectionKind, position: number): Prior {
  const base = SECTION_PRIORS[kind] ?? SECTION_PRIORS.other;
  const shift = positionPrior(kind, position);
  const out = { ...base };
  for (const zone of CONTENT_ZONES) out[zone] = Math.min(1, base[zone] + (shift[zone] ?? 0));
  return out;
}

// ──────────────────────────── Prototype similarity ──────────────────────────

/**
 * Measured on real papers with all-MiniLM-L6-v2, the best prototype cosine
 * of a sentence has a median around 0.32 and rarely exceeds 0.6; rescaling
 * 0.15–0.55 to 0..1 puts it on the same scale as the cue scores.
 */
const COSINE_FLOOR = 0.15;
const COSINE_CEIL = 0.55;

type PrototypeVectors = Record<ContentZone, Float32Array[]>;
const prototypeCache = new WeakMap<EmbedFn, Promise<PrototypeVectors>>();

/** Embeds the prototypes once per embedding function. */
function prototypeVectors(embed: EmbedFn): Promise<PrototypeVectors> {
  let cached = prototypeCache.get(embed);
  if (!cached) {
    cached = embedPrototypes(embed);
    prototypeCache.set(embed, cached);
    cached.catch(() => prototypeCache.delete(embed)); // let a later call retry
  }
  return cached;
}

async function embedPrototypes(embed: EmbedFn): Promise<PrototypeVectors> {
  const texts = CONTENT_ZONES.flatMap((zone) => ZONE_PROTOTYPES[zone]);
  const vectors = await embed(texts);
  if (vectors.length !== texts.length) throw new Error("Embedding count mismatch for zone prototypes");
  const out = {} as PrototypeVectors;
  let offset = 0;
  for (const zone of CONTENT_ZONES) {
    out[zone] = vectors.slice(offset, offset + ZONE_PROTOTYPES[zone].length);
    offset += ZONE_PROTOTYPES[zone].length;
  }
  return out;
}

function prototypeScores(vector: Float32Array, prototypes: PrototypeVectors): Prior {
  const out = {} as Prior;
  for (const zone of CONTENT_ZONES) {
    const best = Math.max(...prototypes[zone].map((p) => cosine(vector, p)));
    out[zone] = Math.min(1, Math.max(0, (best - COSINE_FLOOR) / (COSINE_CEIL - COSINE_FLOOR)));
  }
  return out;
}

/** Sentence and prototype vectors, or null when the model is unavailable or misbehaves. */
async function embedForZoning(
  sentences: Sentence[],
  embed: EmbedFn | null,
): Promise<{ sentences: Float32Array[]; prototypes: PrototypeVectors } | null> {
  if (!embed || sentences.length === 0) return null;
  try {
    const [prototypes, vectors] = await Promise.all([prototypeVectors(embed), embed(sentences.map((s) => s.text))]);
    return vectors.length === sentences.length ? { sentences: vectors, prototypes } : null;
  } catch {
    return null;
  }
}

// ────────────────────────────────── Scoring ─────────────────────────────────

/** Weights at full prototype confidence; they sum to 1. */
const WEIGHTS = { cue: 0.45, prototype: 0.35, prior: 0.2 };

/**
 * The prototype vote counts in proportion to how close the sentence is to
 * any prototype at all. Equations, tables and prompt examples sit far from
 * every prototype, so their (meaningless) similarities should not dilute the
 * cue and section evidence; the unused weight goes back to those two.
 */
function blendWeights(closeness: number): { cue: number; prototype: number; prior: number } {
  const prototype = WEIGHTS.prototype * closeness;
  const rest = (1 - prototype) / (WEIGHTS.cue + WEIGHTS.prior);
  return { cue: WEIGHTS.cue * rest, prototype, prior: WEIGHTS.prior * rest };
}

/**
 * Score "other" must beat to win. With the weights above, a sentence with no
 * cue phrase is labelled from its section only when the prior is strong
 * (>= ~0.72: method, related-work, results, limitations sections); one cue
 * phrase or a close prototype is enough anywhere.
 */
const OTHER_BASELINE = 0.22;
/** Softmax temperature: a 0.1 score gap gives roughly a 3:1 odds ratio. */
const TEMPERATURE = 0.09;

function softmax(scores: ZoneScores): ZoneScores {
  const max = Math.max(...ALL_ZONES.map((z) => scores[z]));
  const exp = emptyScores();
  let total = 0;
  for (const zone of ALL_ZONES) {
    exp[zone] = Math.exp((scores[zone] - max) / TEMPERATURE);
    total += exp[zone];
  }
  for (const zone of ALL_ZONES) exp[zone] /= total;
  return exp;
}

/** Relative position of each sentence inside its section, 0 (first) .. 1 (last). */
function sectionPositions(sentences: Sentence[]): number[] {
  const counts = new Map<string, number>();
  for (const s of sentences) counts.set(s.sectionId, (counts.get(s.sectionId) ?? 0) + 1);
  const seen = new Map<string, number>();
  return sentences.map((s) => {
    const n = seen.get(s.sectionId) ?? 0;
    seen.set(s.sectionId, n + 1);
    const total = counts.get(s.sectionId) ?? 1;
    return total > 1 ? n / (total - 1) : 0;
  });
}

function rank(probabilities: ZoneScores): [Zone, number, number] {
  const sorted = [...ALL_ZONES].sort((a, b) => probabilities[b] - probabilities[a]);
  return [sorted[0], probabilities[sorted[0]], probabilities[sorted[1]]];
}

/**
 * A sentence that announces a list ("Our contributions are as follows:",
 * "This work has two limitations:") lends its zone to the next few
 * sentences of the same section, which are usually the list items.
 */
const LIST_ZONES = new Set<Zone>(["contribution", "limitation", "future"]);
const LIST_ITEMS = 5;
/** Cue strength credited to a list item, about one medium cue phrase. */
const LIST_ITEM_CUE = 0.55;

interface ListCarry {
  zone: Zone;
  sectionId: string;
  remaining: number;
}

function listIntro(sentence: Sentence, cues: ZoneScores): ListCarry | null {
  if (!/:\s*$/u.test(sentence.text)) return null;
  const zone = CONTENT_ZONES.reduce((best, z) => (cues[z] > cues[best] ? z : best), CONTENT_ZONES[0]);
  if (!LIST_ZONES.has(zone) || cues[zone] < 0.5) return null;
  return { zone, sectionId: sentence.sectionId, remaining: LIST_ITEMS };
}

/** Labels every sentence with a zone, softmax scores and a margin-based confidence. */
export async function zoneSentences(sentences: Sentence[], embed: EmbedFn | null): Promise<ZonedSentence[]> {
  const embedded = await embedForZoning(sentences, embed);
  const positions = sectionPositions(sentences);
  const zoned: ZonedSentence[] = [];
  let carry: ListCarry | null = null;

  sentences.forEach((sentence, i) => {
    const cues = cueScores(sentence.text);
    const priors = sectionPrior(sentence.sectionKind, positions[i]);
    const prototypes = embedded ? prototypeScores(embedded.sentences[i], embedded.prototypes) : null;
    const weights = blendWeights(prototypes ? Math.max(...CONTENT_ZONES.map((z) => prototypes[z])) : 0);
    const combined = emptyScores();
    for (const zone of CONTENT_ZONES) {
      combined[zone] =
        weights.cue * cues[zone] + weights.prior * priors[zone] + weights.prototype * (prototypes?.[zone] ?? 0);
    }
    combined.other = OTHER_BASELINE + weights.cue * cues.other;
    if (carry && carry.sectionId === sentence.sectionId && carry.remaining > 0) {
      combined[carry.zone] += weights.cue * LIST_ITEM_CUE;
      carry.remaining--;
    }
    carry = listIntro(sentence, cues) ?? (carry?.sectionId === sentence.sectionId ? carry : null);

    const probabilities = softmax(combined);
    const [zone, top, second] = rank(probabilities);
    zoned.push({ ...sentence, zone, confidence: top - second, scores: probabilities });
  });
  return zoned;
}
