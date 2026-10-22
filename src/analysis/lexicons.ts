/**
 * Curated lexicons behind the reviewer rules.
 *
 * Everything PaperTrail "notices" in a paper starts here: named benchmarks,
 * metric names, hedging and boosting vocabulary, and the stock phrasings
 * authors use when they report variance, seeds, compute, hyperparameters,
 * ablations, baselines, human studies or limitations. Keeping the patterns in
 * one file makes every downstream finding auditable: a finding can always be
 * traced to the exact pattern that fired and the sentence it fired on.
 *
 * Matching conventions:
 *  - Benchmark names are matched case-sensitively ("DROP", "RACE", "MATH"
 *    would otherwise collide with ordinary words); spaces and hyphens inside
 *    a name are interchangeable ("CIFAR-10" = "CIFAR10" = "CIFAR 10").
 *  - Accelerator names (GPU, TPU) are case-sensitive too: "TPU" hides inside
 *    "ouTPUt" when matched case-insensitively.
 *  - Patterns that are counted (hedges, boosters) carry the global flag;
 *    patterns that only test a sentence do not, so `test` stays stateless.
 */

// ───────────────────────────────── Benchmarks ───────────────────────────────

export type DatasetDomain =
  | "language"
  | "reasoning"
  | "code"
  | "qa"
  | "dialogue"
  | "safety"
  | "long-context"
  | "vision"
  | "multimodal"
  | "rl"
  | "speech"
  | "graph"
  | "retrieval"
  | "science"
  | "tabular"
  | "pretraining";

export interface DatasetEntry {
  /** Canonical display name. */
  name: string;
  /** Other spellings; the canonical name is always tried too. */
  aliases?: string[];
  domain: DatasetDomain;
  /** Training corpora are data, but not evaluation benchmarks. */
  corpus?: boolean;
  /** Only for multi-word names that cannot collide with ordinary prose. */
  caseInsensitive?: boolean;
}

const d = (name: string, domain: DatasetDomain, aliases: string[] = [], extra: Partial<DatasetEntry> = {}): DatasetEntry => ({
  name,
  domain,
  aliases,
  ...extra,
});

/** ~290 widely used datasets and benchmarks across ML sub-fields. */
export const DATASETS: DatasetEntry[] = [
  // General language understanding
  d("GLUE", "language"),
  d("SuperGLUE", "language"),
  d("SST-2", "language", ["SST2", "Stanford Sentiment Treebank"]),
  d("MNLI", "language", ["MultiNLI"]),
  d("QNLI", "language"),
  d("QQP", "language"),
  d("RTE", "language"),
  d("CoLA", "language"),
  d("MRPC", "language"),
  d("STS-B", "language"),
  d("WNLI", "language"),
  d("SNLI", "language"),
  d("ANLI", "language", ["Adversarial NLI"]),
  d("XNLI", "language"),
  d("PAWS-X", "language"),
  d("BoolQ", "qa"),
  d("COPA", "reasoning"),
  d("XCOPA", "reasoning"),
  d("WiC", "language"),
  d("WSC", "reasoning", ["Winograd Schema Challenge", "WSC273"]),
  d("MultiRC", "qa"),
  d("ReCoRD", "qa"),
  d("IMDb", "language", ["IMDB"]),
  d("AG News", "language", ["AGNews"]),
  d("Yelp", "language", ["Yelp Reviews"]),
  d("CoNLL-2003", "language", ["CoNLL03", "CoNLL 2003"]),
  d("OntoNotes", "language"),
  d("Penn Treebank", "language", ["PTB"]),
  d("BLiMP", "language"),
  d("LAMBADA", "language"),
  d("WikiText-103", "language", ["WikiText103"]),
  d("WikiText-2", "language", ["WikiText2"]),
  d("WikiText", "language"),
  d("enwik8", "language"),
  d("text8", "language"),
  d("One Billion Word", "language", ["LM1B"]),
  // Commonsense and reasoning
  d("HellaSwag", "reasoning"),
  d("WinoGrande", "reasoning"),
  d("PIQA", "reasoning"),
  d("SIQA", "reasoning", ["Social IQa", "SocialIQA"]),
  d("ARC", "reasoning", ["ARC-Challenge", "ARC-Easy", "ARC-c", "ARC-e", "AI2 Reasoning Challenge"]),
  d("OpenBookQA", "reasoning", ["OBQA"]),
  d("CommonsenseQA", "reasoning", ["CSQA"]),
  d("StrategyQA", "reasoning"),
  d("SWAG", "reasoning"),
  d("Story Cloze", "reasoning", ["StoryCloze", "ROCStories"]),
  d("BIG-Bench Hard", "reasoning", ["BBH", "BIG-bench Hard"]),
  d("BIG-bench", "reasoning", ["BIG-Bench", "BigBench"]),
  d("MMLU", "reasoning"),
  d("MMLU-Pro", "reasoning"),
  d("AGIEval", "reasoning"),
  d("GPQA", "reasoning"),
  d("LogiQA", "reasoning"),
  d("ReClor", "reasoning"),
  d("HANS", "reasoning"),
  d("Date Understanding", "reasoning", [], { caseInsensitive: true }),
  d("Sports Understanding", "reasoning", [], { caseInsensitive: true }),
  d("Last Letter Concatenation", "reasoning", [], { caseInsensitive: true }),
  d("Coin Flip", "reasoning", [], { caseInsensitive: true }),
  d("SayCan", "reasoning"),
  d("Belebele", "language"),
  // Math
  d("GSM8K", "reasoning", ["GSM-8K"]),
  d("GSM-Hard", "reasoning"),
  d("MGSM", "reasoning"),
  d("MATH", "reasoning", ["MATH-500", "MATH500"]),
  d("SVAMP", "reasoning"),
  d("ASDiv", "reasoning"),
  d("AQuA", "reasoning", ["AQuA-RAT"]),
  d("MAWPS", "reasoning"),
  d("MathQA", "reasoning"),
  d("TheoremQA", "reasoning"),
  d("AIME", "reasoning"),
  d("MiniF2F", "reasoning", ["miniF2F"]),
  // Code
  d("HumanEval", "code", ["HumanEval+"]),
  d("MBPP", "code", ["MBPP+"]),
  d("APPS", "code"),
  d("CodeContests", "code"),
  d("DS-1000", "code"),
  d("MultiPL-E", "code"),
  d("SWE-bench", "code", ["SWE-Bench"]),
  d("LiveCodeBench", "code"),
  d("CodeXGLUE", "code"),
  d("Spider", "code"),
  d("WikiSQL", "code"),
  d("BIRD-SQL", "code"),
  // Question answering and reading comprehension
  d("SQuAD", "qa", ["SQuAD 2.0", "SQuAD v2", "SQuAD 1.1", "SQuAD2.0"]),
  d("Natural Questions", "qa", ["NaturalQuestions", "NQ-Open"], { caseInsensitive: true }),
  d("TriviaQA", "qa"),
  d("WebQuestions", "qa", ["WebQSP"]),
  d("HotpotQA", "qa"),
  d("2WikiMultihopQA", "qa", ["2WikiMQA"]),
  d("MuSiQue", "qa"),
  d("Bamboogle", "qa"),
  d("PopQA", "qa"),
  d("ASQA", "qa"),
  d("ELI5", "qa"),
  d("RACE", "qa"),
  d("DROP", "qa"),
  d("QuAC", "qa"),
  d("CoQA", "qa"),
  d("NarrativeQA", "qa"),
  d("QuALITY", "qa"),
  d("QASPER", "qa", ["Qasper"]),
  d("TabFact", "tabular"),
  d("WikiTableQuestions", "tabular"),
  d("FEVER", "qa"),
  d("PubHealth", "qa"),
  d("TruthfulQA", "safety"),
  d("SciQ", "science"),
  d("ScienceQA", "science"),
  d("MedQA", "science"),
  d("MedMCQA", "science"),
  d("PubMedQA", "science"),
  d("LegalBench", "language"),
  d("XQuAD", "qa"),
  d("MLQA", "qa"),
  d("TyDi QA", "qa", ["TyDiQA"]),
  // Summarisation, translation and dialogue
  d("CNN/DailyMail", "language", ["CNN/DM", "CNN-DailyMail", "CNN/Daily Mail"]),
  d("XSum", "language"),
  d("SAMSum", "dialogue"),
  d("Reddit TL;DR", "language", ["TL;DR", "TL;DR summarization"]),
  d("WMT", "language", ["WMT14", "WMT16", "WMT19", "WMT20", "WMT22", "WMT'14"]),
  d("IWSLT", "language", ["IWSLT14", "IWSLT'14"]),
  d("FLORES", "language", ["FLORES-200", "FLORES-101"]),
  d("Anthropic HH", "dialogue", ["Anthropic Helpful and Harmless", "HH-RLHF", "Anthropic-HH"]),
  d("AlpacaEval", "dialogue", ["AlpacaEval 2.0", "AlpacaEval2"]),
  d("MT-Bench", "dialogue", ["MT-bench", "MTBench"]),
  d("Arena-Hard", "dialogue", ["Arena Hard"]),
  d("Chatbot Arena", "dialogue", ["LMSYS Chatbot Arena"]),
  d("FLASK", "dialogue"),
  d("IFEval", "dialogue"),
  d("ShareGPT", "dialogue"),
  d("UltraChat", "dialogue"),
  d("UltraFeedback", "dialogue"),
  d("OpenAssistant", "dialogue", ["OASST1", "OASST"]),
  d("Alpaca", "dialogue", [], { corpus: true }),
  d("Dolly", "dialogue", ["databricks-dolly-15k"], { corpus: true }),
  d("FLAN", "dialogue", ["Flan Collection"], { corpus: true }),
  d("Super-NaturalInstructions", "dialogue", ["Natural Instructions", "SuperNI"], { corpus: true }),
  d("Self-Instruct", "dialogue", [], { corpus: true }),
  // Long context
  d("LongBench", "long-context"),
  d("Needle-in-a-Haystack", "long-context", ["Needle in a Haystack", "NIAH"]),
  d("RULER", "long-context"),
  d("InfiniteBench", "long-context"),
  d("SCROLLS", "long-context", ["ZeroSCROLLS"]),
  d("L-Eval", "long-context"),
  d("Long Range Arena", "long-context", ["LRA"]),
  d("PG-19", "long-context", ["PG19"]),
  d("StreamEval", "long-context"),
  d("Passkey Retrieval", "long-context", ["passkey retrieval"]),
  d("BookSum", "long-context"),
  // Safety, bias and toxicity
  d("RealToxicityPrompts", "safety"),
  d("ToxiGen", "safety"),
  d("CrowS-Pairs", "safety"),
  d("WinoGender", "safety", ["Winogender"]),
  d("WinoBias", "safety"),
  d("BBQ", "safety"),
  d("BOLD", "safety"),
  d("StereoSet", "safety"),
  d("HarmBench", "safety"),
  d("AdvBench", "safety"),
  d("Do-Not-Answer", "safety"),
  d("XSTest", "safety"),
  // Retrieval
  d("MS MARCO", "retrieval", ["MSMARCO"]),
  d("BEIR", "retrieval"),
  d("TREC", "retrieval", ["TREC-COVID", "TREC DL"]),
  d("MTEB", "retrieval"),
  d("LoTTE", "retrieval"),
  d("KILT", "retrieval"),
  // Pre-training corpora
  d("The Pile", "pretraining", ["the Pile"], { corpus: true }),
  d("C4", "pretraining", [], { corpus: true }),
  d("Common Crawl", "pretraining", ["CommonCrawl"], { corpus: true }),
  d("RedPajama", "pretraining", [], { corpus: true }),
  d("SlimPajama", "pretraining", [], { corpus: true }),
  d("RefinedWeb", "pretraining", [], { corpus: true }),
  d("FineWeb", "pretraining", [], { corpus: true }),
  d("Dolma", "pretraining", [], { corpus: true }),
  d("OpenWebText", "pretraining", [], { corpus: true }),
  d("BookCorpus", "pretraining", ["BooksCorpus"], { corpus: true }),
  d("ROOTS", "pretraining", [], { corpus: true }),
  d("The Stack", "pretraining", ["the Stack"], { corpus: true }),
  d("TinyStories", "pretraining", [], { corpus: true }),
  // Vision
  d("ImageNet", "vision", ["ImageNet-1K", "ImageNet-1k", "ILSVRC", "ILSVRC-2012"]),
  d("ImageNet-21K", "vision", ["ImageNet-21k", "ImageNet-22K", "ImageNet-22k"]),
  d("CIFAR-10", "vision"),
  d("CIFAR-100", "vision"),
  d("MNIST", "vision"),
  d("Fashion-MNIST", "vision"),
  d("SVHN", "vision"),
  d("STL-10", "vision"),
  d("Tiny ImageNet", "vision", ["TinyImageNet"]),
  d("COCO", "vision", ["MS COCO", "MS-COCO", "MSCOCO"]),
  d("Pascal VOC", "vision", ["PASCAL VOC", "VOC2007", "VOC2012"]),
  d("ADE20K", "vision"),
  d("Cityscapes", "vision"),
  d("LVIS", "vision"),
  d("Open Images", "vision", ["OpenImages"]),
  d("Places365", "vision"),
  d("CelebA", "vision", ["CelebA-HQ"]),
  d("FFHQ", "vision"),
  d("LSUN", "vision"),
  d("Kinetics", "vision", ["Kinetics-400", "Kinetics-700"]),
  d("Something-Something", "vision", ["Something-Something v2", "SSv2"]),
  d("UCF101", "vision", ["UCF-101"]),
  d("HMDB51", "vision", ["HMDB-51"]),
  d("KITTI", "vision"),
  d("nuScenes", "vision"),
  d("Waymo Open Dataset", "vision", ["Waymo Open"]),
  d("ShapeNet", "vision"),
  d("ModelNet40", "vision", ["ModelNet"]),
  d("ScanNet", "vision"),
  d("LAION", "vision", ["LAION-5B", "LAION-400M", "LAION-2B"], { corpus: true }),
  d("CC3M", "vision", ["Conceptual Captions"], { corpus: true }),
  d("CC12M", "vision", [], { corpus: true }),
  d("JFT-300M", "vision", ["JFT"], { corpus: true }),
  // Multimodal
  d("VQAv2", "multimodal", ["VQA v2", "VQA-v2"]),
  d("GQA", "multimodal"),
  d("TextVQA", "multimodal"),
  d("DocVQA", "multimodal"),
  d("ChartQA", "multimodal"),
  d("OK-VQA", "multimodal", ["OKVQA"]),
  d("VizWiz", "multimodal"),
  d("MMMU", "multimodal"),
  d("MathVista", "multimodal"),
  d("MMBench", "multimodal"),
  d("SEED-Bench", "multimodal"),
  d("POPE", "multimodal"),
  d("MME", "multimodal"),
  d("LLaVA-Bench", "multimodal"),
  d("AI2D", "multimodal"),
  d("BLINK", "multimodal"),
  d("Video-MME", "multimodal", ["VideoMME"]),
  d("Flickr30k", "multimodal", ["Flickr30K"]),
  d("NoCaps", "multimodal", ["nocaps"]),
  d("RefCOCO", "multimodal", ["RefCOCO+", "RefCOCOg"]),
  d("Visual Genome", "multimodal"),
  d("CLEVR", "multimodal"),
  // Reinforcement learning and agents
  d("Atari", "rl", ["Atari 2600", "Arcade Learning Environment"]),
  d("MuJoCo", "rl"),
  d("D4RL", "rl"),
  d("DeepMind Control Suite", "rl", ["DMControl", "DM Control"]),
  d("Procgen", "rl"),
  d("MineRL", "rl"),
  d("NetHack", "rl"),
  d("Meta-World", "rl", ["MetaWorld"]),
  d("OpenAI Gym", "rl"),
  d("ALFWorld", "rl"),
  d("WebShop", "rl"),
  d("WebArena", "rl"),
  d("AgentBench", "rl"),
  d("MiniGrid", "rl"),
  d("BabyAI", "rl"),
  d("StarCraft II", "rl", ["SMAC"]),
  d("Habitat", "rl"),
  // Speech and audio
  d("LibriSpeech", "speech"),
  d("Common Voice", "speech", ["CommonVoice"]),
  d("VCTK", "speech"),
  d("LJSpeech", "speech"),
  d("AudioSet", "speech"),
  d("ESC-50", "speech"),
  d("SC09", "speech", ["Speech Commands"]),
  d("VoxCeleb", "speech", ["VoxCeleb1", "VoxCeleb2"]),
  d("TIMIT", "speech"),
  d("Switchboard", "speech"),
  d("YouTubeMix", "speech"),
  d("MusicCaps", "speech"),
  // Graphs, molecules and biology
  d("Cora", "graph"),
  d("Citeseer", "graph", ["CiteSeer"]),
  d("ogbn-arxiv", "graph"),
  d("ogbn-products", "graph"),
  d("OGB", "graph", ["Open Graph Benchmark"]),
  d("QM9", "graph"),
  d("ZINC", "graph"),
  d("MoleculeNet", "graph"),
  d("TUDataset", "graph"),
  d("PDB", "science", ["Protein Data Bank"]),
  d("CASP", "science", ["CASP14", "CASP15"]),
  d("HG38", "science", ["hg38"]),
  d("Genomic Benchmarks", "science"),
  d("MIMIC-III", "science", ["MIMIC-IV", "MIMIC-CXR"]),
  d("CheXpert", "science"),
  d("UK Biobank", "science"),
  // Tabular, recommendation and time series
  d("MovieLens", "tabular", ["MovieLens-1M", "ML-1M"]),
  d("Criteo", "tabular"),
  d("Netflix Prize", "tabular"),
  d("UCI", "tabular", ["UCI Machine Learning Repository"]),
  d("OpenML", "tabular"),
  d("ETTh1", "tabular", ["ETTh2", "ETTm1", "ETTm2", "ETT"]),
  d("M4", "tabular"),
];

/** Lowercased letters and digits only: "CIFAR-10" and "cifar10" compare equal. */
export function datasetKey(name: string): string {
  return name.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
}

/** One alias as a pattern: spaces and hyphens are interchangeable or optional. */
function aliasPattern(alias: string): string {
  return alias
    .split(/[\s-]+/)
    .map(escapeRegExp)
    .join("[\\s-]?");
}

export interface DatasetMatcher {
  entry: DatasetEntry;
  re: RegExp;
}

/**
 * Compiled dataset patterns. The trailing guard rejects a following letter,
 * digit or "-digit" so "CIFAR-10" does not fire inside "CIFAR-100" and
 * "MMLU" does not fire inside "MMLU-Pro"; longer aliases are tried first.
 */
export const DATASET_MATCHERS: DatasetMatcher[] = DATASETS.map((entry) => {
  const spellings = [entry.name, ...(entry.aliases ?? [])].sort((a, b) => b.length - a.length);
  const body = spellings.map(aliasPattern).join("|");
  const flags = entry.caseInsensitive ? "iu" : "u";
  return { entry, re: new RegExp(`(?<![\\p{L}\\p{N}])(?:${body})(?![\\p{L}\\p{N}]|-[\\p{L}\\p{N}])`, flags) };
});

/**
 * Datasets outside the curated list are recognised from how they are
 * introduced: a distinctive name right before "dataset", "benchmark" or
 * "corpus" ("the StreamEval benchmark"). The name must look like a proper
 * identifier (an inner capital or digit), which excludes "the dataset".
 */
export const NAMED_DATASET_RE =
  /\b([A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\s+(?:dataset|benchmark|corpus|suite|test set|leaderboard)s?\b/gu;

/** Identifier-like words that are never datasets: hardware, model families, field acronyms. */
export const NOT_A_DATASET = new Set(
  `LLM LLMs NLP AI ML RL GPU GPUs TPU TPUs CPU API APIs SOTA OOD IID QA VQA KV MoE SFT RLHF DPO PPO NLU NLG ASR
  TTS OCR CV GAN GANs VAE RNN CNN LSTM MLP BERT GPT GPT-2 GPT-3 GPT-4 GPT-3.5 LLaMA Llama Llama-2 PaLM T5 OPT
  BLOOM Mistral Mixtral Falcon Vicuna Gemma Qwen Phi CLIP ViT HuggingFace PyTorch JAX FP16 BF16 INT8 INT4 FLOPs
  ICLR NeurIPS ICML ACL EMNLP NAACL CVPR ICCV ECCV AAAI IJCAI COLING EACL TMLR JMLR`
    .split(/\s+/)
    .filter(Boolean),
);

// ─────────────────────────────────── Metrics ────────────────────────────────

export interface MetricEntry {
  name: string;
  re: RegExp;
  /** Text-quality metrics imply a generation task (relevant to human evaluation). */
  generation?: boolean;
}

export const METRICS: MetricEntry[] = [
  { name: "accuracy", re: /\baccurac(?:y|ies)\b|\btop-?[15] acc/i },
  { name: "exact match", re: /\bexact[- ]match\b|\bEM\b/ },
  { name: "F1", re: /\bF1\b|\bF-?scores?\b|\bF-measure\b|\bmacro-F1\b/ },
  { name: "precision", re: /\bprecision\b/i },
  { name: "recall", re: /\brecall\b(?!@)/i },
  { name: "recall@k", re: /\b(?:recall|R|hits?)@\d+\b/i },
  { name: "BLEU", re: /\b(?:Sacre)?BLEU\b/, generation: true },
  { name: "ROUGE", re: /\bROUGE(?:-(?:1|2|L|Lsum))?\b/, generation: true },
  { name: "METEOR", re: /\bMETEOR\b/, generation: true },
  { name: "chrF", re: /\bchrF\+*/, generation: true },
  { name: "CIDEr", re: /\bCIDEr\b/, generation: true },
  { name: "SPICE", re: /\bSPICE\b/, generation: true },
  { name: "BERTScore", re: /\bBERTScore\b/, generation: true },
  { name: "MAUVE", re: /\bMAUVE\b/, generation: true },
  { name: "win rate", re: /\bwin[- ]rates?\b/i, generation: true },
  { name: "Elo", re: /\bElo\b/ },
  { name: "perplexity", re: /\bperplexit(?:y|ies)\b|\bPPL\b/i },
  { name: "bits per byte", re: /\bbits?[- ]per[- ](?:byte|character|char)\b|\bBPB\b|\bBPC\b/i },
  { name: "pass@k", re: /\bpass@\d+\b/i },
  { name: "AUC", re: /\bAU(?:ROC|PRC|C)\b/ },
  { name: "mAP", re: /\bmAP\b/ },
  { name: "IoU", re: /\bm?IoU\b/ },
  { name: "FID", re: /\bFID\b/ },
  { name: "Inception Score", re: /\b[Ii]nception [Ss]core\b/ },
  { name: "WER", re: /\bWER\b|\bword error rate\b/i },
  { name: "CER", re: /\bCER\b|\bcharacter error rate\b/i },
  { name: "MRR", re: /\bMRR\b/ },
  { name: "NDCG", re: /\bn?DCG(?:@\d+)?\b/ },
  { name: "MSE", re: /\b(?:R?MSE|MAE)\b|\bmean (?:squared|absolute) error\b/i },
  { name: "success rate", re: /\bsuccess rates?\b/i },
  { name: "reward", re: /\b(?:average|mean|episodic|episode|cumulative|normali[sz]ed) (?:returns?|rewards?)\b/i },
  { name: "correlation", re: /\b(?:Spearman|Pearson|Kendall)(?:'s)?\b/ },
  { name: "latency", re: /\blatenc(?:y|ies)\b/i },
  { name: "throughput", re: /\bthroughput\b/i },
  { name: "memory", re: /\b(?:peak |GPU )?memory (?:usage|footprint|consumption|cost|savings?|reduction)\b/i },
  { name: "speed-up", re: /\bspeed-?ups?\b/i },
  { name: "FLOPs", re: /\bFLOPs\b/ },
  { name: "toxicity", re: /\btoxicity\b/i },
  { name: "calibration (ECE)", re: /\bECE\b|\bexpected calibration error\b/i },
  { name: "FActScore", re: /\bFActScore\b/i, generation: true },
];

// ──────────────────────────────── Calibration ───────────────────────────────

/** Hedges: the author signals uncertainty (counted, so global flag). */
export const HEDGES =
  /\b(?:may|might|could|possibly|perhaps|likely|unlikely|probably|presumably|arguably|potentially|plausibly|appears? to|seems? to|tends? to|suggests?|suggesting|we (?:believe|hypothesi[sz]e|speculate|conjecture|suspect)|to (?:some|a certain) extent|in part|partially|somewhat|in most cases|it is possible)\b/gi;

/**
 * Boosters: intensifiers that overstate certainty or magnitude. Counted only
 * in claim sentences (contributions, results, abstract, conclusion), where
 * they inflate what the evidence shows.
 */
export const BOOSTERS =
  /\b(?:clearly|obviously|undoubtedly|unquestionably|definitely|indisputably|unprecedented|revolutionary|ground-?breaking|game-changing|breakthrough|paradigm shift|dramatically|drastically|vastly|massively|tremendous(?:ly)?|enormous(?:ly)?|immensely|remarkabl[ey]|impressive(?:ly)?|striking(?:ly)?|astonishing(?:ly)?|extraordinar(?:y|ily)|perfectly|by a (?:large|wide|huge|substantial) margin|far (?:superior|better)|first-ever|for the first time)\b/gi;

/**
 * "Significantly better" is a booster unless the sentence reports a
 * statistical test; the guard is applied by the caller with STATISTICS_RE.
 */
export const SIGNIFICANTLY_RE =
  /\b(?:significantly|substantially|considerably|greatly) (?:better|worse|outperform\w*|improv\w*|superior|surpass\w*|exceed\w*|higher|lower|faster|reduc\w*|boost\w*)/gi;

// ───────────────────────────── Evidence of rigour ───────────────────────────

/** Spread or significance: ±, standard deviation/error, confidence intervals, p-values, named tests. */
export const STATISTICS_RE =
  /±|\+\/-|\bstandard (?:deviations?|errors?)\b|\bstd\.?(?=[\s,)]|$)|\bs\.d\.|\bs\.e\.m\.?|\bconfidence intervals?\b|\b(?:9[059])\s?% (?:CI|confidence)\b|\bCIs?\b(?= of| \(|\s*[=:])|\bp\s*[<≤=]\s*0?\.\d+|\bp-values?\b|\bstatistical(?:ly)? significan\w*|\bsignificance (?:tests?|testing|levels?)\b|\bt-tests?\b|\bWilcoxon\b|\bpermutation tests?\b|\bbootstrap(?:ped|ping)? (?:confidence|resampl\w*|samples|estimates?|intervals?)\b|\bMcNemar\b|\bMann-Whitney\b|\bANOVA\b|\bBonferroni\b|\berror bars?\b|\bvariance (?:across|over|between) (?:runs|seeds|trials)\b|\binterquartile\b|\bshaded (?:region|area)s? (?:denote|show|indicate|represent)/i;

const NUMBER_WORD = "(?:\\d+|two|three|four|five|six|seven|eight|nine|ten|twenty|multiple|several|different|random|independent)";
/** Repetition: "5 random seeds", "averaged over 3 runs", "repeated ten times". */
export const SEEDS_RE = new RegExp(
  `\\b${NUMBER_WORD} (?:random |different |independent )?(?:seeds?|runs|trials|repetitions|restarts|initiali[sz]ations)\\b` +
    `|\\baverage[sd]? (?:over|across) ${NUMBER_WORD}\\b` +
    `|\\b(?:repeated|ran|run|conducted) (?:\\w+ ){0,3}(?:\\d+|two|three|four|five|ten) times\\b` +
    "|\\brandom seeds?\\b",
  "i",
);

/**
 * Hardware, accelerator time and energy. Case-sensitive on purpose (see header). "Trained for"
 * counts only with a duration: "trained for 3 epochs" is a hyperparameter, not a compute budget.
 */
export const COMPUTE_RE =
  /\b(?:A100|H100|H800|A800|V100|P100|K80|T4|A6000|A40|A10G?|L40S?|L4|RTX ?\d{4}|TITAN ?[A-Z]\w*|TPU ?v\d\w*|MI250X?|MI300X?|Gaudi\d?)\b|\b(?:\d+|one|two|four|eight|sixteen|single|a single) (?:NVIDIA |Nvidia )?(?:GPUs?|TPUs?|TPU cores|GPU nodes|nodes)\b|\b(?:GPU|TPU)[- ](?:hours?|days?|years?)\b|\b(?:petaflops?|PF-days|zettaFLOPs|FLOPs budget|compute budget)\b|\b(?:[Tt]raining|[Pp]re-?training) (?:took|takes|lasted|time was|required)\b|\b[Tt]rained for (?:(?:about|approximately|roughly|around|~) ?)?(?:\d[\d,.]*|an?|one|two|three|four|five|six|seven|several) ?(?:hours?|days?|weeks?|months?)\b|\b[Ww]all-?clock\b|\b[Cc]arbon (?:footprint|emissions?)\b|\btCO2(?:eq)?\b|\b[kM]Wh\b/u;

/** Optimisation and decoding settings that a replication needs. */
export const HYPERPARAMETER_RE =
  /\blearning rates?\b|\bbatch sizes?\b|\bwarm-?up (?:steps|ratio|period)\b|\bweight decay\b|\bdropout\b|\b(?:Adam|AdamW|SGD|Adafactor|Lion|RMSprop|LAMB)\b|\b\d+ epochs?\b|\bhyper-?parameters?\b|\bcosine (?:schedule|decay|learning)\b|\bgradient clipping\b|\b(?:β|beta)\s*(?:=|of|is set to)\s*\d|\b(?:sampling )?temperature (?:of|=|is set to|was set to) \d|\btop-?p (?:of|=)\s*\d|\bgreedy decoding\b|\bnucleus sampling\b|\b(?:group|window|chunk|block|residual) (?:size|length) (?:of|=|is set to|was set to) \d|\b(?:is|are|was|were) set to \d|\bwe set (?:the )?\w+(?: \w+)? to \d/i;

/** Isolating components: ablations, "w/o X", removal, sensitivity sweeps. */
export const ABLATION_RE =
  /\bablat(?:e|es|ed|ing|ion|ions)\b|\bw\/o\b|\bwithout (?:the |our |any |a )?[\w-]+(?: [\w-]+)? (?:component|module|loss|term|step|stage|layer|mechanism|block|objective|regulari[sz]ation)s?\b|\bremov(?:e|es|ed|ing|al of) (?:the |our |each |this )?[\w-]+(?: [\w-]+)? (?:component|module|loss|term|step|stage|mechanism|block|objective)s?\b|\bcontribution of (?:each|individual|the different) (?:component|module|part)s?\b|\bsensitivity (?:analysis|study|to)\b|\bwe (?:vary|varied|varying|sweep|swept)\b|\b(?:effect|impact|influence) of (?:the |varying |different |using )?(?:\w+ ){0,2}(?:size|number|choice|ratio|length|rate|depth|width|components?)\b/i;

/** Comparison to other systems. */
export const BASELINE_RE =
  /\bbaselines?\b|\bcompar(?:e|ed|es|ing|ison) (?:\w+ ){0,3}(?:with|to|against)\b|\boutperform\w*\b|\bsurpass\w*\b|\b(?:vs\.?|versus)\s|\bover (?:the |all )?(?:prior|previous|existing|competing|strong|state-of-the-art) (?:methods|approaches|baselines|work|models)\b|\brelative to\b|\bcompeting (?:methods|approaches|systems)\b|\b(?:matches|matching|on par with|competitive with)\b/i;

/** State-of-the-art claims (only counted when the authors claim them for their own work). */
export const SOTA_RE =
  /\bstate[- ]of[- ]the[- ]art\b|\bSOTA\b|\bnew (?:state of the art|best|record)\b|\bbest[- ](?:performing|reported|known|published) (?:results?|performance|method|model|score)\b|\boutperforms? (?:all|every|existing|previous|prior)\b|\b(?:surpass|surpasses|surpassing|beat|beats|beating) (?:all|every|existing|previous|prior)\b/i;

export const OWN_WORK_RE = /\b(?:we|our|ours|this (?:paper|work|method|approach))\b/i;

/**
 * Human judgement in the evaluation. Bare "human preferences" and "human labelers" usually describe
 * training data or an alignment goal (RLHF), so preference wording needs an evaluation noun.
 */
export const HUMAN_EVAL_RE =
  /\bhuman (?:evaluations?|evaluators?|judges?|judgments?|judgements?|raters?|annotators?|stud(?:y|ies)|assessments?|ratings?|participants|subjects)\b|\bhuman preference (?:evaluations?|stud(?:y|ies)|judg\w+|ratings?)\b|\buser stud(?:y|ies)\b|\bcrowd-?(?:workers?|sourc\w+)\b|\bMechanical Turk\b|\bMTurk\b|\bProlific\b|\bLikert\b|\bexpert annotat\w+|\binter-?annotator\b|\binter-?rater\b|\bCohen'?s kappa\b|\bKrippendorff/i;

/** Text-generation tasks where automatic metrics are known to be weak proxies for quality. */
export const GENERATION_TASK_RE =
  /\b(?:summari[sz]ation|abstractive|dialog(?:ue)? (?:generation|response|system)s?|single-turn dialog(?:ue)?|translation|story generation|text generation|open-ended (?:generation|text|questions|responses)|caption(?:ing)?|instruction[- ]following|chat(?:bot)? (?:responses|assistants?|models?)|creative writing)\b/i;

/** A language model used as the grader. */
export const LLM_JUDGE_RE =
  /\b(?:GPT-4o?|GPT-3\.5|ChatGPT|PaLM 2|Gemini|an? (?:strong )?LLM|LLMs)\b[^.;]{0,50}\b(?:as (?:an? |the )?(?:judge|evaluator|grader|annotator|referee)s?|to (?:judge|evaluate|grade|rate|score|compare|annotate))\b|\b(?:judged|evaluated|graded|rated|scored|computed|annotated) (?:\w+ ){0,2}by (?:GPT-4o?|GPT-3\.5|ChatGPT|an? LLM)\b|\bLLM[- ]as[- ]an?[- ]judge\b|\b(?:GPT-4|LLM)[- ](?:based )?(?:judge|evaluator|evaluation|grading|win rates?)\b/i;

/** Benchmarks that use a language model as judge by design. */
export const JUDGE_BENCHMARKS = new Set(["AlpacaEval", "MT-Bench", "Arena-Hard", "FLASK"]);

/** Test-set contamination checks. */
export const CONTAMINATION_RE =
  /\b(?:de)?contaminat\w+|\b(?:data|test[- ]set|benchmark|evaluation) (?:leakage|overlap)\b|\bn-gram overlap\b|\boverlap (?:between|with) (?:the )?(?:training|pre-?training) (?:data|corpus|set)\b|\bmemori[sz]ation of (?:the )?(?:test|benchmark)/i;

/** Large-scale pre-training on web text: the setting where contamination matters most. */
export const WEB_SCALE_RE =
  /\b\d+(?:\.\d+)?\s?(?:T|B|trillion|billion) tokens\b|\b(?:pre-?train\w*|trained) on\b[^.]{0,80}\b(?:web|Common ?Crawl|C4|the Pile|internet|tokens)\b/i;

/** Claims of breadth that need a broad evaluation behind them. */
export const GENERALITY_RE =
  /\bgenerali[sz]es? (?:well |better |robustly )?(?:to|across)\b|\bgeneral-purpose\b|\b(?:domain|task|model|architecture)-agnostic\b|\buniversal(?:ly)?\b|\bany (?:model|task|domain|architecture|language|dataset)\b|\bbroadly applicable\b|\bacross (?:a wide (?:range|variety) of|diverse|many|all) (?:\w+ )?(?:tasks|domains|settings|models|benchmarks|languages)\b/i;

/** Training happens in the paper (compute and hyperparameters then matter for replication). */
export const TRAINING_RE =
  /\bwe (?:then |also |further )?(?:train|fine-?tune|pre-?train|finetune|retrain)\b|\b(?:is|are|was|were) (?:then )?(?:trained|fine-?tuned|pre-?trained|finetuned)\b|\btraining (?:run|steps|tokens|data|set|procedure|details|recipe|budget)\b/i;

/** Efficiency claims (hardware must be stated for them to be interpretable). */
export const EFFICIENCY_RE =
  /\b(?:speed-?ups?|throughput|latenc(?:y|ies)|wall-?clock|memory (?:usage|footprint|consumption|savings?|reduction)|faster|tokens\/s|tokens per second|inference (?:cost|time|speed)|peak memory)\b/i;

/** Failure-case or error analysis. */
export const ERROR_ANALYSIS_RE =
  /\b(?:error|failure) (?:analys[ie]s|cases?|modes?)\b|\bqualitative (?:analysis|examples?)\b|\bwhere (?:\w+ ){0,3}fails?\b|\bfails? (?:on|when|in)\b/i;

// ───────────────────────────── Links and openness ───────────────────────────

/** Repositories and model hubs that host code. */
export const CODE_HOST_RE =
  /\b(?:github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|huggingface\.co\/(?!datasets\/)|anonymous\.4open\.science|paperswithcode\.com|colab\.research\.google\.com|sourceforge\.net)/i;

/** Places that host data. */
export const DATA_HOST_RE =
  /\b(?:huggingface\.co\/datasets\/|zenodo\.org|kaggle\.com|figshare\.com|osf\.io|dataverse\.|data\.mendeley\.com|physionet\.org|archive\.ics\.uci\.edu)|\/datasets?\b/i;

export const URL_RE = /\bhttps?:\/\/[^\s<>"')\]]+|\b(?:www\.)?(?:github\.com|huggingface\.co|gitlab\.com)\/[^\s<>"')\]]+/gi;

/**
 * Statements that code or models are (or will be) released. Negations ("no pre-trained model is
 * available", "the model is not publicly available") say the opposite, so a "no/not/never" shortly
 * before the subject or among the words after the verb rejects the match.
 */
export const CODE_STATEMENT_RE =
  /\b(?<!\b(?:no|not|never)\b[^.;]{0,30})(?:code|implementation|source code|models?|weights|checkpoints)(?: and (?:\w+ ){0,2}(?:code|data|models?|weights|checkpoints))? (?:is|are|will be|has been|have been) (?:(?!not\b|never\b|no\b)\w+ ){0,2}(?:publicly |freely |openly )?(?:available|released|open-?sourced?)\b|\b(?:we|our team) (?:will )?(?:release|open-?source|publicly release|make (?:\w+ ){0,3}available)\b[^.]{0,60}\b(?:code|implementation|models?|weights|checkpoints|data(?:set)?s?)\b/i;

// ────────────────────────────── Evaluation sizes ────────────────────────────

/** "on 200 test examples", "n = 50", "150 annotators". The noun says what was counted. */
export const SAMPLE_SIZE_RE =
  /\b(?:n|N)\s*=\s*(\d[\d,]*)\b|\b(\d[\d,]*)\s+(?:randomly[- ]selected |randomly[- ]sampled |held-out |test |evaluation |unique |distinct )?(participants|subjects|annotators|raters|crowdworkers|workers|users|judges|evaluators|experts|examples|samples|instances|questions|prompts|problems|test cases|items|dialogues|conversations|images|videos|episodes|trajectories|documents|articles|queries|utterances)\b/g;

/** People rather than items: small human studies are judged on a different scale. */
export const HUMAN_SUBJECT_NOUNS = new Set([
  "participants", "subjects", "annotators", "raters", "crowdworkers", "workers", "users", "judges", "evaluators", "experts",
]);

/** The sentence talks about evaluation, so a count in it is an evaluation size, not a training detail. */
export const EVALUATION_CONTEXT_RE =
  /\bevaluat\w*|\btest(?:ed|ing| set| split)?\b|\bhuman\b|\bannotat\w*|\brated\b|\bjudg\w*|\bstudy\b|\bbenchmark\w*|\bheld-out\b|\brandomly (?:sample|select)\w*|\bwe (?:sample|select)\w*|\bassess\w*|\bmanual(?:ly)? (?:inspect|analy|review)\w*/i;

// ──────────────────────────────── Limitations ───────────────────────────────

/** Explicit admissions; stronger than the zoner's generic deficiency cues. */
export const LIMITATION_CUE_RE =
  /\blimitations?\b|\bweakness(?:es)?\b|\bdrawbacks?\b|\bshortcomings?\b|\bcaveats?\b|\b(?:beyond|outside) (?:the )?scope\b|\b(?:is|are|was|were|remains?) (?:currently |still )?(?:limited|restricted|confined) to\b|\bwe (?:only|mostly|mainly) (?:evaluate|evaluated|test|tested|consider|considered|focus|focused|study|studied|experiment\w*|restrict\w*)\b|\bonly (?:evaluated|tested|considered|studied)\b|\bmay not (?:generali[sz]e|hold|transfer|scale|apply|extend)\b|\b(?:does|do|did) not (?:generali[sz]e|scale|transfer|address|handle|support|consider|evaluate)\b|\bfails? to\b|\bwe (?:did|do|could|have) not (?:evaluate|test|explore|investigate|consider|compare|study|verify|use|scale)\b|\bnot (?:yet )?(?:been )?(?:evaluated|tested|explored|investigated|verified)\b|\b(?:requires?|relies on|depends on) (?:\w+ ){0,3}(?:access to|additional|large|careful|manual)\b|\boccasionally (?:fails?|generates?|produces?)\b|\bunreliable\b/i;

/** Concessions inside future-work sentences ("while we only evaluate up to 6B ...") are limitations too. */
export const CONCESSION_RE =
  /\b(?:while|although|though) (?:we|our|this)\b[^.]{0,80}\b(?:only|up to|limited|restricted|small|single|few)\b|\bwe (?:only|have only) (?:\w+ ){0,3}(?:evaluate|evaluated|test|tested|consider|considered|explore|explored)\b/i;

/**
 * Section titles that announce a limitations discussion. "Failure modes" is left out: such sections
 * are error analyses (often of the baselines), a separate signal, not the authors' own admissions.
 */
export const LIMITATIONS_TITLE_RE = /\blimitations?\b|\bweakness(?:es)?\b|\bshortcomings?\b/i;

export const RELATED_TITLE_RE = /\brelated works?\b|\bprior works?\b|\bliterature review\b|\bbackground and related\b/i;

export const ABLATION_TITLE_RE = /\bablat/i;
