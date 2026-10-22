/** One-click examples on the home screen. All papers are openly licensed arXiv preprints. */
export interface PaperSample {
  input: string;
  label: string;
  hint: string;
}

export const PAPER_SAMPLES: PaperSample[] = [
  { input: "2201.11903", label: "Chain-of-Thought", hint: "Wei et al., 2022" },
  { input: "2210.03629", label: "ReAct", hint: "Yao et al., 2022" },
  { input: "2305.18290", label: "DPO", hint: "Rafailov et al., 2023" },
  { input: "2312.00752", label: "Mamba", hint: "Gu & Dao, 2023" },
  { input: "2402.02750", label: "KIVI", hint: "Liu et al., 2024" },
];

export const TOPIC_SAMPLE = "kv cache compression";
export const DOI_SAMPLE = { input: "10.18653/v1/N19-1423", label: "BERT (DOI)" };

/** CoT, ReAct, Self-RAG, RAPTOR and DPO: reasoning, retrieval and alignment. */
export const SURVEY_SAMPLE = ["2201.11903", "2210.03629", "2310.11511", "2401.18059", "2305.18290"];
