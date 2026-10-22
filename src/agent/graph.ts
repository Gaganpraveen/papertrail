/**
 * A small, explicit state-graph runtime (nodes + edges + shared state).
 *
 * It mirrors the ideas of LangGraph without the framework: every node is an
 * async function that receives the current state and returns a partial update,
 * edges are either fixed or decided by a router function, and every step is
 * observable through trace events. Checkpoints let a caller persist state after
 * each node so a run can be inspected or resumed.
 */
import type { TraceEvent } from "@/types";

export const START = "__start__";
export const END = "__end__";

export interface NodeContext {
  /** Report progress or a human-readable log line for this node. */
  log(message: string): void;
  signal?: AbortSignal;
}

export type NodeFn<S> = (state: S, ctx: NodeContext) => Promise<Partial<S>> | Partial<S>;
export type Router<S> = (state: S) => string;

export interface NodeSpec<S> {
  name: string;
  label: string;
  description: string;
  fn: NodeFn<S>;
  retries: number;
}

export interface RunOptions<S> {
  onEvent?: (event: TraceEvent) => void;
  /** Called with the merged state after every node. */
  onState?: (state: S, node: string) => void | Promise<void>;
  signal?: AbortSignal;
  /** Start somewhere other than the entry node (resume from a checkpoint). */
  from?: string;
  maxSteps?: number;
}

export interface GraphDescription {
  nodes: { name: string; label: string; description: string }[];
  edges: { from: string; to: string; conditional: boolean }[];
}

export class GraphError extends Error {
  constructor(
    message: string,
    readonly node: string,
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = "GraphError";
  }
}

export class StateGraph<S extends object> {
  private nodes = new Map<string, NodeSpec<S>>();
  private edges = new Map<string, string>();
  private routers = new Map<string, { router: Router<S>; targets: string[] }>();
  private entry: string | null = null;

  addNode(
    name: string,
    fn: NodeFn<S>,
    opts: { label?: string; description?: string; retries?: number } = {},
  ): this {
    if (this.nodes.has(name)) throw new Error(`Duplicate node "${name}"`);
    this.nodes.set(name, {
      name,
      fn,
      label: opts.label ?? name,
      description: opts.description ?? "",
      retries: opts.retries ?? 0,
    });
    return this;
  }

  addEdge(from: string, to: string): this {
    if (from === START) {
      this.entry = to;
      return this;
    }
    this.edges.set(from, to);
    return this;
  }

  addConditionalEdges(from: string, router: Router<S>, targets: string[]): this {
    this.routers.set(from, { router, targets });
    return this;
  }

  /** Validates the topology and returns a runnable graph. */
  compile(): CompiledGraph<S> {
    if (!this.entry) throw new Error("Graph has no entry edge from START");
    const known = (n: string) => n === END || this.nodes.has(n);
    if (!known(this.entry)) throw new Error(`Unknown entry node "${this.entry}"`);
    for (const [from, to] of this.edges) {
      if (!this.nodes.has(from) || !known(to)) throw new Error(`Bad edge ${from} -> ${to}`);
    }
    for (const [from, { targets }] of this.routers) {
      if (!this.nodes.has(from)) throw new Error(`Router on unknown node "${from}"`);
      for (const t of targets) if (!known(t)) throw new Error(`Bad route ${from} -> ${t}`);
    }
    for (const name of this.nodes.keys()) {
      if (!this.edges.has(name) && !this.routers.has(name)) {
        throw new Error(`Node "${name}" has no outgoing edge`);
      }
    }
    return new CompiledGraph(this.entry, this.nodes, this.edges, this.routers);
  }
}

export class CompiledGraph<S extends object> {
  constructor(
    private readonly entry: string,
    private readonly nodes: Map<string, NodeSpec<S>>,
    private readonly edges: Map<string, string>,
    private readonly routers: Map<string, { router: Router<S>; targets: string[] }>,
  ) {}

  describe(): GraphDescription {
    const edges: GraphDescription["edges"] = [{ from: START, to: this.entry, conditional: false }];
    for (const [from, to] of this.edges) edges.push({ from, to, conditional: false });
    for (const [from, { targets }] of this.routers) {
      for (const to of targets) edges.push({ from, to, conditional: true });
    }
    return {
      nodes: [...this.nodes.values()].map(({ name, label, description }) => ({
        name,
        label,
        description,
      })),
      edges,
    };
  }

  async run(initial: S, opts: RunOptions<S> = {}): Promise<S> {
    const { onEvent, onState, signal, maxSteps = 64 } = opts;
    const emit = (e: Omit<TraceEvent, "at">) => onEvent?.({ ...e, at: Date.now() });

    let state = initial;
    let current = opts.from ?? this.entry;
    let steps = 0;

    while (current !== END) {
      if (signal?.aborted) throw new GraphError("Run cancelled", current);
      if (++steps > maxSteps) throw new GraphError(`Exceeded ${maxSteps} steps`, current);

      const spec = this.nodes.get(current);
      if (!spec) throw new GraphError(`Unknown node "${current}"`, current);

      const ctx: NodeContext = {
        log: (message) => emit({ type: "log", node: spec.name, message }),
        signal,
      };

      let update: Partial<S> | undefined;
      for (let attempt = 0; ; attempt++) {
        const started = performance.now();
        emit({ type: "node:start", node: spec.name });
        try {
          update = await spec.fn(state, ctx);
          emit({ type: "node:end", node: spec.name, durationMs: performance.now() - started });
          break;
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          if (attempt < spec.retries && !signal?.aborted) {
            emit({ type: "node:retry", node: spec.name, message });
            await sleep(400 * 2 ** attempt);
            continue;
          }
          emit({ type: "node:error", node: spec.name, message });
          throw new GraphError(message, spec.name, err);
        }
      }

      state = { ...state, ...update };
      await onState?.(state, spec.name);

      const routed = this.routers.get(current);
      const next = routed ? routed.router(state) : this.edges.get(current);
      if (!next) throw new GraphError(`No edge out of "${current}"`, current);
      if (routed && !routed.targets.includes(next)) {
        throw new GraphError(`Router chose undeclared target "${next}"`, current);
      }
      if (routed) emit({ type: "route", node: current, to: next });
      current = next;
    }
    return state;
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
