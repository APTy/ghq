export interface AnalysisOptions {
  timeMs?: number;
  maxDepth?: number;
  beamWidth?: number;
  multiPv?: number;
}

export interface AnalysisLine {
  moves: string[];
  turnMoves: string[];
  /** Positive favors Red. One point is approximately one infantry. */
  score: number;
}

export interface AnalysisResult {
  engineVersion: string;
  state: string;
  score: number;
  bestMove: string | null;
  turnMoves: string[];
  lines: AnalysisLine[];
  /** Completed search depth, measured in player turns. */
  depth: number;
  nodes: number;
  elapsedMs: number;
  selective: true;
  terminal: { winner: "RED" | "BLUE" | null; reason: string } | null;
  timedOut: boolean;
}

interface PendingAnalysis {
  id: number;
  state: string;
  cacheKey: string;
  resolve: (result: AnalysisResult) => void;
  reject: (error: Error) => void;
  onUpdate?: (result: AnalysisResult) => void;
  timeout: ReturnType<typeof setTimeout>;
}

/** A reusable, independently cancellable search worker. Never runs Python on the UI thread. */
export class CpuAnalysisClient {
  private worker: Worker | null = null;
  private pending: PendingAnalysis | null = null;
  private nextId = 0;
  private cache = new Map<string, AnalysisResult>();

  constructor(
    private createWorker: () => Worker = () => new Worker("/analysis-worker.js")
  ) {}

  analyze(
    state: string,
    options: AnalysisOptions = {},
    onUpdate?: (result: AnalysisResult) => void
  ): Promise<AnalysisResult> {
    this.cancel();
    const bounded = {
      timeMs: Math.max(25, Math.min(15000, options.timeMs ?? 1800)),
      maxDepth: Math.max(1, Math.min(5, options.maxDepth ?? 3)),
      beamWidth: Math.max(2, Math.min(24, options.beamWidth ?? 8)),
      multiPv: Math.max(1, Math.min(5, options.multiPv ?? 3)),
    };
    const cacheKey = JSON.stringify([state, bounded]);
    const cached = this.cache.get(cacheKey);
    if (cached) {
      this.cache.delete(cacheKey);
      this.cache.set(cacheKey, cached);
      onUpdate?.(cached);
      return Promise.resolve(cached);
    }

    return new Promise((resolve, reject) => {
      try {
        if (!this.worker) {
          const worker = this.createWorker();
          this.worker = worker;
          worker.onmessage = (event) => {
            if (this.worker === worker) this.receive(event.data);
          };
          worker.onerror = () => {
            if (this.worker === worker)
              this.fail(
                new Error(
                  "The analysis worker could not start. Please try again."
                )
              );
          };
          worker.onmessageerror = () => {
            if (this.worker === worker)
              this.fail(
                new Error("The analysis worker returned an unreadable result.")
              );
          };
        }
        const id = ++this.nextId;
        // Loading Pyodide is separate from the bounded search budget.
        const timeout = setTimeout(() => {
          this.fail(
            new Error(
              "The analysis engine took too long to load. Please try again."
            )
          );
        }, 60000 + bounded.timeMs);
        this.pending = {
          id,
          state,
          cacheKey,
          resolve,
          reject,
          onUpdate,
          timeout,
        };
        this.worker.postMessage({ id, state, options: bounded });
      } catch (error) {
        const failure =
          error instanceof Error ? error : new Error(String(error));
        if (this.pending) this.fail(failure);
        else reject(failure);
      }
    });
  }

  private receive(message: {
    id: number;
    type: string;
    result?: AnalysisResult;
    error?: string;
  }) {
    const pending = this.pending;
    if (!pending || message.id !== pending.id) return;
    if (message.type === "error") {
      this.fail(new Error(message.error || "Analysis failed."));
      return;
    }
    const result = message.result;
    if (!result || result.state !== pending.state) return;
    if (message.type === "progress") {
      pending.onUpdate?.(result);
      return;
    }
    if (message.type !== "result") return;
    clearTimeout(pending.timeout);
    this.pending = null;
    this.cache.set(pending.cacheKey, result);
    if (this.cache.size > 32)
      this.cache.delete(this.cache.keys().next().value!);
    pending.resolve(result);
  }

  private fail(error: Error) {
    const pending = this.pending;
    this.pending = null;
    this.terminateWorker();
    if (pending) {
      clearTimeout(pending.timeout);
      pending.reject(error);
    }
  }

  private terminateWorker() {
    if (!this.worker) return;
    this.worker.onmessage = null;
    this.worker.onerror = null;
    this.worker.onmessageerror = null;
    this.worker.terminate();
    this.worker = null;
  }

  cancel() {
    if (!this.pending) return;
    // A synchronous Python search cannot receive a cancellation message. Terminating
    // its worker really interrupts it, including during Pyodide initialization.
    this.fail(new DOMException("Analysis cancelled", "AbortError"));
  }

  dispose() {
    this.cancel();
    this.terminateWorker();
    this.cache.clear();
  }
}
