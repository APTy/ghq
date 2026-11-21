import { AnalysisResult, CpuAnalysisClient } from "./analysis";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from "@jest/globals";

class FakeWorker {
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessageerror: (() => void) | null = null;
  postMessage = jest.fn<(message: { id: number; state: string }) => void>();
  terminate = jest.fn();
  emit(data: unknown) {
    this.onmessage?.({ data });
  }
}

const result = (state: string): AnalysisResult => ({
  engineVersion: "test",
  state,
  score: 1.2,
  bestMove: "a1a2",
  turnMoves: ["a1a2", "skip"],
  lines: [{ moves: ["a1a2", "skip"], turnMoves: ["a1a2", "skip"], score: 1.2 }],
  depth: 2,
  nodes: 100,
  elapsedMs: 100,
  selective: true,
  terminal: null,
  timedOut: false,
});

describe("CPU analysis worker lifecycle", () => {
  let workers: FakeWorker[];
  let client: CpuAnalysisClient;
  beforeEach(() => {
    workers = [];
    client = new CpuAnalysisClient(() => {
      const worker = new FakeWorker();
      workers.push(worker);
      return worker as unknown as Worker;
    });
  });
  afterEach(() => client.dispose());

  it("streams progress, rejects mismatched positions, and caches only completed results", async () => {
    const update = jest.fn();
    const promise = client.analyze("position-a", {}, update);
    const { id } = workers[0].postMessage.mock.calls[0][0];
    workers[0].emit({ id, type: "progress", result: result("position-b") });
    expect(update).not.toHaveBeenCalled();
    workers[0].emit({ id, type: "progress", result: result("position-a") });
    expect(update).toHaveBeenCalledTimes(1);
    workers[0].emit({ id, type: "result", result: result("position-a") });
    expect(await promise).toEqual(result("position-a"));
    expect(await client.analyze("position-a")).toEqual(result("position-a"));
    expect(workers[0].postMessage).toHaveBeenCalledTimes(1);
  });

  it("interrupts the actual worker when superseded and ignores its late result", async () => {
    const first = client.analyze("position-a").catch((error) => error);
    const oldWorker = workers[0];
    const second = client.analyze("position-b");
    expect((await first).name).toBe("AbortError");
    expect(oldWorker.terminate).toHaveBeenCalledTimes(1);
    oldWorker.emit({ id: 1, type: "result", result: result("position-a") });
    workers[1].emit({ id: 2, type: "result", result: result("position-b") });
    expect((await second).state).toBe("position-b");
  });

  it("can retry after a worker initialization failure", async () => {
    const failed = client.analyze("position-a");
    workers[0].emit({ id: 1, type: "error", error: "Download failed" });
    await expect(failed).rejects.toThrow("Download failed");
    const retry = client.analyze("position-a");
    workers[1].emit({ id: 2, type: "result", result: result("position-a") });
    expect((await retry).bestMove).toBe("a1a2");
  });

  it("ignores errors already queued by a cancelled worker", async () => {
    const first = client.analyze("position-a").catch((error) => error);
    const lateError = workers[0].onerror!;
    const lateMessageError = workers[0].onmessageerror!;
    const second = client.analyze("position-b");
    await first;
    lateError();
    lateMessageError();
    expect(workers[1].terminate).not.toHaveBeenCalled();
    workers[1].emit({ id: 2, type: "result", result: result("position-b") });
    expect((await second).state).toBe("position-b");
  });

  it("does not reuse a short-budget result for a deeper search", async () => {
    const quick = client.analyze("position-a", { timeMs: 100 });
    workers[0].emit({ id: 1, type: "result", result: result("position-a") });
    await quick;
    const deep = client.analyze("position-a", { timeMs: 1000 });
    expect(workers[0].postMessage).toHaveBeenCalledTimes(2);
    workers[0].emit({ id: 2, type: "result", result: result("position-a") });
    await deep;
  });
});
