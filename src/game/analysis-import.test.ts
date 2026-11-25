import { describe, it, jest, expect } from "@jest/globals";
import { prepareAnalysisImport } from "./analysis-import";
import type { GameEngine } from "./engine-v2";

function engine() {
  let red = true;
  return {
    BaseBoard: jest.fn(() => ({
      is_red_turn: () => red,
      is_legal: (move: string) => move === "skip",
      push: () => {
        red = !red;
      },
    })),
    Move: { from_uci: (uci: string) => uci },
  } as unknown as GameEngine;
}

describe("analysis imports", () => {
  it("loads current numbered exports and results", () => {
    const result = prepareAnalysisImport(
      engine(),
      "PGN",
      "1. skip {[%emt 1.0]}\n0-1 {[%sts resign]}"
    );
    expect(result.fen).toBeUndefined();
    expect(result.initialTurns).toHaveLength(2);
    expect(result.initialTurns?.[1]).toMatchObject({
      status: "resign",
      winner: "BLUE",
    });
  });
  it.each([
    "",
    "skip",
    "2. skip {[%emt 0.0]}",
    "1. a1a2 {[%emt 0.0]}",
    "1-0 {[%sts resign]}\n2. skip {[%emt 0.0]}",
  ])("rejects invalid or legacy PGN: %s", (text) => {
    expect(() => prepareAnalysisImport(engine(), "PGN", text)).toThrow();
  });
  it("loads a FEN without carrying PGN history", () => {
    const fen = "q7/8/8/8/8/8/8/7Q - - b";
    expect(prepareAnalysisImport(engine(), "FEN", fen)).toEqual({
      fen,
      initialTurns: undefined,
    });
  });
  it.each(["", "8 - - r", "9/8/8/8/8/8/8/8 - - r", "8/8/8/8/8/8/8/8 - - x"])(
    "rejects malformed FEN: %s",
    (text) => {
      expect(() => prepareAnalysisImport(engine(), "FEN", text)).toThrow();
    }
  );
});
