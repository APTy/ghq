import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from "@jest/globals";
import { BotMultiplayer } from "./engine-v2-multiplayer";
import type { AnalysisResult } from "./analysis";
import type { GameEngine, Turn } from "./engine-v2";
import { allowedMoveFromUci } from "./notation-uci";

jest.mock("@/lib/api", () => ({ ghqFetch: jest.fn() }));

// A tiny turn-state machine isolates async delivery/transaction behavior from
// Python's separately tested rules and search.
function fakeEngine(): GameEngine {
  const create = (saved?: string) => {
    let state = saved
      ? JSON.parse(saved)
      : { blue: false, moves: 0, ended: false, draw: false, winner: true };
    return {
      serialize: () => JSON.stringify(state),
      is_red_turn: () => !state.blue,
      is_blue_turn: () => state.blue,
      outcome: () =>
        state.ended
          ? {
              winner: state.draw ? undefined : state.winner,
              termination: state.draw ? "double-skip" : "hq-capture",
            }
          : undefined,
      is_legal: (move: { uci: () => string }) =>
        !state.ended &&
        ((!state.blue && ["skip", "a1a2xb8"].includes(move.uci())) ||
          (state.blue &&
            ["sba1", "a8a7", "b8b7", "c8c7", "a8a7xb1", "skip"].includes(
              move.uci()
            ))),
      push: (move: { uci: () => string }) => {
        const uci = move.uci();
        if (uci === "a1a2xb8") {
          state.ended = true;
          state.winner = false;
          return;
        }
        if (uci === "a8a7xb1") {
          state.ended = true;
          return;
        }
        if (uci.startsWith("sb")) return;
        if (uci === "skip" && state.blue) {
          state.ended = true;
          state.draw = true;
          return;
        }
        state.moves++;
        if (uci === "skip" || state.moves === 3) {
          state.blue = !state.blue;
          state.moves = 0;
        }
      },
    };
  };
  const BaseBoard = Object.assign(() => create(), { deserialize: create });
  return {
    BaseBoard,
    Move: { from_uci: (uci: string) => ({ uci: () => uci }) },
  } as unknown as GameEngine;
}

function answer(moves = ["sba1", "a8a7", "b8b7", "c8c7"]): AnalysisResult {
  return { turnMoves: moves, bestMove: moves[0] } as AnalysisResult;
}

describe("CPU bot integration", () => {
  let bot: BotMultiplayer;
  const human: Turn = {
    turn: 1,
    moves: [{ name: "Skip", args: [] }],
    elapsedSecs: 1,
  };
  beforeEach(() => {
    jest.useFakeTimers();
    localStorage.clear();
  });
  afterEach(() => {
    bot?.dispose();
    jest.useRealTimers();
  });

  it("delivers a full coordinated turn after the human turn completes, including free captures", async () => {
    const analyze = jest
      .fn<() => Promise<AnalysisResult>>()
      .mockResolvedValue(answer());
    bot = new BotMultiplayer(fakeEngine(), "test", undefined, {
      analyze,
      dispose: jest.fn(),
    });
    const delivered = jest.fn();
    bot.onTurnPlayed(delivered);
    await bot.sendTurn(human);
    expect(analyze).toHaveBeenCalledTimes(1);
    expect(delivered).not.toHaveBeenCalled();
    jest.runOnlyPendingTimers();
    expect(delivered).toHaveBeenCalledWith(
      expect.objectContaining({ turn: 2, moves: expect.any(Array) })
    );
    const reply = delivered.mock.calls[0][0] as Turn;
    expect(reply.moves).toHaveLength(4);
    expect(reply.moves[0].name).toBe("AutoCapture");
  });

  it("does not commit a failed search, so sending the same turn can be retried", async () => {
    const analyze = jest
      .fn<() => Promise<AnalysisResult>>()
      .mockRejectedValueOnce(new Error("Worker unavailable"))
      .mockResolvedValueOnce(answer());
    bot = new BotMultiplayer(fakeEngine(), "test", undefined, {
      analyze,
      dispose: jest.fn(),
    });
    bot.onTurnPlayed(jest.fn());
    await expect(bot.sendTurn(human)).rejects.toThrow("Worker unavailable");
    expect(localStorage.getItem("bot_game_history")).toBeNull();
    await bot.sendTurn(human);
    expect(analyze).toHaveBeenCalledTimes(2);
    const history = JSON.parse(localStorage.getItem("bot_game_history")!);
    expect(history.games).toHaveLength(1);
    expect(history.games[0].pgn).toContain("a8a7");
  });

  it("stops at HQ capture and reports the terminal outcome", async () => {
    bot = new BotMultiplayer(fakeEngine(), "test", undefined, {
      analyze: jest
        .fn<() => Promise<AnalysisResult>>()
        .mockResolvedValue(answer(["a8a7xb1", "invalid"])),
      dispose: jest.fn(),
    });
    const delivered = jest.fn();
    bot.onTurnPlayed(delivered);
    await bot.sendTurn(human);
    jest.runOnlyPendingTimers();
    const reply = delivered.mock.calls[0][0] as Turn;
    expect(reply.moves).toHaveLength(1);
    expect(reply.status).toBe("hq-capture");
    expect(reply.winner).toBe("BLUE");
    const saved = bot.loadGameTurns("test");
    expect(saved).toHaveLength(2);
    expect(saved[1]).toEqual(reply);
  });

  it("persists a human terminal move and outcome without requesting a bot reply", async () => {
    const analyze = jest.fn<() => Promise<AnalysisResult>>();
    bot = new BotMultiplayer(fakeEngine(), "test", undefined, {
      analyze,
      dispose: jest.fn(),
    });
    const delivered = jest.fn();
    bot.onTurnPlayed(delivered);
    await bot.sendTurn({
      turn: 1,
      moves: [allowedMoveFromUci("a1a2xb8")],
      elapsedSecs: 1,
    });
    expect(analyze).not.toHaveBeenCalled();
    expect(delivered).not.toHaveBeenCalled();
    expect(bot.loadGameTurns("test")).toEqual([
      {
        turn: 1,
        moves: [allowedMoveFromUci("a1a2xb8")],
        elapsedSecs: 1,
        status: "hq-capture",
        winner: "RED",
      },
    ]);
  });

  it("reports a drawn reply and cancels pending delivery on disposal", async () => {
    bot = new BotMultiplayer(fakeEngine(), "test", undefined, {
      analyze: jest
        .fn<() => Promise<AnalysisResult>>()
        .mockResolvedValue(answer(["skip"])),
      dispose: jest.fn(),
    });
    const delivered = jest.fn();
    bot.onTurnPlayed(delivered);
    await bot.sendTurn(human);
    bot.dispose();
    jest.runOnlyPendingTimers();
    expect(delivered).not.toHaveBeenCalled();
    expect(localStorage.getItem("bot_game_history")).toContain("skip");
    expect(bot.loadGameTurns("test")[1].status).toBe("double-skip");
    expect(bot.loadGameTurns("test")[1].winner).toBeUndefined();
  });
});
