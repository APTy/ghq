import { describe, expect, it } from "@jest/globals";
import { GameClient, type GameEngine, type Turn } from "./engine-v2";
import { allowedMoveFromUci } from "./notation-uci";
import { pgnToTurns } from "./pgn";
import type { Multiplayer } from "./engine-v2-multiplayer";

declare const jest: typeof import("@jest/globals").jest;
jest.mock("@/lib/console", () => ({ printWelcome: () => undefined }));

// Rule legality lives in Python tests; this models only turn and outcome state
// needed to verify that GameClient retains terminal records in its exports.
function fakeEngine(): GameEngine {
  const create = (serialized?: string) => {
    const state = serialized
      ? JSON.parse(serialized)
      : { blue: false, count: 0, winner: undefined };
    return {
      serialize: () => JSON.stringify(state),
      is_red_turn: () => !state.blue,
      generate_legal_moves: () => ["sba8", "sbc8"].map((uci) => ({
        name: "AutoCapture", auto_capture_type: "bombard", uci: () => uci,
      })),
      outcome: () =>
        state.winner
          ? { winner: state.winner === "BLUE", termination: "hq-capture" }
          : undefined,
      push: (move: { uci: () => string }) => {
        if (move.uci() === "sba8") {
          state.winner = "RED";
          return;
        }
        if (move.uci().includes("xb8")) state.winner = "RED";
        if (move.uci() === "skip" || ++state.count === 3) {
          state.blue = !state.blue;
          state.count = 0;
        }
      },
    };
  };
  return {
    BaseBoard: Object.assign(() => create(), { deserialize: create }),
    Move: { from_uci: (uci: string) => ({ uci: () => uci }) },
  } as unknown as GameEngine;
}

describe("GameClient terminal history", () => {
  it.each([true, false])("submits terminal bombardment without Skip (bot=%s)", async (bot) => {
    const submitted: Turn[] = [];
    const game = new GameClient({
      engine: fakeEngine(), bot, playerId: "0",
      multiplayer: {
        initGame: async () => undefined,
        onTurnPlayed: () => undefined,
        sendTurn: async (turn) => { submitted.push({ ...turn, moves: [...turn.moves] }); },
      },
    });
    expect(game.clearBombardments()).toBe(true);
    expect(game.moves).toEqual([allowedMoveFromUci("sba8")]);
    expect(game.clearBombardments()).toBe(false);
    expect(game.needsTurnConfirmation).toBe(false);
    expect(game.getGameover()?.winner).toBe("RED");
    await game.endTurn();
    expect(submitted).toHaveLength(1);
    expect(submitted[0].moves).toEqual([allowedMoveFromUci("sba8")]);
    expect(game.getThisTurnMoves()).toEqual([]);
    if (bot) expect(game.gameover?.winner).toBe("RED");
    else expect(game.gameover).toBeUndefined(); // Live results arrive from the server.
  });

  it.each([
    { status: "resign", winner: "BLUE", moves: [] },
    { status: "hq-capture", winner: "BLUE", moves: ["a8a7xb1"] },
    { status: "double-skip", winner: undefined, moves: ["skip"] },
  ] as const)(
    "exports received $status with its moves and outcome",
    ({ status, winner, moves }) => {
      const game = new GameClient({
        engine: fakeEngine(),
        bot: true,
        playerId: "0",
      });
      const turn: Turn = {
        turn: 1,
        moves: moves.map(allowedMoveFromUci),
        elapsedSecs: moves.length ? 2.4 : 0,
        status,
        winner,
      };
      game.pushTurn(turn);
      expect(pgnToTurns(game.pgn())).toEqual([turn]);
      expect(game.gameover).toMatchObject({
        status: winner ? "WIN" : "DRAW",
        reason: status,
        winner,
      });
      expect(game.getGameover()).toEqual(game.gameover);
      expect(game.turn).toBe(2);
    }
  );

  it("preserves a resignation result even after the clock expires", () => {
    const game = new GameClient({
      engine: fakeEngine(),
      isPassAndPlayMode: true,
      timeControl: { time: 60, bonus: 0 },
      gameStartTimeMs: Date.now() - 120_000,
    });
    game.pushTurn({
      turn: 1,
      moves: [],
      elapsedSecs: 0,
      status: "resign",
      winner: "RED",
    });
    expect(game.getGameover()).toEqual({
      status: "WIN",
      winner: "RED",
      reason: "resign",
    });
  });

  it("keeps replay outcome detection disabled for terminal history", () => {
    const game = new GameClient({ engine: fakeEngine(), isReplayMode: true });
    game.pushTurn({
      turn: 1,
      moves: [],
      elapsedSecs: 0,
      status: "resign",
      winner: "BLUE",
    });
    expect(game.getGameover()).toBeUndefined();
  });

  it("exports a locally detected terminal outcome", async () => {
    const game = new GameClient({
      engine: fakeEngine(),
      bot: true,
      playerId: "0",
    });
    game.push(allowedMoveFromUci("a1a2xb8"));
    await game.endTurn();
    const [saved] = pgnToTurns(game.pgn());
    expect(saved.status).toBe("hq-capture");
    expect(saved.winner).toBe("RED");
    expect(saved.moves).toEqual([allowedMoveFromUci("a1a2xb8")]);
    expect(game.gameover?.winner).toBe("RED");
  });

  it("submits an early human HQ capture without a post-terminal Skip", async () => {
    const submitted: Turn[] = [];
    const multiplayer: Multiplayer = {
      initGame: async () => undefined,
      onTurnPlayed: () => undefined,
      sendTurn: async (turn) => {
        submitted.push({ ...turn, moves: [...turn.moves] });
      },
    };
    const game = new GameClient({
      engine: fakeEngine(),
      bot: true,
      playerId: "0",
      multiplayer,
    });
    const capture = allowedMoveFromUci("a1a2xb8");
    game.push(capture);
    expect(game.needsTurnConfirmation).toBe(false);
    await game.endTurn();
    expect(submitted).toHaveLength(1);
    expect(submitted[0].moves).toEqual([capture]);
    expect(game.moves).toEqual([capture]);
    expect(pgnToTurns(game.pgn())[0]).toMatchObject({
      moves: [capture],
      status: "hq-capture",
      winner: "RED",
    });
  });
});

describe("analysis initialization", () => {
  it("retains a partial imported turn and supports seeking through it", () => {
    const pgn = "1. a1a2 {[%emt 2.0]}";
    const game = new GameClient({ engine: fakeEngine(), isPassAndPlayMode: true, initialTurns: pgnToTurns(pgn) });
    expect(game.turn).toBe(1);
    expect(game.moves).toHaveLength(1);
    expect(game.pgn()).toBe(pgn);
    game.applyMoves(pgn, 0);
    expect(game.moves).toHaveLength(0);
    game.applyMoves(pgn, 1);
    expect(game.moves).toHaveLength(1);
  });

  it("retains completed turns and terminal results", () => {
    const pgn = "1. skip {[%emt 1.0]}\n0-1 {[%sts resign]}";
    const game = new GameClient({ engine: fakeEngine(), initialTurns: pgnToTurns(pgn) });
    expect(game.pgn()).toBe(pgn);
    expect(game.gameover).toMatchObject({ winner: "BLUE", reason: "resign" });
    game.reset();
    expect(game.gameover).toBeUndefined();
    expect(game.moves).toHaveLength(0);
  });

  it("seeks from the imported FEN", () => {
    const engine = fakeEngine();
    const original = engine.BaseBoard;
    const base = jest.fn(original);
    engine.BaseBoard = Object.assign(base, { deserialize: original.deserialize });
    const game = new GameClient({ engine, fen: "custom-position" });
    game.applyMoves("", 0);
    expect(base).toHaveBeenLastCalledWith("custom-position");
  });
});
