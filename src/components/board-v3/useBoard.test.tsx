import { afterEach, beforeEach, describe, expect, it } from "@jest/globals";
import { act, renderHook } from "@testing-library/react";
import { toast } from "sonner";
import useBoard from "./useBoard";
import type { GameClient } from "@/game/engine-v2";
import type { Settings } from "./SettingsMenu";
import type { UserActionState } from "./state";
import { defaultBoard } from "@/game/engine";
import type { AllowedMove, GameoverState } from "@/game/engine";
import { allowedMoveFromUci } from "@/game/notation-uci";

declare const jest: typeof import("@jest/globals").jest;
jest.mock("@/game/audio", () => ({
  playCaptureSound: jest.fn(),
  playMoveSound: jest.fn(),
  playNextTurnSound: jest.fn(),
}));
jest.mock("sonner", () => ({ toast: { error: jest.fn() } }));

function fakeGame(needsTurnConfirmation = false) {
  return {
    needsTurnConfirmation,
    isSendingTurn: false,
    isReplayMode: false,
    gameover: undefined,
    turn: 1,
    moves: [],
    getV1Board: () => defaultBoard,
    getLastTurnBoards: () => [],
    getLastTurnMoves: () => [],
    getThisTurnMoves: (): AllowedMove[] => [],
    getGameover: (): GameoverState | undefined => undefined,
    currentPlayer: () => "RED",
    currentPlayerTurn: () => "RED",
    isMyTurn: () => true,
    numMovesThisTurn: () => (needsTurnConfirmation ? 3 : 0),
    clearBombardments: () => false,
    canEndTurn: () => true,
    endTurn: jest.fn<() => Promise<void>>().mockResolvedValue(undefined),
  };
}

const settings = { confirmTurn: false } as Settings;
const userActionState = {} as UserActionState;

describe("automatic turn confirmation", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
  });
  afterEach(() => {
    jest.useRealTimers();
  });

  it("waits for a completed turn instead of skipping on mount", async () => {
    const game = fakeGame();
    const { rerender } = renderHook(() =>
      useBoard({
        game: game as unknown as GameClient,
        settings,
        userActionState,
      })
    );
    await act(async () => jest.advanceTimersByTime(10));
    expect(game.endTurn).not.toHaveBeenCalled();
    game.needsTurnConfirmation = true;
    rerender();
    await act(async () => jest.advanceTimersByTime(1));
    expect(game.endTurn).toHaveBeenCalledTimes(1);
  });

  it.each([false, true])("submits a terminal bombardment with confirmTurn=%s", async (confirmTurn) => {
    const game = fakeGame();
    let moves: AllowedMove[] = [];
    game.getThisTurnMoves = () => moves;
    game.clearBombardments = () => {
      moves = [allowedMoveFromUci("sba8")];
      return true;
    };
    game.getGameover = () => moves.length
      ? { status: "WIN", winner: "RED", reason: "hq-capture" }
      : undefined;
    const { rerender } = renderHook(() => useBoard({
      game: game as unknown as GameClient,
      settings: { ...settings, confirmTurn },
      userActionState,
    }));
    await act(async () => jest.advanceTimersByTime(10));
    expect(game.endTurn).toHaveBeenCalledTimes(1);
    rerender();
    await act(async () => jest.advanceTimersByTime(10));
    expect(game.endTurn).toHaveBeenCalledTimes(1);
  });

  it("does not submit a nonterminal bombardment", async () => {
    const game = fakeGame();
    game.getThisTurnMoves = () => [allowedMoveFromUci("sba7")];
    renderHook(() => useBoard({ game: game as unknown as GameClient, settings, userActionState }));
    await act(async () => jest.advanceTimersByTime(10));
    expect(game.endTurn).not.toHaveBeenCalled();
  });

  it("reports a failed worker request once and leaves manual retry available", async () => {
    const game = fakeGame(true);
    game.endTurn.mockRejectedValue(new Error("Worker unavailable"));
    const { rerender } = renderHook(() =>
      useBoard({
        game: game as unknown as GameClient,
        settings,
        userActionState,
      })
    );
    await act(async () => jest.advanceTimersByTime(1));
    expect(toast.error).toHaveBeenCalledWith(
      expect.stringContaining("Worker unavailable")
    );
    expect(game.needsTurnConfirmation).toBe(true);
    rerender();
    await act(async () => jest.advanceTimersByTime(3000));
    expect(game.endTurn).toHaveBeenCalledTimes(1);
  });

  it("handles an interrupted search after unmount without a stale error toast", async () => {
    const game = fakeGame(true);
    let reject!: (error: Error) => void;
    game.endTurn.mockImplementation(
      () =>
        new Promise<void>((_, rejectPromise) => {
          reject = rejectPromise;
        })
    );
    const { unmount } = renderHook(() =>
      useBoard({
        game: game as unknown as GameClient,
        settings,
        userActionState,
      })
    );
    await act(async () => jest.advanceTimersByTime(1));
    unmount();
    await act(async () => reject(new Error("Worker unavailable")));
    expect(toast.error).not.toHaveBeenCalled();
  });

  it("does not report an intentional cancellation as a failure", async () => {
    const game = fakeGame(true);
    game.endTurn.mockRejectedValue(
      new DOMException("Bot closed", "AbortError")
    );
    renderHook(() =>
      useBoard({
        game: game as unknown as GameClient,
        settings,
        userActionState,
      })
    );
    await act(async () => jest.advanceTimersByTime(1));
    expect(toast.error).not.toHaveBeenCalled();
  });
});
