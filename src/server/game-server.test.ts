/** @jest-environment node */
import { describe, expect, it } from "@jest/globals";
declare const jest: typeof import("@jest/globals").jest;
import {
  addGameServerRoutes,
  createNewV3Match,
  matchLifecycleV3,
} from "./game-server";
import * as store from "./matchv3-store";
import { finalizeMatch } from "./match-finalization";
import { SupabaseMatch, supabaseMatchToMatchV3 } from "./supabase";
import { GameEngine, Turn } from "@/game/engine-v2";
import { pgnToTurns } from "@/game/pgn";
import { hashCredentials } from "./turn-processing";

jest.mock("nanoid", () => ({ nanoid: () => "test-id" }));
jest.mock("./matchv3-store");
jest.mock("./match-finalization");
jest.mock("@/lib/supabase", () => ({ getUser: jest.fn() }));
jest.mock("@/lib/console", () => ({ printWelcome: () => undefined }));

describe("v2 multiplayer finalization", () => {
  it.each(["resignation", "board win", "draw", "already recorded resignation"])(
    "persists completion and cleans up active matches for %s",
    async (ending) => {
      // Disable the background lifecycle; exercise the real POST handler.
      const env = process.env.NODE_ENV;
      let post: (ctx: any) => Promise<void> = async () => {};
      const router = {
        get: jest.fn(),
        post: (_: string, handler: typeof post) => {
          post = handler;
        },
      };
      let moved = false;
      const outcome = () =>
        ending === "draw"
          ? { winner: null, termination: "double-skip" }
          : { winner: false, termination: "hq-capture" };
      const engine = {
        BaseBoard: Object.assign(
          () => ({
            is_red_turn: () => true,
            serialize: () => "board",
            outcome: () => (moved ? outcome() : undefined),
          }),
          {
            deserialize: () => ({
              push: () => {
                moved = true;
              },
              is_red_turn: () => true,
              outcome,
              serialize: () => "board",
            }),
          }
        ),
        Move: { from_uci: () => ({}) },
      } as unknown as GameEngine;
      try {
        Object.assign(process.env, { NODE_ENV: "development" });
        addGameServerRoutes(router as any, engine);
      } finally {
        Object.assign(process.env, { NODE_ENV: env });
      }
      const row = {
        id: "test-match",
        engine_version: 2,
        starting_fen: "custom-start",
        player0_id: "red-user",
        player1_id: "blue-user",
        player_0_credentials_hash: hashCredentials("secret"),
        rated: false,
        is_correspondence: true,
        status: null,
        winner_id: null,
        gameover_reason: null,
        current_turn_player_id: "red-user",
        pgn:
          ending === "already recorded resignation"
            ? "0-1 {[%sts resign]}"
            : "",
      } as SupabaseMatch;
      jest
        .mocked(store.fetchMatchV3)
        .mockResolvedValue(supabaseMatchToMatchV3(row));
      let saved: SupabaseMatch | undefined;
      jest
        .mocked(store.updateMatchPGN)
        .mockImplementation(async (_, updater) => {
          saved = updater(row);
          return supabaseMatchToMatchV3(saved);
        });
      const turn: Turn = {
        turn: 1,
        elapsedSecs: 0,
        moves: [],
        status: ending === "draw" ? "double-skip" : "resign",
        winner: ending === "draw" ? undefined : "BLUE",
      };
      if (ending === "board win" || ending === "draw") {
        delete turn.status;
        delete turn.winner;
        turn.moves = [{ name: "Skip", args: [] }];
      }
      await post({
        state: { auth: { userId: "red-user" } },
        params: { id: row.id },
        request: { body: { playerId: "0", credentials: "secret", turn } },
        throw: (status: number, message: string) => {
          throw new Error(`${status}: ${message}`);
        },
      });
      expect(saved).toMatchObject({
        status: ending === "draw" ? "DRAW" : "WIN",
        winner_id:
          ending === "draw"
            ? null
            : ending === "board win"
            ? "red-user"
            : "blue-user",
        gameover_reason:
          ending === "draw"
            ? "double-skip"
            : ending === "board win"
            ? "hq-capture"
            : "resign",
        current_turn_player_id: null,
      });
      expect(
        pgnToTurns(saved!.pgn!).filter((turn) => turn.status)
      ).toHaveLength(1);
      // Completion is committed by updateMatchPGN via the server transaction.
      expect(saved?.status).toBeTruthy();
    }
  );
});

it.each([true, false])(
  "creates new rated=%s matches on v2 with their settings",
  async (rated) => {
    jest.mocked(store.createMatchV3).mockClear();
    await createNewV3Match({
      user0: { id: "red", username: "red", elo: 1000 },
      user1: { id: "blue", username: "blue", elo: 1100 },
      timeControlName: "rapid",
      timeControl: { time: 900000, bonus: 10000 },
      isCorrespondence: false,
      rated,
      startingFen: "custom-start",
    });
    expect(store.createMatchV3).toHaveBeenCalledWith(
      expect.objectContaining({
        engineVersion: 2,
        rated,
        startingFen: "custom-start",
        pgn: "",
        timeControlAllowedTime: 900000,
        timeControlBonus: 10000,
      })
    );
  }
);

describe("batch timeout checking", () => {
  const snapshot = {
    id: "timeout",
    engine_version: 2,
    status: null,
    pgn: "",
    rated: true,
    player0_id: "red",
    player1_id: "blue",
    starting_fen: "custom-start",
    created_at: new Date(Date.now() - 120000).toISOString(),
    time_control_allowed_time: 60000,
    time_control_bonus: 0,
  } as SupabaseMatch;
  const base = jest.fn(() => ({
    is_red_turn: () => true,
    outcome: () => undefined,
  }));
  const engine = { BaseBoard: base } as unknown as GameEngine;

  function setup(matches: SupabaseMatch[]) {
    jest
      .mocked(store.listInProgressLiveMatches)
      .mockClear()
      .mockResolvedValue(matches);
    jest.mocked(store.updateMatchPGN).mockClear();
    jest.mocked(store.fetchMatchV3).mockClear();
    jest
      .mocked(finalizeMatch)
      .mockReset()
      .mockImplementation(async (_, result) => result);
  }

  it("finalizes terminal candidates directly from their batch snapshot", async () => {
    setup([snapshot]);
    await matchLifecycleV3(engine);
    expect(base).toHaveBeenCalledWith("custom-start");
    expect(finalizeMatch).toHaveBeenCalledWith(
      snapshot,
      expect.objectContaining({
        status: "WIN",
        winner_id: "blue",
        gameover_reason: "timeout",
        current_turn_player_id: null,
      })
    );
    const saved = jest.mocked(finalizeMatch).mock.calls[0][1];
    expect(pgnToTurns(saved.pgn!)).toHaveLength(1);
    expect(store.listInProgressLiveMatches).toHaveBeenCalledTimes(1);
    expect(store.updateMatchPGN).not.toHaveBeenCalled();
    expect(store.fetchMatchV3).not.toHaveBeenCalled();
  });

  it("does no per-match reads or writes for ongoing games", async () => {
    setup(
      [1, 2, 3].map((id) => ({
        ...snapshot,
        id: String(id),
        created_at: new Date().toISOString(),
      }))
    );
    await matchLifecycleV3(engine);
    expect(store.listInProgressLiveMatches).toHaveBeenCalledTimes(1);
    expect(store.updateMatchPGN).not.toHaveBeenCalled();
    expect(store.fetchMatchV3).not.toHaveBeenCalled();
    expect(finalizeMatch).not.toHaveBeenCalled();
  });

  it("leaves a stale candidate for the next tick and continues checking other games", async () => {
    setup([snapshot, { ...snapshot, id: "next" }]);
    jest
      .mocked(finalizeMatch)
      .mockRejectedValueOnce(new Error("Match changed concurrently"));
    const log = jest.spyOn(console, "error").mockImplementation(() => {});
    try {
      await matchLifecycleV3(engine);
      expect(finalizeMatch).toHaveBeenCalledTimes(2);
      expect(jest.mocked(finalizeMatch).mock.calls[1][0].id).toBe("next");
      expect(store.listInProgressLiveMatches).toHaveBeenCalledTimes(1);
      expect(store.updateMatchPGN).not.toHaveBeenCalled();
    } finally {
      log.mockRestore();
    }
  });
});
