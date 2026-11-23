/** @jest-environment node */
import { describe, expect, it } from "@jest/globals";
declare const jest: typeof import("@jest/globals").jest;
import { addGameServerRoutes } from "./game-server";
import * as store from "./matchv3-store";
import { SupabaseMatch, supabaseMatchToMatchV3 } from "./supabase";
import { GameEngine, Turn } from "@/game/engine-v2";
import { pgnToTurns } from "@/game/pgn";
import { hashCredentials } from "./turn-processing";

jest.mock("nanoid", () => ({ nanoid: () => "test-id" }));
jest.mock("./matchv3-store");
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
      const engine = {
        BaseBoard: Object.assign(
          () => ({
            is_red_turn: () => true,
            serialize: () => "board",
            outcome: () =>
              moved ? { winner: false, termination: "hq-capture" } : undefined,
          }),
          {
            deserialize: () => ({
              push: () => {
                moved = true;
              },
              is_red_turn: () => true,
              outcome: () => ({ winner: false, termination: "hq-capture" }),
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
      jest.mocked(store.deleteActiveMatches).mockClear();
      const turn: Turn = {
        turn: 1,
        elapsedSecs: 0,
        moves: [],
        status: ending === "draw" ? "double-skip" : "resign",
        winner: ending === "draw" ? undefined : "BLUE",
      };
      if (ending === "board win") {
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
      expect(store.deleteActiveMatches).toHaveBeenCalledWith(row.id);
    }
  );
});
