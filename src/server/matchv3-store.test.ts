/** @jest-environment node */
import { describe, expect, it, beforeEach } from "@jest/globals";
declare const jest: typeof import("@jest/globals").jest;
import { getAdminSupabase } from "@/lib/supabase-server";
import {
  updateMatchPGN,
  fetchMatchV3,
  listInProgressLiveMatches,
} from "./matchv3-store";
jest.mock("@/lib/supabase-server");
import { finalizeMatch } from "./match-finalization";
jest.mock("./match-finalization");

describe("v2 persistence", () => {
  const row = {
    id: "game",
    engine_version: 2,
    pgn: "",
    status: null,
    rated: true,
  };
  const terminal = {
    ...row,
    status: "WIN",
    winner_id: "blue",
    gameover_reason: "resign",
    pgn: "0-1 {[%sts resign]}",
  };
  let query: any;
  let db: any;
  beforeEach(() => {
    query = {
      select: jest.fn().mockReturnThis(),
      eq: jest.fn().mockReturnThis(),
      is: jest.fn().mockReturnThis(),
      single: jest.fn(async () => ({ data: row })),
      update: jest.fn().mockReturnThis(),
    };
    db = { from: jest.fn().mockReturnValue(query) };
    jest.mocked(getAdminSupabase).mockReturnValue(db);
    jest
      .mocked(finalizeMatch)
      .mockReset()
      .mockResolvedValue(terminal as any);
  });
  it("commits rated completion through the server transaction, not separate writes", async () => {
    const result = await updateMatchPGN("game", () => terminal as any);
    expect(result.status).toBe("WIN");
    expect(finalizeMatch).toHaveBeenCalledWith(row, terminal);
    expect(query.update).not.toHaveBeenCalled();
  });
  it("rejects duplicate completion before attempting rating updates", async () => {
    query.single.mockResolvedValue({ data: terminal });
    await expect(updateMatchPGN("game", () => terminal as any)).rejects.toThrow(
      "already complete"
    );
    expect(finalizeMatch).not.toHaveBeenCalled();
  });
  it("treats a lost completion race as a conflict", async () => {
    jest
      .mocked(finalizeMatch)
      .mockRejectedValue(new Error("changed concurrently"));
    await expect(updateMatchPGN("game", () => terminal as any)).rejects.toThrow(
      "concurrently"
    );
  });
  it("protects ordinary moves against a concurrent timeout", async () => {
    query.select.mockImplementation(() =>
      query.update.mock.calls.length ? Promise.resolve({ data: [] }) : query
    );
    await expect(
      updateMatchPGN("game", (match) => ({ ...match, pgn: "1. skip" }))
    ).rejects.toThrow();
    expect(query.is).toHaveBeenCalledWith("status", null);
  });
  it("only fetches latest-engine matches", async () => {
    await fetchMatchV3("game");
    expect(query.eq).toHaveBeenCalledWith("engine_version", 2);
  });
  it("only includes v2 live games in the timeout worker", async () => {
    query.eq.mockImplementation((key: string) =>
      key === "engine_version" ? Promise.resolve({ data: [] }) : query
    );
    await listInProgressLiveMatches();
    expect(query.eq).toHaveBeenCalledWith("engine_version", 2);
    expect(query.eq).toHaveBeenCalledWith("is_correspondence", false);
  });
});
