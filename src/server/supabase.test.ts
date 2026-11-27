import { matchToSupabaseMatch, supabaseMatchToMatchV3, SupabaseMatch } from "./supabase";

describe("starting position storage", () => {
  it.each([null, "", "   "])("reads and rewrites blank FEN %p as the default position", (starting_fen) => {
    const row = { starting_fen } as SupabaseMatch;
    const match = supabaseMatchToMatchV3(row);
    expect(match.startingFen).toBeUndefined();
    expect(matchToSupabaseMatch({ ...match, startingFen: starting_fen ?? undefined }).starting_fen).toBeNull();
  });

  it("preserves a custom position", () => {
    const starting_fen = "8/8/8/8/8/8/8/8 I i r 0";
    const match = supabaseMatchToMatchV3({ starting_fen } as SupabaseMatch);
    expect(match.startingFen).toBe(starting_fen);
    expect(matchToSupabaseMatch(match).starting_fen).toBe(starting_fen);
  });
});
