import { describe, expect, it } from "@jest/globals";
import { matchPath, LATEST_ENGINE_VERSION } from "./match-engine";

describe("match routing", () => {
  it("routes legacy and latest games by stored version, independently of rating", () => {
    expect(matchPath("old-rated", 1)).toBe("/live/old-rated");
    expect(matchPath("old-unrated", 1)).toBe("/live/old-unrated");
    expect(matchPath("new-rated", LATEST_ENGINE_VERSION)).toBe(
      "/game/new-rated"
    );
    expect(matchPath("old-v2-unrated", 2)).toBe("/game/old-v2-unrated");
  });
  it("fails closed for missing metadata instead of guessing from rated", () => {
    expect(() => matchPath("id", undefined as any)).toThrow();
  });
});
