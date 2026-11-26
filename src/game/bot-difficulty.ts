export const BOT_DIFFICULTIES = {
  easy: { label: "Easy", timeMs: 300, maxDepth: 1, beamWidth: 4 },
  medium: { label: "Medium", timeMs: 1200, maxDepth: 2, beamWidth: 6 },
  hard: { label: "Hard", timeMs: 2400, maxDepth: 3, beamWidth: 8 },
} as const;

export type BotDifficulty = keyof typeof BOT_DIFFICULTIES;

export function parseBotDifficulty(value: string | null): BotDifficulty {
  return value === "easy" || value === "hard" || value === "medium"
    ? value
    : "medium";
}
