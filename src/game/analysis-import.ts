import type { GameClientOptions, GameEngine } from "./engine-v2";
import { pgnToTurns } from "./pgn";
import { allowedMoveToUci } from "./notation-uci";

export type ImportFormat = "PGN" | "FEN";

/** Validate in an isolated board before replacing the visible game. */
export function prepareAnalysisImport(
  engine: GameEngine,
  format: ImportFormat,
  input: string
): Pick<GameClientOptions, "fen" | "initialTurns"> {
  const text = input.trim();
  if (!text) throw new Error(`Paste a ${format} first.`);
  if (format === "FEN") {
    const parts = text.split(/\s+/);
    const ranks = parts[0].split("/");
    const validRanks =
      ranks.length === 8 &&
      ranks.every((rank) => {
        const tokens = rank.match(/[1-8]|[qifprthQIFPRTH][↑↗→↘↓↙←↖]?/g) ?? [];
        return (
          tokens.join("") === rank &&
          tokens.reduce(
            (n, token) => n + (/^[1-8]$/.test(token) ? Number(token) : 1),
            0
          ) === 8
        );
      });
    if (
      !validRanks ||
      parts.length !== 4 ||
      !/^(?:-|[IFPRTH]+)$/.test(parts[1]) ||
      !/^(?:-|[ifprth]+)$/.test(parts[2]) ||
      !/^[rb]$/.test(parts[3])
    ) {
      throw new Error(
        "Invalid FEN. Paste a complete GHQ FEN with eight ranks, both reserves, and the side to move."
      );
    }
    const fen = parts.join(" ");
    engine.BaseBoard(fen);
    return { fen, initialTurns: undefined };
  }

  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  lines.forEach((line, index) => {
    const numbered = line.match(/^(\d+)\.\s/);
    if (
      numbered
        ? Number(numbered[1]) !== index + 1
        : !/^(1-0|0-1|1\/2-1\/2)\s+\{/.test(line)
    ) {
      throw new Error(
        `Invalid PGN at line ${index + 1}. Use the current GHQ export format.`
      );
    }
    if (
      !/\{\[%emt \d+(?:\.\d+)?\](?: \[%sts (?:resign|timeout|hq-capture|double-skip|stalemate|cancelled)\])?\}$/.test(
        line
      ) &&
      !/^(1-0|0-1|1\/2-1\/2) \{\[%sts (?:resign|timeout|hq-capture|double-skip|stalemate|cancelled)\]\}$/.test(
        line
      )
    ) {
      throw new Error(
        `Invalid PGN metadata at line ${
          index + 1
        }. Use the current GHQ export format.`
      );
    }
  });
  const turns = pgnToTurns(text);
  const board = engine.BaseBoard();
  turns.forEach((turn, index) => {
    if (!turn.moves.length && !turn.status) throw new Error(`Turn ${index + 1} has no moves.`);
    if (turn.status && index !== turns.length - 1)
      throw new Error("PGN contains turns after the game result.");
    const red = board.is_red_turn();
    for (const move of turn.moves) {
      const uci = allowedMoveToUci(move);
      const parsed = engine.Move.from_uci(uci);
      if (board.is_red_turn() !== red || !board.is_legal(parsed)) {
        throw new Error(`Illegal move in turn ${index + 1}: ${uci}.`);
      }
      board.push(parsed);
    }
    if (
      !turn.status &&
      index < turns.length - 1 &&
      board.is_red_turn() === red
    ) {
      throw new Error(`Turn ${index + 1} is incomplete.`);
    }
  });
  return { fen: undefined, initialTurns: turns };
}
