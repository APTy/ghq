import type { GameoverReason, GameoverState } from "./engine";
import type { Player, Turn } from "./engine-v2";
import { allowedMoveFromUci, allowedMoveToUci } from "./notation-uci";

export function createPGN(turns: Turn[]): string {
  return turns.map((turn) => turnToString(turn)).join("\n");
}

export function turnToString(turn: Turn): string {
  const moves = turn.moves.map((move) => allowedMoveToUci(move)).join(" ");
  if (turn.status) {
    let outcome = "1/2-1/2";
    if (turn.winner === "BLUE") {
      outcome = "0-1";
    } else if (turn.winner === "RED") {
      outcome = "1-0";
    }
    // A terminal capture or draw acceptance still belongs to its played turn.
    // Keep its actions and clock in the same record so reload can replay them.
    if (turn.moves.length) {
      return `${turn.turn}. ${moves} ${outcome} {[%emt ${turn.elapsedSecs.toFixed(1)}] [%sts ${turn.status}]}`;
    }
    return `${outcome} {[%sts ${turn.status}]}`;
  }
  return `${turn.turn}. ${moves} {[%emt ${turn.elapsedSecs.toFixed(1)}]}`;
}

function parseTurn(line: string, turn: number): Turn {
  const comments = Array.from(
    line.matchAll(/\{([^}]*)\}/g),
    (match) => match[1]
  ).join(" ");
  const elapsedMatch = comments.match(/\[%emt\s+([\d.]+)\]/);
  const statusMatch = comments.match(/\[%sts\s+([\w-]+)\]/);
  const tokens = line
    .replace(/\{[^}]*\}/g, "")
    .trim()
    .replace(/^\d+\.\s*/, "")
    .split(/\s+/)
    .filter(Boolean);
  const outcome = tokens[tokens.length - 1];
  if (["1-0", "0-1", "1/2-1/2"].includes(outcome)) tokens.pop();
  const terminal = statusMatch
    ? gameEndTurn(turn, outcome, statusMatch[1] as GameoverReason)
    : undefined;
  return {
    ...(terminal ?? {}),
    turn,
    moves: tokens.map(allowedMoveFromUci),
    elapsedSecs: elapsedMatch ? parseFloat(elapsedMatch[1]) : 0,
  };
}

export function stringToTurn(line: string): Turn {
  const number = line.trim().match(/^(\d+)\./);
  return parseTurn(line, number ? parseInt(number[1]) : 1);
}

export function pgnToTurns(pgn: string): Turn[] {
  const lines = pgn
    .trim()
    .split("\n")
    .filter((line) => line.trim() !== "");

  return lines.map((line, index) => parseTurn(line, index + 1));
}

export function gameEndTurn(
  turn: number,
  outcome: string,
  status: GameoverReason
): Turn {
  const baseTurn: Turn = {
    turn,
    moves: [],
    elapsedSecs: 0,
    status,
  };

  if (outcome === "1-0") {
    baseTurn.winner = "RED";
  } else if (outcome === "0-1") {
    baseTurn.winner = "BLUE";
  }

  return baseTurn;
}

export function resignationTurn(turn: number, playerResigned: Player): Turn {
  return {
    turn,
    moves: [],
    elapsedSecs: 0,
    status: "resign",
    winner: playerResigned === "RED" ? "BLUE" : "RED",
  };
}

export function gameEndTurnFromGameover(
  turn: number,
  gameover: GameoverState
): Turn {
  return {
    turn,
    moves: [],
    elapsedSecs: 0,
    status: gameover.reason,
    winner: gameover.winner,
  };
}
