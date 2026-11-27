import { TimeControl } from "@/game/constants";
import { GameClient, GameEngine, Turn } from "@/game/engine-v2";
import { SendTurnRequest } from "@/lib/api";
import Router from "@koa/router";
import { Server } from "boardgame.io";
import { nanoid } from "nanoid";
import { PassThrough } from "stream";
import { MatchV3, MatchV3Info, User } from "@/lib/types";
import { createPGN, gameEndTurnFromGameover, pgnToTurns } from "@/game/pgn";
import {
  createActiveMatches,
  createMatchV3,
  fetchMatchV3,
  getActiveMatch,
  listInProgressLiveMatches,
  onMatchChange,
  updateMatchPGN,
} from "./matchv3-store";
import { createHash } from "node:crypto";
import { GameoverState } from "@/game/engine";
import { SupabaseMatch } from "./supabase";
import { finalizeMatch } from "./match-finalization";
import { allowedMoveToUci } from "@/game/notation-uci";
import { isTurnAuthorized } from "./turn-processing";

interface Listener {
  id: string;
  stream: PassThrough;
  gameId: string;
  lastTurnIndex: number;
}

const gameIdsToListenerIds: Record<string, Set<string>> = {};
const listeners: Record<string, Listener> = {};

export function sendTurnsToListeners(gameId: string, turns: Turn[]) {
  const listenerIds = gameIdsToListenerIds[gameId];
  if (!listenerIds) {
    // console.error("no listeners found for game", gameId);
    return;
  }
  // console.log("sending turns to listeners", listenerIds);

  for (const listenerId of listenerIds) {
    sendGameTurns(listenerId, turns);
  }
}

export function sendGameTurns(listenerId: string, turns: Turn[]) {
  const listener = listeners[listenerId];

  if (!listener) {
    console.error("listener is not initialized for client:", listenerId);
    return;
  }

  const newTurns = turns.slice(listener.lastTurnIndex);

  if (newTurns.length === 0) {
    return;
  }

  const data = {
    turns: newTurns,
  };

  // console.log("sending turns to listener", listenerId, {
  //   turns: newTurns,
  // });

  const formattedData = `data: ${JSON.stringify(data)}\n\n`;
  listener.stream.write(formattedData);

  listener.lastTurnIndex = turns.length;
}

async function sendInitialTurns(listenerId: string, gameId: string) {
  const game = await fetchMatchV3(gameId);
  if (!game) {
    return;
  }

  const turns = pgnToTurns(game.pgn);
  sendGameTurns(listenerId, turns);
}

export function addGameServerRoutes(
  router: Router<any, Server.AppCtx>,
  engine: GameEngine
) {
  onMatchChange((match) => {
    const turns = pgnToTurns(match.pgn);
    sendTurnsToListeners(match.id, turns);
  });

  const runMatchLifecycle = () => {
    matchLifecycleV3(engine).finally(() => {
      setTimeout(runMatchLifecycle, 5_000);
    });
  };

  if (process.env.NODE_ENV !== "development") {
    runMatchLifecycle();
  }

  router.get("/v3/match/:id", async (ctx) => {
    const gameId = ctx.params.id as string;
    const game = await fetchMatchV3(gameId);
    if (!game) {
      ctx.throw(404, "Game not found");
      return;
    }
    const userId = ctx.state?.auth?.userId;
    const matchInfo = await getV3MatchInfo(gameId, userId);
    if (!matchInfo) {
      ctx.throw(404, "Game not found");
      return;
    }

    ctx.body = JSON.stringify(matchInfo);
  });

  router.get("/v3/match/:id/turns", async (ctx) => {
    const gameId = ctx.params.id as string;
    const game = await fetchMatchV3(gameId);
    if (!game) {
      ctx.throw(404, "Game not found");
      return;
    }

    const listenerId = nanoid();
    const stream = new PassThrough();

    ctx.set({
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    });

    ctx.body = stream;

    const heartbeatInterval = setInterval(() => {
      stream.write(": keepalive\n\n");
    }, 20_000);

    listeners[listenerId] = {
      id: listenerId,
      stream,
      gameId,
      lastTurnIndex: 0,
    };

    if (!gameIdsToListenerIds[gameId]) {
      gameIdsToListenerIds[gameId] = new Set();
    }
    gameIdsToListenerIds[gameId].add(listenerId);

    ctx.req.on("close", () => {
      clearInterval(heartbeatInterval);
      stream.end();
      delete listeners[listenerId];
      gameIdsToListenerIds[gameId]?.delete(listenerId);
    });

    await sendInitialTurns(listenerId, gameId);
  });

  router.post("/v3/match/:id/turns", async (ctx) => {
    const userId = ctx.state.auth?.userId;
    if (!userId) {
      ctx.throw(401, "Unauthorized");
      return;
    }

    const id = ctx.params.id as string;
    const match = await fetchMatchV3(id);
    if (!match) {
      ctx.throw(404, "Game not found");
      return;
    }

    if (match.status) {
      ctx.throw(409, "Game is already complete");
      return;
    }
    const turnReq = ctx.request.body as SendTurnRequest;

    if (!turnReq.credentials) {
      ctx.throw(400, "Credentials are required");
      return;
    }
    if (!turnReq.playerId) {
      ctx.throw(400, "Player ID is required");
      return;
    }
    if (!turnReq.turn) {
      ctx.throw(400, "Turn is required");
      return;
    }

    if (!(await isTurnAuthorized(userId, turnReq, match))) {
      ctx.throw(401, "Unauthorized");
      return;
    }

    const { turn } = turnReq;

    await updateMatchPGN(id, (match): SupabaseMatch => {
      const turns = pgnToTurns(match.pgn || "");

      const timeControl = getSupabaseMatchTimeControl(match);

      // TODO(tyler): can we cache this game client per match id?
      const game = new GameClient({
        engine,
        fen: match.starting_fen || undefined,
        isPassAndPlayMode: true,
        timeControl,
        gameStartTimeMs: getGameStartTimeMs(match?.created_at),
      });

      // Apply the historical moves to the game state
      try {
        for (const turn of turns) {
          game.pushTurn(turn);
        }
      } catch (error) {
        console.error("Error applying moves to game", error);
        ctx.throw(400, "Invalid move");
      }

      const getWinnerId = (gameover: GameoverState) => {
        return gameover.winner
          ? gameover.winner === "RED"
            ? match.player0_id
            : match.player1_id
          : null;
      };

      // Check for gameover due to timeout or something already on the board.
      const gameoverDueToTimeout = game.getGameover();
      if (gameoverDueToTimeout) {
        if (!turns[turns.length - 1]?.status) {
          turns.push(gameEndTurnFromGameover(turn.turn, gameoverDueToTimeout));
        }
        return {
          ...match,
          pgn: createPGN(turns),
          status: gameoverDueToTimeout.status,
          winner_id: getWinnerId(gameoverDueToTimeout),
          gameover_reason: gameoverDueToTimeout.reason,
          current_turn_player_id: null,
        };
      }

      // debugPrintGameState(game, turn);

      if (turn.turn !== game.turn) ctx.throw(409, "Unexpected turn number");
      // Only resignation is an explicit result. Other results must be derived
      // from the authoritative board, not a client-supplied winner/status.
      const appliedTurn =
        turn.status === "resign"
          ? turn
          : {
              ...turn,
              status: undefined,
              winner: undefined,
            };
      game.pushTurn(appliedTurn);

      // TODO(tyler): validate that time is accurate within 1 second of the actual time

      // TODO(tyler): set a time to check back for game end based on the upcoming player's time left

      turns.push(appliedTurn);

      const updatedMatch = { ...match };

      // Then check gameover again because there could have been bombardment or something else that ended the game.
      const gameover = game.getGameover();
      if (gameover) {
        updatedMatch.status = gameover.status;
        updatedMatch.winner_id = getWinnerId(gameover);
        updatedMatch.gameover_reason = gameover.reason;
        if (!appliedTurn.status) {
          turns.push(gameEndTurnFromGameover(turn.turn + 1, gameover));
        }
      }
      updatedMatch.pgn = createPGN(turns);

      // Update the current player turn user id
      updatedMatch.current_turn_player_id = gameover
        ? null
        : game.currentPlayerTurn() === "RED"
        ? match.player0_id
        : match.player1_id;

      return updatedMatch;
    });

    ctx.body = JSON.stringify({ success: true });
  });
}

export interface CreateNewV3MatchOptions {
  user0: User;
  user1: User;
  timeControlName: string;
  timeControl: TimeControl;
  rated: boolean;
  isCorrespondence: boolean;
  startingFen?: string;
}

export async function createNewV3Match({
  user0,
  user1,
  timeControlName,
  timeControl,
  rated,
  startingFen,
  isCorrespondence,
}: CreateNewV3MatchOptions): Promise<void> {
  const matchId = nanoid();
  const player0Creds = nanoid();
  const player1Creds = nanoid();
  const match: MatchV3 = {
    engineVersion: 2,
    id: matchId,
    createdAt: new Date().toISOString(),
    player0UserId: user0.id,
    player0CredentialsHash: hashCredentials(player0Creds),
    player0Elo: user0.elo,
    player1UserId: user1.id,
    player1CredentialsHash: hashCredentials(player1Creds),
    player1Elo: user1.elo,
    currentPlayerTurnUserId: user0.id,
    timeControlName,
    timeControlAllowedTime: timeControl.time,
    timeControlBonus: timeControl.bonus,
    timeControlVariant: timeControl.variant,
    rated,
    isCorrespondence,
    startingFen,
    pgn: "",
  };
  await createMatchV3(match);
  await createActiveMatches({
    matchId,
    player0UserId: user0.id,
    player1UserId: user1.id,
    player0Credentials: player0Creds,
    player1Credentials: player1Creds,
    isCorrespondence,
  });
}

function hashCredentials(credentials: string): string {
  return createHash("sha256").update(credentials).digest("hex");
}

export interface ActiveMatch {
  id: string;
  playerId: "0" | "1";
  credentials: string;
}

export async function getV3MatchInfo(
  id: string,
  userId?: string | null
): Promise<MatchV3Info | null> {
  const match = await fetchMatchV3(id);
  if (!match) {
    return null;
  }

  if (userId) {
    const activeMatch = await getActiveMatch(userId, match.id);
    if (activeMatch) {
      return {
        match,
        playerInfo: {
          playerId: activeMatch.playerId,
          credentials: activeMatch.credentials,
        },
      };
    }
  }
  return { match };
}

export function debugPrintGameState(game: GameClient, turn: Turn) {
  console.log({
    message: "Allowed moves",
    allowedMoves: game
      .getAllowedMoves()
      .map((m) => allowedMoveToUci(m))
      .join(" "),
    currentPlayer: game.currentPlayer(),
    currentPlayerTurn: game.currentPlayerTurn(),
    isMyTurn: game.isMyTurn(),
    needsTurnConfirmation: game.needsTurnConfirmation,
  });
  console.log({
    message: "Pushing turn",
    board: game.fen(),
    turn: turn.moves.map((m) => allowedMoveToUci(m)),
  });
}

export async function matchLifecycleV3(engine: GameEngine) {
  const matches = await listInProgressLiveMatches();

  for (const match of matches) {
    try {
      await checkAndUpdateMatch(engine, match);
    } catch (error) {
      // A turn or another worker may have won the compare-and-swap. Re-read
      // on the next tick; one match must not stop checks for everyone else.
      console.error("Unable to finalize match", match.id, error);
    }
  }
}

function getSupabaseMatchTimeControl(
  match: SupabaseMatch
): TimeControl | undefined {
  if (match.time_control_allowed_time && match.time_control_allowed_time > 0) {
    return {
      time: match.time_control_allowed_time,
      bonus: match.time_control_bonus ?? 0,
      variant: match.time_control_variant || undefined,
    };
  }

  return undefined;
}

function getGameStartTimeMs(createdAt?: string) {
  return createdAt ? new Date(createdAt).getTime() : undefined;
}

async function checkAndUpdateMatch(engine: GameEngine, match: SupabaseMatch) {
  // Evaluate the batch snapshot without another read for each active game.
  const turns = pgnToTurns(match.pgn || "");
  const game = new GameClient({
    engine,
    fen: match.starting_fen || undefined,
    isPassAndPlayMode: true,
    timeControl: getSupabaseMatchTimeControl(match),
    gameStartTimeMs: getGameStartTimeMs(match.created_at),
  });
  for (const turn of turns) game.pushTurn(turn);
  const gameover = game.getGameover();
  if (!gameover) return;
  if (!turns[turns.length - 1]?.status) {
    turns.push(gameEndTurnFromGameover(turns.length + 1, gameover));
  }
  // Only terminal candidates touch the database again. Finalization checks
  // the snapshot PGN and status under lock; stale results are rejected.
  await finalizeMatch(match, {
    ...match,
    pgn: createPGN(turns),
    status: gameover.status,
    winner_id:
      gameover.winner === "RED"
        ? match.player0_id
        : gameover.winner === "BLUE"
        ? match.player1_id
        : null,
    gameover_reason: gameover.reason,
    current_turn_player_id: null,
  });
}
