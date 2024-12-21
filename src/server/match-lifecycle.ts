import { Player } from "@/game/engine";
import { GHQState } from "@/game/engine";
import { checkTimeForGameover } from "@/game/gameover-logic";
import { SupabaseClient } from "@supabase/supabase-js";
import { StorageAPI } from "boardgame.io";

export async function matchLifecycle({
  db,
  supabase,
  onGameEnd,
}: {
  db: StorageAPI.Async | StorageAPI.Sync;
  supabase: SupabaseClient;
  onGameEnd: ({ ctx, G }: { ctx: any; G: GHQState }) => void;
}) {
  // There isn't a way for the server/master to periodically check and update game state,
  // so we're breaking the database abstraction and doing it manually.
  // For more info, see: https://github.com/boardgameio/boardgame.io/issues/92
  const matchIds = await db.listMatches({
    where: { isGameover: false },
  });

  for (const matchId of matchIds) {
    await checkAndUpdateMatch({ db, supabase, matchId, onGameEnd });
  }
}

export async function checkAndUpdateMatch({
  db,
  supabase,
  matchId,
  onGameEnd,
}: {
  db: StorageAPI.Async | StorageAPI.Sync;
  supabase: SupabaseClient;
  matchId: string;
  onGameEnd: ({ ctx, G }: { ctx: any; G: GHQState }) => void;
}) {
  const { data: matchData, error: matchError } = await supabase
    .from("matches")
    .select("status")
    .eq("id", matchId)
    .single();

  if (matchError) {
    console.log({
      message: "Error fetching match",
      matchId,
      matchError,
    });
    return;
  }

  const { state, metadata } = await db.fetch(matchId, {
    state: true,
    metadata: true,
  });

  // If the match has been aborted, update the gameover state (so the game framework understands the game is over).
  if (matchData.status === "ABORTED") {
    metadata.gameover = { status: matchData.status };
    state.ctx.gameover = { status: matchData.status };
    await db.setMetadata(matchId, metadata);
    await db.setState(matchId, state);
    console.log(`Updated game match to be aborted for matchId=${matchId}.`);
    return;
  }

  // TODO(tyler): mark game as abandoned early if player disconnected for 30+ seconds

  // Let's look at the state and see if it's gameover
  const currentPlayer: Player =
    state.ctx.currentPlayer === "0" ? "RED" : "BLUE";
  const gameover = checkTimeForGameover({ G: state.G, currentPlayer });
  if (gameover) {
    metadata.gameover = gameover;
    metadata.updatedAt = Date.now();
    state.ctx.gameover = gameover;
    state._stateID += 1; // Needed so that setState() works.
    await db.setState(matchId, state);
    await db.setMetadata(matchId, metadata);
    console.log(`Updated gameover state for matchId=${matchId}.`);
    onGameEnd({ ctx: state.ctx, G: state.G });
  }
}
