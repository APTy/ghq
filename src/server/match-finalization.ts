import { Pool } from "pg";
import { calculateElo } from "../game/elo";
import { User } from "../lib/types";
import { updateUserStats } from "./user-stats";
import { SupabaseMatch } from "./supabase";
import { getPostgresPool } from "./postgres";

/** Keep the result, ratings and active-match cleanup atomic and retry-safe. */
export async function finalizeMatch(
  expected: SupabaseMatch,
  result: SupabaseMatch,
  pool: Pool = getPostgresPool()
): Promise<SupabaseMatch> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const { rows } = await client.query<SupabaseMatch>(
      "SELECT * FROM matches WHERE id = $1 FOR UPDATE",
      [expected.id]
    );
    const match = rows[0];
    if (
      !match ||
      match.engine_version !== 2 ||
      match.status !== null ||
      match.pgn !== expected.pgn
    ) {
      throw new Error(
        `Match ${expected.id} changed concurrently or is complete`
      );
    }
    const { status, winner_id: winner } = result;
    if (
      !status ||
      !["WIN", "DRAW", "ABORTED"].includes(status) ||
      (status === "WIN"
        ? !winner || ![match.player0_id, match.player1_id].includes(winner)
        : winner != null)
    ) {
      throw new Error("Invalid terminal result");
    }

    if (match.rated && status !== "ABORTED") {
      // Stable ordering prevents deadlocks between correspondence games with
      // shared players. Read their current ratings while holding these locks.
      const { rows: players } = await client.query(
        'SELECT id, username, elo, "gamesThisMonth", badge FROM users WHERE id = ANY($1::text[]) ORDER BY id FOR UPDATE',
        [[match.player0_id, match.player1_id]]
      );
      if (players.length !== 2 || players.some((p) => p.elo == null))
        throw new Error("Missing player rating");
      const users: User[] = players.map((p) => ({
        ...p,
        elo: Number(p.elo),
        gamesThisMonth: Number(p.gamesThisMonth ?? 0),
      }));
      const red = users.find((p) => p.id === match.player0_id)!;
      const blue = users.find((p) => p.id === match.player1_id)!;
      const score = status === "DRAW" ? 0.5 : winner === red.id ? 1 : 0;
      const ratings = [
        calculateElo(red.elo, blue.elo, score),
        calculateElo(blue.elo, red.elo, 1 - score),
      ];
      for (const [index, user] of [red, blue].entries()) {
        updateUserStats(user);
        await client.query(
          'UPDATE users SET elo = $2, "gamesThisMonth" = $3, badge = $4 WHERE id = $1',
          [user.id, ratings[index], user.gamesThisMonth, user.badge ?? null]
        );
      }
    }

    const saved = await client.query<SupabaseMatch>(
      `UPDATE matches SET pgn = $2, status = $3, winner_id = $4,
        gameover_reason = $5, current_turn_player_id = NULL WHERE id = $1 RETURNING *`,
      [match.id, result.pgn, status, winner, result.gameover_reason]
    );
    await client.query("DELETE FROM active_user_matches WHERE match_id = $1", [
      match.id,
    ]);
    await client.query("COMMIT");
    return saved.rows[0];
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}
