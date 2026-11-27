// Use a disposable local cluster; never reads .env or connects to Supabase.
// GHQ_TEST_PG_SOCKET=/tmp/.../socket node --import tsx scripts/test-match-finalization.ts
import { Client, Pool } from "pg";
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { finalizeMatch } from "../src/server/match-finalization";
import { SupabaseMatch } from "../src/server/supabase";

const socket = process.env.GHQ_TEST_PG_SOCKET;
if (!socket?.startsWith("/tmp/"))
  throw new Error("Disposable local socket required");
const options = { host: socket, port: 55439, user: process.env.USER };
const database = `ghq_test_${Date.now()}`;
const admin = new Client({ ...options, database: "postgres" });
const pool = new Pool({ ...options, database });

async function main() {
  await admin.connect();
  await admin.query(`CREATE DATABASE "${database}"`);
  try {
    await pool.query(`
      CREATE TABLE users (id text PRIMARY KEY, username text, elo numeric, "gamesThisMonth" numeric DEFAULT 0, badge text);
      CREATE TABLE matches (
        id text PRIMARY KEY, player0_id text REFERENCES users, player1_id text REFERENCES users,
        player_0_credentials_hash text, player_1_credentials_hash text, pgn text,
        status text, winner_id text, gameover_reason text, current_turn_player_id text, rated boolean DEFAULT true
      );
      CREATE TABLE active_user_matches (match_id text REFERENCES matches, user_id text);
      INSERT INTO users(id, elo) VALUES ('red',1000),('blue',1000);
      INSERT INTO matches(id, rated) VALUES ('old-rated',true),('old-unrated',false);
      INSERT INTO matches(id, rated, player_0_credentials_hash, player_1_credentials_hash, pgn)
        VALUES ('v2-unrated',false,'a','b',''),('v2-rated',true,'a','b','1. skip');
    `);
    await pool.query(
      readFileSync(
        `${__dirname}/../migrations/002_match_engine_version.sql`,
        "utf8"
      )
    );
    assert.deepEqual(
      (await pool.query("select id, engine_version from matches order by id"))
        .rows,
      [
        { id: "old-rated", engine_version: 1 },
        { id: "old-unrated", engine_version: 1 },
        { id: "v2-rated", engine_version: 2 },
        { id: "v2-unrated", engine_version: 2 },
      ]
    );
    async function seed(id: string, rated = true) {
      const { rows } = await pool.query<SupabaseMatch>(
        "insert into matches(id,engine_version,player0_id,player1_id,pgn,rated,current_turn_player_id) values ($1,2,'red','blue','',$2,'red') returning *",
        [id, rated]
      );
      await pool.query(
        "insert into active_user_matches values ($1,'red'),($1,'blue')",
        [id]
      );
      return rows[0];
    }
    const finish = (
      match: SupabaseMatch,
      status = "WIN",
      winner: string | null = "blue"
    ) =>
      finalizeMatch(
        match,
        {
          ...match,
          status,
          winner_id: winner,
          pgn: "terminal",
          gameover_reason: "resign",
        },
        pool
      );
    const resigned = await seed("resigned");
    const result = await finish(resigned);
    assert.equal(result.current_turn_player_id, null);
    assert.deepEqual(
      (
        await pool.query(
          'select id,elo,"gamesThisMonth" as games,badge from users order by id'
        )
      ).rows,
      [
        { id: "blue", elo: "1016", games: "1", badge: "recruit" },
        { id: "red", elo: "984", games: "1", badge: "recruit" },
      ]
    );
    assert.equal(
      (
        await pool.query(
          "select count(*) from active_user_matches where match_id='resigned'"
        )
      ).rows[0].count,
      "0"
    );
    await assert.rejects(finish(resigned), /complete/);
    await finish(await seed("unrated", false));
    await finish(await seed("abort"), "ABORTED", null);
    const games = async () =>
      (await pool.query('select sum("gamesThisMonth") as games from users'))
        .rows[0].games;
    assert.equal(await games(), "2");
    await finish(await seed("draw"), "DRAW", null);
    assert.equal(await games(), "4");
    const stale = await seed("stale");
    await assert.rejects(finish({ ...stale, pgn: "wrong" }), /concurrently/);
    await assert.rejects(finish(stale, "WIN", "outsider"), /Invalid terminal/);

    // Two completions of the same snapshot must only apply ratings once.
    const racing = await seed("race");
    const raced = await Promise.allSettled([
      finish(racing),
      finish(racing, "WIN", "red"),
    ]);
    assert.equal(raced.filter((r) => r.status === "fulfilled").length, 1);
    assert.equal(await games(), "6");
    const a = await seed("parallel-a"),
      b = await seed("parallel-b");
    await Promise.all([finish(a), finish(b)]);
    assert.equal(await games(), "10");

    // Force the final write to fail after rating writes: all changes roll back.
    const failed = await seed("rollback");
    await pool.query(
      "ALTER TABLE matches ADD CONSTRAINT reject_test_result CHECK (pgn <> 'fail')"
    );
    await assert.rejects(
      finalizeMatch(
        failed,
        { ...failed, pgn: "fail", status: "WIN", winner_id: "blue" },
        pool
      )
    );
    assert.equal(await games(), "10");
    assert.equal(
      (await pool.query("select status from matches where id='rollback'"))
        .rows[0].status,
      null
    );
    assert.equal(
      (
        await pool.query(
          "select count(*) from active_user_matches where match_id='rollback'"
        )
      ).rows[0].count,
      "2"
    );
    console.log(
      "PASS: version backfill, win/draw/unrated/abort, duplicate results, stale PGN, concurrent ratings, failure rollback and cleanup"
    );
  } finally {
    await pool.end();
    await admin.query(`DROP DATABASE "${database}"`);
    await admin.end();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
