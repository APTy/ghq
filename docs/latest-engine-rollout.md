# Latest-engine multiplayer rollout

New live and correspondence games use the full v2 engine/PGN stack (named
`V3` in the existing multiplayer code), regardless of rating. The play menu
offers rated rapid and blitz. New friend challenges are rated. Existing
unrated games and pending challenges retain their rating setting.

Routes are fixed: `/live/:id` always loads the legacy client; `/game/:id`
always loads the latest client. There are no routing lookups or redirects.
`matches.engine_version` is included in the existing match-list SELECT so
links to old correspondence games and history still use the correct route.
Matchmaking always sends players to `/game/:id`, without a version lookup.
Bot, pass-and-play and the `/learn` board already use the v2 engine.

## Deployment

1. Let all legacy live games finish, then pause game creation/traffic while
   replacing the single server and frontend. Existing correspondence games
   can remain in progress. Resuming legacy live games through matchmaking
   after the rollout is intentionally unsupported.
   Do not run the old application against the completed migration: old writers
   do not explicitly set `engine_version`.
2. Apply `migrations/002_match_engine_version.sql`. It classifies existing
   rows from their credential/PGN storage, without rewriting moves, results,
   ratings or credentials. Ambiguous rows fail the transaction for inspection.
3. Deploy the updated server and frontend together, then resume traffic.
   Preserve the legacy boardgame.io server and `Games` table for existing games.
4. Smoke-test new rated rapid/blitz and correspondence games; open a legacy
   correspondence game and a preexisting unrated v2 game through their list
   links; verify that each opens its original engine.

There are no new database functions or triggers. Result finalization is a
TypeScript transaction in `src/server/match-finalization.ts`, using the same
Elo and stats functions as the application. It commits the result, both player
updates and active-match cleanup together. The expected PGN and locked match
status reject stale/duplicate finalization. The timeout worker evaluates the
existing batch snapshot, with no per-game database reads for ongoing games.
Only terminal candidates enter finalization. A candidate whose PGN/status
changed is rejected and reconsidered from the next batch.

The existing two-second matchmaking poll remains. It performs the existing
active-match lookup and does not fetch engine metadata. Match-list requests
include one additional column but no additional query. Normal game-page
loads use their existing data requests without a preliminary version lookup.

The connection uses the existing `POSTGRES_PASSWORD` and Supabase pooler.
Optional `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER` and
`POSTGRES_CA_CERT` override those settings. The bundled public Supabase CA
allows certificate/hostname verification; TLS verification is not disabled.

## Validation

Run `yarn test --runInBand` for unit/regression coverage. Database integration
coverage uses a disposable local PostgreSQL cluster with a Unix socket under
`/tmp/`, port 55439:

```sh
GHQ_TEST_PG_SOCKET=/tmp/your-cluster/socket node --import tsx scripts/test-match-finalization.ts
```

The script creates and removes its own test database. It does not load `.env`
or connect to production. It tests migration classification, results, stats,
duplicate and concurrent finalization, and transaction rollback on failure.

Rollback requires a compatible application build that still routes by stored
engine version and supports v2 rated games. Do not drop metadata or reroute
already-created games based on `rated`.
