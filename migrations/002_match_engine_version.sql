BEGIN;

-- Apply with game creation paused, then start the updated server/frontend.
-- New writers set engine_version explicitly; there are no triggers/defaults.
ALTER TABLE public.matches ADD COLUMN IF NOT EXISTS engine_version smallint;

-- The full v2 multiplayer store has credential hashes and a PGN (including
-- an empty string before turn one). Rating is independent of storage format.
UPDATE public.matches SET engine_version = CASE
  WHEN player_0_credentials_hash IS NOT NULL AND player_1_credentials_hash IS NOT NULL
    AND pgn IS NOT NULL THEN 2
  WHEN player_0_credentials_hash IS NULL AND player_1_credentials_hash IS NULL THEN 1
  ELSE NULL END
WHERE engine_version IS NULL;

-- Ambiguous records cause this transaction to fail instead of guessing.
ALTER TABLE public.matches ALTER COLUMN engine_version SET NOT NULL;
ALTER TABLE public.matches ADD CONSTRAINT matches_engine_version_check CHECK (engine_version IN (1, 2));

COMMIT;
