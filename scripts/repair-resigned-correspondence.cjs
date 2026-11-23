// One-off repair for the confirmed resignation in this unrated v2 match.
// Dry run: node scripts/repair-resigned-correspondence.cjs
// Apply:   node scripts/repair-resigned-correspondence.cjs --apply
require("dotenv").config();
const { createClient } = require("@supabase/supabase-js");

const id = "OHR9HfTzgdWdc_s_fv43Y";
const supabase = createClient(
  "https://wjucmtrnmjcaatbtktxo.supabase.co",
  process.env.SUPABASE_SECRET_KEY
);
const fields = "id,status,winner_id,gameover_reason,current_turn_player_id";

async function main() {
  const { data: match, error } = await supabase
    .from("matches")
    .select(`${fields},player1_id,rated,is_correspondence,pgn`)
    .eq("id", id)
    .single();
  if (error) throw error;
  if (
    match.rated !== false ||
    match.is_correspondence !== true ||
    !/0-1 \{\[%sts resign\]\}\s*$/.test(match.pgn || "")
  ) {
    throw new Error(
      "Match no longer matches the confirmed unrated resignation; refusing repair"
    );
  }
  const patch = {
    status: "WIN",
    winner_id: match.player1_id,
    gameover_reason: "resign",
    current_turn_player_id: null,
  };
  const alreadyRepaired = Object.entries(patch).every(
    ([key, value]) => match[key] === value
  );
  if (
    !alreadyRepaired &&
    (match.status !== null ||
      match.winner_id !== null ||
      match.gameover_reason !== null)
  ) {
    throw new Error("Unexpected completion metadata; refusing to overwrite it");
  }
  console.log(JSON.stringify({ id, alreadyRepaired, patch }));
  if (!process.argv.includes("--apply")) return;

  if (!alreadyRepaired) {
    const { data, error } = await supabase
      .from("matches")
      .update(patch)
      .eq("id", id)
      .eq("pgn", match.pgn)
      .is("status", null)
      .is("winner_id", null)
      .is("gameover_reason", null)
      .select(fields);
    if (error) throw error;
    if (data.length !== 1)
      throw new Error("Match changed concurrently; repair stopped");
  }
  // Same cleanup as normal server finalization. This is safe to retry.
  const { error: cleanupError } = await supabase
    .from("active_user_matches")
    .delete()
    .eq("match_id", id);
  if (cleanupError) throw cleanupError;

  const { data: verified, error: verifyError } = await supabase
    .from("matches")
    .select(`${fields},pgn`)
    .eq("id", id)
    .single();
  if (verifyError) throw verifyError;
  if (
    verified.pgn !== match.pgn ||
    !Object.entries(patch).every(([key, value]) => verified[key] === value)
  ) {
    throw new Error("Completion verification failed");
  }
  const { count, error: activeError } = await supabase
    .from("active_user_matches")
    .select("match_id", { count: "exact", head: true })
    .eq("match_id", id);
  if (activeError) throw activeError;
  if (count !== 0) throw new Error("Active match cleanup verification failed");
  delete verified.pgn;
  console.log(
    JSON.stringify({ verified, pgnUnchanged: true, activeMatchCount: count })
  );
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
