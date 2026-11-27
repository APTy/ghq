import { Pool } from "pg";
import { SUPABASE_CA } from "./supabase-ca";

let pool: Pool | undefined;

export function getPostgresPool(): Pool {
  if (!pool) {
    if (!process.env.POSTGRES_PASSWORD)
      throw new Error("POSTGRES_PASSWORD is not set");
    pool = new Pool({
      host: process.env.POSTGRES_HOST ?? "aws-0-us-east-2.pooler.supabase.com",
      port: Number(process.env.POSTGRES_PORT ?? 6543),
      user: process.env.POSTGRES_USER ?? "postgres.wjucmtrnmjcaatbtktxo",
      password: process.env.POSTGRES_PASSWORD,
      database: "postgres",
      ssl: {
        ca: process.env.POSTGRES_CA_CERT ?? SUPABASE_CA,
        rejectUnauthorized: true,
      },
      max: 5,
      connectionTimeoutMillis: 10_000,
      idleTimeoutMillis: 30_000,
    });
  }
  return pool;
}
