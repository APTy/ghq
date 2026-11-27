export type EngineVersion = 1 | 2;
export const LATEST_ENGINE_VERSION = 2 as const;

export function matchPath(id: string, engineVersion: EngineVersion): string {
  if (engineVersion !== 1 && engineVersion !== 2) {
    throw new Error("Missing or unsupported match engine version");
  }
  return `/${engineVersion === 2 ? "game" : "live"}/${encodeURIComponent(id)}`;
}
