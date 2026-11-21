"use client";

import { useEffect, useRef, useState } from "react";
import { CpuAnalysisClient, type AnalysisResult } from "@/game/analysis";
import { allowedMoveFromUci } from "@/game/notation-uci";
import { coordinateToAlgebraic, degreesToCardinal } from "@/game/notation";

const DEBOUNCE_MS = 250;

/** Distinguish rotations, deployments, and free captures in turn plans. */
export function describeAnalysisMove(uci: string): string {
  try {
    const move = allowedMoveFromUci(uci);
    if (move.name === "Skip") return "End turn";
    if (move.name === "Reinforce") {
      const [unit, to, capture] = move.args;
      return `Deploy ${unit
        .toLowerCase()
        .replace(/_/g, " ")} at ${coordinateToAlgebraic(to)}${
        capture ? `; capture ${coordinateToAlgebraic(capture)}` : ""
      }`;
    }
    if (move.name === "AutoCapture") {
      const [kind, at] = move.args;
      return `${
        kind === "bombard" ? "Bombard" : "Capture"
      } ${coordinateToAlgebraic(at!)} (free)`;
    }
    if (move.name === "MoveAndOrient") {
      const [from, to, orientation] = move.args;
      const start = coordinateToAlgebraic(from);
      const end = coordinateToAlgebraic(to);
      const facing = degreesToCardinal(orientation!);
      return start === end
        ? `Rotate ${start} ${facing}`
        : `${start} → ${end}, face ${facing}`;
    }
    if (move.name === "Move") {
      const [from, to, capture] = move.args;
      return `${coordinateToAlgebraic(from)} → ${coordinateToAlgebraic(to)}${
        capture ? `; capture ${coordinateToAlgebraic(capture)}` : ""
      }`;
    }
    return uci;
  } catch {
    // A future engine action should remain readable even before UI support lands.
    return uci;
  }
}

function scoreLabel(score: number): string {
  if (Math.abs(score) >= 10000)
    return `${score > 0 ? "Red" : "Blue"} winning line`;
  return `${score > 0 ? "+" : ""}${score.toFixed(1)}`;
}

function positionLabel(result: AnalysisResult): string {
  if (result.terminal) {
    if (!result.terminal.winner) return "Draw";
    return result.terminal.winner === "RED" ? "Red wins" : "Blue wins";
  }
  if (Math.abs(result.score) >= 10000) {
    return `Winning line for ${result.score > 0 ? "Red" : "Blue"} (selective)`;
  }
  if (Math.abs(result.score) < 0.05) return "Balanced";
  return `${result.score > 0 ? "Red" : "Blue"} advantage`;
}

/** A null state hides analysis and prevents worker creation. */
export default function AnalysisPanel({
  state,
  isReplay,
}: {
  state: string | null;
  isReplay: boolean;
}) {
  const clientRef = useRef<CpuAnalysisClient | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<{
    state: string | null;
    phase: "searching" | "complete" | "error";
  }>({ state: null, phase: "searching" });
  const [request, setRequest] = useState({ state: "", number: 0 });

  useEffect(() => {
    return () => {
      clientRef.current?.dispose();
      clientRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (state === null) {
      clientRef.current?.cancel();
      setResult(null);
      return;
    }

    let current = true;
    setStatus({ state, phase: "searching" });

    const timer = setTimeout(async () => {
      try {
        const client = (clientRef.current ??= new CpuAnalysisClient());
        const receive = (update: AnalysisResult) => {
          if (current && update.state === state) setResult(update);
        };
        const completed = await client.analyze(
          state,
          {
            timeMs:
              request.state === state
                ? request.number === 1
                  ? 6000
                  : 15000
                : 1800,
            maxDepth:
              request.state === state ? (request.number === 1 ? 4 : 5) : 3,
            multiPv: 3,
          },
          receive
        );
        if (!current) return;
        receive(completed);
        setStatus({ state, phase: "complete" });
      } catch (error: unknown) {
        if (!current || (error instanceof Error && error.name === "AbortError"))
          return;
        setStatus({ state, phase: "error" });
      }
    }, DEBOUNCE_MS);

    return () => {
      current = false;
      clearTimeout(timer);
      clientRef.current?.cancel();
    };
  }, [state, request]);

  if (state === null) {
    return null;
  }

  // Retain the bar while searching, but never show another position's moves.
  const visibleResult = result?.state === state ? result : null;
  const phase = status.state === state ? status.phase : "searching";
  const searching = phase === "searching";
  const stale = result !== null && result.state !== state;
  const redWidth = result
    ? result.terminal
      ? result.terminal.winner === "RED"
        ? 100
        : result.terminal.winner === "BLUE"
        ? 0
        : 50
      : 50 * (1 + Math.tanh(result.score / 5))
    : 50;
  const label = result
    ? positionLabel(result)
    : "Analyzing position";
  const evaluationText =
    label +
    (result &&
    !result.terminal &&
    Math.abs(result.score) < 10000
      ? ` · ${scoreLabel(result.score)}`
      : "");

  return (
    <section
      aria-label="Engine analysis"
      className="flex h-64 min-w-0 flex-col break-words border-y border-slate-200 text-slate-800"
    >
      <div
        role="img"
        aria-label={`Heuristic evaluation: ${evaluationText}.${stale ? " Previous position." : ""} Positive scores favor Red.`}
        aria-busy={searching}
        className="relative h-5 shrink-0 overflow-hidden bg-blue-300 transition-opacity duration-200"
        style={{ opacity: searching || stale ? 0.55 : 1 }}
        title="Heuristic advantage, not a winning probability. Positive scores favor Red; one point is approximately one infantry unit."
      >
        <div
          className="absolute inset-y-0 left-0 bg-red-300 transition-[width] duration-300"
          style={{ width: `${redWidth}%` }}
        />
        <div className="relative text-center text-xs font-semibold leading-5 text-slate-800">
          {evaluationText}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto px-3 py-2 [scrollbar-gutter:stable]">
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="font-semibold">
            {isReplay ? "Replay analysis" : "Position analysis"}
          </span>
          <span role="status" className="text-slate-500">
            {phase === "error"
              ? "Analysis unavailable"
              : searching
              ? "Searching…"
              : "Search complete"}
          </span>
        </div>

        {!visibleResult && phase !== "error" && (
          <p className="text-xs text-slate-500">
            Preparing engine and searching turn plans…
          </p>
        )}
        {phase === "error" && (
          <p className="text-xs text-slate-500">
            The engine could not finish. You can retry analysis.
          </p>
        )}

        {visibleResult && (
          <>
            <p className="text-[11px] text-slate-500">
              Heuristic score · positive favors Red
              {searching && " · provisional"}
            </p>
            {!visibleResult.terminal && visibleResult.lines.length > 0 && (
              <div>
                <div className="mb-1 text-xs font-medium">
                  Candidate turn plans
                </div>
                <ol className="space-y-1">
                  {visibleResult.lines.slice(0, 3).map((line, index) => {
                    const plan = line.turnMoves.length
                      ? line.turnMoves
                      : line.moves.slice(0, 1);
                    const continuation = line.moves.slice(
                      line.turnMoves.length
                    );
                    return (
                      <li key={`${index}-${plan.join(" ")}`}>
                        <details className="rounded border border-slate-200 bg-white/70 text-xs">
                          <summary className="cursor-pointer px-2 py-1.5">
                            <span className="font-semibold">
                              {index + 1}.{" "}
                              {describeAnalysisMove(plan[0] ?? "skip")}
                            </span>
                            <span className="ml-2 whitespace-nowrap tabular-nums text-slate-500">
                              {scoreLabel(line.score)}
                            </span>
                            {plan.length > 1 && (
                              <span className="ml-1 text-slate-500">
                                + {plan.length - 1} more
                              </span>
                            )}
                          </summary>
                          <div className="space-y-2 border-t border-slate-100 px-3 py-2">
                            <ol className="list-inside list-decimal space-y-1">
                              {plan.map((move, step) => (
                                <li key={`${step}-${move}`}>
                                  {describeAnalysisMove(move)}
                                </li>
                              ))}
                            </ol>
                            {continuation.length > 0 && (
                              <div className="text-slate-500">
                                <span className="font-medium">
                                  Continuation:{" "}
                                </span>
                                {continuation
                                  .map(describeAnalysisMove)
                                  .join("; ")}
                              </div>
                            )}
                          </div>
                        </details>
                      </li>
                    );
                  })}
                </ol>
              </div>
            )}
            <p
              className="text-[11px] text-slate-500"
              title={visibleResult.engineVersion}
            >
              {visibleResult.depth === 0
                ? "Initial estimate"
                : `${visibleResult.depth} turn${
                    visibleResult.depth === 1 ? "" : "s"
                  } searched`}
              {visibleResult.depth === 1 &&
                " · opponent reply not yet searched"}
              {" · "}
              {visibleResult.nodes.toLocaleString()} nodes
              {" · "}
              {(visibleResult.elapsedMs / 1000).toFixed(1)}s
              {" · selective search"}
            </p>
          </>
        )}

        {!searching &&
          !visibleResult?.terminal &&
          (phase === "error" ||
            request.state !== state ||
            request.number < 2) && (
            <button
              type="button"
              className="text-xs font-medium text-slate-600 underline underline-offset-2 hover:text-slate-900"
              onClick={() =>
                setRequest((previous) => ({
                  state,
                  number: previous.state === state ? previous.number + 1 : 1,
                }))
              }
            >
              {phase === "error" ? "Retry analysis" : "Analyze deeper"}
            </button>
          )}
      </div>
    </section>
  );
}
