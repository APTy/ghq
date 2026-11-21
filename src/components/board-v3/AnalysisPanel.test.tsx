import "@testing-library/jest-dom/jest-globals";
import { afterEach, beforeEach, describe, expect, it } from "@jest/globals";
import type { Mock } from "jest-mock";
import { act, fireEvent, render, screen } from "@testing-library/react";
import AnalysisPanel, { describeAnalysisMove } from "./AnalysisPanel";
import { CpuAnalysisClient, type AnalysisResult } from "@/game/analysis";

// Keep Jest global so Next's SWC transform hoists the module mock.
declare const jest: typeof import("@jest/globals").jest;
jest.mock("@/game/analysis", () => ({ CpuAnalysisClient: jest.fn() }));

function result(state: string, score = 2): AnalysisResult {
  return {
    engineVersion: "test-cpu",
    state,
    score,
    bestMove: "a1a2",
    turnMoves: ["a1a2", "b1b1↗", "ric1"],
    lines: [
      {
        moves: ["a1a2", "b1b1↗", "ric1", "h8h7"],
        turnMoves: ["a1a2", "b1b1↗", "ric1"],
        score,
      },
    ],
    depth: 2,
    nodes: 1234,
    elapsedMs: 1800,
    selective: true,
    terminal: null,
    timedOut: true,
  };
}

describe("AnalysisPanel", () => {
  let analyze: Mock<CpuAnalysisClient["analyze"]>;
  let cancel: Mock;
  let dispose: Mock;

  beforeEach(() => {
    jest.useFakeTimers();
    analyze = jest.fn<CpuAnalysisClient["analyze"]>(
      () => new Promise(() => {})
    );
    cancel = jest.fn();
    dispose = jest.fn();
    (CpuAnalysisClient as unknown as Mock)
      .mockReset()
      .mockImplementation(() => ({ analyze, cancel, dispose }));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("does not create an engine for a disabled online position", () => {
    render(<AnalysisPanel state={null} isReplay />);
    act(() => jest.advanceTimersByTime(1000));
    expect(CpuAnalysisClient).not.toHaveBeenCalled();
    expect(screen.queryByRole("region", { name: "Engine analysis" })).not.toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("debounces replay changes and ignores late results from the old position", async () => {
    const { rerender, unmount } = render(
      <AnalysisPanel state="live" isReplay={false} />
    );
    act(() => jest.advanceTimersByTime(249));
    expect(analyze).not.toHaveBeenCalled();
    act(() => jest.advanceTimersByTime(1));
    expect(analyze).toHaveBeenCalledWith(
      "live",
      { timeMs: 1800, maxDepth: 3, multiPv: 3 },
      expect.any(Function)
    );
    const oldUpdate = analyze.mock.calls[0][2]!;
    act(() => oldUpdate(result("live", 4)));
    expect(screen.getByRole("img")).toHaveAccessibleName(/\+4.0/);
    const width = (screen.getByRole("img").firstElementChild as HTMLElement).style.width;

    rerender(<AnalysisPanel state="replay" isReplay />);
    expect(cancel).toHaveBeenCalled();
    expect(screen.getByRole("img")).toHaveAccessibleName(/\+4.0.*Previous position/);
    expect(screen.getByRole("img")).toHaveStyle({ opacity: "0.55" });
    expect(screen.getByRole("img").firstElementChild).toHaveStyle({ width });
    expect(screen.queryByText("Candidate turn plans")).not.toBeInTheDocument();
    act(() => oldUpdate(result("live", 9)));
    expect(screen.getByRole("img")).not.toHaveAccessibleName(/\+9.0/);
    act(() => jest.advanceTimersByTime(250));
    expect(analyze.mock.calls[1][0]).toBe("replay");
    act(() => analyze.mock.calls[1][2]!(result("replay", -1.5)));
    expect(screen.getByRole("img")).toHaveAccessibleName(
      /Blue advantage · -1.5/
    );
    expect(screen.getByText("Replay analysis")).toBeInTheDocument();
    expect(CpuAnalysisClient).toHaveBeenCalledTimes(1);
    unmount();
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it("shows turn plans and requests a longer search when asked", async () => {
    analyze.mockResolvedValue(result("position"));
    render(<AnalysisPanel state="position" isReplay={false} />);
    await act(async () => jest.advanceTimersByTime(250));
    expect(screen.getByRole("status")).toHaveTextContent("Search complete");
    expect(screen.getByRole("img")).toHaveStyle({ opacity: "1" });
    expect(screen.getByText("Rotate b1 ↗")).toBeInTheDocument();
    expect(screen.getByText("Deploy infantry at c1")).toBeInTheDocument();
    expect(screen.getByText(/2 turns searched/)).toHaveTextContent(
      "1,234 nodes"
    );
    expect(screen.getByText(/Heuristic score/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Analyze deeper" }));
    expect(screen.getByRole("img")).toHaveStyle({ opacity: "0.55" });
    expect(screen.getByRole("img")).toHaveAccessibleName(/\+2.0/);
    await act(async () => jest.advanceTimersByTime(250));
    expect(analyze.mock.calls[1][1]).toEqual({
      timeMs: 6000,
      maxDepth: 4,
      multiPv: 3,
    });
    fireEvent.click(screen.getByRole("button", { name: "Analyze deeper" }));
    await act(async () => jest.advanceTimersByTime(250));
    expect(analyze.mock.calls[2][1]).toEqual({
      timeMs: 15000,
      maxDepth: 5,
      multiPv: 3,
    });
    expect(
      screen.queryByRole("button", { name: "Analyze deeper" })
    ).not.toBeInTheDocument();
  });

  it("distinguishes a winning search line from an already won game", async () => {
    analyze.mockResolvedValue(result("position", 10000));
    const { rerender } = render(
      <AnalysisPanel state="position" isReplay={false} />
    );
    await act(async () => jest.advanceTimersByTime(250));
    expect(screen.getByRole("img")).toHaveAccessibleName(
      /Winning line for Red \(selective\)/
    );
    expect(screen.getByRole("img")).not.toHaveAccessibleName(/10000/);

    analyze.mockResolvedValue({
      ...result("won", 10000),
      terminal: { winner: "RED", reason: "hq_capture" },
    });
    rerender(<AnalysisPanel state="won" isReplay={false} />);
    await act(async () => jest.advanceTimersByTime(250));
    expect(screen.getByRole("img")).toHaveAccessibleName(/Red wins/);
    expect(screen.queryByText("Candidate turn plans")).not.toBeInTheDocument();
  });

  it("handles worker startup failure and permits retry", async () => {
    (CpuAnalysisClient as unknown as Mock).mockImplementationOnce(() => {
      throw new Error("Worker blocked");
    });
    render(<AnalysisPanel state="position" isReplay={false} />);
    await act(async () => jest.advanceTimersByTime(250));
    expect(screen.getByRole("status")).toHaveTextContent(
      "Analysis unavailable"
    );
    expect(
      screen.getByRole("button", { name: "Retry analysis" })
    ).toBeInTheDocument();
  });
});

describe("analysis move descriptions", () => {
  it.each([
    ["a1a1↗", "Rotate a1 ↗"],
    ["a1b1→", "a1 → b1, face →"],
    ["rfd1", "Deploy armored infantry at d1"],
    ["sbc4", "Bombard c4 (free)"],
    ["sfc4", "Capture c4 (free)"],
    ["a1a2xb2", "a1 → a2; capture b2"],
    ["skip", "End turn"],
  ])("describes %s", (uci, expected) => {
    expect(describeAnalysisMove(uci)).toBe(expected);
  });
});
