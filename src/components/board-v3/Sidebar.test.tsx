import "@testing-library/jest-dom/jest-globals";
import { describe, expect, it } from "@jest/globals";
import { render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import Sidebar from "./Sidebar";
import type { GameClient } from "@/game/engine-v2";

declare const jest: typeof import("@jest/globals").jest;
jest.mock("./AnalysisPanel", () => ({ __esModule: true, default: () => <section aria-label="Engine analysis" /> }));
jest.mock("./HistoryLog", () => ({ HistoryLog: () => null }));
jest.mock("@/components/Header", () => ({ __esModule: true, default: () => null }));
jest.mock("./HomeButton", () => ({ __esModule: true, default: () => null }));
jest.mock("./ResignButton", () => ({ __esModule: true, default: () => null }));
jest.mock("./AbandonButton", () => ({ __esModule: true, default: () => null }));
jest.mock("./SettingsMenu", () => ({ __esModule: true, default: () => null }));
jest.mock("./ShareGameDialog", () => ({ __esModule: true, default: () => null }));
jest.mock("./PlayLocalIconButton", () => ({ __esModule: true, default: () => null }));
jest.mock("@/game/engine-v2", () => ({ gameoverReason: () => "Game complete" }));

function props(isOnline: boolean, seekIndex = -1): ComponentProps<typeof Sidebar> {
  const game = {
    isOnline, turn: 1, moves: [],
    currentPlayerTurn: () => "RED", currentPlayer: () => "RED",
    numMovesThisTurn: () => 0, analysisState: jest.fn(() => "position"),
  } as unknown as GameClient;
  return { game, activeGame: game, seek: jest.fn(), seekIndex, gameover: undefined,
    className: "", settings: { autoFlipBoard: false }, setSettings: jest.fn() };
}

describe("analysis visibility", () => {
  it.each([[false, -1], [false, 0], [true, -1], [true, 0]])(
    "hides active game analysis (online=%s, seek=%s)", (online, seek) => {
      const p = props(online as boolean, seek as number);
      render(<Sidebar {...p} />);
      expect(screen.queryByRole("region", { name: "Engine analysis" })).not.toBeInTheDocument();
      expect(p.activeGame.analysisState).not.toHaveBeenCalled();
    }
  );

  it("shows learning positions but cannot opt an ongoing online game in", () => {
    const { rerender } = render(<Sidebar {...props(false)} isLearning />);
    expect(screen.getByRole("region", { name: "Engine analysis" })).toBeInTheDocument();
    rerender(<Sidebar {...props(true)} isLearning />);
    expect(screen.queryByRole("region", { name: "Engine analysis" })).not.toBeInTheDocument();
  });

  it.each([false, true])("shows finished games and their earlier positions (online=%s)", (online) => {
    const p = props(online, 0);
    const { rerender } = render(<Sidebar {...p} gameover={{ status: "WIN", winner: "RED", reason: "hq-capture" }} />);
    expect(screen.getByRole("region", { name: "Engine analysis" })).toBeInTheDocument();
    rerender(<Sidebar {...p} />);
    expect(screen.queryByRole("region", { name: "Engine analysis" })).not.toBeInTheDocument();
  });
});
