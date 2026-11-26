import { afterEach, expect, it, jest } from "@jest/globals";
import { fireEvent, render, screen, cleanup } from "@testing-library/react";
import { parseBotDifficulty } from "@/game/bot-difficulty";

const push = jest.fn();
jest.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));
const { PlayBotButton } = require("./PlayBotButton") as typeof import("./PlayBotButton");

afterEach(() => { cleanup(); push.mockClear(); });

it.each(["Easy", "Medium", "Hard"])("starts a game with %s selected", (label) => {
  render(<PlayBotButton />);
  fireEvent.click(screen.getByRole("button", { name: /Play Bot/ }));
  expect(push).not.toHaveBeenCalled();
  expect((screen.getByRole("radio", { name: "Medium" }) as HTMLInputElement).checked).toBe(true);
  fireEvent.click(screen.getByRole("radio", { name: label }));
  fireEvent.click(screen.getByRole("button", { name: "Start Game" }));
  expect(push).toHaveBeenCalledWith(`/bot?difficulty=${label.toLowerCase()}`);
});

it("can dismiss the dialog without starting a game", () => {
  render(<PlayBotButton />);
  fireEvent.click(screen.getByRole("button", { name: /Play Bot/ }));
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(screen.queryByRole("dialog")).toBeNull();
  expect(push).not.toHaveBeenCalled();
});

it("defaults missing or invalid URL difficulties to medium", () => {
  for (const value of [null, "", "invalid", "toString"]) {
    expect(parseBotDifficulty(value)).toBe("medium");
  }
  for (const value of ["easy", "medium", "hard"]) {
    expect(parseBotDifficulty(value)).toBe(value);
  }
});
