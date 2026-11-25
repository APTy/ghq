import { it, jest, expect } from "@jest/globals";
import { fireEvent, render, screen } from "@testing-library/react";
import LoadPositionDialog from "./LoadPositionDialog";

it.each(["PGN", "FEN"] as const)("submits pasted %s and closes", (format) => {
  const onLoad = jest.fn();
  const onClose = jest.fn();
  render(
    <LoadPositionDialog format={format} onLoad={onLoad} onClose={onClose} />
  );
  fireEvent.change(screen.getByLabelText(format), {
    target: { value: "pasted data" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Load" }));
  expect(onLoad).toHaveBeenCalledWith(format, "pasted data");
  expect(onClose).toHaveBeenCalledTimes(1);
});

it("keeps invalid input open with an error", () => {
  const onClose = jest.fn();
  render(
    <LoadPositionDialog
      format="PGN"
      onLoad={() => {
        throw new Error("Illegal move in turn 2");
      }}
      onClose={onClose}
    />
  );
  fireEvent.change(screen.getByLabelText("PGN"), {
    target: { value: "bad data" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Load" }));
  expect(screen.getByRole("alert").textContent).toBe("Illegal move in turn 2");
  expect(onClose).not.toHaveBeenCalled();
});

it("cancels without loading", () => {
  const onLoad = jest.fn();
  const onClose = jest.fn();
  render(<LoadPositionDialog format="FEN" onLoad={onLoad} onClose={onClose} />);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(onLoad).not.toHaveBeenCalled();
  expect(onClose).toHaveBeenCalled();
});
