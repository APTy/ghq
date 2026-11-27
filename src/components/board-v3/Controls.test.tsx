import { fireEvent, renderHook } from "@testing-library/react";
import useControls, { ControlsProps } from "./Controls";

jest.mock("react-hotkeys-hook", () => ({ useHotkeys: jest.fn() }));

function controls(overrides: Partial<ControlsProps> = {}): ControlsProps {
  return {
    settings: { autoFlipBoard: false, undoWithMouse: true },
    undo: jest.fn(),
    redo: jest.fn(),
    cancel: jest.fn(),
    skip: jest.fn(),
    canConfirm: true,
    replay: jest.fn(),
    backward: jest.fn(),
    forward: jest.fn(),
    togglePOV: jest.fn(),
    ...overrides,
  };
}

describe("mouse turn confirmation", () => {
  it.each([0, 4])("confirms when button %i completes the Mouse5 + left combo", (button) => {
    const props = controls();
    renderHook(() => useControls(props));
    fireEvent.mouseDown(window, { button, buttons: 17 });
    expect(props.skip).toHaveBeenCalledTimes(1);
    expect(props.redo).not.toHaveBeenCalled();
    expect(props.undo).not.toHaveBeenCalled();
  });

  it("keeps ordinary redo and undo from confirming", () => {
    const props = controls();
    renderHook(() => useControls(props));
    fireEvent.mouseDown(window, { button: 4, buttons: 16 });
    fireEvent.mouseDown(window, { button: 3, buttons: 8 });
    fireEvent.mouseDown(window, { button: 0, buttons: 9 });
    expect(props.redo).toHaveBeenCalledTimes(1);
    expect(props.undo).toHaveBeenCalledTimes(1);
    expect(props.skip).not.toHaveBeenCalled();
  });

  it.each([
    { canConfirm: false },
    { settings: { autoFlipBoard: false, undoWithMouse: false } },
  ])("does not confirm when the shortcut is unavailable: %j", (overrides) => {
    const props = controls(overrides);
    renderHook(() => useControls(props));
    fireEvent.mouseDown(window, { button: 0, buttons: 17 });
    fireEvent.mouseDown(window, { button: 4, buttons: 17 });
    expect(props.skip).not.toHaveBeenCalled();
  });

  it("consumes the confirmation click across a rerender but allows the next click", () => {
    const props = controls();
    const { rerender } = renderHook((p) => useControls(p), { initialProps: props });
    const target = document.createElement("button");
    document.body.appendChild(target);
    const handler = jest.fn();
    target.addEventListener("mousedown", handler);
    target.addEventListener("mouseup", handler);
    target.addEventListener("click", handler);
    try {
      fireEvent.mouseDown(target, { button: 0, buttons: 17 });
      rerender({ ...props, canConfirm: false });
      fireEvent.mouseUp(target, { button: 0, buttons: 16 });
      fireEvent.click(target, { button: 0 });
      expect(handler).not.toHaveBeenCalled();
      expect(props.skip).toHaveBeenCalledTimes(1);
      fireEvent.mouseDown(target, { button: 0, buttons: 1 });
      fireEvent.mouseUp(target, { button: 0 });
      fireEvent.click(target, { button: 0 });
      expect(handler).toHaveBeenCalledTimes(3);
    } finally {
      target.remove();
    }
  });
});
