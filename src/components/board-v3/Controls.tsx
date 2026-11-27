import { useEffect, useRef } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import { Settings } from "./SettingsMenu";

export interface ControlsProps {
  settings: Settings;
  undo: () => void;
  redo: () => void;
  cancel: () => void;
  skip: () => void;
  canConfirm: boolean;
  replay: () => void;
  backward: () => void;
  forward: () => void;
  togglePOV: () => void;
}

export default function useControls({
  settings,
  undo,
  redo,
  cancel,
  skip,
  canConfirm,
  replay,
  backward,
  forward,
  togglePOV,
}: ControlsProps) {
  const suppressConfirmClick = useRef(false);
  useEffect(() => {
    const handleMouseButton = (e: MouseEvent) => {
      if (!settings.undoWithMouse) {
        return;
      }

      if (e.type === "mousedown" && e.button === 0) {
        suppressConfirmClick.current = false;
      }

      // Mouse5 is button 4, and its held-button mask is 16.
      if (
        e.type === "mousedown" &&
        (e.button === 0 || e.button === 4) &&
        (e.buttons & 17) === 17 &&
        canConfirm
      ) {
        e.preventDefault();
        e.stopPropagation();
        suppressConfirmClick.current = true;
        skip();
        return;
      }

      if (
        suppressConfirmClick.current &&
        e.button === 0 &&
        (e.type === "mouseup" || e.type === "click")
      ) {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "click") suppressConfirmClick.current = false;
        return;
      }

      // Most browsers go "back" on mouseup, so we'll prevent default on that and return.
      if (e.type === "mouseup") {
        if (e.button === 3 || e.button === 4) e.preventDefault();
        return;
      }

      // Handle ordinary side-button presses as undo/redo only.
      if (e.type === "mousedown" && (e.button === 3 || e.button === 4)) {
        e.preventDefault();

        if (e.button === 3) {
          undo();
        } else if (e.button === 4) {
          redo();
        }
      }
    };

    window.addEventListener("mousedown", handleMouseButton, true);
    window.addEventListener("mouseup", handleMouseButton, true);
    window.addEventListener("click", handleMouseButton, true);
    return () => {
      window.removeEventListener("mousedown", handleMouseButton, true);
      window.removeEventListener("mouseup", handleMouseButton, true);
      window.removeEventListener("click", handleMouseButton, true);
    };
  }, [undo, redo, skip, canConfirm, settings.undoWithMouse]);
  useHotkeys(
    "escape",
    (e) => {
      e.preventDefault();
      cancel();
    },
    [cancel]
  );

  useHotkeys(
    "left",
    (e) => {
      e.preventDefault();
      undo();
    },
    [undo]
  );

  useHotkeys(
    "right",
    (e) => {
      e.preventDefault();
      redo();
    },
    [redo]
  );

  useHotkeys(
    "space",
    (e) => {
      e.preventDefault();
      replay();
    },
    [replay]
  );

  useHotkeys(
    ".",
    (e) => {
      e.preventDefault();
      skip();
    },
    [skip]
  );

  useHotkeys(
    "enter",
    (e) => {
      e.preventDefault();
      skip();
    },
    [skip]
  );

  useHotkeys(
    "up",
    (e) => {
      e.preventDefault();
      backward();
    },
    [backward]
  );

  useHotkeys(
    "down",
    (e) => {
      e.preventDefault();
      forward();
    },
    [forward]
  );

  useHotkeys(
    "x",
    (e) => {
      e.preventDefault();
      togglePOV();
    },
    [togglePOV]
  );

  return {};
}
