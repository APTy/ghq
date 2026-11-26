"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { ImportFormat } from "@/game/analysis-import";

export default function LoadPositionDialog({
  format,
  onClose,
  onLoad,
}: {
  format: ImportFormat;
  onClose: () => void;
  onLoad: (format: ImportFormat, text: string) => void;
}) {
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Load from {format}</DialogTitle>
          <DialogDescription>
            {format === "PGN"
              ? "Paste a current GHQ PGN export to replace the board and its move history."
              : "Paste a GHQ FEN to replace the position and clear move history."}
          </DialogDescription>
        </DialogHeader>
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            try {
              onLoad(format, text);
              onClose();
            } catch (cause) {
              setError(
                cause instanceof Error
                  ? cause.message
                  : `Unable to load ${format}. Check the pasted text.`
              );
            }
          }}
        >
          <Label htmlFor="load-position">{format}</Label>
          <Textarea
            id="load-position"
            autoFocus
            rows={format === "PGN" ? 8 : 3}
            spellCheck={false}
            className="font-mono"
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setError("");
            }}
            aria-invalid={!!error}
            aria-describedby={error ? "load-error" : undefined}
          />
          {error && (
            <p id="load-error" role="alert" className="text-sm text-red-600 dark:text-red-400">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!text.trim()}>
              Load
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
