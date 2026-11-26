"use client";

import React from "react";

interface MoveProgressBarProps {
  numMoves: number;
}

export default function MoveProgressBar({ numMoves }: MoveProgressBarProps) {
  const progress = Math.min((numMoves / 3) * 100, 100);

  return (
    <div className="mt-1 w-full h-1 shrink-0 rounded-full bg-muted overflow-hidden">
      <div
        className="h-full rounded-full bg-blue-500 transition-all duration-300 ease-in-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}
