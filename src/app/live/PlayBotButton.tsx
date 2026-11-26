"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Button from "./ButtonV2";
import { Button as UIButton } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter,
  DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { BOT_DIFFICULTIES, BotDifficulty } from "@/game/bot-difficulty";

export function PlayBotButton() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [difficulty, setDifficulty] = useState<BotDifficulty>("medium");

  return (
    <>
      <Button onClick={() => setOpen(true)}>🤖 Play Bot</Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Play Bot</DialogTitle>
            <DialogDescription>Choose your opponent’s difficulty.</DialogDescription>
          </DialogHeader>
          <fieldset>
            <legend className="mb-2 text-sm font-medium">Difficulty</legend>
            <div className="grid grid-cols-3 gap-2">
              {(Object.keys(BOT_DIFFICULTIES) as BotDifficulty[]).map((level) => (
                <label key={level} className={`flex cursor-pointer items-center justify-center gap-2 rounded-md border p-3 text-sm font-medium ${difficulty === level ? "border-blue-600 bg-blue-50 text-blue-900" : "border-gray-300"}`}>
                  <input type="radio" name="bot-difficulty" value={level}
                    checked={difficulty === level} onChange={() => setDifficulty(level)}
                    className="accent-blue-600" />
                  {BOT_DIFFICULTIES[level].label}
                </label>
              ))}
            </div>
          </fieldset>
          <DialogFooter>
            <UIButton onClick={() => {
              setOpen(false);
              router.push(`/bot?difficulty=${difficulty}`);
            }}>Start Game</UIButton>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
