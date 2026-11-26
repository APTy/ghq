"use client";

import { ImportFormat } from "@/game/analysis-import";
import React, { useMemo } from "react";
import classNames from "classnames";
import { HistoryLog } from "./HistoryLog";
import AnalysisPanel from "./AnalysisPanel";

import Header from "@/components/Header";
import HomeButton from "./HomeButton";
import ResignButton from "./ResignButton";
import AbandonButton from "./AbandonButton";
import SettingsMenu, { Settings } from "./SettingsMenu";
import { Swords } from "lucide-react";
import { config } from "@/lib/config";
import { GameClient, gameoverReason } from "@/game/engine-v2";
import ShareGameDialog from "./ShareGameDialog";
import { SeekFunc } from "./useSeek";
import { GameoverState } from "@/game/engine";
import PlayLocalIconButton from "./PlayLocalIconButton";

export default function Sidebar({
  game,
  activeGame,
  seek,
  gameover,
  seekIndex,
  className,
  settings,
  setSettings,
  isLearning = false,
  onLoad,
}: {
  // this is the realGame
  game: GameClient;
  // this is the active game, which may be the realGame or the simGame
  activeGame: GameClient;
  seek: SeekFunc;
  gameover: GameoverState | undefined;
  seekIndex: number;
  className: string;
  settings: Settings;
  setSettings: (settings: Settings) => void;
  isLearning?: boolean;
  onLoad?: (format: ImportFormat, text: string) => void;
}) {
  const currentPlayerTurn = useMemo(
    () => game.currentPlayerTurn(),
    [game.turn]
  );
  const currentPlayer = useMemo(() => game.currentPlayer(), [game.turn]);
  const movesLeft = useMemo(
    () => 3 - game.numMovesThisTurn(),
    [game.moves, game.turn]
  );

  // Seeking in an unfinished game is not post-game analysis. Gate on the
  // real game's outcome, never the replay position or pass-and-play setting.
  const showAnalysis = Boolean(gameover) || (isLearning && !game.isOnline);

  return (
    <div
      className={classNames(
        "w-full md:w-[450px] bg-card/80 h-screen overflow-y-auto shadow-lg transition-shadow duration-300 shadow-slate-400 dark:shadow-black/40 space-y-3",
        className
      )}
    >
      <div className="hidden sm:block">
        <Header />
      </div>
      {showAnalysis && (
        <AnalysisPanel state={activeGame.analysisState()} isReplay={seekIndex !== -1} />
      )}
      <HistoryLog
        game={game}
        gameover={gameover}
        seek={seek}
        seekIndex={seekIndex}
      />
      {gameover ? (
        <div className="flex flex-col items-center justify-center gap-0 justify-center items-center">
          <h2
            className={classNames(
              "text-center font-bold",
              gameover.status === "DRAW" && "text-foreground",
              gameover.status === "WIN" && gameover.winner === "RED"
                ? "text-red-500"
                : "text-blue-500"
            )}
          >
            {gameover.status === "DRAW" ? (
              "Draw!"
            ) : (
              <>{gameover.winner === "RED" ? "Red" : "Blue"} Won!</>
            )}
          </h2>
          <div className="text-center text-foreground text-sm">
            {gameoverReason(gameover)}
          </div>
          <div className="flex gap-1 mt-3">
            <PlayLocalIconButton game={activeGame} />
            <ShareGameDialog game={activeGame} />
            <HomeButton />
            <SettingsMenu onLoad={onLoad} settings={settings} setSettings={setSettings} />
          </div>
        </div>
      ) : (
        <div className="text-center flex items-center flex-col justify-center flex-1">
          <div
            className={classNames(
              game.currentPlayer() === "RED" ? "text-red-500" : "text-blue-500",
              "flex items-center gap-1"
            )}
          >
            <div className="flex items-center gap-1 font-semibold">
              <Swords className="w-5 h-5" /> {game.turn}{" "}
            </div>
            <div>
              {currentPlayer === currentPlayerTurn ? "Your" : "Their"} Turn
            </div>
          </div>
          <div className="text-muted-foreground flex gap-1 justify-center items-center font-medium">
            {movesLeft} remaining move
            {movesLeft !== 1 ? "s" : ""}{" "}
          </div>
          <div className="flex gap-1 justify-center items-center mt-2">
            {currentPlayer === currentPlayerTurn || !game.isOnline ? (
              <>
                {/* {G.drawOfferedBy && G.drawOfferedBy !== ctx.currentPlayer ? (
                  <AcceptDrawButton draw={() => moves.AcceptDraw()} />
                ) : (
                  <OfferDrawButton
                    draw={(offer: boolean) => moves.OfferDraw(offer)}
                  />
                )} */}
                <ResignButton resign={() => game.resign()} />
                <ShareGameDialog game={game} />
                <SettingsMenu onLoad={onLoad} settings={settings} setSettings={setSettings} />
              </>
            ) : (
              <>
                {config.useClerk && game.id && (
                  <AbandonButton matchId={game.id} />
                )}
                <ShareGameDialog game={game} />
                <SettingsMenu onLoad={onLoad} settings={settings} setSettings={setSettings} />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
