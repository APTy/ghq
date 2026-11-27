"use client";

import { matchPath } from "@/lib/match-engine";
import { DateTime } from "luxon";
import classNames from "classnames";
import Link from "next/link";
import { MatchModel } from "@/lib/types";
import RatedBadge from "@/components/RatedBadge";

export function MatchLink({ game }: { game: MatchModel }) {
  return (
    <Link
      href={matchPath(game.id, game.engineVersion)}
      className="p-1 px-2 bg-white/50 dark:bg-muted border border-white/50 dark:border-border hover:border-blue-700 flex justify-between hover:bg-blue-100 dark:hover:bg-accent rounded flex-col sm:flex-row"
    >
      <div className="flex gap-1 items-center flex-wrap">
        <RatedBadge rated={game.rated ?? false} />
        <div
          className={classNames(
            game.player1 === game.winner && "text-blue-800 dark:text-blue-300"
          )}
        >
          {game.player1}
        </div>{" "}
        <span className="text-xs">({game.player1Elo})</span>
        <span className="text-gray-500 dark:text-muted-foreground"> vs.</span>
        <div
          className={classNames(
            game.player2 === game.winner && "text-blue-800 dark:text-blue-300"
          )}
        >
          {game.player2}
        </div>{" "}
        <span className="text-xs">({game.player2Elo})</span>
      </div>

      <div className="flex gap-1 items-center text-xs justify-end">
        {game.isYourTurn !== undefined ? (
          <>
            {game.isYourTurn ? (
              <div className="flex gap-2 items-center text-blue-600 dark:text-blue-400">
                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                Your turn
              </div>
            ) : (
              <div className="flex gap-2 items-center text-gray-600 dark:text-muted-foreground">
                <div className="w-2 h-2 bg-gray-600 rounded-full"></div>
                Opponent&apos;s turn
              </div>
            )}
          </>
        ) : (
          <div className="text-gray-500 dark:text-muted-foreground">
            {DateTime.fromISO(game.createdAt).toRelative()}
          </div>
        )}
      </div>
    </Link>
  );
}
