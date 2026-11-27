"use client";

export default function Rules() {
  return (
    <div className="flex flex-col gap-3 text-sm">
      <div>
        GHQ is a game of strategy and tactics for two players. Players alternate
        turns moving pieces on the board, attempting to capture their
        opponent&apos;s HQ to win the game.
      </div>
      <div>
        It was originally designed by{" "}
        <a
          className="text-blue-600 dark:text-blue-400 hover:text-blue-400 dark:hover:text-blue-300"
          href="https://www.vonnegutgame.com/"
          target="_blank"
        >
          Kurt Vonnegut
        </a>
        . The{" "}
        <a
          className="text-blue-600 dark:text-blue-400 hover:text-blue-400 dark:hover:text-blue-300"
          href="https://www.kvmlshop.org/product/kurt-vonnegut-the-lost-board-game-ghq/877"
          target="_blank"
        >
          physical game
        </a>
        {" "}was developed by Geoff Engelstein. This online version is free and
        open-source, built and maintained by Vonnegut and gaming enthusiasts.
      </div>

      <div>
        👾 Join us on{" "}
        <a
          className="text-blue-600 dark:text-blue-400 hover:text-blue-400 dark:hover:text-blue-300"
          href="https://discord.gg/MDaTYTdG5e"
          target="_blank"
        >
          Discord!
        </a>
      </div>
    </div>
  );
}
