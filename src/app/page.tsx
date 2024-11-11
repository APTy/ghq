"use client";

import { useRouter } from "next/navigation";
import { Button } from "./live/Button";
import { API_URL } from "./live/config";
import { PlayOnlineButton } from "./live/PlayOnlineButton";
import { ghqFetch } from "@/lib/api";
import { useEffect, useState } from "react";
import { ClerkLoaded, ClerkLoading, useAuth, useUser } from "@clerk/nextjs";
import { Learn } from "@/components/Learn";
import Image from "next/image";
import { SignInButton, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import { Loader2 } from "lucide-react";
import Header from "@/components/Header";

interface Game {
  id: string;
  player1: string;
  player2: string;
  status: string;
}

function App() {
  const router = useRouter();
  const { isSignedIn, getToken } = useAuth();
  const [games, setGames] = useState<Game[]>([]);

  useEffect(() => {
    if (!isSignedIn) {
      return;
    }

    ghqFetch<{ matches: Game[] }>({
      url: `${API_URL}/matches`,
      getToken,
      method: "GET",
    }).then((data) => {
      setGames(
        data?.matches?.map((match: any) => ({
          id: match.id,
          player1: match.usernames[0],
          player2: match.usernames[1],
          status: `Turn ${match.state.ctx.turn}`,
        })) ?? []
      );
    });
  }, [isSignedIn]);

  async function playLocal() {
    router.push("/local");
  }

  async function playBot() {
    router.push("/bot");
  }

  async function goLearn() {
    router.push("/learn");
  }

  function openSignInDialog() {
    if (!isSignedIn) {
      const signInButton = document.getElementById("sign-in-button");
      if (signInButton) {
        signInButton.click();
      }
    }
  }

  return (
    <div className="p-2 flex flex-col gap-4 lg:px-48">
      <Header />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-2">
        <div className="col-span-2 border rounded p-4 bg-slate-50 lg:order-1 order-2">
          <Learn />
        </div>

        <div className="col-span-1 flex flex-col gap-2 lg:order-2 order-1">
          <div className="flex flex-col gap-2 border rounded p-4 bg-slate-50">
            <div className="text-2xl">Play a game</div>
            <div className="flex flex-wrap gap-2 justify-center items-center">
              <PlayOnlineButton openSignInDialog={openSignInDialog} />
              <Button onClick={playLocal}>👨‍💻 Pass n&apos; Play</Button>
              <Button onClick={playBot}>🤖 Play Bot</Button>
              <Button onClick={goLearn}>📚 Learn</Button>
            </div>
          </div>

          <div className="border rounded p-4 min-h-[400px] bg-slate-50">
            <div className="text-2xl">Live games</div>
            <div className="flex flex-col gap-2">
              {games.map((game: Game) => (
                <a
                  key={game.id}
                  href={`/live/${game.id}`}
                  className="py-2 px-3 bg-white border border-gray-200 rounded-lg shadow hover:shadow-md"
                >
                  <div className="text-xl font-bold tracking-tight text-gray-900">
                    {game.player1} vs {game.player2}
                  </div>
                  <p className="font-normal text-gray-600">{game.status}</p>
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
