"use client";

import { API_URL } from "./live/config";
import { ghqFetch } from "@/lib/api";
import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { StatusIndicator } from "@/components/StatusIndicator";
import { useMatchmaking } from "@/components/MatchmakingProvider";
import { OnlineUser } from "@/lib/types";

interface MatchSummary {
  userId: string;
  username: string;
  elo: number;
  wins: number;
  losses: number;
  draws: number;
  status?: OnlineUser["status"];
}

export default function Leaderboard() {
  const { isSignedIn, getToken } = useAuth();
  const [loading, setLoading] = useState(true);
  const [rawUsers, setRawUsers] = useState<MatchSummary[]>([]);
  const [users, setUsers] = useState<MatchSummary[]>([]);
  const { usersOnline: usersOnlineFromMatchmaking } = useMatchmaking();

  useEffect(() => {
    if (!isSignedIn) {
      setLoading(false);
      return;
    }

    setLoading(true);

    ghqFetch<{ summary: MatchSummary[] }>({
      url: `${API_URL}/match-summary`,
      getToken,
      method: "GET",
    })
      .then((data) => {
        setRawUsers(data.summary ?? []);
      })
      .finally(() => setLoading(false));
  }, [isSignedIn]);

  useEffect(() => {
    const userStatusLookup = new Map<string, OnlineUser["status"]>();
    for (const user of usersOnlineFromMatchmaking?.users ?? []) {
      userStatusLookup.set(user.id, user.status);
    }

    const users: MatchSummary[] = rawUsers.map((user) => {
      return {
        ...user,
        status: userStatusLookup.get(user.userId) ?? "offline",
      };
    });

    setUsers(users);
  }, [usersOnlineFromMatchmaking, rawUsers]);

  return (
    <div className="flex flex-col gap-2 w-full">
      {!isSignedIn && (
        <div className="text-gray-600">Sign in to see the top players!</div>
      )}

      {loading && (
        <div className="flex flex-col gap-2">
          <div className="py-2 px-3 bg-gray-300 animate-pulse border border-gray-200 rounded-lg h-8"></div>
          <div className="py-2 px-3 bg-gray-300 animate-pulse border border-gray-200 rounded-lg h-8"></div>
          <div className="py-2 px-3 bg-gray-300 animate-pulse border border-gray-200 rounded-lg h-8"></div>
        </div>
      )}

      <div className="flex flex-col">
        {users.map((user: MatchSummary) => (
          <div key={user.userId} className="rounded flex justify-between">
            <div className="flex flex-row gap-2 items-center">
              <StatusIndicator status={user.status ?? "offline"} />
              {user.username ?? "Anonymous"}
            </div>
            <div className="flex flex-row gap-2 items-center">
              <span className="text-green-700">{user.wins}</span>
              <span className="text-gray-300">|</span>
              <span className="text-red-700">{user.losses}</span>
              <span className="text-gray-300">|</span>
              <span className="text-gray-700">{user.draws}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
