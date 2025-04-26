"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useCallback,
  useState,
  ReactNode,
} from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";
import { ghqFetch } from "@/lib/api";
import { API_URL } from "@/app/live/config";
import { TIME_CONTROLS } from "@/game/constants";
import { playGameReadySound } from "@/game/audio";
import { UsersOnline } from "@/lib/types";

interface MatchmakingContextType {
  matchmakingMode: keyof typeof TIME_CONTROLS | null;
  startMatchmaking: (mode: keyof typeof TIME_CONTROLS) => void;
  cancelMatchmaking: () => void;
  usersOnline: UsersOnline | null;
}

const MatchmakingContext = createContext<MatchmakingContextType | undefined>(
  undefined
);

export const useMatchmaking = () => {
  const context = useContext(MatchmakingContext);
  if (!context) {
    throw new Error("useMatchmaking must be used within a MatchmakingProvider");
  }
  return context;
};

interface MatchmakingData {
  match: {
    id: string;
    playerId: string;
    credentials: string;
  };
}

export const MatchmakingProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [matchmakingMode, setMatchmakingMode] = useState<
    keyof typeof TIME_CONTROLS | null
  >(null);
  const { isSignedIn, getToken } = useAuth();
  const [usersOnline, setUsersOnline] = useState<UsersOnline | null>(null);
  const router = useRouter();

  const checkMatchmaking = useCallback(async () => {
    try {
      const data = await ghqFetch<MatchmakingData>({
        url: `${API_URL}/matchmaking?mode=${matchmakingMode}`,
        getToken,
        method: "POST",
      });
      if (data.match) {
        router.push(`/live/${data.match.id}`);
        playGameReadySound();
        setMatchmakingMode(null);
      }
    } catch (error) {
      console.error("Error polling matchmaking API:", error);
    }
  }, [router, matchmakingMode]);

  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (matchmakingMode) {
      checkMatchmaking();
      interval = setInterval(() => checkMatchmaking(), 2000); // Poll every 2 seconds
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [matchmakingMode, checkMatchmaking]);

  const startMatchmaking = (mode: keyof typeof TIME_CONTROLS) => {
    setMatchmakingMode(mode);
  };

  const cancelMatchmaking = () => {
    ghqFetch({
      url: `${API_URL}/matchmaking?mode=${matchmakingMode}`,
      getToken,
      method: "DELETE",
    });
    setMatchmakingMode(null);
  };

  // Keep track of online users
  useEffect(() => {
    if (!isSignedIn) {
      return;
    }

    ghqFetch<UsersOnline>({
      url: `${API_URL}/users/online`,
      getToken,
      method: "GET",
    }).then((data) => {
      setUsersOnline(data);
    });
  }, [isSignedIn, getToken]);

  return (
    <MatchmakingContext.Provider
      value={{
        matchmakingMode,
        startMatchmaking,
        cancelMatchmaking,
        usersOnline,
      }}
    >
      {children}
    </MatchmakingContext.Provider>
  );
};
