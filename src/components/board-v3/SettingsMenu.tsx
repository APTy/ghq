"use client";

import * as React from "react";
import { DropdownMenuCheckboxItemProps } from "@radix-ui/react-dropdown-menu";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import LoadPositionDialog from "./LoadPositionDialog";
import type { ImportFormat } from "@/game/analysis-import";
import { useTheme } from "next-themes";
import { Ellipsis } from "lucide-react";

export interface Settings {
  autoFlipBoard: boolean;
  confirmTurn?: boolean;
  undoWithMouse?: boolean;
}

type Checked = DropdownMenuCheckboxItemProps["checked"];

export default function SettingsMenu({
  settings,
  setSettings,
  onLoad,
}: {
  onLoad?: (format: ImportFormat, text: string) => void;
  settings: Settings;
  setSettings: (settings: Settings) => void;
}) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  const [format, setFormat] = React.useState<ImportFormat | null>(null);
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" aria-label="Board settings">
            <Ellipsis />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56">
          <DropdownMenuLabel>Appearance</DropdownMenuLabel>
          <DropdownMenuCheckboxItem
            checked={mounted && theme === "dark"}
            disabled={!mounted}
            onCheckedChange={(checked) => setTheme(checked ? "dark" : "light")}
          >
            Dark mode
          </DropdownMenuCheckboxItem>
          <DropdownMenuSeparator />
          <DropdownMenuLabel>Pass and Play</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuCheckboxItem
            checked={settings.autoFlipBoard}
            onCheckedChange={(checked) =>
              setSettings({ ...settings, autoFlipBoard: !!checked })
            }
          >
            Automatic board flip
          </DropdownMenuCheckboxItem>
          <DropdownMenuLabel>Game</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuCheckboxItem
            checked={settings.confirmTurn}
            onCheckedChange={(checked) =>
              setSettings({ ...settings, confirmTurn: !!checked })
            }
          >
            Confirm turn
          </DropdownMenuCheckboxItem>
          <DropdownMenuCheckboxItem
            checked={settings.undoWithMouse}
            onCheckedChange={(checked) =>
              setSettings({ ...settings, undoWithMouse: !!checked })
            }
          >
            Mouse 4/5 for undo/redo
          </DropdownMenuCheckboxItem>
          {onLoad && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => setFormat("PGN")}>
                Load from PGN
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={() => setFormat("FEN")}>
                Load from FEN
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      {format && onLoad && (
        <LoadPositionDialog
          key={format}
          format={format}
          onLoad={onLoad}
          onClose={() => setFormat(null)}
        />
      )}
    </>
  );
}
