import Image from "next/image";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PIECE_INFO, PieceInfo } from "@/lib/game-info";

export default function LearnBasics() {
  return (
    <>
      <div className="flex flex-wrap gap-2">
        <HowToWinCard />
        <HowToPlayCard />
        <SetupCard />
      </div>
      <div className="flex flex-wrap gap-2">
        <HowInfantryCapture />
        <HowArtilleryCapture />
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.values(PIECE_INFO).map((piece) => (
          <PieceCard key={piece.name} piece={piece} />
        ))}
      </div>
    </>
  );
}

function HowToWinCard() {
  return (
    <Card className="min-w-48 w-80">
      <CardHeader>
        <CardTitle>How to win</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col justify-center items-center gap-8">
          <div className="font-bold text-blue-600 dark:text-blue-400 text-center">
            Capture the enemy HQ
            <br />
            to win!
          </div>
          <div className="flex gap-4 justify-center items-center">
            <Image src="/hq-red.png" alt="hq" width={64} height={64} />
            <Image src="/hq-blue.png" alt="hq" width={64} height={64} />
          </div>
          <div className="text-sm text-card-foreground">
            Use <strong>artillery</strong> and <strong>infantry</strong> to
            defend your HQ and destroy the enemy&apos;s HQ!
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function HowToPlayCard() {
  return (
    <Card className="min-w-48 w-80">
      <CardHeader>
        <CardTitle>How to play</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col justify-center items-center gap-2">
          <div className="font-bold">
            You get <strong>3</strong> moves per turn.
          </div>
          <div className="text-card-foreground">
            <div className="mb-2">Choose any of:</div>
            <div className="text-sm">
              <div>(A) Deploy a piece from your reserve</div>
              <div>(B) Move an infantry and/or capture a piece</div>
              <div>(C) Move and/or rotate an artillery</div>
              <div className="mt-2 text-sm">
                Each piece can move once per turn
              </div>
            </div>
          </div>
          <div className="text-sm text-card-foreground mt-4">
            <div>
              <strong>Adjacent square</strong> means touching sides left, right,
              up, or down.
            </div>
            <div>
              <strong>Any square</strong> means adjacent or diagonal.
            </div>
            <div>
              <strong>Capturing</strong> means destroying an enemy piece.
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SetupCard() {
  return (
    <Card className="min-w-48 w-80">
      <CardHeader>
        <CardTitle>Board setup</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col justify-center items-center gap-2">
          <Image
            src="/rules/board-setup.png"
            alt="board-setup"
            width={300}
            height={300}
          />
        </div>
      </CardContent>
    </Card>
  );
}

function HowInfantryCapture() {
  return (
    <Card className="min-w-48 w-80">
      <CardHeader>
        <CardTitle>How infantry capture</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col justify-center items-center gap-2">
          <div className="text-sm">
            An infantry is <strong>engaged</strong> if it is adjacent to an
            enemy infantry.
          </div>
          <div className="flex gap-2">
            <Image
              className="transform rotate-90"
              src="/regular-infantry-blue.png"
              alt="infantry"
              width={32}
              height={32}
            />
            <Image
              className="transform -rotate-90"
              src="/regular-infantry-red.png"
              alt="infantry"
              width={32}
              height={32}
            />
          </div>
          <div className="text-xs">Engaged, no capture</div>
          <div className="text-sm">
            An infantry may <strong>capture</strong> if it is adjacent to an
            already-engaged enemy infantry.
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex gap-2">
              <Image
                className="transform rotate-90"
                src="/regular-infantry-blue.png"
                alt="infantry"
                width={32}
                height={32}
              />
              <Image
                className="transform -rotate-90"
                src="/regular-infantry-red.png"
                alt="infantry"
                width={32}
                height={32}
              />
            </div>
            <div className="flex gap-2">
              <Image
                src="/regular-infantry-red.png"
                alt="infantry"
                width={32}
                height={32}
              />
              <Image
                className="transform -rotate-90 invisible"
                src="/regular-infantry-red.png"
                alt="infantry"
                width={32}
                height={32}
              />
            </div>
          </div>
          <div className="text-xs">Capture!</div>
          <div className="text-sm">
            Tip: Try to <strong>outnumber </strong>your opponent!
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function HowArtilleryCapture() {
  return (
    <Card className="min-w-48 w-80">
      <CardHeader>
        <CardTitle>How artillery capture</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col justify-center items-center gap-2">
          <div className="text-sm">
            An artillery <strong>bombards</strong> squares in front of itself.
          </div>
          <div className="flex gap-1">
            <Image
              className="transform rotate-90 mr-1"
              src="/regular-artillery-blue.png"
              alt="infantry"
              width={32}
              height={32}
            />
            <div className="stripe-blue-transparent-sm w-10 h-10"></div>
            <div className="stripe-blue-transparent-sm w-10 h-10"></div>
          </div>
          <div className="text-xs">Controlling squares</div>
          <div className="text-sm">
            If your piece is left in a bombarded square, it will be{" "}
            <strong>captured</strong> at the start of your opponent&apos;s turn!
          </div>
          <div className="flex gap-1">
            <Image
              className="transform rotate-90 mr-1"
              src="/regular-artillery-blue.png"
              alt="infantry"
              width={32}
              height={32}
            />
            <div className="stripe-blue-transparent-sm w-10 h-10"></div>
            <div className="stripe-blue-transparent-sm w-10 h-10 flex justify-center items-center">
              <Image
                className="transform rotate-90"
                src="/regular-infantry-red.png"
                alt="infantry"
                width={32}
                height={32}
              />
            </div>
          </div>
          <div className="text-xs">Capture!</div>
          <div className="text-sm">
            Enemy pieces cannot move onto or through bombarded squares.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PieceCard({ piece }: { piece: PieceInfo }) {
  return (
    <Card className="min-w-48 w-80">
      <CardHeader>
        <CardTitle>{piece.name}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col justify-center items-center gap-2">
          <Image
            src={`/${piece.image}-blue.png`}
            alt={piece.name}
            width={48}
            height={48}
          />
          <div className="text-sm text-card-foreground">
            <div>Moves: {piece.moves}</div>
            {piece.captures && <div>Captures: {piece.captures}</div>}
            {piece.bombards && <div>Bombards: {piece.bombards}</div>}
            <div>Captured by: {piece.capturedBy}</div>
            <div className="text-blue-600 dark:text-blue-400">{piece.special}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
