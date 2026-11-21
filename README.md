# Kurt Vonnegut's GHQ
> We like to think that if heaven is real, the most interesting thing going on in Kurt's life is watching all us earthly fools playing his board game. **Is GHQ winnable?** idk...**war winnable**. So it goes. Play on or write some code with us

### Contributing ideas
- Help us build the GHQ bot and figure out if this is a solvable game. It could be 🤷 
- Build an interactive instruction manual with embedded playable boards to grow the community
- Mobile friendly PWA 
- Correspondence games
- Refactor of board logic is needed. We didn't know the reqs when we started so there's a lot to do. 
- Build some cool variants. What does 4 player GHQ look like? Infinite canvas

Huge Kudos to everyone who has contributed to https://github.com/boardgameio/boardgame.io. 

### CPU bot and position analysis

`public/search.py` searches coordinated player turns using `public/engine.py` as
the rules authority. It ranks every legal action at each expanded position,
keeps a bounded set of promising continuations, deduplicates complete rule
states, and runs iterative minimax across turn boundaries. The first narrow
pass prioritizes looking at an opponent reply before widening the search.
Artillery orientations share candidate families so rotations do not consume
the entire shortlist. The shortlist preserves different first decisions and a
quiet alternative, so reordering one attractive capture cannot exclude every
safer plan. It also retains distinct capture targets before filling the
shortlist with different preparations for the same capture. Mandatory captures are resolved before completed plans
are ranked; threatened pieces are also discounted while pruning partial turns.

The browser runs the same search in a dedicated Pyodide Web Worker. Bot turns
get a 2.4-second search budget; the analysis panel starts at 1.8 seconds, with
6- and 15-second deeper-analysis requests. Loading Pyodide is additional startup
time. Search cancellation terminates the worker, and replies are applied only
after every action has been checked against the rules engine. Reload the page
after editing Python sources: existing workers retain their loaded modules.

Scores are heuristic infantry units, positive for Red, including reserve
material, deployment, threats, artillery coverage, mobility, and HQ safety.
Unsupported advances also carry a positional cost: deep units with little
friendly support, few exits, and nearby enemy pressure are risky even when
the search cannot calculate a complete trapping sequence.
Speculative HQ and material threats from immediately capturable batteries
receive reduced weight, so a hanging gun cannot crowd safer plans out of the
search simply by aiming at the HQ. Legal firing zones and bombardments already
due remain unchanged; overlapping fire from a safe battery retains its threat.
Future free-capture threats receive reduced weight when the defender can move
the threatened piece first. Reply search can then compare rescuing that piece
and removing its attacker with taking a different capture target.
They are not calibrated win probabilities. Depth counts player turns, not
individual actions. Candidate plans and winning lines are selective search
results, not proofs of optimal play. Analysis is available in learning mode
and finished games (including their replays), and is hidden during all unfinished games.

Run the complete Python regression suite (including the imported AlphaZero
GHQ tests) with:

```sh
python3 -m pip install -r requirements-test.txt
python3 -B -m pytest tests -q
```

`tests/alpha_zero/` contains the tests and fixtures copied from
`../alpha-zero-general/ghq/`, plus its action encoder, NumPy serializer,
`GHQGame` wrapper, and base `Game` class as test-only support. Imports point to
this project's `public/engine.py`; assertions and fixture expectations are
unchanged. These tests do not require the sibling repository at runtime.

The TypeScript worker, bot, PGN, and UI lifecycle tests run with `npm test`.
Compare playing strength against the retained legacy `ValuePlayer` with:

```sh
python3 -B scripts/benchmark_cpu_bot.py --games 4 --time-ms 1200 --max-turns 60 > /tmp/ghq-arena.json
```

The arena uses paired colors and openings, checks legal actions and input
immutability, records source fingerprints, and reports actual latency and
completed search depth. Games stopped by the turn cap are unfinished, not
draws. A short smoke run is useful for regressions; it does not establish Elo
or an overall strength gain. Native Python throughput also differs from
browser WebAssembly, so test important positions in both runtimes.
