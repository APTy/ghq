from .GHQGame import GHQGame


def sanity_check(game_impl):
    import random

    b = game_impl.getInitBoard()
    player = 1
    for _ in range(100):
        valids = game_impl.getValidMoves(b, player)
        if not valids.any(): break
        a = random.choice([i for i,v in enumerate(valids) if v])
        b, player = game_impl.getNextState(b, player, a)

    for player in (1, -1):
        canon1 = game_impl.getCanonicalForm(b, player)
        canon2 = game_impl.getCanonicalForm(
                    game_impl.getCanonicalForm(b, player), +1)
        assert str(canon1) == str(canon2), "Canonical form is not idempotent!"

def test_ghqgame():
    g = GHQGame()

    for _ in range(10):
        sanity_check(g)
