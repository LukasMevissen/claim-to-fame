# Shadow model comparison

Shadow models see the same case, prompt and price book as the live model; their
quotes are never submitted. Net = replay of the quotes against the real opponents
(`c2f/replay.py`), t inside the revealed bounds; two columns for the two assumptions
about t where only a lower bound is known (1.15x and 1.4x that bound).
In time = answered within 45 s. Compare rows only on the same games.

## claude-fable-5-2@low

- games scored: 15 (1..15), failed: 14
- latency: median 13.8 s, max 13.8 s, in time 1/15
- net, all scored games (failed = 0/0 quotes): -47,066 / -47,066
- on the 10 games we also played live: shadow -20,841 / -20,841 vs live 8,945 / 8,945
- tokens: 21,414 in, 770 out over 1 games

## claude-opus-5-5@low

- games scored: 15 (1..15), failed: 14
- latency: median 13.8 s, max 13.8 s, in time 1/15
- net, all scored games (failed = 0/0 quotes): -48,054 / -48,054
- on the 10 games we also played live: shadow -21,829 / -21,829 vs live 8,945 / 8,945
- tokens: 21,414 in, 788 out over 1 games

## claude-sonnet-5-5@low

- games scored: 15 (1..15), failed: 14
- latency: median 11.0 s, max 11.0 s, in time 1/15
- net, all scored games (failed = 0/0 quotes): -51,478 / -51,478
- on the 10 games we also played live: shadow -25,253 / -25,253 vs live 8,945 / 8,945
- tokens: 21,414 in, 798 out over 1 games

## gpt-6.2-sol@low

- games scored: 14 (1..14), failed: 0
- latency: median 14.2 s, max 20.3 s, in time 14/14
- net, all scored games (failed = 0/0 quotes): 38,276 / 54,308
- on the 9 games we also played live: shadow 9,969 / 26,001 vs live 7,145 / 7,145

