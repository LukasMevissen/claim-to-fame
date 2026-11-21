# Shadow model comparison

Shadow models see the same case, prompt and price book as the live model; their
quotes are never submitted. Net = replay of the quotes against the real opponents
(`c2f/replay.py`), t inside the revealed bounds; two columns for the two assumptions
about t where only a lower bound is known (1.15x and 1.4x that bound).
In time = answered within 45 s. Compare rows only on the same games.

## claude-fable-5-2@low

- games scored: 2 (12..13), failed: 2
- latency: median - s, max - s, in time 0/2
- net, all scored games (failed = 0/0 quotes): -9,809 / -9,809
- on the 2 games we also played live: shadow -9,809 / -9,809 vs live 3,838 / 3,838

## gpt-6-luna@low

- games scored: 2 (12..13), failed: 0
- latency: median 9.7 s, max 9.7 s, in time 2/2
- net, all scored games (failed = 0/0 quotes): 3,604 / 3,604
- on the 2 games we also played live: shadow 3,604 / 3,604 vs live 3,838 / 3,838

