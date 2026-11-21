# Shadow model comparison

Shadow models see the same case, prompt and price book as the live model; their
quotes are never submitted. Net = replay of the quotes against the real opponents
(`c2f/replay.py`), t inside the revealed bounds; two columns for the two assumptions
about t where only a lower bound is known (1.15x and 1.4x that bound).
In time = answered within 45 s. Compare rows only on the same games.

## claude-fable-5-2@low

- games answered: 15 of 15 attempted
- latency: median 12.2 s, max 15.4 s, in time 15/15
- net on the answered games: 44,980 / 60,738
- on the 10 of those we also played live: shadow 14,513 / 30,271 vs live 8,945 / 8,945
- tokens: 337,182 in, 8,078 out over 15 games

## claude-opus-5-5@low

- games answered: 15 of 15 attempted
- latency: median 11.2 s, max 15.3 s, in time 15/15
- net on the answered games: 40,747 / 56,916
- on the 10 of those we also played live: shadow 16,093 / 32,262 vs live 8,945 / 8,945
- tokens: 337,182 in, 7,608 out over 15 games

## claude-sonnet-5-5@low

- games answered: 15 of 15 attempted
- latency: median 10.6 s, max 14.4 s, in time 15/15
- net on the answered games: 29,381 / 45,139
- on the 10 of those we also played live: shadow 1,869 / 17,627 vs live 8,945 / 8,945
- tokens: 337,182 in, 8,786 out over 15 games

## gpt-6.2-sol@low

- games answered: 14 of 14 attempted
- latency: median 14.2 s, max 20.3 s, in time 14/14
- net on the answered games: 38,276 / 54,308
- on the 9 of those we also played live: shadow 9,969 / 26,001 vs live 7,145 / 7,145

