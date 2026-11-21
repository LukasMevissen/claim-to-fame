# Shadow model comparison

Shadow models see the same case, prompt and price book as the live model; their
quotes are never submitted. Net = replay of the quotes against the real opponents
(`c2f/replay.py`), t inside the revealed bounds; two columns for the two assumptions
about t where only a lower bound is known (1.15x and 1.4x that bound).
In time = answered within 45 s. Compare rows only on the same games.

## claude-fable-5-2@high

- games answered: 19 of 19 attempted
- latency: median 22.0 s, max 37.2 s, in time 19/19
- net on the answered games: 63,251 / 79,557
- on the 14 of those we also played live: shadow 31,676 / 47,982 vs live 14,689 / 14,689
- tokens: 441,555 in, 24,520 out over 19 games

## claude-fable-5-2@low

- games answered: 19 of 19 attempted
- latency: median 12.1 s, max 28.6 s, in time 19/19
- net on the answered games: 59,498 / 75,667
- on the 14 of those we also played live: shadow 32,554 / 48,723 vs live 14,689 / 14,689
- tokens: 441,555 in, 13,342 out over 19 games

## claude-opus-5-5@high

- games answered: 19 of 19 attempted
- latency: median 21.7 s, max 36.5 s, in time 19/19
- net on the answered games: 59,648 / 59,648
- on the 14 of those we also played live: shadow 33,129 / 33,129 vs live 14,689 / 14,689
- tokens: 441,555 in, 22,632 out over 19 games

## claude-opus-5-5@low

- games answered: 19 of 19 attempted
- latency: median 11.9 s, max 28.6 s, in time 19/19
- net on the answered games: 62,556 / 78,588
- on the 14 of those we also played live: shadow 37,132 / 53,164 vs live 14,689 / 14,689
- tokens: 441,555 in, 12,299 out over 19 games

## claude-sonnet-5-5@high

- games answered: 19 of 19 attempted
- latency: median 18.3 s, max 37.6 s, in time 19/19
- net on the answered games: 34,317 / 34,317
- on the 14 of those we also played live: shadow 5,061 / 5,061 vs live 14,689 / 14,689
- tokens: 441,555 in, 17,833 out over 19 games

## claude-sonnet-5-5@low

- games answered: 19 of 19 attempted
- latency: median 11.0 s, max 22.9 s, in time 19/19
- net on the answered games: 34,382 / 50,139
- on the 14 of those we also played live: shadow 6,869 / 22,627 vs live 14,689 / 14,689
- tokens: 441,555 in, 14,053 out over 19 games

## gpt-6.2-sol@high

- games answered: 19 of 19 attempted
- latency: median 20.8 s, max 38.8 s, in time 19/19
- net on the answered games: 44,072 / 60,241
- on the 14 of those we also played live: shadow 15,055 / 31,224 vs live 14,689 / 14,689

## gpt-6.2-sol@low

- games answered: 14 of 14 attempted
- latency: median 14.2 s, max 20.3 s, in time 14/14
- net on the answered games: 38,276 / 54,308
- on the 9 of those we also played live: shadow 9,969 / 26,001 vs live 7,145 / 7,145

