#!/bin/bash
# usage: watch.sh GAME HHMMSS(utc, ~80s after start)
cd /home/diego/Documents/claim-to-fame
until [ "$(date -u +%H%M%S)" \> "$2" ]; do sleep 5; done
journalctl --user -u c2f-bot.service --since "-2 min" --no-pager | cut -c60-200 | grep -E "game $1\]|FAILED|error|Traceback" | grep -v "stage fixed"
.venv/bin/python -m c2f market 2>&1 | grep -v -E "^table|clean|refit" | tail -3
.venv/bin/python scripts/observer/review.py $1 | cut -c1-210
curl -s "https://c2f.public.quantco.cloud/leaderboard/api/matrix?page=1&game_limit=1" | .venv/bin/python -c "
import json,sys
d=json.load(sys.stdin)['items']
print(' | '.join(f\"{i}.{t['team_name'][:8]} {round(t['total']/1000,1)}k\" for i,t in enumerate(d,1)))"
systemctl --user is-active c2f-bot.service
