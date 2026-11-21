# QuantCo: Claim to Fame (case brief, team copy)

Source: START Hack League Munich 2026 case page
https://app.hackleague.org/events/start/cases/3e0c84d3-2ee8-44d8-bc60-8e257fd6cdaa
(copied 2026-11-21 for internal team use; check the page for updates)

Room **1200** · Discord **#quantco** · Team: **GemischtesHack** (Lukas Mevissen, Diegocentrico, Natalia Milanova)

## The scenario

In insurance, a *claim* is a request by a policyholder for payment after damage
(burst pipe, car accident, ...). Handymen and repair garages issue invoices,
policyholders submit them, and the insurer decides whether to pay.

For every invoiced line item the insurer must judge whether it is:

- **Covered** by the policy,
- **Related** to the reported damage, and
- **Reasonably priced** (total per line item = quantity × unit price).

## Round mechanics

1. **A case drops**: an insurance policy, a damage description, an invoice without prices, sometimes photos.
2. **You price it**: send a charge price and an acceptance limit for every line item to the QuantCo API **within exactly one minute**.
3. **Everyone plays everyone**: our prices meet every other team's, in both roles.

Details:

- A case is released on a schedule. Once out, fetch the **decryption key** from the API, decrypt the case `.zip`, analyse, submit, all within one minute.
- Each case contains `policy.txt`, `description.txt`, `invoices.pdf` and sometimes `images.png`.
- **Later submissions overwrite earlier ones.** Pick one person to run and submit, and plan how to schedule runs across the whole tournament.

## Our two roles (per line item)

| Role | Variable | Meaning |
|---|---|---|
| Invoice issuer (handyman) | **a** (charge price) | amount we charge the other teams |
| Claim handler (insurer) | **b** (acceptance limit) | max we pay when another team bills us for the same item |

- Every line item has a **secret fair-value threshold t**: the max price a claims expert considers appropriate.
- Items **not covered** by the policy have **t = 0**.
- **Always submit the gross total** (*brutto Gesamtbetrag*, incl. VAT) for the whole line item. Never net, never per unit.
- Defaults if nothing is submitted: **a = 0, b = 0**.
- Secret **payment cap c ≥ 4t** per item.

## Payoffs (amount the insurer pays the handyman)

| | a ≤ t (price okay) | a > t (price fraudulent) |
|---|---|---|
| **a ≤ b (accepted)** | a | min(a, c) |
| **a > b (rejected)** | 1.5·a (wrongful reject: a + 0.5a damages) | 0 |

Overcharge and you earn nothing; set b too low and you pay 1.5× penalties. Find t and you win on both sides.

## Setup (from the brief)

1. Register the team to get the **API key**. Do not share it. (Lives in `.env`, never committed.)
2. Get the cases from the **shared Drive folder** (encrypted `.zip` per case, plus test **case 0**). Put them in `data/cases/`.
3. Run the provided **`starter_script.py`** (needs `TEAM_API_KEY`, `OPENAI_KEY`, `requests`, `openai`, and 7-Zip).
   Easiest: `pixi install` then `pixi run python starter_script.py`. Details in **`API_HANDBOOK.md`**.
   Put both files in `vendor/quantco/` (see `vendor/quantco/README.md`).
4. Any language, library or framework is fine.

Leaderboard, game schedule, matchups, transactions: https://c2f.public.quantco.cloud/leaderboard/
(at 13:28 on 2026-11-21 the first game was ~26 min away; no completed games yet)

## Fair play

- No cross-team collaboration on prices, limits, strategies or case analysis.
- No sharing keys; never submit for another team.
- No attacking the platform (no early keys, no peeking at other submissions or thresholds, no overloading the API).
- Inside the team anything goes: tooling, LLMs, manual work, domain research.
- Cheating means disqualification. Unsure? Ask the QuantCo team.

## Hand-in

Shortly before the challenge ends: **short write-up (max. 1 page)** and a **pitch**. Draft lives in `writeup/`.

## Judging

- **Tournament result (70%)**: ranked by cumulative net payoff (received as handyman minus paid as insurer) across all games. Top team 70 points, others scaled linearly by rank to 0. Unsubmitted items count as a = 0, b = 0.
- **Presentation and method (30%)**, jury 0 to 30 points for write-up and pitch:
  - Method and uncertainty: how we estimated t and derived a and b
  - Evidence for our choices
  - Resourcefulness: smart and economical beats brute force, no token maxing
  - Reliability and automation: runs dependably within the one minute
  - Clarity and honesty: strategy and actual results match, strengths and weaknesses stated

## Prizes

1st: AirPods (QuantCo) + €300 (START Munich) · 2nd: €200 · 3rd: €100 · all teams: QuantCo merch.
