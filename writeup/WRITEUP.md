# Claim to Fame: GemischtesHack (max. 1 page)

*Automatically generated from dashboard, submission logs, and replay evidence (not hand-typed totals).*

**Problem.** For each invoice line, infer the hidden fair threshold t, then set charge a (handyman) and
acceptance limit b (insurer) within 60 s against 11 other teams.

**Method.**
1. One LLM call per case (policy, claim, invoice, photo) returns per line: coverage, relatedness, fair
   gross price m, and uncertainty. The prompt adds a **price book** (revealed t from past games),
   **price priors** for recurring line types, and—since game 30—similar past cases first (`rel` analyst).
2. Belief about t: if likely not covered, we put probability on t = 0; otherwise a lognormal around m.
3. **Acceptance limit b:** accepting is right iff P(t ≥ a) ≥ 2/3, so b is the 1/3 quantile (b = 0 if
   likely uncovered). Exact for the insurer role; does not depend on opponents.
4. **Charge a:** maximise expected revenue against opponent limits learned from public transactions,
   with floor 0.87·m. Staged submit (fixed → fast `gpt-4.1` → full `gpt-6.2-sol`). A learned charge
   agent runs in **shadow** only (not submitted).

**Evidence.** Replay games 1–8 vs real opponents: `gpt-6.2-sol` +26k / +41k net vs `gpt-4.1-mini`
+2k / +17k (oracle 71k). Floor 0.87: +39k / +55k on 12-game replay. Verifiable lines games 1–12:
coverage 18/18 indemnified, 10/10 uncovered; median log price error 0.08. Live ranks games 10–14:
1, 1, 5, 1, 3 after policy-first prompt + price book (previously 6th–10th on mini). Through game 90:
cumulative net **+493,457**, **rank 6/12** (games 1–5 unplayed left a early deficit). Offline since
then: similar-case ordering (+60k replay 30–55), item prior +27k A/B; charge agent kept shadow (3/8
round wins vs fixed rule on games 56–63).

**Reliability.** Unattended `python -m c2f run`: **92/96** logged submits OK. Median time to final
quotes **18 s** (fast **14 s**, fixed **0.4 s**); maximum **50 s** within the 54 s budget. Primary
path is OpenAI stages; Anthropic backup when configured but disabled when credits fail. Version stamp
and timings in `runs/submissions.jsonl`.

**Strengths / weaknesses.** Strengths: policy-first reading, learning from public transactions, exact
b-rule, replay-gated changes, automated pipeline. Weaknesses: missed games 1–5; 0.87 floor tuned on
limited games; lines without price-book precedent often ~30% off; single lognormal per line cannot
split discrete policy cases; leaders ahead on cumulative net; charge agent not yet safe to ship live.