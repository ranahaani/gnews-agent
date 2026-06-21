---
name: gnews
description: Search published journalism (Reuters / BBC / AP / TechCrunch and 141 countries via Google News) and produce cited briefs. Complements /last30days, which covers social signal (Reddit, X, YouTube, TikTok).
version: 0.1.0
author: Muhammad Abdullah <ranahaani@gmail.com>
allowed-tools: Bash, Read, Write
---

# /gnews — journalism layer for Claude Code

`/last30days` answers "what is the internet saying about X?". `/gnews` answers
**"what did journalists report about X?"**. Use them together for the full
picture: social signal + published record.

## When to invoke

- The user asks for news, headlines, coverage, a brief, a sentiment read,
  or a what-changed-since check on a topic, company, person, or event.
- The user says "what does the press say about X" or "what was the
  reporting on X".
- The user wants citations to real published articles, not Reddit threads
  or tweets.

Do **not** invoke for:

- General-knowledge questions answerable from Wikipedia or training data.
- Stock prices, market data, OHLC (not news).
- Social-media reactions (use `/last30days` instead).

## How to invoke

The skill calls the installed `gnews-agent` CLI. The launcher script handles
install detection and falls back to a clear instruction message if the
package is missing.

```bash
bash skills/gnews/scripts/gnews_agent.py ingest "OpenAI"
bash skills/gnews/scripts/gnews_agent.py search "GPT-5 safety" --days 7 --limit 5
bash skills/gnews/scripts/gnews_agent.py brief "OpenAI this week" --days 7
bash skills/gnews/scripts/gnews_agent.py timeline "OpenAI" --days 30
```

`brief` and `sentiment` require an LLM API key (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, etc). `search`, `ingest`, and `timeline` work with no
key — they only use local storage + GNews RSS.

## Output format laws

Every `/gnews` response must follow this shape. No invented `##` section
headers. No emoji. No filler.

```
📰 gnews-agent v0.1.0 · synced YYYY-MM-DD

What journalism says:

<200-400 word prose summary with inline markdown citations to the URLs
returned by gnews-agent. Bold-lead-in paragraphs are fine. Do not
hallucinate facts beyond what's in the cited articles. If sources
disagree, say so explicitly.>

KEY STORIES from the research:
1. [<headline>](<url>) — <publisher>, <date> — <one sentence on why it matters>
2. [<headline>](<url>) — <publisher>, <date> — <one sentence on why it matters>
3. ...

Engine: gnews-agent · GNews 0.8.2 · <n> articles across <n> publishers ·
country=<XX> · language=<xx>
```

## Citation discipline

- Every factual claim in the prose must trace to a `[...](url)` link from
  the JSON returned by `gnews-agent brief`.
- If `brief` returns `article_count: 0`, say so plainly. Do not paper over
  empty results with general knowledge.
- If sources contradict, surface the disagreement and cite both.

## Standard recipe

1. `bash skills/gnews/scripts/gnews_agent.py ingest "<topic>"` — refresh the
   local store.
2. `bash skills/gnews/scripts/gnews_agent.py brief "<topic>" --days 7` —
   get the cited summary.
3. Reformat the JSON response into the output-format-laws shape above.
4. If the user asked about sentiment, also call
   `bash skills/gnews/scripts/gnews_agent.py sentiment "<topic>" --days 14`.
5. Surface any `status: rate_limited` from the ingest step verbatim so the
   user knows coverage may be partial.
