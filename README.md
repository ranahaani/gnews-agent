# gnews-agent

> Persistent, semantic, evaluated news intelligence layer for AI agents — built on top of [GNews](https://github.com/ranahaani/GNews) (106k+ monthly PyPI downloads, 141 countries, 41 languages).

`/last30days` searches **what people say** (Reddit, X, YouTube, TikTok). `gnews-agent` searches **what journalism reports** (Reuters, BBC, AP, TechCrunch, and the 141-country Google News graph). Use them together for the complete picture.

```
/last30days OpenAI GPT-5     → social signal
/gnews      OpenAI GPT-5     → published record
```

## What you get

- `pip install gnews-agent` and `from gnews_agent import NewsMemory` — drop into any LangGraph, CrewAI, or vanilla-Python pipeline.
- Persistent SQLite + ChromaDB storage — no fetch-and-discard. Articles dedup across runs on `title-slug + publisher`, so Reuters and BBC covering the same event stay as two distinct rows (multi-outlet framings preserved).
- Semantic vector search out of the box (sentence-transformers, 384-dim, no server, no sidecar).
- LLM-cited briefs + sentiment via [LiteLLM](https://github.com/BerriAI/litellm) — OpenAI / Anthropic / Groq / Ollama, switchable by config.
- FastMCP server (`gnews-agent serve`) — Claude Desktop, Cursor, Windsurf, LangGraph can call the tools directly.
- Claude Code skill (`/gnews`) — installable from the marketplace, complements `/last30days`.

## Install

```bash
pip install gnews-agent

# optional extras
pip install "gnews-agent[openai]"     # OpenAI embedding backend
pip install "gnews-agent[fulltext]"   # full-article extraction via trafilatura
pip install "gnews-agent[lance]"      # LanceDB vector backend
pip install "gnews-agent[qdrant]"     # Qdrant server backend
pip install "gnews-agent[evals]"      # DeepEval + Langfuse (v2)
```

## BYO LLM key

`search`, `ingest`, and `timeline` are **keyless** — they only need GNews (free, no API key) and local storage.

`brief` and `sentiment` are **LLM-powered** and need an API key. Set whichever provider you use:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk_...
# or run Ollama locally — no key needed
```

## Three ways to use it

### 1. Python library

```python
from gnews_agent import NewsMemory

memory = NewsMemory()                              # SQLite + Chroma, persistent
memory.ingest("OpenAI", method="get_news")         # fetch + dedup + embed + store
results = memory.search("GPT-5 safety", days=7)    # semantic re-ranked by recency
timeline = memory.timeline("OpenAI", days=30)      # SQL-only, keyless
brief = memory.brief("OpenAI this week", days=7)   # cited LLM summary (needs key)
sentiment = memory.sentiment("Tesla", days=14)     # (needs key)
print(memory.stats())
```

### 2. CLI

```bash
gnews-agent ingest "OpenAI" --method get_news
gnews-agent search "GPT-5 safety" --days 7 --limit 5
gnews-agent brief  "OpenAI this week" --days 7
gnews-agent sentiment "Tesla" --days 14 --timeline
gnews-agent timeline  "OpenAI" --days 30
gnews-agent stats
gnews-agent serve --transport stdio
```

All commands emit JSON to stdout by default. Add `--no-pretty` for one-line output (pipe through `jq` or feed to another agent).

### 3. MCP server

`gnews-agent serve --transport stdio` exposes five tools (`search_news`, `get_brief`, `get_sentiment`, `get_timeline`, `monitor_topic`) and three resources (`news://latest/{topic}`, `news://sentiment/{topic}`, `news://timeline/{topic}`).

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the Windows equivalent:

```json
{
  "mcpServers": {
    "gnews-agent": {
      "command": "gnews-agent",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

Restart Claude Desktop and ask: *"Show me what journalism reported about OpenAI this week."*

### 4. Docker (self-host / team)

```bash
cd docker
docker compose up --build
# HTTP MCP listening on http://localhost:8000
```

The image pre-downloads `all-MiniLM-L6-v2` at build time so first ingest doesn't pay the ~80MB cold-start.

## Critical design choices (so you know what you're getting)

- **Dedup key = `sha256(title_slug + "|" + publisher_norm)`** — URL excluded. Google News surfaces the same article under multiple URL variants (locale params, tracking suffixes, redirector vs resolved). Title-slug + publisher catches those without merging legitimately distinct outlets.
- **No semantic cosine dedup at ingestion.** Reuters and BBC reporting the same event are kept as separate rows. Cosine similarity is reserved for query-time re-ranking inside `brief()`.
- **Embedding model + dimension are recorded on every vector row** — the store refuses cross-model queries via `EmbeddingDimMismatchError`. To switch from sentence-transformers to OpenAI embeddings, use a separate `vector_path` or wipe the existing collection.
- **Story clustering is v2** — `brief()` ranks by recency + similarity and passes the top-N to the LLM, no clustering.
- **Webhook URL validation for `monitor_topic`** — RFC1918, loopback, link-local, and cloud-metadata IPs are refused. HTTPS required in MCP-exposed mode.
- **GNews ≥0.8.2** is required for built-in 429 retry + exponential backoff. On top of that, `gnews-agent` adds per-topic 1s spacing and topic cooldown when retries are exhausted.

## Environment variables

| Var | Meaning |
|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GROQ_API_KEY` | Provider key for `brief`/`sentiment`. Pick the one that matches `--llm-provider`. |
| `GNEWS_AGENT_LLM_PROVIDER` | Default `--llm-provider` flag value. |
| `GNEWS_AGENT_LLM_MODEL` | Default `--llm-model` flag value. |
| `GNEWS_AGENT_HOME` | Used by the Docker image as the persistence root (mounts `/data`). |

## Status

- **v0.1.0** — Library + CLI + MCP server + Claude Code skill scaffold. End-to-end ingest → search → brief works against a real `OPENAI_API_KEY`. 77/77 tests pass.
- **v2 deferred** — DeepEval `FaithfulnessMetric`, Langfuse tracing, CI eval gate, story clustering, `get_entities` MCP tool, Telegram/Slack/Email alert channels, LanceDB + Qdrant backends, multi-language summaries.

## License

MIT — see [LICENSE](LICENSE).
