# gnews-agent

> Persistent, semantic news intelligence layer for AI agents — built on top of [GNews](https://github.com/ranahaani/GNews) (106k+ monthly PyPI downloads, 141 countries, 41 languages).

The journalism layer your AI agent is missing. Fetch published news from Reuters, BBC, AP, TechCrunch, and the 141-country Google News graph; dedup it; embed it; store it persistently; and query it semantically — over a Python API, a CLI, or an MCP server that drops straight into Claude.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [CLI](#cli)
  - [MCP server](#mcp-server)
  - [Claude Code skill](#claude-code-skill)
  - [Python library](#python-library)
- [Configuration](#configuration)
- [Docker](#docker)
- [Design notes](#design-notes)
- [Status](#status)
- [Pairs well with `/last30days`](#pairs-well-with-last30days)
- [License](#license)

## Installation

```bash
pip install gnews-agent
```

Optional extras:

```bash
pip install "gnews-agent[openai]"     # OpenAI embedding backend
pip install "gnews-agent[fulltext]"   # full-article extraction via trafilatura
pip install "gnews-agent[lance]"      # LanceDB vector backend
pip install "gnews-agent[qdrant]"     # Qdrant server backend
pip install "gnews-agent[evals]"      # DeepEval + Langfuse (v2)
```

Verify the install:

```bash
gnews-agent --version
gnews-agent stats              # → {"total_articles": 0, ...}
```

## Usage

The library exposes the same six capabilities through every surface:
`ingest`, `search`, `timeline`, `brief`, `sentiment`, `stats`. Pick the
surface that matches how the rest of your system already works.

### CLI

```bash
gnews-agent ingest "OpenAI" --method get_news
gnews-agent search "GPT-5 safety" --days 7 --limit 5
gnews-agent brief  "OpenAI this week" --days 7
gnews-agent sentiment "Tesla" --days 14 --timeline
gnews-agent timeline  "OpenAI" --days 30
gnews-agent stats
gnews-agent serve --transport stdio
```

Every command emits JSON to stdout. Use `--no-pretty` for one-line output
(pipe through `jq` or feed straight into another agent).

### MCP server

`gnews-agent serve` exposes five tools — `search_news`, `get_brief`,
`get_sentiment`, `get_timeline`, `monitor_topic` — and three resources —
`news://latest/{topic}`, `news://sentiment/{topic}`,
`news://timeline/{topic}`. Works in any MCP client.

**Claude Code (CLI):**

```bash
claude mcp add gnews-agent -- gnews-agent serve --transport stdio
```

Then ask Claude:

> *"Use gnews-agent to ingest the latest reporting on OpenAI, then give me a cited brief on what changed this week."*

**Claude Desktop / Cursor / Windsurf:**

Add this block to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or the platform equivalent:

```json
{
  "mcpServers": {
    "gnews-agent": {
      "command": "gnews-agent",
      "args": ["serve", "--transport", "stdio"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "GNEWS_AGENT_LLM_PROVIDER": "anthropic",
        "GNEWS_AGENT_LLM_MODEL": "claude-3-5-haiku-latest"
      }
    }
  }
}
```

Restart the client. If `gnews-agent` isn't on `$PATH`, use the absolute
path from `which gnews-agent`.

**HTTP transport (LangGraph, custom agents):**

```bash
gnews-agent serve --transport http --port 8000
```

### Claude Code skill

The `/gnews` skill wraps the MCP server with prompt scaffolding and a
citation-formatted output template.

```
/plugin marketplace add ranahaani/gnews-agent
/gnews OpenAI this week
/gnews Pakistan economy sentiment
/gnews Tesla vs Rivian coverage last 30 days
```

Requires `pip install gnews-agent` on the same machine.

### Python library

Drop into any LangGraph, CrewAI, or vanilla Python pipeline.

```python
from gnews_agent import NewsMemory

memory = NewsMemory()                              # SQLite + Chroma, persistent
memory.ingest("OpenAI", method="get_news")         # fetch + dedup + embed + store
results = memory.search("GPT-5 safety", days=7)    # semantic re-ranked by recency
timeline = memory.timeline("OpenAI", days=30)      # SQL-only
brief = memory.brief("OpenAI this week", days=7)   # cited LLM summary
sentiment = memory.sentiment("Tesla", days=14)
print(memory.stats())
```

`NewsMemory` accepts a `NewsMemoryConfig` for full control over paths,
embedder, vector backend, and LLM provider:

```python
from gnews_agent import NewsMemory, NewsMemoryConfig

cfg = NewsMemoryConfig(
    db_path="~/news/news.db",
    vector_path="~/news/chroma",
    embed_model="all-MiniLM-L6-v2",
    llm_provider="anthropic",
    llm_model="claude-3-5-haiku-latest",
)
memory = NewsMemory(config=cfg)
```

## Configuration

`search`, `ingest`, `timeline`, and `stats` are keyless — they only need
GNews (free, no API key) and local storage. `brief` and `sentiment` are
LLM-powered and need a provider key.

| Variable | Used by | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `brief`, `sentiment`, `gnews-agent[openai]` embeddings | Pass via env or `LLMConfig`. |
| `ANTHROPIC_API_KEY` | `brief`, `sentiment` | Recommended for higher TPM on the free tier. |
| `GROQ_API_KEY` | `brief`, `sentiment` | Cheapest + fastest; tighter token-per-minute limits. |
| `GEMINI_API_KEY` | `brief`, `sentiment` | |
| `GNEWS_AGENT_LLM_PROVIDER` | CLI default for `--llm-provider` | `anthropic` / `openai` / `groq` / `gemini` / `ollama`. |
| `GNEWS_AGENT_LLM_MODEL` | CLI default for `--llm-model` | Provider-qualified or bare model id. |
| `GNEWS_AGENT_HOME` | Docker entrypoint | Mounts `/data` for SQLite + Chroma persistence. |

Copy [`.env.example`](.env.example) to `.env` and fill in only the keys
you need. Ollama runs locally with no key at all.

## Docker

```bash
cd docker
docker compose up --build
# HTTP MCP listening on http://localhost:8000
```

The image pre-downloads `all-MiniLM-L6-v2` at build time so the first
ingest doesn't pay the ~80MB cold-start.

## Design notes

- **Dedup key = `sha256(title_slug + "|" + publisher_norm)`.** URL is
  excluded because Google News surfaces the same article under multiple
  URL variants (locale params, tracking suffixes, redirector vs
  resolved). Title-slug + publisher catches those without merging
  legitimately distinct outlets.
- **No semantic cosine dedup at ingestion.** Reuters and BBC reporting
  the same event are kept as separate rows. Cosine similarity is
  reserved for query-time re-ranking inside `brief()`.
- **Embedding model + dimension are recorded on every vector row.** The
  store refuses cross-model queries via `EmbeddingDimMismatchError`. To
  switch from sentence-transformers to OpenAI embeddings, use a
  separate `vector_path` or wipe the existing collection.
- **Story clustering is v2.** `brief()` ranks by recency + similarity
  and passes the top-N to the LLM, no clustering.
- **Webhook URL validation for `monitor_topic`.** RFC1918, loopback,
  link-local, and cloud-metadata IPs are refused. HTTPS required in
  MCP-exposed mode.
- **GNews ≥0.8.2** is required for the built-in 429 retry +
  exponential backoff. `gnews-agent` layers per-topic 1s spacing and
  topic cooldown on top of that.

## Status

- **v0.1.0** — Library + CLI + MCP server + Claude Code skill scaffold.
  End-to-end ingest → search → brief works against a real LLM key.
  83 unit + 24 integration tests pass.
- **v2 deferred** — DeepEval `FaithfulnessMetric`, Langfuse tracing,
  CI eval gate, story clustering, `get_entities` MCP tool,
  Telegram/Slack/Email alert channels, LanceDB + Qdrant backends,
  multi-language summaries.

## Pairs well with `/last30days`

`/last30days` searches **what people say** (Reddit, X, YouTube, TikTok,
Hacker News). `gnews-agent` searches **what journalism reports**
(Reuters, BBC, AP, TechCrunch). Different signals from different
sources:

```
/last30days OpenAI GPT-5     → social signal
/gnews      OpenAI GPT-5     → published record
```

Install both and ask Claude to combine them for the full picture.

## License

MIT — see [LICENSE](LICENSE).
