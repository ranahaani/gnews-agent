-- gnews-agent SQLite schema (v1)
--
-- Dedup key is `sha256(title_slug + "|" + publisher_norm)` per system design.
-- URL is intentionally excluded from the composite key (Google News surfaces
-- the same article under multiple URL variants); the UNIQUE index on
-- articles.url_hash is a backstop for the exact-URL collision case only.

CREATE TABLE IF NOT EXISTS articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    url             TEXT    NOT NULL,
    url_hash        TEXT    NOT NULL UNIQUE,
    publisher_name  TEXT,
    publisher_href  TEXT,
    published_date  TEXT,                  -- original publisher date string (may be RFC822)
    published_at    TEXT,                  -- normalised ISO-8601 ("YYYY-MM-DDTHH:MM:SS") for sort/filter
    summary         TEXT,
    full_text       TEXT,
    country         TEXT,
    language        TEXT,
    topic           TEXT,
    embed_model     TEXT    NOT NULL,
    embed_dim       INTEGER NOT NULL,
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_articles_topic ON articles(topic);
CREATE INDEX IF NOT EXISTS idx_articles_date  ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_lang  ON articles(language, country);

CREATE TABLE IF NOT EXISTS dedup_index (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    composite_key   TEXT    NOT NULL UNIQUE,    -- sha256(title_slug + "|" + publisher_norm)
    title_slug      TEXT    NOT NULL,           -- title.lower().strip().replace(' ', '-')
    publisher_norm  TEXT    NOT NULL,           -- publisher.lower().strip()
    article_id      INTEGER REFERENCES articles(id) ON DELETE SET NULL,
    seen_count      INTEGER DEFAULT 1,
    first_seen      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_dedup_title_pub ON dedup_index(title_slug, publisher_norm);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    topic             TEXT,
    method            TEXT,                       -- get_news | get_top_news | get_news_by_topic | ...
    fetched           INTEGER DEFAULT 0,
    new_articles      INTEGER DEFAULT 0,
    skipped_dupes     INTEGER DEFAULT 0,
    status            TEXT CHECK(status IN ('success', 'failed', 'partial', 'rate_limited')),
    error_message     TEXT,
    started_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_seconds  REAL
);

-- FTS5 mirror for keyword-fallback search (Stage 2 uses this).
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    title,
    summary,
    full_text,
    content='articles',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS articles_fts_ai AFTER INSERT ON articles BEGIN
    INSERT INTO articles_fts(rowid, title, summary, full_text)
    VALUES (new.id, new.title, new.summary, new.full_text);
END;

CREATE TRIGGER IF NOT EXISTS articles_fts_ad AFTER DELETE ON articles BEGIN
    INSERT INTO articles_fts(articles_fts, rowid, title, summary, full_text)
    VALUES('delete', old.id, old.title, old.summary, old.full_text);
END;

CREATE TRIGGER IF NOT EXISTS articles_fts_au AFTER UPDATE ON articles BEGIN
    INSERT INTO articles_fts(articles_fts, rowid, title, summary, full_text)
    VALUES('delete', old.id, old.title, old.summary, old.full_text);
    INSERT INTO articles_fts(rowid, title, summary, full_text)
    VALUES (new.id, new.title, new.summary, new.full_text);
END;
