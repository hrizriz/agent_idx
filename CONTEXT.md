# Agent IDX

A Telegram bot that answers questions about the Indonesian stock market (IDX). It acquires data by driving a logged-in Stockbit session with Playwright, stores it as parquet/DuckDB/Chroma, and answers through an LLM tool-calling loop.

This file is the project's shared language. When a term below has an `_Avoid_` list, the listed words mean the same thing but should not be used — in code, in prompts, or in conversation. Indonesian words appear only where the domain itself is Indonesian.

## Language

### Market

**Symbol**:
An IDX listing code, uppercase, 3–5 letters, no `.JK` suffix. The canonical identifier for everything the project stores.
_Avoid_: ticker, simbol, kode saham, emiten (Indonesian, acceptable in user-facing text only)

**Universe**:
The full set of farmable **Symbols**, defined by `data/symbols_list.txt`.
_Avoid_: symbols list, semua emiten, watchlist

**IHSG**:
The IDX composite index, treated as a pseudo-**Symbol** with its own chart directory.

**Timeframe**:
A bar size, written as an uppercase code: `1M 5M 15M 30M 1H 4H 1D 1W`.
_Avoid_: resolution (reserve that for the TradingView widget label), interval, tf (fine as a local variable, not in prose)

**Bar**:
One OHLCV row at a given **Timeframe**.
_Avoid_: candle, ohlc, kline

**Net foreign**:
`foreign_buy - foreign_sell` for one **Symbol** on one trading day. Chartbit does not carry it; it comes from the overview page or the daily panel.
_Avoid_: foreign flow (use for the topic, not the number), asing, net asing

**Daily panel**:
The one-row-per-symbol-per-day table exposed as the DuckDB view `daily_stock`, backed by `data/parquet/daily_stock_summary_*.parquet`. Its `date` column is an INTEGER `YYYYMMDD`.
_Avoid_: daily summary, market panel, panel (ambiguous — see **Backtest panel**)

**WIB**:
`Asia/Jakarta`, the only timezone in the project. Every timestamp, cron schedule, and trading date is WIB.

### Acquisition

**Farm**:
A bulk acquisition run over many **Symbols**, paced with jitter, tolerant of partial failure, reporting progress as it goes. Charts and fundamentals each have one.
_Avoid_: bulk scrape, harvest, batch job

**Scrape**:
A single page visit that produces markdown. The unit a **Farm** repeats.
_Avoid_: fetch, crawl, pull

**Section**:
One tab of a **Symbol**'s Stockbit page: `overview`, `keystats`, `financials`, `profile`, `chartbit`, `bit`. `company` and `chart` are input aliases for `profile` and `chartbit`; normalise them at the edge and use the canonical six everywhere inside.
_Avoid_: page, tab, sub-page

**Deep scrape**:
A **Section** extractor that scrolls, clicks period/statement toggles, and runs injected JS, rather than dumping page text. Only `keystats`, `financials`, and `profile` have one.

**Chartbit**:
Stockbit's TradingView chart page, and the provenance label for all **Bars** in the project (`source = 'stockbit_chartbit'`). Chartbit carries no **Net foreign**.

**Intercept**:
Capturing **Bars** by listening to Stockbit's XHR JSON responses instead of reading the DOM. A 401/403 on an intercepted URL is the project's evidence that the session expired.

**Stream**:
A Stockbit account timeline that gets scraped as a corpus. Two exist: `reports` (`@StockbitReports`) and `official` (`@Stockbit`, the one that carries midday **Net foreign**). `source=both` scrapes each.
_Avoid_: Stockbit Reports as a name for the corpus (it means only one of the two accounts), feed, timeline

**Post**:
One item in a **Stream**, identified by `posted_at` + `title`, where the title is the first non-empty line of the body.

**Window**:
The date range a **Stream** scrape covers, expressed either as `days` or as `date_from`/`date_to`.
_Avoid_: range, period, rentang

**Browser session**:
The single Playwright Chromium singleton that every scrape shares, owned by one worker thread because Playwright's sync API is thread-affine.

**Persistent profile**:
The Chromium user-data directory (`data/stockbit_profile`) that keeps the Stockbit login alive across bot restarts. Distinct from the `profile` **Section** and from a Stockbit account page.
_Avoid_: user data dir, browser profile, session dir

**Auth sentinel**:
A bare string in tool output signalling that a human must intervene: `NEED_STOCKBIT_CREDENTIALS`, `NEED_STOCKBIT_OTP`. Related stop reasons follow the same shape: `CHARTBIT_BLANK_WHILE_LOGGED_IN`, `NO_OHLCV_INTERCEPTED`, `BROWSER_CLOSED`, `NO_FILE_WRITTEN`.
_Avoid_: error code, flag

**GAP_DATA**:
The required admission that a figure is unavailable. Emitting it is always correct; inventing the number never is.

**FILE: line**:
The line `FILE: <path>` in tool output, meaning "attach this artifact to the Telegram reply". `SAVED:` is the deliberate opposite — written to disk, not attached.

**Sidecar**:
A small JSON metadata record beside a large artifact: `data/fundamentals/{SYMBOL}.json` for a fundamentals **Scrape**, `data/charts/_meta/{SYMBOL}.json` for a chart **Farm**.

### Storage

**Chart store**:
`data/charts.duckdb`, table `ohlcv`, keyed `(symbol, timeframe, ts)`, kept in sync from `data/charts/{SYMBOL}/{TF}.parquet`. Parquet is the source of truth; DuckDB is the query surface.

**Transform store**:
`data/transforms.duckdb`, holding `fundamental_metrics`, `foreign_flow_daily`, and `news_sentiment` — everything derived from scraped text.

**Doc store**:
`data/documents.duckdb`, holding user-supplied documents (PDF/text), deduplicated by content hash.

**Vector store**:
The Chroma collection at `data/chroma` holding **Stream** **Posts** for semantic search. The only embedding-based retrieval in the project.

**Knowledge base**:
The markdown tree under `knowledge/`, searched by keyword occurrence, not embeddings. Categories: `curriculum`, `daily`, `news`, `lessons`, `notes`, `docs`.
_Avoid_: RAG, knowledge store (it is not a vector store — see **Vector store**)

**Curriculum**:
The hand-written IDX market-theory material in `knowledge/curriculum/`. The only **Knowledge base** category that gets a retrieval score boost.

**Metric**:
One `(section, metric, value_raw, value_num)` row parsed out of a keystats or profile **Scrape**.

**Metric group**:
The markdown heading a **Metric** was found under (`current valuation`, `per share`, `shareholder > 1%`, `series:*`, `statement:*`, `ratio:*`). Stored in the `section` column, but not the same concept as a **Section**.

**Indicator**:
A derived technical column on a **Bar**: `sma_5/20/50/200`, `ema_12/26`, `macd*`, `rsi_14`, `vol_sma_20`. The four SMAs together are the **MA ladder**.
_Avoid_: feature, technicals, signal

### Agent runtime

**Analyst agent**:
The single-agent tool-calling loop that answers every question outside `/council`.
_Avoid_: agent (bare, when the **Council** is also in scope), assistant

**Council**:
The three-**Role** deliberation loop reached by `/council`, running until the Critic approves or the round cap is hit.
_Avoid_: multi-agent, 3-agent, panel

**Role**:
One seat in the **Council**. `Researcher` gathers evidence and may call tools; `Analyst` writes the thesis and calls none; `Critic` returns a **Critique verdict**.

**Evidence Pack**:
The Researcher's structured output — request summary, market data, knowledge, news, gaps, tools used. The **Council**'s internal input, never shown as the answer.

**Critique verdict**:
The Critic's `APPROVE` or `REVISE` on a draft. This is the only thing "verdict" means here; it is not a market call.

**Lesson**:
A takeaway the Critic emits that gets written into the **Knowledge base**. The project's only self-improvement write path.

**Query plan**:
The routing decision built by deterministic regex before any LLM call: intent, focus, symbols, sources, retrieval, and the tools that turn is allowed to use.
_Avoid_: plan (bare), route, classification

**Intent**:
The category a question falls into — one of fifteen, e.g. `market_analysis`, `stockbit_chart`, `composite_report`, `bot_meta`. Carried on the **Query plan**.

**Pipeline**:
A system-defined, ordered list of tool calls run in place of letting the LLM choose. Bound to an **Intent**, executed within one chat turn.
_Avoid_: workflow, chain, graph

**Forced path**:
A deterministic handler on the **Analyst agent** that answers without entering the tool loop at all (`_try_force_*`). Sits between a **Pipeline** and the free-form loop.

**Staged action**:
A mutating operation that is described to the user and executed only after a `ya`. Covers project-file writes, **Stream** posts, both **Farms**, and document ingest. Staging is keyed by Telegram chat id and cancelled with `/cancel`.
_Avoid_: pending job, pending write, confirmation gate

**Bulk threshold**:
More than three **Symbols** in one **Farm** request makes it bulk, and bulk always becomes a **Staged action**.

**Composite report**:
A fused multi-source answer — market, news, **Stream**, **Knowledge base** — assembled in Python, then written up by one LLM call. Delivered in chat, never as a PDF.

**PDF report**:
A rendered PDF file, produced only when the user explicitly asks. "Do not make a PDF" is treated as a permanent stored preference.

**Digest**:
A scheduled markdown artifact written into the **Knowledge base** by a **Cron job**: `market`, `news`, or `weekly`.
_Avoid_: report (see **Composite report** / **PDF report**), summary, study sheet

**Learning note**:
A timestamped markdown file capturing something the user told the bot to remember.

### Scheduling

**Cron job**:
A clock-triggered unit of work that runs unattended, writes one markdown artifact, and calls no LLM. Six exist. Distinct from a **Staged action**, which is user-triggered.
_Avoid_: task, scheduled task, job (bare, when staging is also in scope)

**Cron switch**:
Setting a job's crontab env var to `off`/`false`/`0`/`disabled`/empty to unregister it. Only the three scraping jobs honour it.

**Progress callback**:
The throttled channel a **Farm** uses to narrate itself back into Telegram from its worker thread — 120s between chart updates, 900s for fundamentals.

### Evaluation

**Trading scenario**:
A named entry/exit rule evaluated against historical data (foreign-flow streaks, drop bounces, breakouts, MA cross, and so on).
_Avoid_: setup, strategy, scenario (bare — collides with **QA scenario**)

**QA scenario**:
One of the 200 agent-behaviour regression cases: a question plus seed history plus expectations about routing, history-dropping, and answer content. Nothing to do with trading.
_Avoid_: backtest, test case, scenario (bare)

**Backtest panel**:
The long symbol-by-date DataFrame that backtests run over, optionally supplemented from Yahoo Finance where parquet coverage is thin. Distinct from the **Daily panel**.

**Win band**:
The bounded win-rate filter (default 75–85%). The upper bound is deliberate: a win rate above it is read as overfitting, not as a better setup.

## Relationships

- A **Farm** repeats a **Scrape** across the **Universe**; a **Cron job** or a **Staged action** invokes it.
- Every **Scrape** goes through the one **Browser session**, which is kept logged in by the **Persistent profile**.
- A **Scrape** writes markdown plus a **Sidecar**; the markdown becomes **Metrics** in the **Transform store**, **Bars** in the **Chart store**, or **Posts** in the **Vector store**.
- A question becomes a **Query plan**, which carries an **Intent**, which may bind a **Pipeline** or hit a **Forced path**; anything left over runs the **Analyst agent** tool loop.
- `/council` bypasses that entirely: Researcher produces an **Evidence Pack**, Analyst produces a thesis, Critic returns a **Critique verdict** and possibly a **Lesson**.

## Flagged ambiguities

Unresolved collisions in the current code. Prefer the term named here; treat the rest as legacy.

- **"report"** carries five unrelated meanings. Resolved: the fused answer is a **Composite report**, the file is a **PDF report**, the scraped corpus is a **Stream**, the cron artifact is a **Digest**, the Council's research output is an **Evidence Pack**. Only `@StockbitReports` keeps the word, as an account name.
- **"job"** means both a **Cron job** and a **Staged action** awaiting confirmation. Qualify it every time.
- **"scenario"** means both a **Trading scenario** and a **QA scenario**, and "backtest" covers both strategy simulation and agent QA. Always qualify.
- **"universe"** is also used for the backtest's market-cap filter mode. Resolved: **Universe** is the symbol list; the filter is a cap filter.
- **"profile"** is the Chromium **Persistent profile**, the `profile` **Section**, and a Stockbit account page. Always qualify.
- **"section"** is a **Section** in the scraper and a **Metric group** in the transform layer, sharing a column name.
- **"farm" vs "scrape"**: charts keep the distinction, fundamentals do not (`fund_farm.scrape_one`). The distinction above is the intended one.
- **No structured market call exists.** There is no buy/sell/hold enum, conviction score, or target-price field anywhere — every recommendation is prose under a "Pandangan" heading. Do not write code that assumes a parseable stance.
