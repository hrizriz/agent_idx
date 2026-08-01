# IDX data sources on GitHub (streaming + technicals)

Research note: how open-source IDX projects typically acquire market data for
charts, “realtime”, and technical indicators. Claims below cite primary
READMEs / source trees, not secondary blogs.

## Pattern summary

| Tier | Typical source | Streaming? | Technicals |
|------|----------------|------------|------------|
| Free unofficial | Yahoo Finance (`.JK`) | Poll / rare Yahoo WS | Computed locally |
| Session scrape | Stockbit Exodus REST (+ JWT) | HTTP poll, not public WS | Local on bars |
| Static dump | Dataset-Saham-IDX (from idx.co.id) | No | Offline only |
| Paid vendor | Invezgo, Kun/StockerAPI, iTick | Often true WS or vendor intraday | Client-side, sometimes vendor series |

Dominant free stack: **Yahoo bars → local indicators**. True tick WebSocket for
IDX is rare and usually commercial. Stockbit appears mostly as a **broker /
orderbook** side-channel, not as Chartbit OHLCV authority (except stacks like
`agent_idx`).

## Repos inspected

### YogaSakti/mcp-idx
- Prices/OHLCV: `yfinance` only.
- Indicators: `pandas-ta` on Yahoo bars (RSI, MACD, SMA/EMA, BB, ADX, Ichimoku, …).
- Foreign flow tools are Yahoo proxies, labeled not real BEI foreign flow.
- No streaming; intraday = polled Yahoo history.
- https://github.com/YogaSakti/mcp-idx

### baguskto/saham-mcp
- HIGH: GitHub dataset (daily history) + Yahoo Finance (`yahoo-finance2`).
- MEDIUM: web scraper stub (mostly unimplemented).
- Indicators: hand-rolled TypeScript SMA/EMA/RSI/MACD/BB.
- No streaming (HTTP + cache TTLs).
- https://github.com/baguskto/saham-mcp

### wildangunawan/Dataset-Saham-IDX
- Static daily CSV universe for research (OHLCV + foreign buy/sell, etc.).
- Provenance: processed from idx.co.id; BEI ToS now restricts scraping;
  updates are manual/periodic.
- Not for intraday or commercial redistribution.
- https://github.com/wildangunawan/Dataset-Saham-IDX

### StockerAPI / Kun
- Commercial token API: HTTP historical OHLCV + snapshot; WebSocket live quotes.
- Repo is product/docs, not an open feed implementation.
- Indicators left to the client.
- https://github.com/StockerAPI/indonesia-stock-market-api

### Invezgo/invezgo-go-sdk
- Paid REST (API key). Intraday chart, order book, running trade, broker
  packages, vendor indicator charts (BDM/foreign/ratio/ritel).
- No WebSocket in the public SDK tree; “realtime” is vendor REST during session.
- https://github.com/Invezgo/invezgo-go-sdk

### FadelSearr/Dellmology
- Price fallback: Stockbit orderbook → Yahoo 1m chart API → fail.
- Stockbit Exodus REST with Bearer JWT; “stream” endpoints are HTTP polls.
- Local Go engine can simulate ticks / SSE; real Stockbit WS is aspirational.
- Indicators local; Yahoo for bars, Stockbit for broker.
- https://github.com/FadelSearr/Dellmology

### Other
- [dann2907/idxterminal](https://github.com/dann2907/idxterminal) — idx.co.id +
  yfinance; Lightweight Charts; own WebSocket to push UI (not exchange feed).
- [sulthonzh/idx-finance](https://github.com/sulthonzh/idx-finance) — Yahoo +
  idx.co.id; no tick stream.
- [VYDev37/go-tvscanner-api](https://github.com/VYDev37/go-tvscanner-api) —
  TradingView screener for Indonesia (poll, not bar stream).
- [sukirman1901/Pulse-CLI](https://github.com/sukirman1901/Pulse-CLI) — Yahoo TA +
  manual Stockbit JWT for broker.
- [yahoofinancelive/yliveticker](https://github.com/yahoofinancelive/yliveticker)
  — Yahoo Finance WebSocket (global; usable with `.JK`).

## Relevance to agent_idx

`agent_idx` is unusual among GitHub IDX projects: **Chartbit is the bar
source of truth**, indicators are local, and Stockbit session is a persistent
Playwright profile rather than a pasted JWT. Closest OSS cousins for the
Stockbit half are Dellmology and Pulse-CLI, but they still default Yahoo for
OHLCV.

Practical upgrade paths if “streaming lengkap + teknikal” is required:

1. Keep Chartbit bars + local indicators; add a paid WS only for live quotes.
2. Replace OHLCV feed with Invezgo/Kun/iTick; keep local TA.
3. Hybrid like Dellmology: Yahoo/IDX bars + Stockbit broker/foreign only.

## Sources consulted

Primary: repository READMEs and source files linked above (fetched/inspected
2026-08-01 via GitHub). Secondary marketing pages (kun.pro, iTick blog) only
where the GitHub repo itself is documentation-only.
