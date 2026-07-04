from agent_idx.config import load_settings
from agent_idx.data import StockDataStore


def main() -> None:
    settings = load_settings()
    store = StockDataStore(settings.parquet_dir)
    try:
        print(store.summarize_period("20260602", "20260703", "BBCA"))
        print("---")
        print(
            store.run_sql(
                """
                SELECT date, open_price, high, low, close, change, volume, value,
                       foreign_buy, foreign_sell,
                       foreign_buy - foreign_sell AS net_foreign
                FROM daily_stock
                WHERE symbol = 'BBCA' AND date BETWEEN 20260602 AND 20260703
                ORDER BY date
                """
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
