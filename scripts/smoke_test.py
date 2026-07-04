from agent_idx.config import load_settings
from agent_idx.data import StockDataStore


def main() -> None:
    settings = load_settings()
    store = StockDataStore(settings.parquet_dir, max_rows=settings.max_sql_rows)
    try:
        print(store.list_date_range())
        print("--- BBCA 2022 ---")
        print(store.summarize_period("2022-01-01", "2022-12-31", "BBCA"))
        print("--- top volume 2022 ---")
        print(
            store.run_sql(
                """
                SELECT symbol, SUM(volume) AS vol
                FROM daily_stock
                WHERE date BETWEEN 20220101 AND 20221231
                GROUP BY symbol
                ORDER BY vol DESC
                LIMIT 5
                """
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
