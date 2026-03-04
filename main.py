# main.py

import yaml
from loguru import logger
from data import (
    MarketDataFetcher,
    IndicatorEngine,
    DataNormalizer,
    SyntheticDataGenerator
)


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError("❌ config.yaml is empty!")
    return config


def main():
    config = load_config()
    logger.info("🚀 Trading Bot Starting...")

    # ── Test synthetic data ────────────────────
    generator = SyntheticDataGenerator(config)

    # Generate one of each regime
    bull     = generator.generate("bull",     n_days=365)
    bear     = generator.generate("bear",     n_days=365)
    sideways = generator.generate("sideways", n_days=365)
    volatile = generator.generate("volatile", n_days=365)

    # Generate mixed regime (most realistic)
    mixed = generator.generate_mixed_regime(n_days=1000)

    # ── Show price ranges for each regime ─────
    logger.info("\n📊 Regime Summary:")
    for name, df in [
        ("Bull",     bull),
        ("Bear",     bear),
        ("Sideways", sideways),
        ("Volatile", volatile),
        ("Mixed",    mixed)
    ]:
        start = df["close"].iloc[0]
        end   = df["close"].iloc[-1]
        high  = df["high"].max()
        low   = df["low"].min()
        chg   = ((end / start) - 1) * 100

        logger.info(
            f"   {name:10} | "
            f"Start: ${start:7.2f} | "
            f"End: ${end:7.2f} | "
            f"Change: {chg:+6.1f}% | "
            f"Range: ${low:.2f}-${high:.2f}"
        )

    # ── Run the full pipeline on synthetic data ─
    logger.info("\n🔄 Running full pipeline on synthetic bull data...")

    engine     = IndicatorEngine(config)
    normalizer = DataNormalizer(config)

    enriched   = engine.compute_all(bull)
    normalized = normalizer.normalize(enriched)

    logger.info(f"\n✅ Synthetic data pipeline complete!")
    logger.info(f"   Raw rows:        {len(bull)}")
    logger.info(f"   After indicators:{len(enriched)}")
    logger.info(f"   After normalize: {len(normalized)}")
    logger.info(f"   Columns:         {len(normalized.columns)}")
    logger.info(
        f"   Value range:     "
        f"{normalized.min().min():.3f} to "
        f"{normalized.max().max():.3f}"
    )


if __name__ == "__main__":
    main()