# main.py

import yaml
from loguru import logger
from data import MarketDataFetcher, IndicatorEngine, DataNormalizer
from ai   import AgentTrainer, InferenceEngine


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError("❌ config.yaml is empty!")
    return config


def main():
    config = load_config()
    logger.info("🚀 Trading Bot Starting...")

    # ── Data pipeline ──────────────────────────
    fetcher    = MarketDataFetcher(config)
    engine     = IndicatorEngine(config)
    normalizer = DataNormalizer(config)

    raw        = fetcher.fetch_stock("AAPL")
    enriched   = engine.compute_all(raw)
    raw_prices = raw["close"].reindex(enriched.index)
    normalized = normalizer.normalize(enriched)
    raw_prices = raw_prices.reindex(normalized.index).dropna()
    normalized = normalized.loc[raw_prices.index]

    # ── PHASE A: Train and save ────────────────
    logger.info("\n" + "="*50)
    logger.info("PHASE A: Training")
    logger.info("="*50)

    trainer = AgentTrainer(config)
    results = trainer.train_and_evaluate(
        normalized_df = normalized,
        raw_prices    = raw_prices,
        timesteps     = 100_000,
    )

    # ── PHASE B: Load and run inference ───────
    logger.info("\n" + "="*50)
    logger.info("PHASE B: Load saved model & run inference")
    logger.info("="*50)

    engine_inf = InferenceEngine(config)
    engine_inf.load_model()

    # Run backtest with the LOADED model
    # (proves save/load works — not just in-memory)
    summary = engine_inf.run_backtest(normalized, raw_prices)

    # ── PHASE C: Single prediction demo ───────
    logger.info("\n" + "="*50)
    logger.info("PHASE C: Single-step prediction (live trading demo)")
    logger.info("="*50)

    # Simulate "today's" observation = last row of data
    last_obs = normalized.iloc[-1].values.astype("float32")

    # Pad with portfolio state (5 zeros = no position, full cash)
    portfolio_state = [1.0, 0.0, 0.5, 0.5, 0.5, 0.0]
    full_obs = __import__("numpy").concatenate(
        [last_obs, portfolio_state]
    ).astype("float32")

    decision = engine_inf.predict(
        observation     = full_obs,
        portfolio_value = config["broker"]["initial_capital"],
    )

    logger.info(f"\n🤖 Bot decision for TODAY:")
    logger.info(f"   Action:     {decision['action_name']}")
    logger.info(f"   Confidence: {decision['confidence']}%")
    logger.info(f"   Certain:    {decision['certain']}")
    logger.info(f"   All probs:")
    for name, pct in decision["probs"].items():
        bar = "█" * int(pct / 5)
        logger.info(f"      {name}: {pct:5.1f}% {bar}")
    
    # ── PHASE C: Single prediction demo ───────
    logger.info("\n" + "="*50)
    logger.info("PHASE C: Single-step prediction (live trading demo)")
    logger.info("="*50)

    last_obs = normalized.iloc[-1].values.astype("float32")
    portfolio_state = [1.0, 0.0, 0.5, 0.5, 0.5, 0.0]
    full_obs = __import__("numpy").concatenate(
        [last_obs, portfolio_state]
    ).astype("float32")

    decision = engine_inf.predict(
        observation     = full_obs,
        portfolio_value = config["broker"]["initial_capital"],
    )

    logger.info(f"\n🤖 Bot decision for TODAY:")
    logger.info(f"   Action:     {decision['action_name']}")
    logger.info(f"   Confidence: {decision['confidence']}%")
    logger.info(f"   Certain:    {decision['certain']}")
    logger.info(f"   All probs:")
    for name, pct in decision["probs"].items():
        bar = "█" * int(pct / 5)
        logger.info(f"      {name}: {pct:5.1f}% {bar}")

    # ── PHASE D: Position Sizing Demo ─────────   ← INSIDE main()
    logger.info("\n" + "="*50)
    logger.info("PHASE D: Position Sizing Demo")
    logger.info("="*50)

    from risk import PositionSizer
    methods = ["fixed_pct", "kelly", "volatility", "fixed_dollar"]
    for method in methods:
        config["risk"]["position_sizing"] = method
        sizer = PositionSizer(config)
        result = sizer.calculate(
            portfolio_value   = 10_000,
            current_price     = 227.48,
            win_rate          = 0.55,
            avg_win           = 1.8,
            avg_loss          = 1.0,
            recent_volatility = 0.018,
        )
        logger.info(
            f"\n   [{method}]"
            f"\n   Fraction:    {result['fraction']*100:.1f}%"
            f"\n   Dollar size: ${result['dollar_size']:,.0f}"
            f"\n   Units:       {result['units']:.2f} shares"
            f"\n   Reasoning:   {result['reasoning']}"
        )


if __name__ == "__main__":
    main()
