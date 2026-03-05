# ai/trainer.py

import pandas as pd
import numpy as np
from loguru import logger
from .agent import TradingAgent


class AgentTrainer:
    """
    Manages the complete training workflow.

    Single responsibility: orchestrate the
    train → evaluate → save pipeline cleanly.

    Separating trainer from agent keeps each
    class focused on one job.
    """

    def __init__(self, config: dict):
        self.config = config
        logger.info("🎓 AgentTrainer ready")

    def train_and_evaluate(
        self,
        normalized_df: pd.DataFrame,
        raw_prices:    pd.Series,
        timesteps:     int = None,
    ) -> dict:
        """
        Full pipeline:
        1. Build agent
        2. Evaluate BEFORE training (baseline)
        3. Train
        4. Evaluate AFTER training (improvement)
        5. Save model
        6. Return comparison report

        Returns dict with before/after metrics.
        """

        # ── Step 1: Split data into train/test ────
        # First principles: never evaluate on the
        # same data you trained on.
        # Like studying the exact exam questions —
        # you'd score 100% but learn nothing.
        split = self.config["data"]["train_split"]  # 0.8
        split_idx = int(len(normalized_df) * split)

        train_df     = normalized_df.iloc[:split_idx].copy()
        test_df      = normalized_df.iloc[split_idx:].copy()
        train_prices = raw_prices.iloc[:split_idx].copy()
        test_prices  = raw_prices.iloc[split_idx:].copy()

        logger.info(
            f"\n📊 Data split:"
            f"\n   Train: {len(train_df)} rows "
            f"(${train_prices.min():.2f}–${train_prices.max():.2f})"
            f"\n   Test:  {len(test_df)} rows "
            f"(${test_prices.min():.2f}–${test_prices.max():.2f})"
        )

        # ── Step 2: Build agent on TRAINING data ──
        agent = TradingAgent(self.config)
        agent.build(
            normalized_df = train_df,
            raw_prices    = train_prices,
        )

        # ── Step 3: Baseline — BEFORE training ────
        logger.info("\n📊 Evaluating BEFORE training...")
        before = agent.run_episode(test_df, test_prices)
        logger.info(
            f"   Return:    {before['total_return']:+.2f}%"
        )
        logger.info(
            f"   Drawdown:  {before['max_drawdown']:.2f}%"
        )
        logger.info(
            f"   Trades:    {before['total_trades']}"
        )
        logger.info(
            f"   Win rate:  {before['win_rate']:.1f}%"
        )

        # ── Step 4: TRAIN ──────────────────────────
        steps = timesteps or self.config["ai"]["training_timesteps"]
        agent.train(total_timesteps=steps)

        # ── Step 5: Evaluate AFTER training ───────
        logger.info("\n📊 Evaluating AFTER training...")
        after = agent.run_episode(test_df, test_prices)
        logger.info(
            f"   Return:    {after['total_return']:+.2f}%"
        )
        logger.info(
            f"   Drawdown:  {after['max_drawdown']:.2f}%"
        )
        logger.info(
            f"   Trades:    {after['total_trades']}"
        )
        logger.info(
            f"   Win rate:  {after['win_rate']:.1f}%"
        )

        # ── Step 6: Save the trained model ────────
        agent.save()

        # ── Step 7: Build comparison report ───────
        report = self._build_report(before, after)
        self._print_report(report)

        return {
            "agent":  agent,
            "before": before,
            "after":  after,
            "report": report,
        }

    def _build_report(
        self,
        before: dict,
        after:  dict
    ) -> dict:
        """
        Compare before vs after metrics.
        Calculate improvement for each metric.
        """
        def improvement(b, a, higher_is_better=True):
            diff = a - b
            if higher_is_better:
                direction = "✅" if diff > 0 else "❌"
            else:
                direction = "✅" if diff < 0 else "❌"
            return diff, direction

        return_diff,   return_flag   = improvement(
            before["total_return"], after["total_return"]
        )
        drawdown_diff, drawdown_flag = improvement(
            before["max_drawdown"], after["max_drawdown"],
            higher_is_better=False   # lower drawdown = better
        )
        winrate_diff,  winrate_flag  = improvement(
            before["win_rate"], after["win_rate"]
        )

        return {
            "return_before":   before["total_return"],
            "return_after":    after["total_return"],
            "return_diff":     return_diff,
            "return_flag":     return_flag,

            "drawdown_before": before["max_drawdown"],
            "drawdown_after":  after["max_drawdown"],
            "drawdown_diff":   drawdown_diff,
            "drawdown_flag":   drawdown_flag,

            "winrate_before":  before["win_rate"],
            "winrate_after":   after["win_rate"],
            "winrate_diff":    winrate_diff,
            "winrate_flag":    winrate_flag,

            "trades_before":   before["total_trades"],
            "trades_after":    after["total_trades"],
        }

    def _print_report(self, report: dict):
        """
        Print a clean before vs after comparison.
        """
        logger.info("\n" + "="*55)
        logger.info("📋 TRAINING REPORT — BEFORE vs AFTER")
        logger.info("="*55)
        logger.info(
            f"{'Metric':<20} {'Before':>10} "
            f"{'After':>10} {'Change':>10}"
        )
        logger.info("-"*55)
        logger.info(
            f"{'Total Return':<20} "
            f"{report['return_before']:>9.2f}% "
            f"{report['return_after']:>9.2f}% "
            f"{report['return_diff']:>+9.2f}% "
            f"{report['return_flag']}"
        )
        logger.info(
            f"{'Max Drawdown':<20} "
            f"{report['drawdown_before']:>9.2f}% "
            f"{report['drawdown_after']:>9.2f}% "
            f"{report['drawdown_diff']:>+9.2f}% "
            f"{report['drawdown_flag']}"
        )
        logger.info(
            f"{'Win Rate':<20} "
            f"{report['winrate_before']:>9.1f}% "
            f"{report['winrate_after']:>9.1f}% "
            f"{report['winrate_diff']:>+9.1f}% "
            f"{report['winrate_flag']}"
        )
        logger.info(
            f"{'Total Trades':<20} "
            f"{report['trades_before']:>10} "
            f"{report['trades_after']:>10}"
        )
        logger.info("="*55)