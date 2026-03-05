# ai/reward.py

import numpy as np
from collections import deque
from loguru import logger


class RewardFunction:
    """
    Carefully designed reward function for trading.

    First principle: the bot does EXACTLY what
    you reward. Design every component intentionally.

    Components:
    1. Portfolio change   → reward making money
    2. Drawdown penalty   → punish losing it
    3. Cost penalty       → punish over-trading
    4. Sharpe bonus       → reward consistency
    """

    def __init__(self, config: dict):
        """
        Load reward weights from config.
        Each weight controls how much each
        component influences the final reward.
        """
        # We'll add these to config.yaml shortly
        self.portfolio_weight  = 1.0    # main signal
        self.drawdown_weight   = 0.5    # drawdown penalty
        self.cost_weight       = 0.1    # transaction cost penalty
        self.sharpe_weight     = 0.3    # consistency bonus

        self.initial_capital   = config["broker"]["initial_capital"]

        # Rolling window of recent returns
        # Used to calculate mini Sharpe ratio
        self.return_window     = 20
        self.recent_returns    = deque(maxlen=self.return_window)

        # Track peak portfolio value for drawdown calc
        self.peak_value        = self.initial_capital

        logger.info("🏆 RewardFunction ready")
        logger.info(f"   Portfolio weight:  {self.portfolio_weight}")
        logger.info(f"   Drawdown weight:   {self.drawdown_weight}")
        logger.info(f"   Cost weight:       {self.cost_weight}")
        logger.info(f"   Sharpe weight:     {self.sharpe_weight}")

    def reset(self):
        """
        Clear history at the start of each episode.
        Fresh slate for new training run.
        """
        self.recent_returns = deque(maxlen=self.return_window)
        self.peak_value     = self.initial_capital

    def calculate(
        self,
        prev_value:   float,
        curr_value:   float,
        trade_cost:   float,
        trade_taken:  bool,
    ) -> tuple:
        """
        Calculate the full reward for one step.

        prev_value:   portfolio value last step
        curr_value:   portfolio value this step
        trade_cost:   dollar cost of any trade this step
        trade_taken:  did we actually trade this step?

        Returns:
            reward:      final combined reward value
            breakdown:   dict showing each component
                         (useful for debugging and logging)
        """

        # ── Component 1: Portfolio Change ─────────
        # How much did our portfolio change this step
        # as a fraction of initial capital?
        # Using initial_capital keeps scale consistent
        # throughout training regardless of account size.

        pct_change = (
            (curr_value - prev_value) / self.initial_capital
        )
        reward_portfolio = self.portfolio_weight * pct_change

        # ── Component 2: Drawdown Penalty ─────────
        # Update peak value (peak only goes UP)
        if curr_value > self.peak_value:
            self.peak_value = curr_value

        # Current drawdown from peak
        if self.peak_value > 0:
            drawdown = (
                self.peak_value - curr_value
            ) / self.peak_value
        else:
            drawdown = 0.0

        # Squared penalty — small drawdowns barely matter,
        # large drawdowns matter A LOT
        reward_drawdown = (
            -self.drawdown_weight * (drawdown ** 2)
        )

        # ── Component 3: Transaction Cost Penalty ─
        # Penalize the actual cost of trading
        # as a fraction of initial capital
        cost_fraction    = trade_cost / self.initial_capital
        reward_cost      = -self.cost_weight * cost_fraction

        # ── Component 4: Sharpe Bonus ─────────────
        # Add this step's return to rolling window
        self.recent_returns.append(pct_change)

        reward_sharpe = 0.0

        # Only calculate Sharpe once we have enough history
        if len(self.recent_returns) >= 5:
            returns_arr = np.array(self.recent_returns)
            mean_ret    = np.mean(returns_arr)
            std_ret     = np.std(returns_arr)

            if std_ret > 1e-8:   # avoid division by zero
                sharpe = mean_ret / std_ret

                # Only reward POSITIVE Sharpe
                # We don't want to reward consistently
                # losing money in a stable way!
                if sharpe > 0:
                    reward_sharpe = self.sharpe_weight * sharpe

        # ── Final Reward ───────────────────────────
        reward = (
            reward_portfolio
            + reward_drawdown
            + reward_cost
            + reward_sharpe
        )

        # Scale down to keep values small and stable
        # Large reward values make neural network learning
        # unstable — small consistent values work better
        reward = reward * 1e-3

        # ── Breakdown for debugging ────────────────
        breakdown = {
            "reward_total":     round(reward,            8),
            "reward_portfolio": round(reward_portfolio,  6),
            "reward_drawdown":  round(reward_drawdown,   6),
            "reward_cost":      round(reward_cost,       6),
            "reward_sharpe":    round(reward_sharpe,     6),
            "drawdown_pct":     round(drawdown * 100,    2),
            "portfolio_change": round(pct_change * 100,  4),
        }

        return reward, breakdown