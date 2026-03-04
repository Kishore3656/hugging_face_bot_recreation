# main.py

import yaml
from loguru import logger

# ── ADD THESE LINES AT THE TOP ──────────────
import gymnasium as gym
print(f"✅ Gymnasium version: {gym.__version__}")

from stable_baselines3 import PPO, A2C, DDPG, TD3
print(f"✅ Stable Baselines3 imported successfully")
print(f"✅ All 4 algorithms available: PPO, A2C, DDPG, TD3")
# ────────────────────────────────────────────


def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError("❌ config.yaml is empty!")
    return config

def main():
    config = load_config()
    logger.info("🚀 Trading Bot Starting...")
    # ... rest of your existing main.py code

if __name__ == "__main__":
    main()