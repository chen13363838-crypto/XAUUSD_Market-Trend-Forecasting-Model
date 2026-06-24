"""
main_multi_tf.py - Multi-timeframe demo (M5+M15+M30)

Usage:
    python main_multi_tf.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data.feed import DataFeeder
from core.multi_tf_agent import MultiTFAgent

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 64)
print("  Multi-Timeframe Regime Detection (GMM)")
print("=" * 64)

# Generate synthetic data for 3 timeframes
feeder = DataFeeder()
df_m30 = feeder.make_synthetic(n_bars=1000, mode="mixed", seed=1)
df_m15 = feeder.make_synthetic(n_bars=2000, mode="mixed", seed=2)
df_m5  = feeder.make_synthetic(n_bars=4000, mode="mixed", seed=3)

# Rename 'time' to 'datetime' for multi-TF module
tf_data = {
    "M30": df_m30.rename(columns={"time": "datetime"}),
    "M15": df_m15.rename(columns={"time": "datetime"}),
    "M5":  df_m5.rename(columns={"time": "datetime"}),
}

print(f"\n  M30: {len(df_m30)} bars")
print(f"  M15: {len(df_m15)} bars")
print(f"  M5 : {len(df_m5)} bars")

# Train
print("\n[1/3] Training multi-TF GMM model...")
agent = MultiTFAgent(mode="learn")
agent.train(tf_data, n_states=3)

# Decode history
print("\n[2/3] Decoding history...")
labels = agent.decode_history(tf_data)
valid = labels.dropna()
counts = valid.value_counts()
total = len(valid)
for regime, cnt in counts.items():
    pct = cnt / total * 100
    bar = "#" * int(pct / 2)
    print(f"  {str(regime):<14}: {cnt:>5} ({pct:5.1f}%)  {bar}")

# Predict
print("\n[3/3] Latest regime...")
recent = {k: v.tail(200).reset_index(drop=True) for k, v in tf_data.items()}
result = agent.run(recent)
print(f"  Regime     : {result.regime}")
print(f"  Confidence : {result.confidence:.2f}")
print(f"  Score      : {result.score:+.3f}")

# Save
agent.save_model(os.path.join(MODEL_DIR, "multi_tf_gmm_demo.pkl"))
print(f"\n  Model saved to saved_models/multi_tf_gmm_demo.pkl")
print("=" * 64)
print("  Done!")
print("=" * 64)
