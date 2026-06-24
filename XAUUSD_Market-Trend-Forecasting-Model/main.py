"""
jud_model/main.py - Entry point & usage examples

Usage:
    pip install -r requirements.txt
    python main.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data.feed import DataFeeder
from core.agent import JudgeAgent
from utils.logger import get_logger

_log = get_logger("main")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


def demo_rule_mode():
    """Rule mode: no training needed."""
    print("=" * 60)
    print("Example 1: Rule mode")
    print("=" * 60)
    agent = JudgeAgent(mode="rule")
    for m in ("trend", "oscillation", "mixed"):
        df = DataFeeder.make_synthetic(n_bars=500, mode=m)
        r = agent.run(df)
        print(f"  [{m.upper():>12}] -> {r.regime:<12}  conf={r.confidence:.2f}")


def demo_learn_mode():
    """Learn mode: GMM from synthetic data."""
    print("\n" + "=" * 60)
    print("Example 2: Learn mode (GMM)")
    print("=" * 60)
    train_df = DataFeeder.make_synthetic(n_bars=2000, mode="mixed", seed=0)
    agent = JudgeAgent(mode="learn")
    agent.train(train_df, model_type="gmm", n_states=2)
    agent.save_model(os.path.join(MODEL_DIR, "gmm_demo.pkl"))
    for m in ("trend", "oscillation", "mixed"):
        df = DataFeeder.make_synthetic(n_bars=300, mode=m, seed=99)
        r = agent.run(df)
        print(f"  [{m.upper():>12}] -> {r.regime:<12}  conf={r.confidence:.2f}")


def demo_load_and_predict():
    """Load existing model."""
    mp = os.path.join(MODEL_DIR, "gmm_demo.pkl")
    if not os.path.exists(mp):
        return
    print("\n" + "=" * 60)
    print("Example 3: Load & predict")
    print("=" * 60)
    agent = JudgeAgent.from_pretrained(mp)
    df = DataFeeder.make_synthetic(n_bars=400, mode="trend", seed=77)
    r = agent.run(df)
    print(f"  {r}")


def demo_decode_history():
    """History decoding for backtest."""
    mp = os.path.join(MODEL_DIR, "gmm_demo.pkl")
    if not os.path.exists(mp):
        return
    print("\n" + "=" * 60)
    print("Example 4: History decoding")
    print("=" * 60)
    agent = JudgeAgent.from_pretrained(mp)
    df = DataFeeder.make_synthetic(n_bars=200, mode="mixed", seed=55)
    labels = agent.decode_history(df)
    counts = labels.value_counts()
    total = labels.notna().sum()
    for regime, cnt in counts.items():
        pct = cnt / total * 100
        print(f"  {str(regime):<14} {cnt:>4} ({pct:.1f}%)")


def demo_csv_template():
    """Real data from CSV - replace path to run."""
    fp = r"data\your_data.csv"
    if not os.path.exists(fp):
        print("\n  Example 5: Skipped (replace CSV path to run)")
        return
    feeder = DataFeeder()
    df = feeder.feed_from_csv(fp, sep="\t")
    split = int(len(df) * 0.8)
    agent = JudgeAgent(mode="hybrid")
    agent.train(df.iloc[:split], model_type="gmm", n_states=2)
    r = agent.run(df.iloc[split:])
    print(f"  {r}")


if __name__ == "__main__":
    demo_rule_mode()
    demo_learn_mode()
    demo_load_and_predict()
    demo_decode_history()
    demo_csv_template()
