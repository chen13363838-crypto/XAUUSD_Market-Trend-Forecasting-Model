# jud_model — XAUUSD Market Regime Detection Framework

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Stars](https://img.shields.io/github/stars/chen13363838-crypto/XAUUSD_Market-Trend-Forecasting-Model)
![Last Commit](https://img.shields.io/github/last-commit/chen13363838-crypto/XAUUSD_Market-Trend-Forecasting-Model)

**一个专注于黄金（XAUUSD）的市场状态（Regime）检测框架**，帮助交易者自动判断当前是**趋势市场**还是**震荡市场**，从而智能切换不同的EA策略。

---

## ✨ 核心功能

**jud_model** 将市场实时分类为四大状态：

| Regime     | 描述                     | 适合策略         |
|------------|--------------------------|------------------|
| **TREND**  | 强趋势（突破行情）       | 趋势跟随、突破EA |
| **OSCILLATION** | 震荡/区间行情         | 震荡策略、均值回归 |
| **FUZZY**  | 过渡状态                 | 观望或降低仓位   |
| **UNKNOWN**| 置信度不足               | 停止交易         |

### 支持四种工作模式
1. **Rule Mode** —— 纯规则评分（无需训练，稳定可靠）
2. **Learn Mode** —— 无监督机器学习（GMM / HMM）
3. **Hybrid Mode** —— **推荐**：ML为主，低置信时回退规则
4. **Multi-Timeframe Mode** —— M5 + M15 + M30 多周期特征融合（21维）

---

## 🚀 快速开始

```bash
git clone https://github.com/chen13363838-crypto/XAUUSD_Market-Trend-Forecasting-Model.git
cd XAUUSD_Market-Trend-Forecasting-Model

# 推荐直接进入项目目录
cd XAUUSD_Market-Trend-Forecasting-Model
pip install -r requirements.txt
python main.py
