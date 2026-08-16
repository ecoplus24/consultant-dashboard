"""
규칙 기반 주식매매 백테스트 엔진

국내(KOSPI/KOSDAQ)와 해외(NYSE/NASDAQ) 주식을 같은 코드로 검증한다.

핵심 원칙:
  - 전략은 "목표 비중"만 말하고, 리스크 관리와 체결은 엔진이 강제한다
  - 신호는 당일 종가, 체결은 익일 시가 (미래참조 차단)
  - 수수료·거래세·슬리피지·정수 주식·현금 제약을 전부 반영한다

기본 사용법:

    from trading import quickstart
    result = quickstart(["005930", "000660"], market="kr", strategy="momentum")
    print(result.report)
"""

from .broker.base import Account, Broker, BrokerError, Holding, Order, OrderResult
from .costs import KR_COSTS, US_COSTS, ZERO_COSTS, CostModel, costs_for
from .data import PriceData, get_provider
from .engine import Backtest, BacktestResult, run_backtest
from .live import LiveTrader, SafetyLimits, TradePlan, format_plan, is_market_open
from .metrics import Performance, required_cagr, summarize, years_to_target
from .report import compare, format_performance, goal_analysis, monte_carlo
from .risk import PRESETS, RiskManager, preset
from .strategies import Strategy, build

__version__ = "0.1.0"

__all__ = [
    "CostModel", "KR_COSTS", "US_COSTS", "ZERO_COSTS", "costs_for",
    "PriceData", "get_provider",
    "Backtest", "BacktestResult", "run_backtest",
    "Performance", "summarize", "required_cagr", "years_to_target",
    "format_performance", "compare", "goal_analysis", "monte_carlo",
    "RiskManager", "PRESETS", "preset",
    "Strategy", "build",
    "Account", "Broker", "BrokerError", "Holding", "Order", "OrderResult",
    "LiveTrader", "SafetyLimits", "TradePlan", "format_plan", "is_market_open",
    "load_data", "quickstart",
]


def load_data(symbols, market="kr", start="2015-01-01", end="2025-12-31", **kwargs):
    """시장 코드로 가격 데이터를 받아 PriceData로 돌려준다."""
    provider = get_provider(market, **kwargs)
    return provider.fetch(list(symbols), start, end)


def quickstart(
    symbols,
    market="kr",
    strategy="momentum",
    risk_preset="balanced",
    start="2015-01-01",
    end="2025-12-31",
    initial_capital=1_000_000.0,
    strategy_params=None,
):
    """데이터 수집 → 백테스트 → 리포트까지 한 번에."""
    data = load_data(symbols, market=market, start=start, end=end)
    strat = build(strategy, **(strategy_params or {}))
    result = run_backtest(
        data,
        strategy=strat,
        risk=preset(risk_preset),
        costs=costs_for(market),
        initial_capital=initial_capital,
    )
    result.report = format_performance(result)
    return result
