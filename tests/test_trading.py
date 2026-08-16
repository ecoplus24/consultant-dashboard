"""
백테스트 엔진 검증

여기서 제일 중요한 건 수익률 계산이 맞느냐가 아니라,
"백테스트가 거짓말하지 않느냐"를 잡는 테스트들이다:
미래참조, 현금 초과 매수, 비용 누락, 손절 미작동.
"""

import numpy as np
import pandas as pd
import pytest

from trading import costs, metrics, risk
from trading.data.base import PriceData
from trading.data.providers import SyntheticProvider
from trading.engine import Backtest, run_backtest
from trading.strategies import BuyAndHold, DualMomentum, MovingAverageCross, build
from trading.strategies.base import Strategy
from trading.strategies import indicators as ind


# --- 픽스처 -----------------------------------------------------------------


@pytest.fixture
def data():
    provider = SyntheticProvider(seed=1)
    return provider.fetch(["AAA", "BBB", "CCC"], "2018-01-01", "2024-12-31")


def constant_frames(prices: dict[str, list[float]], start="2020-01-01"):
    """지정한 종가 경로를 그대로 갖는 PriceData. 시가=종가라 체결가가 예측 가능하다."""
    frames = {}
    for symbol, closes in prices.items():
        idx = pd.bdate_range(start=start, periods=len(closes))
        frames[symbol] = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes,
             "close": closes, "volume": [1e6] * len(closes)},
            index=idx,
        )
    return PriceData(frames)


# --- 지표 -------------------------------------------------------------------


def test_sma_matches_manual_average():
    df = pd.DataFrame({"X": [1.0, 2, 3, 4, 5]})
    result = ind.sma(df, 3)["X"]
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_rsi_is_100_when_price_only_rises():
    df = pd.DataFrame({"X": np.arange(1.0, 60.0)})
    assert ind.rsi(df, 14)["X"].iloc[-1] == pytest.approx(100.0)


def test_rsi_stays_in_bounds(data):
    values = ind.rsi(data.close, 14).to_numpy()
    values = values[np.isfinite(values)]
    assert values.min() >= 0.0 and values.max() <= 100.0


def test_atr_is_positive(data):
    atr = ind.atr(data.high, data.low, data.close, 14).dropna()
    assert (atr.to_numpy() > 0).all()


# --- 성과지표 ---------------------------------------------------------------


def test_cagr_on_known_doubling():
    idx = pd.to_datetime(["2020-01-01", "2024-12-31"])
    equity = pd.Series([100.0, 200.0], index=idx)
    # 5년에 2배 -> 2^(1/5) - 1 ≈ 14.87%
    assert metrics.cagr(equity) == pytest.approx(0.1487, abs=1e-3)


def test_max_drawdown_is_measured_from_peak():
    equity = pd.Series(
        [100.0, 150.0, 75.0, 120.0],
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    # 150 -> 75 이므로 -50%
    assert metrics.max_drawdown(equity) == pytest.approx(-0.5)


def test_required_cagr_for_100x_over_10_years():
    """100만원 -> 1억, 10년. 연 58%가 필요하다는 걸 못 박아 둔다."""
    need = metrics.required_cagr(1_000_000, 100_000_000, 10)
    assert need == pytest.approx(0.5849, abs=1e-3)


def test_years_to_target_at_realistic_return():
    """연 15%로 100배를 만들려면 33년이 걸린다."""
    years = metrics.years_to_target(1_000_000, 100_000_000, 0.15)
    assert years == pytest.approx(32.9, abs=0.5)


def test_years_to_target_is_infinite_when_not_growing():
    assert metrics.years_to_target(100, 200, 0.0) == float("inf")


def test_future_value_with_zero_return_is_just_the_sum_of_deposits():
    fv = metrics.future_value(1_000_000, years=10, annual_return=0.0,
                              monthly_contribution=500_000)
    assert fv == pytest.approx(1_000_000 + 500_000 * 120)


def test_required_cagr_with_contributions_round_trips():
    """구한 수익률로 다시 굴리면 목표 금액이 나와야 한다."""
    need = metrics.required_cagr_with_contributions(1_000_000, 300_000_000, 20, 500_000)
    fv = metrics.future_value(1_000_000, 20, need, 500_000)
    assert fv == pytest.approx(300_000_000, rel=1e-4)


def test_monthly_contributions_lower_the_required_return():
    """적립을 하면 필요한 수익률이 내려간다 — 이게 100배보다 현실적인 경로다."""
    without = metrics.required_cagr(1_000_000, 100_000_000, 20)
    with_dca = metrics.required_cagr_with_contributions(1_000_000, 100_000_000, 20, 500_000)
    assert with_dca < without
    assert without == pytest.approx(0.2589, abs=1e-3)  # 원금만: 연 25.9%


def test_savings_alone_reaching_target_is_flagged(data):
    """납입액만으로 목표를 넘으면 '전략 덕분'이라고 말하면 안 된다."""
    from trading.report import goal_analysis

    result = run_backtest(data, BuyAndHold(), risk=risk.preset("balanced"),
                          costs=costs.KR_COSTS, initial_capital=1_000_000)
    text = goal_analysis(result, target=10_000_000, years=10,
                         monthly_contribution=500_000, simulations=500)
    assert "저축만 해도 목표를 넘는다" in text
    assert "납입액 덕분" in text


# --- 비용 -------------------------------------------------------------------


def test_slippage_hurts_both_directions():
    model = costs.CostModel(slippage_bps=10)
    assert model.fill_price(100, "buy") > 100
    assert model.fill_price(100, "sell") < 100


def test_sell_tax_applies_only_to_sells():
    model = costs.CostModel(commission_bps=1.5, sell_tax_bps=18.0)
    assert model.transaction_cost(1_000_000, "sell") > model.transaction_cost(1_000_000, "buy")


def test_korean_sell_cost_is_about_20bps():
    total = costs.KR_COSTS.transaction_cost(1_000_000, "sell")
    assert total == pytest.approx(1_000_000 * 0.00195, rel=1e-6)


# --- 데이터 -----------------------------------------------------------------


def test_pricedata_aligns_symbols_to_one_calendar(data):
    assert len(data.symbols) == 3
    assert data.close.shape == (len(data.calendar), 3)
    assert not data.close.isna().to_numpy().any()


def test_view_cannot_see_the_future(data):
    """전략에 넘어가는 뷰는 오늘까지만 잘려 있어야 한다."""
    view = data.view(100)
    assert len(view.close) == 101
    assert view.close.index[-1] == data.calendar[100]
    assert view.date == data.calendar[100]


def test_missing_days_are_not_tradable():
    frames = {
        "AAA": pd.DataFrame(
            {"open": [1.0, 2, 3], "high": [1.0, 2, 3], "low": [1.0, 2, 3],
             "close": [1.0, 2, 3], "volume": [1, 1, 1]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        ),
        "BBB": pd.DataFrame(
            {"open": [5.0], "high": [5.0], "low": [5.0], "close": [5.0], "volume": [1]},
            index=pd.to_datetime(["2020-01-03"]),
        ),
    }
    pd_data = PriceData(frames)
    assert not pd_data.tradable.iloc[0]["BBB"]  # 아직 상장 전
    assert pd_data.tradable.iloc[2]["BBB"]


# --- 엔진: 정확성 -----------------------------------------------------------


def test_flat_market_with_no_costs_preserves_capital():
    """가격이 안 움직이고 비용이 0이면 자산도 안 움직여야 한다."""
    data = constant_frames({"AAA": [100.0] * 300})
    result = run_backtest(
        data, BuyAndHold(), risk=risk.preset("none"),
        costs=costs.ZERO_COSTS, initial_capital=1_000_000,
    )
    assert result.equity.iloc[-1] == pytest.approx(1_000_000, rel=1e-9)


def test_costs_reduce_returns():
    """같은 전략·같은 데이터에서 비용을 켜면 성과가 반드시 나빠진다."""
    data = constant_frames({"AAA": list(np.linspace(100, 200, 400))})
    kwargs = dict(strategy=MovingAverageCross(fast=5, slow=20, max_positions=1),
                  risk=risk.preset("none"), initial_capital=1_000_000)

    free = run_backtest(data, costs=costs.ZERO_COSTS, **kwargs)
    paid = run_backtest(data, costs=costs.KR_COSTS, **kwargs)

    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.performance.total_costs > 0


def test_never_spends_more_cash_than_available(data):
    """현금이 마이너스로 내려가면 몰래 레버리지를 쓴 것이다."""
    bt = Backtest(
        data, BuyAndHold(), risk.preset("none"), costs.KR_COSTS,
        initial_capital=1_000_000,
    )
    bt.run()
    assert bt.cash >= -1e-6


def test_gap_up_shrinks_the_order_instead_of_dropping_it():
    """
    주문 수량은 어제 종가로 계산되는데 오늘 시가가 갭상승할 수 있다.
    예산이 모자란다고 주문을 통째로 버리면 상승장에서 아무것도 못 산다.
    """
    prices = list(np.linspace(100, 200, 300))  # 매일 조금씩 오르는 시장
    data = constant_frames({"AAA": prices})

    bt = Backtest(data, BuyAndHold(), risk.preset("none"), costs.ZERO_COSTS,
                  initial_capital=1_000_000)
    result = bt.run()

    assert len(bt.trades) > 0, "상승장에서 매수가 한 건도 없다"
    assert bt.cash >= -1e-6
    assert result.equity.iloc[-1] > 1_000_000


def test_integer_shares_by_default():
    """소수점 주식을 끄면 보유수량은 항상 정수여야 한다."""
    data = constant_frames({"AAA": list(np.linspace(100, 300, 300))})
    bt = Backtest(
        data, BuyAndHold(), risk.preset("none"), costs.ZERO_COSTS,
        initial_capital=1_000_000, fractional_shares=False,
    )
    bt.run()
    for pos in bt.positions.values():
        assert pos.quantity == int(pos.quantity)


def test_expensive_stock_is_unaffordable_with_small_capital():
    """100만원으로 200만원짜리 주식은 못 산다. 소액 계좌의 실제 제약이다."""
    data = constant_frames({"AAA": [2_000_000.0] * 200})
    bt = Backtest(
        data, BuyAndHold(), risk.preset("none"), costs.ZERO_COSTS,
        initial_capital=1_000_000,
    )
    result = bt.run()
    assert len(bt.positions) == 0
    assert result.equity.iloc[-1] == pytest.approx(1_000_000)


def test_no_lookahead_orders_fill_at_next_open():
    """
    신호가 뜬 날의 종가가 아니라 '다음날 시가'에 체결되는지 확인한다.

    급등 전날 신호가 잡혀도 급등한 가격에 사야 한다는 뜻이다.
    """
    closes = [100.0] * 30 + [100.0, 200.0] + [200.0] * 30
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    # 시가는 그날 종가와 같게 두되, 급등일 시가도 200으로 둔다
    frame = pd.DataFrame(
        {"open": closes, "high": closes, "low": closes,
         "close": closes, "volume": [1e6] * len(closes)},
        index=idx,
    )
    data = PriceData({"AAA": frame})

    class BuyOnDay30(Strategy):
        warmup = 1

        def target_weights(self, view):
            return {"AAA": 1.0} if view.index >= 30 else {}

    bt = Backtest(data, BuyOnDay30(), risk.preset("none"), costs.ZERO_COSTS,
                  initial_capital=1_000_000)
    bt.run()

    fills = [t for t in bt.trades if t["side"] == "buy"]
    assert len(fills) == 1
    # 30번 인덱스 종가(100)에 신호 -> 31번 시가(200)에 체결
    assert fills[0]["price"] == pytest.approx(200.0)


# --- 엔진: 리스크 -----------------------------------------------------------


def test_stop_loss_exits_a_falling_position():
    """-15% 손절이 걸린 채로 계속 떨어지면 반드시 청산돼야 한다."""
    closes = [100.0] * 20 + list(np.linspace(100, 40, 60))
    data = constant_frames({"AAA": closes})

    manager = risk.RiskManager(
        max_position_weight=1.0, stop_loss_pct=0.15,
        trailing_stop_pct=None, max_drawdown_stop=None, vol_target=None,
    )
    stopped = run_backtest(data, BuyAndHold(), risk=manager, costs=costs.ZERO_COSTS,
                           initial_capital=1_000_000, rebalance_days=1)
    unstopped = run_backtest(data, BuyAndHold(), risk=risk.preset("none"),
                             costs=costs.ZERO_COSTS, initial_capital=1_000_000,
                             rebalance_days=1)

    assert "손절" in set(stopped.stop_events["reason"])
    # 손절이 있으면 같은 하락장에서 낙폭이 더 작고 원금이 더 남아야 한다
    assert stopped.performance.max_drawdown > unstopped.performance.max_drawdown
    assert stopped.equity.iloc[-1] > unstopped.equity.iloc[-1]


def test_circuit_breaker_moves_everything_to_cash():
    """계좌 낙폭이 한계를 넘으면 전량 현금화된다."""
    closes = [100.0] * 10 + list(np.linspace(100, 20, 80))
    data = constant_frames({"AAA": closes})

    manager = risk.RiskManager(
        max_position_weight=1.0, stop_loss_pct=None, trailing_stop_pct=None,
        max_drawdown_stop=0.20, cooldown_days=1000, vol_target=None,
    )
    bt = Backtest(data, BuyAndHold(), manager, costs.ZERO_COSTS,
                  initial_capital=1_000_000, rebalance_days=1)
    result = bt.run()

    assert "서킷브레이커" in set(result.stop_events["reason"])
    assert len(bt.positions) == 0  # 냉각기간이 끝까지 이어져 재진입 없음


def test_position_weight_cap_prevents_concentration():
    data = constant_frames({s: list(np.linspace(100, 200, 300)) for s in ["AAA", "BBB", "CCC", "DDD"]})
    manager = risk.RiskManager(
        max_position_weight=0.25, stop_loss_pct=None, trailing_stop_pct=None,
        max_drawdown_stop=None, vol_target=None,
    )

    class AllIn(Strategy):
        warmup = 1

        def target_weights(self, view):
            return {"AAA": 1.0}

    bt = Backtest(data, AllIn(), manager, costs.ZERO_COSTS, initial_capital=10_000_000)
    result = bt.run()

    close = data.close.iloc[-1]["AAA"]
    held_value = bt.positions["AAA"].quantity * close
    assert held_value / result.equity.iloc[-1] <= 0.26


def test_leverage_is_rejected():
    with pytest.raises(ValueError, match="레버리지"):
        risk.RiskManager(max_gross_exposure=2.0)


def test_risk_presets_are_independent_instances():
    """프리셋이 상태를 공유하면 두 번째 백테스트가 오염된다."""
    a, b = risk.preset("balanced"), risk.preset("balanced")
    a.stop_events.append({"date": None, "symbol": "X", "reason": "손절"})
    assert len(b.stop_events) == 0


# --- 전략 -------------------------------------------------------------------


def test_dual_momentum_holds_cash_when_everything_falls():
    """전 종목이 하락 중이면 절대 모멘텀 필터가 전부 현금으로 뺀다."""
    falling = list(np.linspace(200, 100, 400))
    data = constant_frames({"AAA": falling, "BBB": falling})

    result = run_backtest(
        data, DualMomentum(lookback=126, top_n=2),
        risk=risk.preset("none"), costs=costs.ZERO_COSTS, initial_capital=1_000_000,
    )
    # 한 주도 사지 않았으므로 원금 그대로
    assert result.equity.iloc[-1] == pytest.approx(1_000_000)
    assert len(result.trades) == 0


def test_dual_momentum_without_absolute_filter_buys_the_falling_knife():
    """절대 모멘텀 필터를 끄면 하락장에서도 '덜 빠진 종목'을 산다."""
    falling = list(np.linspace(200, 100, 400))
    data = constant_frames({"AAA": falling, "BBB": falling})

    result = run_backtest(
        data, DualMomentum(lookback=126, top_n=2, absolute_filter=False),
        risk=risk.preset("none"), costs=costs.ZERO_COSTS, initial_capital=1_000_000,
    )
    assert len(result.trades) > 0
    assert result.equity.iloc[-1] < 1_000_000


def test_all_registered_strategies_run(data):
    for name in build.__globals__["REGISTRY"]:
        result = run_backtest(
            data, build(name), risk=risk.preset("balanced"),
            costs=costs.KR_COSTS, initial_capital=1_000_000,
        )
        assert len(result.equity) == len(data.calendar)
        assert result.equity.notna().all()
        assert (result.equity.to_numpy() >= 0).all()


def test_strategy_weights_never_exceed_one(data):
    """어떤 전략이든 비중 합이 1을 넘으면 레버리지다."""
    for name in ["buyhold", "macross", "momentum", "rsi"]:
        strategy = build(name)
        view = data.view(len(data.calendar) - 1)
        assert sum(strategy.target_weights(view).values()) <= 1.0 + 1e-9


# --- 리포트 -----------------------------------------------------------------


def test_monte_carlo_probabilities_are_valid(data):
    from trading.report import monte_carlo

    result = run_backtest(data, BuyAndHold(), risk=risk.preset("balanced"),
                          costs=costs.KR_COSTS, initial_capital=1_000_000)
    mc = monte_carlo(result, years=10, target=100_000_000, simulations=500)

    for key in ("prob_target", "prob_loss", "prob_ruin"):
        assert 0.0 <= mc[key] <= 1.0
    assert mc["p5"] <= mc["median"] <= mc["p95"]


def test_goal_analysis_flags_unrealistic_targets(data):
    from trading.report import goal_analysis

    result = run_backtest(data, BuyAndHold(), risk=risk.preset("balanced"),
                          costs=costs.KR_COSTS, initial_capital=1_000_000)
    text = goal_analysis(result, target=100_000_000, years=10, simulations=500)

    assert "현실적인 목표가 아니다" in text
    assert "목표 달성 확률" in text


def test_report_renders(data):
    from trading.report import compare, format_performance

    results = [
        run_backtest(data, build(n), risk=risk.preset("balanced"),
                     costs=costs.KR_COSTS, initial_capital=1_000_000)
        for n in ["buyhold", "momentum"]
    ]
    assert "CAGR" in format_performance(results[0])
    assert "전략" in compare(results)
