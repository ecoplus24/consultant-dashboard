"""
성과지표

"얼마 벌었나"만 보면 안 된다. 같은 수익률이라도 도중에 -70%를 맞은 전략과
-15%로 버틴 전략은 완전히 다른 물건이다. 실제로 사람이 견딜 수 있는지를
결정하는 건 MDD와 낙폭 지속기간이다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# --- 개별 지표 ---------------------------------------------------------------


def total_return(equity: pd.Series) -> float:
    """전체 기간 누적 수익률."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def years_elapsed(equity: pd.Series) -> float:
    """실제 달력 기준 경과 연수."""
    if len(equity) < 2:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    return max(days / 365.25, 1e-9)


def cagr(equity: pd.Series) -> float:
    """연평균 복리 수익률."""
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (1.0 / years_elapsed(equity)) - 1.0)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """전고점 대비 낙폭 시계열 (음수)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """최대 낙폭(MDD). 음수로 반환한다."""
    if len(equity) < 2:
        return 0.0
    return float(drawdown_series(equity).min())


def longest_drawdown_days(equity: pd.Series) -> int:
    """전고점을 회복하지 못한 최장 기간(일). 심리적으로 가장 중요한 숫자."""
    if len(equity) < 2:
        return 0
    peak = equity.cummax()
    at_peak = equity >= peak
    longest = 0
    last_peak_date = equity.index[0]
    for date, is_peak in at_peak.items():
        if is_peak:
            last_peak_date = date
        else:
            longest = max(longest, (date - last_peak_date).days)
    return int(longest)


def volatility(returns: pd.Series) -> float:
    """연율화 변동성."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """샤프지수. risk_free는 연율 기준으로 넣는다."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """하방 변동성만 위험으로 보는 지수. 상승 변동을 벌주지 않는다."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def calmar(equity: pd.Series) -> float:
    """CAGR / |MDD|. 낙폭 한 단위당 수익을 본다."""
    mdd = abs(max_drawdown(equity))
    if mdd < 1e-9:
        return 0.0
    return float(cagr(equity) / mdd)


def required_cagr(start_value: float, target_value: float, years: float) -> float:
    """
    목표 금액에 도달하려면 연 몇 %가 필요한지.

    "100만원으로 1억"이 어떤 요구인지 숫자로 확인할 때 쓴다.
    """
    if start_value <= 0 or target_value <= 0 or years <= 0:
        raise ValueError("start_value, target_value, years는 모두 0보다 커야 한다")
    return float((target_value / start_value) ** (1.0 / years) - 1.0)


def future_value(
    start_value: float, years: float, annual_return: float, monthly_contribution: float = 0.0
) -> float:
    """원금 + 매월 적립을 연 annual_return으로 굴렸을 때의 미래가치."""
    months = int(round(years * 12))
    m = (1.0 + annual_return) ** (1 / 12) - 1.0
    if abs(m) < 1e-12:
        return start_value + monthly_contribution * months
    growth = (1.0 + m) ** months
    return start_value * growth + monthly_contribution * (growth - 1.0) / m


def required_cagr_with_contributions(
    start_value: float, target_value: float, years: float, monthly_contribution: float
) -> float:
    """
    매월 적립을 감안했을 때 목표에 필요한 연수익률.

    닫힌 해가 없어 이분법으로 푼다. 적립액만으로 목표를 넘으면 음수가 나오는데,
    그건 "수익이 아니라 저축으로 달성되는 목표"라는 뜻이라 그대로 돌려준다.
    """
    if monthly_contribution <= 0:
        return required_cagr(start_value, target_value, years)

    lo, hi = -0.99, 10.0
    if future_value(start_value, years, hi, monthly_contribution) < target_value:
        return float("inf")  # 어떤 수익률로도 도달 불가한 수준

    for _ in range(200):
        mid = (lo + hi) / 2
        if future_value(start_value, years, mid, monthly_contribution) < target_value:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def years_to_target(start_value: float, target_value: float, annual_return: float) -> float:
    """주어진 연수익률로 목표 금액까지 걸리는 연수."""
    if start_value <= 0 or target_value <= 0:
        raise ValueError("start_value와 target_value는 0보다 커야 한다")
    if annual_return <= -1.0:
        raise ValueError("annual_return은 -100%보다 커야 한다")
    if annual_return <= 0:
        return float("inf")
    return float(np.log(target_value / start_value) / np.log(1.0 + annual_return))


# --- 묶음 -------------------------------------------------------------------


@dataclass
class Performance:
    """백테스트 결과 요약."""

    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    start_equity: float
    end_equity: float
    total_return: float
    cagr: float
    volatility: float
    max_drawdown: float
    longest_drawdown_days: int
    sharpe: float
    sortino: float
    calmar: float
    trades: int
    win_rate: float
    profit_factor: float
    total_costs: float
    exposure: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = str(self.start.date())
        d["end"] = str(self.end.date())
        return d


def summarize(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    exposure_series: pd.Series | None = None,
    total_costs: float = 0.0,
) -> Performance:
    """자산곡선(+거래내역)에서 성과 요약을 만든다."""
    equity = equity.dropna()
    if len(equity) < 2:
        raise ValueError("자산곡선이 너무 짧아 성과를 계산할 수 없다")

    returns = equity.pct_change().dropna()

    n_trades, win_rate, profit_factor = 0, 0.0, 0.0
    if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
        closed = trades[trades["pnl"].notna()]
        n_trades = len(closed)
        if n_trades > 0:
            wins = closed[closed["pnl"] > 0]["pnl"]
            losses = closed[closed["pnl"] < 0]["pnl"]
            win_rate = len(wins) / n_trades
            gross_loss = abs(losses.sum())
            # 손실이 하나도 없으면 profit factor는 정의상 무한대 -> inf로 둔다
            profit_factor = float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf")

    exposure = float(exposure_series.mean()) if exposure_series is not None and len(exposure_series) else 0.0

    return Performance(
        start=equity.index[0],
        end=equity.index[-1],
        years=years_elapsed(equity),
        start_equity=float(equity.iloc[0]),
        end_equity=float(equity.iloc[-1]),
        total_return=total_return(equity),
        cagr=cagr(equity),
        volatility=volatility(returns),
        max_drawdown=max_drawdown(equity),
        longest_drawdown_days=longest_drawdown_days(equity),
        sharpe=sharpe(returns),
        sortino=sortino(returns),
        calmar=calmar(equity),
        trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_costs=float(total_costs),
        exposure=exposure,
    )
