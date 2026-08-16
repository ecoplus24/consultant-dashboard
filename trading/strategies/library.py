"""
전략 모음

전부 "규칙이 한 줄로 설명되는" 것들만 넣었다. 파라미터가 많고 설명이 복잡한 전략은
과거 데이터에 맞춰 깎은 것일 확률이 높고, 실전에서 그대로 무너진다.
"""

from __future__ import annotations

import pandas as pd

from ..data.base import MarketView
from . import indicators as ind
from .base import Strategy, equal_weight


class BuyAndHold(Strategy):
    """
    전 종목 동일비중으로 한 번 사고 그대로 둔다.

    벤치마크다. 어떤 전략이든 이걸 비용까지 감안해서 못 이기면 만들 이유가 없다.
    한 번 매수한 뒤로는 None(유지)을 돌려주므로 리밸런싱 거래가 발생하지 않는다.
    """

    warmup = 1

    def __init__(self, rebalance: bool = False):
        # rebalance=True면 동일비중을 계속 유지한다(정기 리밸런싱 벤치마크)
        self.rebalance = rebalance
        self._bought = False

    def target_weights(self, view: MarketView) -> dict[str, float] | None:
        if self.rebalance:
            return equal_weight(view.tradable_symbols())
        if self._bought:
            return None  # 유지
        symbols = view.tradable_symbols()
        if not symbols:
            return {}
        self._bought = True
        return equal_weight(symbols)

    def describe(self) -> str:
        kind = "정기 리밸런싱" if self.rebalance else "매수 후 보유"
        return f"동일비중 {kind} (벤치마크)"


class MovingAverageCross(Strategy):
    """
    단기 이동평균이 장기 이동평균 위에 있는 종목만 보유.

    추세추종의 가장 단순한 형태. 상승장에서 따라가고 하락장에서 현금으로 빠진다.
    대신 횡보장에서는 속임수 신호로 야금야금 깎인다.
    """

    def __init__(self, fast: int = 50, slow: int = 200, max_positions: int = 10):
        if fast >= slow:
            raise ValueError("fast는 slow보다 짧아야 한다")
        self.fast = fast
        self.slow = slow
        self.max_positions = max_positions
        self.warmup = slow + 1

    def target_weights(self, view: MarketView) -> dict[str, float]:
        close = view.close
        fast_ma = ind.sma(close, self.fast).iloc[-1]
        slow_ma = ind.sma(close, self.slow).iloc[-1]

        candidates = [
            s for s in view.tradable_symbols()
            if pd.notna(fast_ma[s]) and pd.notna(slow_ma[s]) and fast_ma[s] > slow_ma[s]
        ]

        # 신호가 너무 많으면 추세가 가장 강한 순으로 자른다
        if len(candidates) > self.max_positions:
            strength = {s: fast_ma[s] / slow_ma[s] - 1 for s in candidates}
            candidates = sorted(candidates, key=lambda s: strength[s], reverse=True)
            candidates = candidates[: self.max_positions]

        return equal_weight(candidates)

    def describe(self) -> str:
        return f"이동평균 교차 ({self.fast}일 > {self.slow}일, 최대 {self.max_positions}종목)"


class DualMomentum(Strategy):
    """
    듀얼 모멘텀 (상대 + 절대).

    1) 상대 모멘텀: 최근 lookback 기간 수익률 상위 N종목을 고른다.
    2) 절대 모멘텀: 그중에서도 수익률이 0(또는 현금수익률) 이상인 것만 산다.

    2번이 핵심이다. 이게 없으면 시장 전체가 무너질 때 "덜 빠진 종목"을
    끝까지 들고 있게 된다. 하락장에서 현금으로 빠지는 장치다.
    """

    def __init__(
        self,
        lookback: int = 126,
        top_n: int = 5,
        absolute_filter: bool = True,
        min_return: float = 0.0,
        skip_recent: int = 0,
    ):
        self.lookback = lookback
        self.top_n = top_n
        self.absolute_filter = absolute_filter
        self.min_return = min_return
        # 최근 1개월을 빼고 모멘텀을 재는 관행. 단기 반전 효과를 피하려는 것.
        self.skip_recent = skip_recent
        self.warmup = lookback + skip_recent + 1

    def target_weights(self, view: MarketView) -> dict[str, float]:
        close = view.close
        end = close.iloc[-1 - self.skip_recent]
        start = close.iloc[-1 - self.skip_recent - self.lookback]
        momentum = (end / start - 1.0).dropna()

        tradable = set(view.tradable_symbols())
        momentum = momentum[[s for s in momentum.index if s in tradable]]

        if self.absolute_filter:
            momentum = momentum[momentum > self.min_return]

        if momentum.empty:
            return {}  # 전부 현금

        winners = momentum.sort_values(ascending=False).head(self.top_n).index.tolist()

        # 절대 모멘텀에 걸려 종목 수가 모자라면, 그 몫은 현금으로 남긴다.
        # 남은 종목에 몰아주면 하락장에서 오히려 집중도가 올라간다.
        return equal_weight(winners, total=len(winners) / self.top_n)

    def describe(self) -> str:
        f = "절대모멘텀 필터 ON" if self.absolute_filter else "상대모멘텀만"
        return f"듀얼 모멘텀 ({self.lookback}일 수익률 상위 {self.top_n}종목, {f})"


class RSIMeanReversion(Strategy):
    """
    장기 상승추세에 있는 종목이 단기 과매도로 빠졌을 때만 산다.

    추세 필터 없이 RSI만 보고 사면 "떨어지는 칼날"을 계속 받게 된다.
    장기 이평선 위라는 조건이 그 필터다.
    """

    def __init__(
        self,
        rsi_window: int = 14,
        oversold: float = 30.0,
        exit_level: float = 55.0,
        trend_ma: int = 200,
        max_positions: int = 5,
    ):
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.exit_level = exit_level
        self.trend_ma = trend_ma
        self.max_positions = max_positions
        self.warmup = max(trend_ma, rsi_window * 3) + 1
        self._held: set[str] = set()

    def target_weights(self, view: MarketView) -> dict[str, float]:
        close = view.close
        rsi_now = ind.rsi(close, self.rsi_window).iloc[-1]
        trend = ind.sma(close, self.trend_ma).iloc[-1]
        price = close.iloc[-1]
        tradable = set(view.tradable_symbols())

        # 청산: RSI가 회복됐거나 추세가 깨진 종목은 뺀다
        for s in list(self._held):
            if s not in tradable:
                continue
            if pd.isna(rsi_now[s]) or rsi_now[s] >= self.exit_level or price[s] < trend[s]:
                self._held.discard(s)

        # 진입: 추세 위 + 과매도
        for s in tradable:
            if len(self._held) >= self.max_positions:
                break
            if s in self._held:
                continue
            if pd.isna(rsi_now[s]) or pd.isna(trend[s]):
                continue
            if price[s] > trend[s] and rsi_now[s] < self.oversold:
                self._held.add(s)

        return equal_weight(sorted(self._held & tradable))

    def describe(self) -> str:
        return (
            f"RSI 평균회귀 ({self.trend_ma}일선 위 & RSI<{self.oversold} 매수, "
            f"RSI>{self.exit_level} 청산)"
        )


REGISTRY: dict[str, type[Strategy]] = {
    "buyhold": BuyAndHold,
    "macross": MovingAverageCross,
    "momentum": DualMomentum,
    "rsi": RSIMeanReversion,
}


def build(name: str, **kwargs) -> Strategy:
    """이름으로 전략을 만든다."""
    key = name.strip().lower()
    if key not in REGISTRY:
        raise ValueError(f"알 수 없는 전략: {name} (가능: {', '.join(REGISTRY)})")
    return REGISTRY[key](**kwargs)
