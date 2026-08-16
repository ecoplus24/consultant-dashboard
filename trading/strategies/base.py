"""
전략 인터페이스

전략은 "무엇을 얼마나 들고 있을 것인가"(목표 비중)만 말한다.
얼마를 살지, 수수료가 얼마인지, 손절을 어떻게 걸지는 엔진과 리스크 계층의 몫이다.
이렇게 나눠야 전략을 갈아끼워도 리스크 관리가 항상 동일하게 적용된다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data.base import MarketView


class Strategy(ABC):
    """모든 전략의 부모."""

    #: 지표 계산에 필요한 최소 봉 개수. 이만큼 쌓이기 전에는 호출되지 않는다.
    warmup: int = 1

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def target_weights(self, view: MarketView) -> dict[str, float] | None:
        """
        오늘 종가 기준으로 판단한 목표 비중.

        {종목: 비중} 형태이고 비중은 총자산 대비다. 합이 1보다 작으면 나머지는 현금.
        합이 1을 넘으면 엔진이 1로 정규화한다(레버리지 금지).

        None을 돌려주면 "현 상태 유지"다. 빈 dict({})와는 다르다 —
        빈 dict는 전량 청산이고, None은 아무것도 건드리지 말라는 뜻이다.
        유지를 표현할 수 없으면 매수 후 보유 전략조차 리밸런싱으로 계속 거래하게 된다.
        (유지 중이라도 리스크 계층의 손절은 그대로 작동한다)
        """

    def describe(self) -> str:
        """리포트에 찍을 한 줄 설명."""
        return self.name


def equal_weight(symbols: list[str], total: float = 1.0) -> dict[str, float]:
    """선택된 종목에 동일비중을 배분한다."""
    if not symbols:
        return {}
    w = total / len(symbols)
    return {s: w for s in symbols}
