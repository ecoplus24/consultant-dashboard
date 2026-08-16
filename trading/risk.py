"""
리스크 관리 계층

전략이 "사고 싶다"고 말한 것을 실제로 얼마나 살지 결정하고, 필요하면 거부한다.
전략과 분리해 둔 이유는 단순하다 — 전략은 바꿔 끼워도 리스크 규칙은 항상
동일하게 걸려 있어야 하기 때문이다.

여기 들어 있는 규칙들이 "100만원으로 1억"류 발상과 정면으로 충돌하는 부분이다.
파산을 막는 장치는 전부 기대수익률을 깎는다. 그게 정상이고, 그 대가로
계좌가 0이 되지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.base import MarketView


@dataclass
class RiskContext:
    """리스크 판단에 필요한 현재 상태. 엔진이 채워서 넘긴다."""

    view: MarketView
    equity: float
    peak_equity: float
    positions: dict[str, float]  # 종목 -> 보유수량
    entry_price: dict[str, float]  # 종목 -> 평균 매입가
    high_water: dict[str, float]  # 종목 -> 보유 중 최고가


@dataclass
class RiskManager:
    """
    비중을 깎고, 손절을 걸고, 최악의 경우 전부 현금화한다.

    max_position_weight : 한 종목 최대 비중. 몰빵 방지.
    stop_loss_pct       : 매입가 대비 손절선 (0.15 = -15%)
    trailing_stop_pct   : 보유 중 최고가 대비 하락 손절선
    max_drawdown_stop   : 계좌 전체 낙폭이 이만큼이면 전량 청산 후 냉각기간
    cooldown_days       : 손절/서킷 발동 후 재진입 금지 일수
    vol_target          : 목표 연변동성. 지정하면 총 노출을 여기 맞춰 조절한다
    max_gross_exposure  : 총 노출 상한. 1.0이면 레버리지 없음
    """

    max_position_weight: float = 0.25
    stop_loss_pct: float | None = 0.15
    trailing_stop_pct: float | None = 0.25
    max_drawdown_stop: float | None = 0.30
    cooldown_days: int = 5
    vol_target: float | None = None
    max_gross_exposure: float = 1.0
    vol_window: int = 20

    _blocked: dict[str, int] = field(default_factory=dict, init=False)
    _circuit_until: int = field(default=-1, init=False)
    stop_events: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self):
        if not 0 < self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight는 0과 1 사이여야 한다")
        if self.max_gross_exposure > 1.0:
            raise ValueError(
                "max_gross_exposure가 1을 넘으면 레버리지다. "
                "이 엔진은 현금 담보 매매만 지원한다."
            )

    def reset(self) -> None:
        self._blocked.clear()
        self._circuit_until = -1
        self.stop_events.clear()

    # --- 개별 종목 손절 ------------------------------------------------------

    def _stopped_out(self, symbol: str, ctx: RiskContext) -> str | None:
        """손절 사유를 돌려준다. 걸리지 않았으면 None."""
        if ctx.positions.get(symbol, 0) <= 0:
            return None

        price = float(ctx.view.last_close()[symbol])
        if not np.isfinite(price) or price <= 0:
            return None

        entry = ctx.entry_price.get(symbol)
        if self.stop_loss_pct and entry and price <= entry * (1 - self.stop_loss_pct):
            return "손절"

        peak = ctx.high_water.get(symbol)
        if self.trailing_stop_pct and peak and price <= peak * (1 - self.trailing_stop_pct):
            return "추적손절"

        return None

    # --- 변동성 타겟팅 -------------------------------------------------------

    def _exposure_scale(self, targets: dict[str, float], ctx: RiskContext) -> float:
        """
        포트폴리오 예상 변동성이 목표보다 높으면 전체 비중을 줄인다.

        종목 간 상관을 1로 가정한 보수적 근사다. 실제보다 변동성을 크게 보므로
        노출이 과하게 잡히는 쪽으로는 틀리지 않는다.
        """
        if not self.vol_target or not targets:
            return 1.0

        close = ctx.view.close
        if len(close) < self.vol_window + 1:
            return 1.0

        recent = close.pct_change().iloc[-self.vol_window:]
        vols = recent.std(ddof=1) * np.sqrt(252)

        weighted = sum(
            w * float(vols[s]) for s, w in targets.items()
            if s in vols.index and np.isfinite(vols[s])
        )
        if weighted <= 1e-9:
            return 1.0

        # 위로는 늘리지 않는다. 변동성이 낮다고 비중을 키우면
        # 조용한 시장에서 최대로 물린 채 폭락을 맞는다.
        return float(min(1.0, self.vol_target / weighted))

    # --- 메인 --------------------------------------------------------------

    def adjust(self, targets: dict[str, float], ctx: RiskContext) -> dict[str, float]:
        """전략의 목표 비중에 리스크 규칙을 전부 적용한 최종 비중."""
        bar = ctx.view.index

        # 1. 계좌 전체 서킷브레이커 — 다른 모든 판단보다 우선한다
        if self.max_drawdown_stop and ctx.peak_equity > 0:
            dd = ctx.equity / ctx.peak_equity - 1.0
            if dd <= -self.max_drawdown_stop and bar > self._circuit_until:
                self._circuit_until = bar + self.cooldown_days
                self.stop_events.append(
                    {"date": ctx.view.date, "symbol": "*PORTFOLIO*",
                     "reason": "서킷브레이커", "drawdown": dd}
                )
        if bar <= self._circuit_until:
            return {}  # 전량 현금

        # 2. 개별 손절 — 보유 중인 종목만 검사한다
        for symbol in list(ctx.positions):
            reason = self._stopped_out(symbol, ctx)
            if reason:
                self._blocked[symbol] = bar + self.cooldown_days
                self.stop_events.append(
                    {"date": ctx.view.date, "symbol": symbol, "reason": reason,
                     "price": float(ctx.view.last_close()[symbol])}
                )

        # 3. 냉각 중인 종목 제외
        adjusted = {
            s: w for s, w in targets.items()
            if w > 0 and self._blocked.get(s, -1) < bar
        }
        if not adjusted:
            return {}

        # 4. 종목당 비중 상한
        adjusted = {s: min(w, self.max_position_weight) for s, w in adjusted.items()}

        # 5. 변동성 타겟팅
        adjusted = {s: w * self._exposure_scale(adjusted, ctx) for s, w in adjusted.items()}

        # 6. 총 노출 상한 (레버리지 차단)
        gross = sum(adjusted.values())
        if gross > self.max_gross_exposure:
            scale = self.max_gross_exposure / gross
            adjusted = {s: w * scale for s, w in adjusted.items()}

        return {s: w for s, w in adjusted.items() if w > 1e-6}

    def events_frame(self) -> pd.DataFrame:
        """발동된 손절·서킷 기록."""
        if not self.stop_events:
            return pd.DataFrame(columns=["date", "symbol", "reason"])
        return pd.DataFrame(self.stop_events)


# 자주 쓰는 프리셋 ------------------------------------------------------------
#
# RiskManager는 손절 이력 같은 상태를 들고 있으므로 인스턴스를 공유하면 안 된다.
# 호출할 때마다 새로 만들어 주는 팩토리로 둔다.

PRESETS = {
    "conservative": lambda: RiskManager(
        max_position_weight=0.20, stop_loss_pct=0.10, trailing_stop_pct=0.15,
        max_drawdown_stop=0.20, vol_target=0.12,
    ),
    "balanced": lambda: RiskManager(
        max_position_weight=0.25, stop_loss_pct=0.15, trailing_stop_pct=0.25,
        max_drawdown_stop=0.30, vol_target=0.18,
    ),
    "aggressive": lambda: RiskManager(
        max_position_weight=0.50, stop_loss_pct=0.25, trailing_stop_pct=0.35,
        max_drawdown_stop=0.45, vol_target=None,
    ),
    # 리스크 관리를 끈 상태. 비교 목적으로만 존재한다.
    # 실제로 이걸로 운용하면 언젠가 계좌가 사라진다.
    "none": lambda: RiskManager(
        max_position_weight=1.0, stop_loss_pct=None, trailing_stop_pct=None,
        max_drawdown_stop=None, vol_target=None,
    ),
}


def preset(name: str) -> RiskManager:
    """이름으로 리스크 프리셋을 새로 만든다."""
    key = name.strip().lower()
    if key not in PRESETS:
        raise ValueError(f"알 수 없는 리스크 프리셋: {name} (가능: {', '.join(PRESETS)})")
    return PRESETS[key]()
