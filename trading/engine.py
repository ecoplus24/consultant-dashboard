"""
백테스트 엔진

설계에서 타협하지 않은 것 세 가지:

1. 미래참조 차단 — 신호는 t일 종가로 만들고 주문은 t+1일 시가에 체결된다.
   당일 종가로 신호를 만들어 당일 종가에 체결하는 백테스트는 전부 거짓말이다.
2. 정수 주식 — 100만원으로 30만원짜리 주식을 0.3주 살 수는 없다.
   소액 계좌일수록 이 제약이 성과를 크게 바꾼다.
3. 현금 제약 — 가진 돈보다 많이 못 산다. 갭상승으로 예산이 모자라면 주문이 줄어든다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .data.base import PriceData
from .metrics import Performance, summarize
from .risk import RiskContext, RiskManager
from .strategies.base import Strategy


@dataclass
class Position:
    """보유 종목 하나."""

    quantity: float = 0.0
    avg_cost: float = 0.0  # 수수료 포함 평균 매입단가
    high_water: float = 0.0  # 보유 기간 중 최고 종가


@dataclass
class BacktestResult:
    """백테스트 산출물 전부."""

    equity: pd.Series
    performance: Performance
    trades: pd.DataFrame
    positions: pd.DataFrame
    exposure: pd.Series
    stop_events: pd.DataFrame
    strategy_name: str
    strategy_desc: str
    initial_capital: float
    metadata: dict = field(default_factory=dict)


class Backtest:
    """일봉 기준 이벤트 루프 백테스터."""

    def __init__(
        self,
        data: PriceData,
        strategy: Strategy,
        risk: RiskManager,
        costs: CostModel,
        initial_capital: float = 1_000_000.0,
        rebalance_days: int = 5,
        min_trade_value: float = 0.0,
        fractional_shares: bool = False,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital은 0보다 커야 한다")

        self.data = data
        self.strategy = strategy
        self.risk = risk
        self.costs = costs
        self.initial_capital = float(initial_capital)
        self.rebalance_days = max(1, rebalance_days)
        # 자잘한 리밸런싱은 수수료만 먹는다. 기본값은 자본의 0.5%.
        self.min_trade_value = (
            min_trade_value if min_trade_value > 0 else initial_capital * 0.005
        )
        self.fractional_shares = fractional_shares

        self.cash = self.initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[dict] = []

    # --- 체결 --------------------------------------------------------------

    def _market_value(self, prices: pd.Series) -> float:
        return sum(
            pos.quantity * float(prices[s])
            for s, pos in self.positions.items()
            if pos.quantity > 0 and np.isfinite(prices[s])
        )

    def _affordable(self, fill: float) -> float:
        """현재 현금으로 살 수 있는 최대 수량. 수수료까지 빼고 계산한다."""
        if fill <= 0:
            return 0.0
        unit_cost = fill + self.costs.transaction_cost(fill, "buy")
        qty = self.cash / unit_cost
        return qty if self.fractional_shares else math.floor(qty)

    def _buy(self, symbol: str, quantity: float, price: float, date) -> None:
        if quantity <= 0:
            return
        fill = self.costs.fill_price(price, "buy")

        # 주문 수량은 어제 종가로 계산했는데 오늘 시가가 갭상승했을 수 있다.
        # 그럴 땐 주문을 버리지 말고 살 수 있는 만큼으로 줄인다.
        # (예산 초과분을 그냥 무시하면 상승장에서 매수가 통째로 사라진다)
        quantity = min(quantity, self._affordable(fill))
        if quantity <= 0:
            return  # 한 주도 못 산다. 빚내서 사지는 않는다.

        notional = quantity * fill
        fee = self.costs.transaction_cost(notional, "buy")
        if notional + fee > self.cash + 1e-9:
            return

        self.cash -= notional + fee
        pos = self.positions.setdefault(symbol, Position())
        total_cost = pos.avg_cost * pos.quantity + notional + fee
        pos.quantity += quantity
        pos.avg_cost = total_cost / pos.quantity
        pos.high_water = max(pos.high_water, price)

        self.trades.append(
            {"date": date, "symbol": symbol, "side": "buy", "quantity": quantity,
             "price": fill, "cost": fee, "pnl": None}
        )

    def _sell(self, symbol: str, quantity: float, price: float, date) -> None:
        pos = self.positions.get(symbol)
        if pos is None or quantity <= 0:
            return
        quantity = min(quantity, pos.quantity)
        if quantity <= 0:
            return

        fill = self.costs.fill_price(price, "sell")
        notional = quantity * fill
        fee = self.costs.transaction_cost(notional, "sell")
        pnl = notional - fee - pos.avg_cost * quantity

        self.cash += notional - fee
        pos.quantity -= quantity
        if pos.quantity <= 1e-9:
            self.positions.pop(symbol, None)

        self.trades.append(
            {"date": date, "symbol": symbol, "side": "sell", "quantity": quantity,
             "price": fill, "cost": fee, "pnl": pnl}
        )

    # --- 주문 생성 ----------------------------------------------------------

    def _orders_from_targets(
        self, targets: dict[str, float], prices: pd.Series, equity: float
    ) -> dict[str, float]:
        """목표 비중 -> 종목별 수량 증감."""
        orders: dict[str, float] = {}
        symbols = set(targets) | set(self.positions)

        for symbol in symbols:
            price = float(prices.get(symbol, np.nan))
            if not np.isfinite(price) or price <= 0:
                continue

            target_value = equity * targets.get(symbol, 0.0)
            target_qty = target_value / price
            if not self.fractional_shares:
                target_qty = math.floor(target_qty)

            current_qty = self.positions.get(symbol, Position()).quantity
            delta = target_qty - current_qty
            if abs(delta) < (1e-9 if self.fractional_shares else 1):
                continue

            # 비중을 0으로 만드는 청산은 금액이 작아도 반드시 실행한다.
            # (손절이 min_trade_value에 막히면 리스크 관리가 무력화된다)
            is_exit = targets.get(symbol, 0.0) == 0.0 and current_qty > 0
            if not is_exit and abs(delta) * price < self.min_trade_value:
                continue

            orders[symbol] = delta

        return orders

    def _execute(self, orders: dict[str, float], prices: pd.Series, tradable: pd.Series, date) -> None:
        """매도 먼저, 그다음 매수. 매도 대금이 있어야 매수 예산이 생긴다."""
        sells = {s: q for s, q in orders.items() if q < 0}
        buys = {s: q for s, q in orders.items() if q > 0}

        for symbol, qty in sells.items():
            if not tradable.get(symbol, False):
                continue  # 거래정지·미상장일에는 못 판다
            self._sell(symbol, -qty, float(prices[symbol]), date)

        # 매수는 큰 주문부터 넣는다. 현금이 모자라면 뒤쪽이 잘린다.
        ordered = sorted(buys.items(), key=lambda kv: -kv[1] * float(prices[kv[0]]))
        for symbol, qty in ordered:
            if not tradable.get(symbol, False):
                continue
            self._buy(symbol, qty, float(prices[symbol]), date)

    # --- 메인 루프 ----------------------------------------------------------

    def run(self) -> BacktestResult:
        data = self.data
        calendar = data.calendar
        n = len(calendar)
        warmup = max(1, self.strategy.warmup)

        if n <= warmup + 1:
            raise ValueError(
                f"데이터가 부족하다. 최소 {warmup + 2}일 필요한데 {n}일뿐이다. "
                f"기간을 늘리거나 전략의 lookback을 줄여라."
            )

        self.risk.reset()

        equity_curve = np.full(n, np.nan)
        exposure_curve = np.full(n, 0.0)
        position_log: list[dict] = []
        pending: dict[str, float] = {}
        peak_equity = self.initial_capital
        last_rebalance = -10**9

        for i in range(n):
            date = calendar[i]
            open_px = data.open.iloc[i]
            close_px = data.close.iloc[i]
            tradable = data.tradable.iloc[i]

            # 1) 어제 낸 주문을 오늘 시가에 체결
            if pending:
                self._execute(pending, open_px, tradable, date)
                pending = {}

            # 2) 종가로 평가하고 보유 종목 고점 갱신
            for symbol, pos in self.positions.items():
                px = float(close_px[symbol])
                if np.isfinite(px):
                    pos.high_water = max(pos.high_water, px)

            holdings = self._market_value(close_px)
            equity = self.cash + holdings
            equity_curve[i] = equity
            exposure_curve[i] = holdings / equity if equity > 0 else 0.0
            peak_equity = max(peak_equity, equity)

            if self.positions:
                position_log.append(
                    {"date": date, "cash": self.cash, "holdings": holdings,
                     "equity": equity, "n_positions": len(self.positions)}
                )

            # 계좌가 사실상 소멸하면 더 돌릴 이유가 없다
            if equity <= self.initial_capital * 0.01:
                equity_curve[i:] = equity
                break

            # 3) 오늘 종가 기준으로 내일 낼 주문을 만든다
            if i < warmup or i >= n - 1:
                continue
            if i - last_rebalance < self.rebalance_days:
                continue

            view = data.view(i)
            targets = self.strategy.target_weights(view)

            # None은 "현 상태 유지". 지금 실제 비중을 그대로 목표로 삼으면
            # 리밸런싱 거래는 안 생기면서 손절 같은 리스크 규칙은 계속 걸린다.
            if targets is None:
                targets = {
                    s: (p.quantity * float(close_px[s])) / equity
                    for s, p in self.positions.items()
                    if equity > 0 and np.isfinite(close_px[s])
                }

            ctx = RiskContext(
                view=view,
                equity=equity,
                peak_equity=peak_equity,
                positions={s: p.quantity for s, p in self.positions.items()},
                entry_price={s: p.avg_cost for s, p in self.positions.items()},
                high_water={s: p.high_water for s, p in self.positions.items()},
            )
            targets = self.risk.adjust(targets, ctx)

            pending = self._orders_from_targets(targets, close_px, equity)
            if pending:
                last_rebalance = i

        equity_series = pd.Series(equity_curve, index=calendar, name="equity").ffill()
        exposure_series = pd.Series(exposure_curve, index=calendar, name="exposure")
        trades_df = pd.DataFrame(self.trades)
        total_costs = float(trades_df["cost"].sum()) if len(trades_df) else 0.0

        performance = summarize(
            equity_series,
            trades=trades_df,
            exposure_series=exposure_series,
            total_costs=total_costs,
        )

        return BacktestResult(
            equity=equity_series,
            performance=performance,
            trades=trades_df,
            positions=pd.DataFrame(position_log),
            exposure=exposure_series,
            stop_events=self.risk.events_frame(),
            strategy_name=self.strategy.name,
            strategy_desc=self.strategy.describe(),
            initial_capital=self.initial_capital,
            metadata={
                "symbols": data.symbols,
                "rebalance_days": self.rebalance_days,
                "fractional_shares": self.fractional_shares,
            },
        )


def run_backtest(
    data: PriceData,
    strategy: Strategy,
    risk: RiskManager | None = None,
    costs: CostModel | None = None,
    initial_capital: float = 1_000_000.0,
    **kwargs,
) -> BacktestResult:
    """백테스트 한 번 돌리는 단축 함수."""
    from .costs import KR_COSTS
    from .risk import PRESETS

    return Backtest(
        data=data,
        strategy=strategy,
        risk=risk if risk is not None else PRESETS["balanced"](),
        costs=costs if costs is not None else KR_COSTS,
        initial_capital=initial_capital,
        **kwargs,
    ).run()
