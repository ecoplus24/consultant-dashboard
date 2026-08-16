"""
실전/모의 주문 실행 계층

백테스트와 같은 전략·리스크 코드를 그대로 쓰되, 가격만 실시간 계좌에서 온다.
전략을 두 번 구현하면 백테스트와 실전이 갈라지고, 그 순간 검증은 무의미해진다.

기본 동작은 항상 **드라이런**이다. 주문은 계산해서 보여 주기만 하고 내지 않는다.
실제 주문은 execute()를 명시적으로 부를 때만 나간다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

from .broker.base import Account, Broker, Order, OrderResult
from .data.base import DataProvider, PriceData
from .risk import RiskContext, RiskManager
from .strategies.base import Strategy


KST = timezone(timedelta(hours=9))
STATE_DIR = Path.home() / ".cache" / "trading"

# KRX 호가단위 (2023년 개정 기준). 이 단위에 안 맞는 지정가는 주문이 거부된다.
TICK_TABLE = [
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
    (float("inf"), 1_000),
]


def tick_size(price: float) -> int:
    """해당 가격대의 호가단위."""
    for upper, tick in TICK_TABLE:
        if price < upper:
            return tick
    return 1_000


def round_to_tick(price: float, side: str) -> int:
    """
    호가단위에 맞춰 가격을 자른다.

    매수는 내림, 매도는 올림 — 항상 나에게 불리한 쪽이 아니라
    '체결을 서두르지 않는' 쪽으로 맞춘다. 급하게 체결하려고 반대로 하면
    슬리피지가 그대로 비용이 된다.
    """
    tick = tick_size(price)
    if side == "buy":
        return int(price // tick * tick)
    return int(-(-price // tick) * tick)


@dataclass
class SafetyLimits:
    """
    주문 실행 전에 통과해야 하는 관문.

    리스크 계층(risk.py)이 "얼마나 들고 있을지"를 정한다면,
    여기는 "한 번에 얼마나 잘못될 수 있는지"를 막는다.
    코드 버그, 데이터 오류, 손가락 실수로 큰 주문이 나가는 걸 차단하는 게 목적이다.
    """

    max_order_value: float = 1_000_000      # 주문 1건 최대 금액
    max_total_order_value: float = 5_000_000  # 실행 1회 총 주문 금액
    max_orders_per_run: int = 20
    min_order_value: float = 50_000         # 이보다 작으면 수수료가 아깝다
    min_price: float = 1_000                # 동전주 제외
    allow_market_orders: bool = False       # 시장가는 슬리피지를 통제할 수 없다
    limit_offset: float = 0.003             # 지정가를 현재가 대비 얼마나 벌려 낼지
    trading_hours_only: bool = True
    cash_buffer: float = 0.02               # 현금의 2%는 남긴다 (수수료·세금 여유)

    def check_order(self, order: Order) -> str | None:
        """통과하지 못하는 이유. 문제없으면 None."""
        if order.order_type == "market" and not self.allow_market_orders:
            return "시장가 주문이 비활성화돼 있다 (allow_market_orders=True 필요)"
        if order.price > 0 and order.price < self.min_price:
            return f"주가 {order.price:,.0f}원이 하한 {self.min_price:,.0f}원 미만"
        value = order.notional
        if value > self.max_order_value:
            return f"주문금액 {value:,.0f}원이 1건 한도 {self.max_order_value:,.0f}원 초과"
        # 청산(매도)은 금액이 작아도 막으면 안 된다. 손절이 무력화된다.
        if order.side == "buy" and 0 < value < self.min_order_value:
            return f"주문금액 {value:,.0f}원이 최소 {self.min_order_value:,.0f}원 미만"
        return None


def is_market_open(now: datetime | None = None) -> bool:
    """
    KRX 정규장(평일 09:00~15:30 KST) 여부.

    공휴일은 판별하지 않는다. 휴장일에 주문을 내면 증권사가 거부하므로
    치명적이지는 않지만, 그것까지 막고 싶으면 휴장일 달력을 붙여야 한다.
    """
    now = now or datetime.now(KST)
    if now.weekday() >= 5:
        return False
    return dtime(9, 0) <= now.time() <= dtime(15, 30)


@dataclass
class TradePlan:
    """오늘 낼 주문과 그 근거."""

    orders: list[Order] = field(default_factory=list)
    target_weights: dict[str, float] = field(default_factory=dict)
    account: Account | None = None
    rejected: list[tuple[Order, str]] = field(default_factory=list)
    as_of: datetime = field(default_factory=lambda: datetime.now(KST))

    @property
    def total_buy(self) -> float:
        return sum(o.notional for o in self.orders if o.side == "buy")

    @property
    def total_sell(self) -> float:
        return sum(o.notional for o in self.orders if o.side == "sell")


class LiveTrader:
    """
    전략 → 목표비중 → 리스크 조정 → 주문 생성 → (선택적) 제출.

    손절 추적과 계좌 고점은 실행 사이에 기억해야 하므로 파일에 저장한다.
    이게 없으면 매번 새로 시작한 것처럼 굴어서 추적손절과 서킷브레이커가 작동하지 않는다.
    """

    def __init__(
        self,
        broker: Broker,
        strategy: Strategy,
        risk: RiskManager,
        provider: DataProvider,
        symbols: list[str],
        limits: SafetyLimits | None = None,
        lookback_days: int = 500,
        state_path: Path | None = None,
    ):
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.provider = provider
        self.symbols = list(symbols)
        self.limits = limits or SafetyLimits()
        self.lookback_days = lookback_days
        self.state_path = state_path or (STATE_DIR / "live_state.json")
        self.state = self._load_state()

    # --- 상태 저장 ----------------------------------------------------------

    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"high_water": {}, "peak_equity": 0.0, "history": []}

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state["history"] = self.state.get("history", [])[-200:]
            self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))
        except OSError:
            pass

    # --- 시세 --------------------------------------------------------------

    def _fetch_history(self) -> PriceData:
        """전략 워밍업에 필요한 만큼의 일봉을 받는다."""
        end = datetime.now(KST).date()
        start = end - timedelta(days=self.lookback_days)
        data = self.provider.fetch(self.symbols, str(start), str(end))

        if len(data.calendar) < self.strategy.warmup + 1:
            raise ValueError(
                f"데이터가 부족하다. {self.strategy.warmup + 1}일 필요한데 "
                f"{len(data.calendar)}일뿐이다. lookback_days를 늘려라."
            )
        return data

    # --- 계획 --------------------------------------------------------------

    def plan(self) -> TradePlan:
        """오늘 낼 주문을 계산한다. 주문은 내지 않는다."""
        data = self._fetch_history()
        view = data.view(len(data.calendar) - 1)
        account = self.broker.get_account()

        # 계좌 고점은 실행 사이에 이어져야 서킷브레이커가 의미를 갖는다
        peak_equity = max(float(self.state.get("peak_equity", 0.0)), account.equity)
        self.state["peak_equity"] = peak_equity

        # 보유 종목 고점 갱신 (추적손절용)
        high_water = dict(self.state.get("high_water", {}))
        for symbol, holding in account.holdings.items():
            high_water[symbol] = max(high_water.get(symbol, 0.0), holding.current_price)
        # 이미 판 종목의 기록은 지운다
        high_water = {s: v for s, v in high_water.items() if s in account.holdings}
        self.state["high_water"] = high_water

        targets = self.strategy.target_weights(view)
        if targets is None:  # 유지
            targets = {s: account.weight_of(s) for s in account.holdings}

        ctx = RiskContext(
            view=view,
            equity=account.equity,
            peak_equity=peak_equity,
            positions={s: float(h.quantity) for s, h in account.holdings.items()},
            entry_price={s: h.avg_price for s, h in account.holdings.items()},
            high_water=high_water,
        )
        adjusted = self.risk.adjust(targets, ctx)

        orders, rejected = self._build_orders(adjusted, account)
        self._save_state()

        return TradePlan(
            orders=orders, target_weights=adjusted,
            account=account, rejected=rejected,
        )

    def _limit_price(self, symbol: str, side: str, reference: float) -> int:
        """지정가. 체결 가능성을 조금 높이되 호가단위에 맞춘다."""
        offset = 1 + self.limits.limit_offset if side == "buy" else 1 - self.limits.limit_offset
        return round_to_tick(reference * offset, side)

    def _build_orders(
        self, targets: dict[str, float], account: Account
    ) -> tuple[list[Order], list[tuple[Order, str]]]:
        """목표비중과 현재 보유를 비교해 주문을 만든다."""
        equity = account.equity
        orders: list[Order] = []
        rejected: list[tuple[Order, str]] = []

        if equity <= 0:
            return orders, rejected

        candidates: list[tuple[str, int, float]] = []  # (종목, 증감수량, 기준가)
        for symbol in set(targets) | set(account.holdings):
            holding = account.holdings.get(symbol)
            price = holding.current_price if holding else self.broker.get_price(symbol)
            if price <= 0:
                continue

            target_qty = int((equity * targets.get(symbol, 0.0)) // price)
            delta = target_qty - (holding.quantity if holding else 0)
            if delta != 0:
                candidates.append((symbol, delta, price))

        # 매도를 먼저 처리해야 매수 예산이 생긴다
        candidates.sort(key=lambda c: c[1])

        available = account.cash * (1 - self.limits.cash_buffer)
        total_value = 0.0

        for symbol, delta, price in candidates:
            side = "buy" if delta > 0 else "sell"
            limit = self._limit_price(symbol, side, price)
            quantity = abs(delta)

            if side == "buy":
                # 예산 안에서만 산다
                affordable = int(available // limit) if limit > 0 else 0
                quantity = min(quantity, affordable)
                if quantity <= 0:
                    continue

            try:
                order = Order(symbol=symbol, side=side, quantity=quantity,
                              order_type="limit", price=float(limit))
            except ValueError:
                continue  # 수량 0이나 가격 0 — 낼 수 없는 주문이다

            reason = self.limits.check_order(order)
            if reason is None and len(orders) >= self.limits.max_orders_per_run:
                reason = f"실행당 주문 건수 한도 {self.limits.max_orders_per_run}건 초과"
            if reason is None and total_value + order.notional > self.limits.max_total_order_value:
                reason = (
                    f"총 주문금액 한도 {self.limits.max_total_order_value:,.0f}원 초과"
                )

            if reason:
                rejected.append((order, reason))
                continue

            orders.append(order)
            total_value += order.notional
            if side == "buy":
                available -= order.notional

        return orders, rejected

    # --- 실행 --------------------------------------------------------------

    def execute(self, plan: TradePlan, confirm: bool = False) -> list[OrderResult]:
        """
        계획된 주문을 실제로 낸다.

        실전 계좌(is_live=True)에서는 confirm=True 없이는 절대 나가지 않는다.
        실수로 실행되는 걸 막는 마지막 관문이다.
        """
        if self.broker.is_live and not confirm:
            raise PermissionError(
                "실전 계좌다. 주문을 내려면 confirm=True를 명시해야 한다.\n"
                "모의투자로 충분히 검증한 뒤에만 실행해라."
            )
        if self.limits.trading_hours_only and not is_market_open():
            raise RuntimeError(
                "정규장(평일 09:00~15:30 KST)이 아니다. "
                "테스트하려면 SafetyLimits(trading_hours_only=False)로 끄면 되지만, "
                "장외 시간 주문은 증권사가 거부한다."
            )

        results = []
        for order in plan.orders:
            result = self.broker.submit(order)
            results.append(result)
            self.state.setdefault("history", []).append(
                {
                    "at": datetime.now(KST).isoformat(),
                    "symbol": order.symbol, "side": order.side,
                    "quantity": order.quantity, "price": order.price,
                    "ok": result.ok, "order_id": result.order_id,
                    "message": result.message,
                }
            )
            # 하나라도 거부되면 멈춘다. 원인 모른 채 계속 밀어 넣지 않는다.
            if not result.ok:
                break

        self._save_state()
        return results


def format_plan(plan: TradePlan, broker: Broker) -> str:
    """주문 계획을 사람이 검토할 수 있는 형태로."""
    account = plan.account
    kind = "실전투자" if broker.is_live else "모의투자"

    lines = [
        "=" * 66,
        f" 주문 계획 — {kind}",
        f" 기준시각: {plan.as_of.strftime('%Y-%m-%d %H:%M:%S')} KST",
        "=" * 66,
    ]

    if account:
        lines += [
            "",
            " [ 계좌 ]",
            f"   주문가능현금   {account.cash:>15,.0f} 원",
            f"   주식평가액     {account.holdings_value:>15,.0f} 원",
            f"   총평가액       {account.equity:>15,.0f} 원",
        ]
        if account.holdings:
            lines.append("")
            lines.append(" [ 보유종목 ]")
            for h in account.holdings.values():
                lines.append(
                    f"   {h.symbol} {h.name[:10]:<10} {h.quantity:>6,}주 "
                    f"평가 {h.value:>12,.0f}원  손익 {h.pnl_pct * 100:>+6.2f}%"
                )

    lines += ["", " [ 목표 비중 ]"]
    if plan.target_weights:
        for symbol, weight in sorted(plan.target_weights.items(), key=lambda kv: -kv[1]):
            lines.append(f"   {symbol}  {weight * 100:>6.2f}%")
        cash_weight = 1 - sum(plan.target_weights.values())
        lines.append(f"   {'현금':<8}{cash_weight * 100:>6.2f}%")
    else:
        lines.append("   전량 현금 (진입 조건 미충족 또는 리스크 규칙 발동)")

    lines += ["", " [ 주문 ]"]
    if plan.orders:
        for order in plan.orders:
            lines.append(f"   {order}   ({order.notional:,.0f}원)")
        lines.append("")
        lines.append(f"   매수 합계 {plan.total_buy:>15,.0f} 원")
        lines.append(f"   매도 합계 {plan.total_sell:>15,.0f} 원")
    else:
        lines.append("   없음 — 현재 보유가 목표와 일치한다")

    if plan.rejected:
        lines += ["", " [ 안전장치에 걸린 주문 ]"]
        for order, reason in plan.rejected:
            lines.append(f"   {order}")
            lines.append(f"      → {reason}")

    lines.append("=" * 66)
    return "\n".join(lines)
