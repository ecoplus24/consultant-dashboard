"""
증권사 연동 공통 인터페이스

백테스트 엔진과 실전 주문을 같은 어휘로 다루기 위한 계층이다.
증권사가 바뀌어도 위쪽 코드(전략·리스크·실행기)는 그대로 쓸 수 있어야 한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Order:
    """주문 하나."""

    symbol: str
    side: str  # "buy" / "sell"
    quantity: int
    order_type: str = "limit"  # "limit" 지정가 / "market" 시장가
    price: float = 0.0  # 시장가면 무시된다

    def __post_init__(self):
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side는 buy 또는 sell이어야 한다: {self.side}")
        if self.order_type not in ("limit", "market"):
            raise ValueError(f"order_type은 limit 또는 market이어야 한다: {self.order_type}")
        if self.quantity <= 0:
            raise ValueError(f"수량은 1주 이상이어야 한다: {self.quantity}")
        if self.order_type == "limit" and self.price <= 0:
            raise ValueError("지정가 주문에는 가격이 필요하다")

    @property
    def notional(self) -> float:
        """주문 금액(추정). 시장가는 가격을 모르므로 price를 넣어 둔 경우만 유효하다."""
        return self.quantity * self.price

    def __str__(self) -> str:
        kind = "시장가" if self.order_type == "market" else f"{self.price:,.0f}원"
        action = "매수" if self.side == "buy" else "매도"
        return f"{self.symbol} {action} {self.quantity:,}주 @ {kind}"


@dataclass
class OrderResult:
    """주문 접수 결과. 체결 결과가 아니라 '접수'되었는지다."""

    ok: bool
    order: Order
    order_id: str = ""
    message: str = ""
    raw: dict = field(default_factory=dict)
    submitted_at: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        mark = "OK" if self.ok else "실패"
        return f"[{mark}] {self.order} — {self.message}"


@dataclass
class Holding:
    """보유 종목."""

    symbol: str
    name: str
    quantity: int
    avg_price: float
    current_price: float

    @property
    def value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return self.current_price / self.avg_price - 1.0


@dataclass
class Account:
    """계좌 스냅샷."""

    cash: float  # 주문가능현금
    holdings: dict[str, Holding] = field(default_factory=dict)

    @property
    def holdings_value(self) -> float:
        return sum(h.value for h in self.holdings.values())

    @property
    def equity(self) -> float:
        return self.cash + self.holdings_value

    def weight_of(self, symbol: str) -> float:
        if self.equity <= 0:
            return 0.0
        h = self.holdings.get(symbol)
        return (h.value / self.equity) if h else 0.0


class Broker(ABC):
    """증권사 어댑터."""

    #: 실제 돈이 오가는 연결인지. 실행기가 안전장치를 켤 때 본다.
    is_live: bool = False

    @abstractmethod
    def get_account(self) -> Account:
        """잔고와 보유종목을 조회한다."""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """현재가를 조회한다."""

    @abstractmethod
    def submit(self, order: Order) -> OrderResult:
        """주문을 낸다."""

    @abstractmethod
    def cancel(self, order_id: str, symbol: str, quantity: int) -> OrderResult:
        """미체결 주문을 취소한다."""


class BrokerError(RuntimeError):
    """증권사 API가 거부했거나 응답이 이상할 때."""
