"""
증권사 연동 검증

실제 KIS 서버에는 붙지 않는다. HTTP 응답을 가짜로 만들어서
"우리 코드가 응답을 올바로 해석하는가"와 "안전장치가 실제로 막는가"를 본다.

여기서 확인하는 건 수익이 아니라 사고 방지다:
실전 계좌 오발주, 호가단위 위반, 예산 초과 매수, 장외 시간 주문.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trading.broker.base import Account, Broker, BrokerError, Holding, Order, OrderResult
from trading.broker.kis import KISBroker, KISClient, KISConfig
from trading.data.providers import SyntheticProvider
from trading.live import (
    LiveTrader,
    SafetyLimits,
    format_plan,
    is_market_open,
    round_to_tick,
    tick_size,
)
from trading.risk import RiskManager, preset
from trading.strategies import BuyAndHold
from trading.strategies.base import Strategy


# --- 가짜 HTTP --------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class FakeSession:
    """요청을 기록하고 미리 정한 응답을 돌려준다."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def _resolve(self, url):
        for key, payload in self.responses.items():
            if key in url:
                return payload
        raise AssertionError(f"준비되지 않은 요청: {url}")

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return FakeResponse(self._resolve(url))

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeResponse(self._resolve(url))


TOKEN_RESPONSE = {"access_token": "test-token", "expires_in": 86400}
HASHKEY_RESPONSE = {"HASH": "test-hash"}

BALANCE_RESPONSE = {
    "rt_cd": "0",
    "msg1": "정상처리",
    "output1": [
        {"pdno": "005930", "prdt_name": "삼성전자", "hldg_qty": "10",
         "pchs_avg_pric": "70000", "prpr": "75000"},
        {"pdno": "000660", "prdt_name": "SK하이닉스", "hldg_qty": "0",
         "pchs_avg_pric": "0", "prpr": "150000"},
    ],
    "output2": [{"prvs_rcdl_excc_amt": "500000", "dnca_tot_amt": "900000"}],
    "ctx_area_fk100": "", "ctx_area_nk100": "",
}

PRICE_RESPONSE = {"rt_cd": "0", "output": {"stck_prpr": "75000"}}
ORDER_RESPONSE = {"rt_cd": "0", "msg1": "주문 전송 완료", "output": {"ODNO": "0000123456"}}


@pytest.fixture
def config(tmp_path):
    return KISConfig(app_key="key12345", app_secret="secret", account="12345678-01",
                     env="paper")


@pytest.fixture
def client(config, monkeypatch):
    c = KISClient(config, use_token_cache=False)
    c.session = FakeSession({
        "/oauth2/tokenP": TOKEN_RESPONSE,
        "/uapi/hashkey": HASHKEY_RESPONSE,
        "inquire-balance": BALANCE_RESPONSE,
        "inquire-price": PRICE_RESPONSE,
        "order-cash": ORDER_RESPONSE,
    })
    # 테스트에서 호출 간 대기까지 하면 느려진다
    monkeypatch.setattr(c._limiter, "min_interval", 0.0)
    return c


@pytest.fixture
def broker(client):
    return KISBroker(client=client)


# --- 설정 -------------------------------------------------------------------


def test_paper_is_the_default_environment():
    config = KISConfig(app_key="k", app_secret="s", account="12345678-01")
    assert config.env == "paper"
    assert "openapivts" in config.host  # 모의투자 서버


def test_live_and_paper_use_different_hosts_and_tr_ids():
    paper = KISConfig(app_key="k", app_secret="s", account="12345678-01", env="paper")
    live = KISConfig(app_key="k", app_secret="s", account="12345678-01", env="live")

    assert paper.host != live.host
    assert paper.tr("buy") != live.tr("buy")
    assert paper.tr("buy").startswith("V")  # 모의투자 TR은 V로 시작
    assert live.tr("buy").startswith("T")


def test_account_number_is_split_correctly():
    config = KISConfig(app_key="k", app_secret="s", account="12345678-01")
    assert config.cano == "12345678"
    assert config.product_code == "01"


def test_malformed_account_number_is_rejected():
    with pytest.raises(ValueError, match="계좌번호"):
        KISConfig(app_key="k", app_secret="s", account="12345678")


def test_tr_ids_can_be_overridden_without_editing_code():
    """증권사가 TR ID를 개정해도 설정으로 대응할 수 있어야 한다."""
    config = KISConfig(app_key="k", app_secret="s", account="12345678-01",
                       tr_overrides={"buy": "TTTC0012U"})
    assert config.tr("buy") == "TTTC0012U"
    assert config.tr("sell") == "VTTC0801U"  # 나머지는 기본값 유지


def test_paper_rate_limit_is_stricter_than_live():
    paper = KISConfig(app_key="k", app_secret="s", account="12345678-01", env="paper")
    live = KISConfig(app_key="k", app_secret="s", account="12345678-01", env="live")
    assert paper.min_interval > live.min_interval


def test_missing_env_vars_give_a_useful_message(monkeypatch):
    for key in ("KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ACCOUNT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(BrokerError, match="KIS_APP_KEY"):
        KISConfig.from_env()


# --- 토큰 -------------------------------------------------------------------


def test_token_is_reused_within_the_session(client):
    first = client.token()
    second = client.token()
    assert first == second == "test-token"
    token_calls = [c for c in client.session.calls if "tokenP" in c[1]]
    assert len(token_calls) == 1, "토큰을 두 번 받으면 KIS가 발급 제한으로 막는다"


def test_token_cache_file_is_owner_only(config, monkeypatch, tmp_path):
    """토큰은 계좌 접근 권한이다. 다른 사용자가 읽을 수 있으면 안 된다."""
    monkeypatch.setattr("trading.broker.kis.TOKEN_CACHE_DIR", tmp_path)
    c = KISClient(config, use_token_cache=True)
    c.session = FakeSession({"/oauth2/tokenP": TOKEN_RESPONSE})
    monkeypatch.setattr(c._limiter, "min_interval", 0.0)
    c.token()

    files = list(tmp_path.glob("token_*.json"))
    assert len(files) == 1
    assert files[0].stat().st_mode & 0o077 == 0, "토큰 파일이 다른 사용자에게 노출된다"


def test_expired_cached_token_is_refreshed(config, monkeypatch, tmp_path):
    monkeypatch.setattr("trading.broker.kis.TOKEN_CACHE_DIR", tmp_path)
    c = KISClient(config, use_token_cache=True)
    path = c._token_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "access_token": "stale",
        "expires_at": (datetime.now() - timedelta(hours=1)).isoformat(),
    }))

    c.session = FakeSession({"/oauth2/tokenP": TOKEN_RESPONSE})
    monkeypatch.setattr(c._limiter, "min_interval", 0.0)
    assert c.token() == "test-token"


# --- 응답 해석 --------------------------------------------------------------


def test_balance_parses_holdings_and_skips_zero_quantity(broker):
    account = broker.get_account()
    assert "005930" in account.holdings
    assert "000660" not in account.holdings, "보유수량 0은 보유종목이 아니다"

    holding = account.holdings["005930"]
    assert holding.quantity == 10
    assert holding.value == pytest.approx(750_000)
    assert holding.pnl_pct == pytest.approx(75_000 / 70_000 - 1)


def test_balance_uses_settlement_cash_not_deposit(broker):
    """
    예수금(dnca_tot_amt)에는 미결제 금액이 섞여 있다.
    그걸로 주문하면 증권사가 거부하므로 주문가능금액을 써야 한다.
    """
    account = broker.get_account()
    assert account.cash == 500_000  # prvs_rcdl_excc_amt
    assert account.equity == pytest.approx(500_000 + 750_000)


def test_api_rejection_raises_instead_of_returning_silently(client):
    client.session = FakeSession({
        "inquire-price": {"rt_cd": "1", "msg_cd": "40580000", "msg1": "종목코드 오류"},
        "/oauth2/tokenP": TOKEN_RESPONSE,
    })
    with pytest.raises(BrokerError, match="종목코드 오류"):
        client.current_price("999999")


def test_order_failure_returns_a_result_not_an_exception(broker):
    """주문 실패는 예외가 아니라 결과로 돌려서 나머지 흐름을 통제한다."""
    broker.client.session = FakeSession({
        "/oauth2/tokenP": TOKEN_RESPONSE,
        "/uapi/hashkey": HASHKEY_RESPONSE,
        "order-cash": {"rt_cd": "1", "msg_cd": "40310000", "msg1": "주문가능금액 부족"},
    })
    result = broker.submit(Order("005930", "buy", 10, "limit", 75_000))
    assert not result.ok
    assert "주문가능금액 부족" in result.message


def test_submit_sends_hashkey_and_correct_tr_id(broker):
    broker.submit(Order("005930", "buy", 3, "limit", 75_000))
    order_calls = [c for c in broker.client.session.calls if "order-cash" in c[1]]
    assert len(order_calls) == 1

    headers = order_calls[0][2]["headers"]
    assert headers["tr_id"] == "VTTC0802U"  # 모의투자 매수
    assert headers["hashkey"] == "test-hash"

    body = order_calls[0][2]["json"]
    assert body["ORD_DVSN"] == "00"  # 지정가
    assert body["ORD_QTY"] == "3"
    assert body["ORD_UNPR"] == "75000"


def test_market_order_sends_zero_price(broker):
    broker.submit(Order("005930", "buy", 1, "market"))
    body = [c for c in broker.client.session.calls if "order-cash" in c[1]][0][2]["json"]
    assert body["ORD_DVSN"] == "01"
    assert body["ORD_UNPR"] == "0"


# --- 주문 검증 --------------------------------------------------------------


def test_invalid_orders_are_rejected_at_construction():
    with pytest.raises(ValueError):
        Order("005930", "buy", 0, "limit", 100)       # 수량 0
    with pytest.raises(ValueError):
        Order("005930", "hold", 1, "limit", 100)      # 잘못된 side
    with pytest.raises(ValueError):
        Order("005930", "buy", 1, "limit", 0)         # 지정가인데 가격 없음


# --- 호가단위 ---------------------------------------------------------------


@pytest.mark.parametrize("price,expected", [
    (1_500, 1), (3_000, 5), (12_000, 10),
    (35_000, 50), (75_000, 100), (300_000, 500), (700_000, 1_000),
])
def test_tick_size_table(price, expected):
    assert tick_size(price) == expected


def test_limit_price_snaps_to_tick():
    """호가단위에 안 맞는 가격은 주문이 거부된다."""
    assert round_to_tick(75_123, "buy") == 75_100    # 100원 단위 내림
    assert round_to_tick(75_123, "sell") == 75_200   # 100원 단위 올림
    assert round_to_tick(3_002, "buy") == 3_000      # 5원 단위
    assert round_to_tick(1_234, "buy") == 1_234      # 1원 단위는 그대로


# --- 안전장치 ---------------------------------------------------------------


def test_market_orders_are_blocked_by_default():
    limits = SafetyLimits()
    reason = limits.check_order(Order("005930", "buy", 1, "market"))
    assert reason is not None and "시장가" in reason


def test_oversized_order_is_blocked():
    limits = SafetyLimits(max_order_value=1_000_000)
    reason = limits.check_order(Order("005930", "buy", 100, "limit", 75_000))
    assert reason is not None and "한도" in reason


def test_penny_stock_is_blocked():
    limits = SafetyLimits(min_price=1_000)
    reason = limits.check_order(Order("123456", "buy", 100, "limit", 500))
    assert reason is not None and "하한" in reason


def test_small_sell_is_allowed_but_small_buy_is_not():
    """청산을 최소금액으로 막으면 손절이 무력화된다."""
    limits = SafetyLimits(min_order_value=50_000)
    assert limits.check_order(Order("005930", "sell", 1, "limit", 10_000)) is None
    assert limits.check_order(Order("005930", "buy", 1, "limit", 10_000)) is not None


# --- 실행기 -----------------------------------------------------------------


class FakeBroker(Broker):
    """계좌를 흉내내는 브로커. 주문은 기록만 한다."""

    def __init__(self, cash=1_000_000, holdings=None, is_live=False):
        self.account = Account(cash=cash, holdings=holdings or {})
        self.is_live = is_live
        self.submitted: list[Order] = []

    def get_account(self):
        return self.account

    def get_price(self, symbol):
        h = self.account.holdings.get(symbol)
        return h.current_price if h else 50_000.0

    def submit(self, order):
        self.submitted.append(order)
        return OrderResult(ok=True, order=order, order_id="X1", message="접수됨")

    def cancel(self, order_id, symbol, quantity):
        return OrderResult(ok=True, order=Order(symbol, "sell", quantity, "market"),
                           message="취소됨")


@pytest.fixture
def trader_parts(tmp_path):
    provider = SyntheticProvider(seed=3)
    return provider, tmp_path / "state.json"


def make_trader(broker, provider, state_path, strategy=None, risk=None, limits=None):
    return LiveTrader(
        broker=broker,
        strategy=strategy or BuyAndHold(),
        risk=risk or preset("none"),
        provider=provider,
        symbols=["AAA", "BBB"],
        limits=limits or SafetyLimits(trading_hours_only=False),
        lookback_days=400,
        state_path=state_path,
    )


def test_plan_does_not_submit_anything(trader_parts):
    provider, state = trader_parts
    broker = FakeBroker(cash=1_000_000)
    plan = make_trader(broker, provider, state).plan()

    assert broker.submitted == [], "plan()은 절대 주문을 내면 안 된다"
    assert isinstance(plan.orders, list)


def test_live_account_refuses_to_execute_without_confirm(trader_parts):
    provider, state = trader_parts
    broker = FakeBroker(cash=1_000_000, is_live=True)
    trader = make_trader(broker, provider, state)
    plan = trader.plan()

    with pytest.raises(PermissionError, match="confirm"):
        trader.execute(plan, confirm=False)
    assert broker.submitted == []


def test_orders_stay_within_cash(trader_parts):
    provider, state = trader_parts
    broker = FakeBroker(cash=1_000_000)
    limits = SafetyLimits(trading_hours_only=False, max_order_value=10_000_000,
                          max_total_order_value=10_000_000)
    plan = make_trader(broker, provider, state, limits=limits).plan()

    buy_total = sum(o.notional for o in plan.orders if o.side == "buy")
    assert buy_total <= 1_000_000, "가진 현금보다 많이 사려 한다"


def test_total_order_cap_blocks_the_excess(trader_parts):
    provider, state = trader_parts
    broker = FakeBroker(cash=100_000_000)
    limits = SafetyLimits(trading_hours_only=False, max_order_value=100_000_000,
                          max_total_order_value=1_000_000)
    plan = make_trader(broker, provider, state, limits=limits).plan()

    assert sum(o.notional for o in plan.orders) <= 1_000_000
    assert plan.rejected, "한도를 넘은 주문은 거부 목록에 남아야 한다"


def test_all_limit_prices_are_on_valid_ticks(trader_parts):
    provider, state = trader_parts
    broker = FakeBroker(cash=5_000_000)
    limits = SafetyLimits(trading_hours_only=False, max_order_value=5_000_000,
                          max_total_order_value=5_000_000)
    plan = make_trader(broker, provider, state, limits=limits).plan()

    assert plan.orders, "주문이 하나도 없으면 검증 의미가 없다"
    for order in plan.orders:
        assert order.price % tick_size(order.price) == 0, f"{order} 호가단위 위반"


def test_stop_loss_produces_a_sell_order(trader_parts):
    """평단 대비 크게 하락한 보유 종목은 청산 주문이 나와야 한다."""
    provider, state = trader_parts
    holdings = {
        "AAA": Holding("AAA", "가상A", quantity=10, avg_price=100_000,
                       current_price=70_000),
    }
    broker = FakeBroker(cash=100_000, holdings=holdings)

    manager = RiskManager(max_position_weight=1.0, stop_loss_pct=0.15,
                          trailing_stop_pct=None, max_drawdown_stop=None,
                          vol_target=None)

    class HoldEverything(Strategy):
        warmup = 1

        def target_weights(self, view):
            return None  # 유지

    plan = make_trader(broker, provider, state,
                       strategy=HoldEverything(), risk=manager).plan()

    sells = [o for o in plan.orders if o.side == "sell" and o.symbol == "AAA"]
    assert sells, "손절 조건인데 매도 주문이 없다"
    assert sells[0].quantity == 10


def test_state_persists_peak_equity_across_runs(trader_parts):
    """
    계좌 고점을 기억하지 못하면 서킷브레이커가 매 실행마다 초기화된다.
    """
    provider, state = trader_parts
    rich = FakeBroker(cash=10_000_000)
    make_trader(rich, provider, state).plan()

    saved = json.loads(Path(state).read_text())
    assert saved["peak_equity"] == pytest.approx(10_000_000)

    poor = FakeBroker(cash=5_000_000)
    trader = make_trader(poor, provider, state)
    trader.plan()
    assert trader.state["peak_equity"] == pytest.approx(10_000_000), "고점이 낮아졌다"


def test_execute_stops_after_the_first_failure(trader_parts):
    provider, state = trader_parts

    class FailingBroker(FakeBroker):
        def submit(self, order):
            self.submitted.append(order)
            return OrderResult(ok=False, order=order, message="거부됨")

    broker = FailingBroker(cash=5_000_000)
    limits = SafetyLimits(trading_hours_only=False, max_order_value=5_000_000,
                          max_total_order_value=5_000_000)
    trader = make_trader(broker, provider, state, limits=limits)
    plan = trader.plan()

    if len(plan.orders) < 2:
        pytest.skip("주문이 2건 미만이라 중단 동작을 볼 수 없다")

    results = trader.execute(plan, confirm=True)
    assert len(results) == 1, "첫 실패 후에도 계속 주문을 밀어 넣고 있다"


def test_outside_trading_hours_execution_is_blocked(trader_parts):
    provider, state = trader_parts
    broker = FakeBroker(cash=1_000_000)
    trader = make_trader(broker, provider, state,
                         limits=SafetyLimits(trading_hours_only=True))
    plan = trader.plan()

    saturday = datetime(2025, 1, 4, 11, 0)  # 토요일
    assert not is_market_open(saturday)

    if is_market_open():
        pytest.skip("지금이 정규장이라 차단 동작을 검증할 수 없다")
    with pytest.raises(RuntimeError, match="정규장"):
        trader.execute(plan, confirm=True)


def test_market_hours_boundaries():
    assert is_market_open(datetime(2025, 1, 6, 9, 0))      # 월 09:00 개장
    assert is_market_open(datetime(2025, 1, 6, 15, 30))    # 15:30 종료 시점
    assert not is_market_open(datetime(2025, 1, 6, 8, 59))
    assert not is_market_open(datetime(2025, 1, 6, 15, 31))
    assert not is_market_open(datetime(2025, 1, 5, 11, 0))  # 일요일


def test_plan_renders(trader_parts):
    provider, state = trader_parts
    broker = FakeBroker(cash=1_000_000)
    trader = make_trader(broker, provider, state)
    text = format_plan(trader.plan(), broker)

    assert "모의투자" in text
    assert "주문 계획" in text
