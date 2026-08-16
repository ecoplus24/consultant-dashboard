"""
한국투자증권 KIS Open API 연동

계좌 개설 후 https://apiportal.koreainvestment.com 에서 앱키/시크릿을 발급받아
환경변수로 넣어 두고 쓴다.

    export KIS_APP_KEY="..."
    export KIS_APP_SECRET="..."
    export KIS_ACCOUNT="12345678-01"    # 계좌번호 8자리-상품코드 2자리
    export KIS_ENV="paper"              # paper(모의투자) / live(실전)

기본값은 항상 모의투자다. 실전 전환은 명시적으로 KIS_ENV=live 를 넣거나
KISConfig(env="live")를 직접 써야만 된다.

※ TR ID와 응답 필드는 한국투자증권이 개정할 수 있다. 실전 투입 전에
  API 포털의 최신 문서와 대조하고, 반드시 모의투자로 먼저 검증할 것.
  TR ID는 KISConfig.tr_overrides로 코드 수정 없이 바꿀 수 있게 해 뒀다.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import requests

from .base import Account, Broker, BrokerError, Holding, Order, OrderResult


PAPER_HOST = "https://openapivts.koreainvestment.com:29443"
LIVE_HOST = "https://openapi.koreainvestment.com:9443"

TOKEN_CACHE_DIR = Path.home() / ".cache" / "kis"

# 거래 관련 TR ID는 실전/모의가 다르다. (조회성 시세 TR은 공통)
TR_IDS = {
    "live": {
        "buy": "TTTC0802U",
        "sell": "TTTC0801U",
        "cancel": "TTTC0803U",
        "balance": "TTTC8434R",
        "orderable": "TTTC8908R",
    },
    "paper": {
        "buy": "VTTC0802U",
        "sell": "VTTC0801U",
        "cancel": "VTTC0803U",
        "balance": "VTTC8434R",
        "orderable": "VTTC8908R",
    },
}

TR_PRICE = "FHKST01010100"  # 주식현재가 시세
TR_DAILY = "FHKST03010100"  # 국내주식 기간별 시세(일/주/월/년)


@dataclass
class KISConfig:
    """접속 설정."""

    app_key: str
    app_secret: str
    account: str  # "12345678-01"
    env: str = "paper"
    timeout: int = 10
    tr_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.env not in ("paper", "live"):
            raise ValueError("env는 paper 또는 live여야 한다")
        if "-" not in self.account:
            raise ValueError("계좌번호 형식은 '12345678-01' 이다 (종합계좌-상품코드)")
        if not self.app_key or not self.app_secret:
            raise ValueError("app_key와 app_secret이 필요하다")

    @classmethod
    def from_env(cls) -> "KISConfig":
        """환경변수에서 읽는다. 없는 값이 있으면 무엇이 빠졌는지 알려 준다."""
        missing = [
            k for k in ("KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ACCOUNT") if not os.getenv(k)
        ]
        if missing:
            raise BrokerError(
                f"환경변수가 없다: {', '.join(missing)}\n"
                "  export KIS_APP_KEY=...\n"
                "  export KIS_APP_SECRET=...\n"
                "  export KIS_ACCOUNT=12345678-01\n"
                "  export KIS_ENV=paper   # 기본값. 실전은 live"
            )
        return cls(
            app_key=os.environ["KIS_APP_KEY"],
            app_secret=os.environ["KIS_APP_SECRET"],
            account=os.environ["KIS_ACCOUNT"],
            env=os.getenv("KIS_ENV", "paper").strip().lower(),
        )

    @property
    def host(self) -> str:
        return LIVE_HOST if self.env == "live" else PAPER_HOST

    @property
    def cano(self) -> str:
        """종합계좌번호 8자리."""
        return self.account.split("-")[0]

    @property
    def product_code(self) -> str:
        """계좌상품코드 2자리."""
        return self.account.split("-")[1]

    def tr(self, action: str) -> str:
        return self.tr_overrides.get(action) or TR_IDS[self.env][action]

    # 모의투자는 초당 2건, 실전은 초당 20건이 상한이다.
    @property
    def min_interval(self) -> float:
        return 0.5 if self.env == "paper" else 0.06


class _RateLimiter:
    """호출 간 최소 간격을 강제한다. 초과하면 KIS가 곧바로 차단한다."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()


class KISClient:
    """KIS Open API 저수준 클라이언트. 토큰 관리와 호출 제한을 책임진다."""

    def __init__(self, config: KISConfig, use_token_cache: bool = True):
        self.config = config
        self.use_token_cache = use_token_cache
        self.session = requests.Session()
        self._limiter = _RateLimiter(config.min_interval)
        self._token: str | None = None
        self._token_expires: datetime | None = None

    # --- 토큰 --------------------------------------------------------------

    @property
    def _token_path(self) -> Path:
        # 앱키 전체를 파일명에 넣지 않는다. 앞 8자리만 식별용으로 쓴다.
        return TOKEN_CACHE_DIR / f"token_{self.config.env}_{self.config.app_key[:8]}.json"

    def _load_cached_token(self) -> bool:
        if not self.use_token_cache or not self._token_path.exists():
            return False
        try:
            data = json.loads(self._token_path.read_text())
            expires = datetime.fromisoformat(data["expires_at"])
        except (ValueError, KeyError, OSError):
            return False

        # 만료 10분 전부터는 새로 받는다
        if expires - timedelta(minutes=10) <= datetime.now():
            return False

        self._token = data["access_token"]
        self._token_expires = expires
        return True

    def _save_token(self, token: str, expires: datetime) -> None:
        if not self.use_token_cache:
            return
        try:
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(
                json.dumps({"access_token": token, "expires_at": expires.isoformat()})
            )
            # 토큰은 계좌 접근 권한이다. 소유자만 읽게 막는다.
            self._token_path.chmod(0o600)
        except OSError:
            pass  # 캐시 실패는 치명적이지 않다

    def token(self) -> str:
        """
        접근토큰. 24시간짜리라 파일에 캐싱한다.

        KIS는 토큰 발급 자체에도 호출 제한(분당 1회)을 두므로,
        캐싱하지 않으면 재실행할 때마다 막힌다.
        """
        if self._token and self._token_expires and datetime.now() < self._token_expires:
            return self._token
        if self._load_cached_token():
            return self._token  # type: ignore[return-value]

        self._limiter.wait()
        resp = self.session.post(
            f"{self.config.host}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": self.config.app_key,
                "appsecret": self.config.app_secret,
            },
            timeout=self.config.timeout,
        )
        if resp.status_code != 200:
            raise BrokerError(f"토큰 발급 실패 ({resp.status_code}): {resp.text[:300]}")

        body = resp.json()
        token = body.get("access_token")
        if not token:
            raise BrokerError(f"토큰이 응답에 없다: {body}")

        expires = datetime.now() + timedelta(seconds=int(body.get("expires_in", 86400)))
        self._token, self._token_expires = token, expires
        self._save_token(token, expires)
        return token

    def hashkey(self, body: dict) -> str:
        """주문 본문 위변조 검증용 해시. 주문 계열 POST에 필요하다."""
        self._limiter.wait()
        resp = self.session.post(
            f"{self.config.host}/uapi/hashkey",
            json=body,
            headers={
                "content-type": "application/json",
                "appkey": self.config.app_key,
                "appsecret": self.config.app_secret,
            },
            timeout=self.config.timeout,
        )
        if resp.status_code != 200:
            raise BrokerError(f"hashkey 발급 실패 ({resp.status_code}): {resp.text[:300]}")
        return resp.json().get("HASH", "")

    # --- 공통 호출 ----------------------------------------------------------

    def _headers(self, tr_id: str, extra: dict | None = None) -> dict:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.token()}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "tr_id": tr_id,
            "custtype": "P",  # 개인
        }
        headers.update(extra or {})
        return headers

    def request(
        self, method: str, path: str, tr_id: str,
        params: dict | None = None, body: dict | None = None,
        retries: int = 2,
    ) -> dict:
        """
        API 호출 한 번. rt_cd가 '0'이 아니면 예외를 던진다.

        조용히 실패하면 "주문이 나간 줄 알았는데 안 나간" 상황이 생긴다.
        여기서는 항상 시끄럽게 실패시킨다.
        """
        url = f"{self.config.host}{path}"
        extra = {}
        if body is not None:
            extra["hashkey"] = self.hashkey(body)

        last_error = ""
        for attempt in range(retries + 1):
            self._limiter.wait()
            try:
                resp = self.session.request(
                    method, url,
                    headers=self._headers(tr_id, extra),
                    params=params, json=body,
                    timeout=self.config.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"네트워크 오류: {exc}"
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 500 and attempt < retries:
                last_error = f"서버 오류 500: {resp.text[:200]}"
                time.sleep(2 ** attempt)
                continue

            if resp.status_code != 200:
                raise BrokerError(f"{path} 실패 ({resp.status_code}): {resp.text[:300]}")

            data = resp.json()
            if data.get("rt_cd") != "0":
                raise BrokerError(
                    f"{path} 거부 [{data.get('msg_cd', '?')}] {data.get('msg1', '')}"
                )
            return data

        raise BrokerError(f"{path} 재시도 실패: {last_error}")

    # --- 시세 --------------------------------------------------------------

    def current_price(self, symbol: str) -> float:
        """현재가(원)."""
        data = self.request(
            "GET", "/uapi/domestic-stock/v1/quotations/inquire-price", TR_PRICE,
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol},
        )
        price = data.get("output", {}).get("stck_prpr")
        if price is None:
            raise BrokerError(f"{symbol} 현재가를 받지 못했다: {data.get('output')}")
        return float(price)

    def daily_bars(self, symbol: str, start: str, end: str, adjusted: bool = True) -> list[dict]:
        """
        일봉을 받는다. 한 번에 최대 100건이라 기간을 쪼개서 여러 번 부른다.

        start/end는 'YYYY-MM-DD' 또는 'YYYYMMDD'.
        """
        start_dt = datetime.strptime(start.replace("-", ""), "%Y%m%d")
        end_dt = datetime.strptime(end.replace("-", ""), "%Y%m%d")
        if start_dt > end_dt:
            raise ValueError("start가 end보다 늦다")

        rows: dict[str, dict] = {}
        cursor = end_dt
        while cursor >= start_dt:
            # 영업일 100건 ≈ 달력 140일. 여유를 두고 자른다.
            chunk_start = max(start_dt, cursor - timedelta(days=140))
            data = self.request(
                "GET",
                "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                TR_DAILY,
                params={
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": symbol,
                    "FID_INPUT_DATE_1": chunk_start.strftime("%Y%m%d"),
                    "FID_INPUT_DATE_2": cursor.strftime("%Y%m%d"),
                    "FID_PERIOD_DIV_CODE": "D",
                    "FID_ORG_ADJ_PRC": "0" if adjusted else "1",
                },
            )
            chunk = [r for r in (data.get("output2") or []) if r and r.get("stck_bsop_date")]
            if not chunk:
                break

            for row in chunk:
                rows[row["stck_bsop_date"]] = row

            oldest = min(r["stck_bsop_date"] for r in chunk)
            next_cursor = datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)
            if next_cursor >= cursor:
                break  # 커서가 안 밀리면 무한루프다
            cursor = next_cursor

        return [rows[k] for k in sorted(rows)]


class KISBroker(Broker):
    """
    Broker 인터페이스의 한국투자증권 구현.

    주의: submit()은 실제 주문을 낸다. 모의투자(env="paper")가 기본이지만
    env="live"면 진짜 돈이 움직인다.
    """

    def __init__(self, config: KISConfig | None = None, client: KISClient | None = None):
        self.config = config or (client.config if client else KISConfig.from_env())
        self.client = client or KISClient(self.config)
        self.is_live = self.config.env == "live"

    def __repr__(self) -> str:
        kind = "실전투자" if self.is_live else "모의투자"
        return f"KISBroker({kind}, 계좌 {self.config.cano[:4]}****)"

    # --- 조회 --------------------------------------------------------------

    def get_price(self, symbol: str) -> float:
        return self.client.current_price(symbol)

    def get_account(self) -> Account:
        """잔고 조회. 보유종목이 많으면 연속조회로 이어 받는다."""
        holdings: dict[str, Holding] = {}
        cash = 0.0
        fk, nk = "", ""

        for _ in range(20):  # 연속조회 상한. 무한루프 방지.
            data = self.client.request(
                "GET", "/uapi/domestic-stock/v1/trading/inquire-balance",
                self.config.tr("balance"),
                params={
                    "CANO": self.config.cano,
                    "ACNT_PRDT_CD": self.config.product_code,
                    "AFHR_FLPR_YN": "N",       # 시간외단일가 미반영
                    "OFL_YN": "",
                    "INQR_DVSN": "02",         # 종목별
                    "UNPR_DVSN": "01",
                    "FUND_STTL_ICLD_YN": "N",
                    "FNCG_AMT_AUTO_RDPT_YN": "N",
                    "PRCS_DVSN": "00",         # 전일매매 포함
                    "CTX_AREA_FK100": fk,
                    "CTX_AREA_NK100": nk,
                },
            )

            for row in data.get("output1") or []:
                qty = int(float(row.get("hldg_qty", 0) or 0))
                if qty <= 0:
                    continue
                symbol = row.get("pdno", "")
                holdings[symbol] = Holding(
                    symbol=symbol,
                    name=row.get("prdt_name", ""),
                    quantity=qty,
                    avg_price=float(row.get("pchs_avg_pric", 0) or 0),
                    current_price=float(row.get("prpr", 0) or 0),
                )

            summary = data.get("output2") or []
            if summary:
                # dnca_tot_amt(예수금)이 아니라 주문가능현금을 쓴다.
                # 예수금에는 아직 결제되지 않은 금액이 섞여 있어 그대로 주문하면 거부된다.
                first = summary[0]
                cash = float(first.get("prvs_rcdl_excc_amt") or first.get("dnca_tot_amt") or 0)

            # tr_cont가 F/M이면 뒤에 더 있다
            fk = data.get("ctx_area_fk100", "").strip()
            nk = data.get("ctx_area_nk100", "").strip()
            if not nk:
                break

        return Account(cash=cash, holdings=holdings)

    def orderable_cash(self) -> float:
        """주문가능현금만 따로 조회한다."""
        return self.get_account().cash

    # --- 주문 --------------------------------------------------------------

    def submit(self, order: Order) -> OrderResult:
        """
        현금 주문을 낸다.

        지정가는 ORD_DVSN "00", 시장가는 "01"이고 시장가일 때 단가는 0을 보낸다.
        """
        body = {
            "CANO": self.config.cano,
            "ACNT_PRDT_CD": self.config.product_code,
            "PDNO": order.symbol,
            "ORD_DVSN": "01" if order.order_type == "market" else "00",
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": "0" if order.order_type == "market" else str(int(order.price)),
        }

        try:
            data = self.client.request(
                "POST", "/uapi/domestic-stock/v1/trading/order-cash",
                self.config.tr(order.side), body=body,
            )
        except BrokerError as exc:
            return OrderResult(ok=False, order=order, message=str(exc))

        output = data.get("output") or {}
        return OrderResult(
            ok=True,
            order=order,
            order_id=output.get("ODNO", ""),
            message=data.get("msg1", "접수됨"),
            raw=data,
        )

    def cancel(self, order_id: str, symbol: str, quantity: int) -> OrderResult:
        """미체결 주문 전량 취소."""
        dummy = Order(symbol=symbol, side="sell", quantity=quantity, order_type="market")
        body = {
            "CANO": self.config.cano,
            "ACNT_PRDT_CD": self.config.product_code,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": order_id,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 02 = 취소
            "ORD_QTY": str(int(quantity)),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y",
        }
        try:
            data = self.client.request(
                "POST", "/uapi/domestic-stock/v1/trading/order-rvsecncl",
                self.config.tr("cancel"), body=body,
            )
        except BrokerError as exc:
            return OrderResult(ok=False, order=dummy, message=str(exc))

        return OrderResult(
            ok=True, order=dummy, order_id=order_id,
            message=data.get("msg1", "취소 접수됨"), raw=data,
        )
