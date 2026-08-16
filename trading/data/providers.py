"""
시장별 데이터 어댑터

국내는 pykrx, 해외는 yfinance를 쓴다. 두 라이브러리 모두 컬럼명·타입이 제각각이라
여기서 open/high/low/close/volume 한 가지 형태로 정규화한다.

네트워크가 막힌 환경(CI 등)에서도 엔진을 검증할 수 있도록 CSV 공급자와
합성 데이터 공급자를 함께 둔다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .base import DataProvider, OHLCV


CACHE_DIR = Path.home() / ".cache" / "trading-data"


class CachedProvider(DataProvider):
    """받아온 데이터를 로컬에 캐싱한다. 같은 백테스트를 반복할 때 훨씬 빠르다."""

    name = "base"

    def __init__(self, use_cache: bool = True, cache_dir: Path | None = None):
        self.use_cache = use_cache
        self.cache_dir = cache_dir or CACHE_DIR

    def _cache_path(self, symbol: str, start: str, end: str) -> Path:
        # 종목코드에 /나 ^ 같은 문자가 섞여 있어 파일명으로 못 쓰므로 해시를 붙인다
        key = hashlib.sha1(f"{self.name}|{symbol}|{start}|{end}".encode()).hexdigest()[:16]
        safe = "".join(c if c.isalnum() else "_" for c in symbol)[:24]
        return self.cache_dir / f"{self.name}_{safe}_{key}.csv"

    def fetch_one(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self._cache_path(symbol, start, end)
        if self.use_cache and path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)

        df = self._download(symbol, start, end)
        df = normalize(df, symbol)

        if self.use_cache and not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
        return df

    def _download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError


def normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """어떤 소스에서 왔든 표준 OHLCV 프레임으로 맞춘다."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV)

    df = df.copy()

    # yfinance가 단일 종목에도 MultiIndex 컬럼을 주는 경우가 있다
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename = {
        "시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        "Adj Close": "adj_close",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: 정규화 후에도 컬럼이 없다 {missing}")

    df = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    # 종가가 0이거나 없는 행은 거래정지·데이터오류다. 그대로 두면 수익률이 폭발한다.
    df = df[df["close"].notna() & (df["close"] > 0)]
    return df.sort_index()


class KRXProvider(CachedProvider):
    """
    국내 주식 (KOSPI/KOSDAQ). pykrx 사용.

    최신 pykrx는 KRX 오픈API 계정을 요구한다. 환경변수 KRX_ID / KRX_PW를
    설정해 두지 않으면 빈 결과가 조용히 돌아오므로 여기서 잡아서 알려 준다.
    """

    name = "krx"

    def _download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import os

        try:
            from pykrx import stock
        except ImportError as exc:
            raise RuntimeError("pykrx가 필요하다: pip install pykrx") from exc

        fromdate = pd.Timestamp(start).strftime("%Y%m%d")
        todate = pd.Timestamp(end).strftime("%Y%m%d")
        # 액면분할·유상증자 보정을 위해 수정주가로 받는다
        df = stock.get_market_ohlcv(fromdate, todate, symbol, adjusted=True)

        if (df is None or df.empty) and not (os.getenv("KRX_ID") and os.getenv("KRX_PW")):
            raise RuntimeError(
                "KRX에서 데이터를 받지 못했다. 환경변수 KRX_ID / KRX_PW가 필요할 수 있다 "
                "(data.krx.co.kr 오픈API 계정). 계정 없이 쓰려면 --market us 를 쓰거나 "
                "CSVProvider로 보유 데이터를 붙여라."
            )
        return df


class USProvider(CachedProvider):
    """해외 주식/ETF (NYSE/NASDAQ). yfinance 사용."""

    name = "us"

    def _download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance가 필요하다: pip install yfinance") from exc

        # auto_adjust=True면 배당·분할이 반영된 가격이 온다. 장기 백테스트에 필수.
        return yf.download(
            symbol, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
        )


class KISProvider(CachedProvider):
    """
    국내 주식 일봉을 한국투자증권 KIS API로 받는다.

    pykrx와 달리 증권 계좌만 있으면 되고, 실전 주문에 쓰는 시세와 같은 출처라
    백테스트와 실전의 데이터가 어긋나지 않는다.
    """

    name = "kis"

    def __init__(self, client=None, use_cache: bool = True, cache_dir: Path | None = None):
        super().__init__(use_cache=use_cache, cache_dir=cache_dir)
        self._client = client

    @property
    def client(self):
        # 실제로 데이터를 받을 때까지 인증을 미룬다 (캐시만 쓰는 경우 키가 없어도 된다)
        if self._client is None:
            from ..broker.kis import KISClient, KISConfig

            self._client = KISClient(KISConfig.from_env())
        return self._client

    def _download(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        rows = self.client.daily_bars(symbol, start, end, adjusted=True)
        if not rows:
            return pd.DataFrame(columns=OHLCV)

        frame = pd.DataFrame(
            [
                {
                    "date": r["stck_bsop_date"],
                    "open": r.get("stck_oprc"),
                    "high": r.get("stck_hgpr"),
                    "low": r.get("stck_lwpr"),
                    "close": r.get("stck_clpr"),
                    "volume": r.get("acml_vol"),
                }
                for r in rows
            ]
        )
        frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
        return frame.set_index("date")


class CSVProvider(DataProvider):
    """`{디렉터리}/{종목}.csv`에서 읽는다. 자체 보유 데이터를 붙일 때 쓴다."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def fetch_one(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        path = self.directory / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} 없음")
        df = normalize(pd.read_csv(path, index_col=0, parse_dates=True), symbol)
        return df.loc[start:end]


class SyntheticProvider(DataProvider):
    """
    기하 브라운 운동으로 만든 가상 시세.

    네트워크 없이 엔진을 검증하기 위한 것이다. 여기서 나온 수익률은
    아무 의미가 없다 — 전략의 우열을 이걸로 판단하면 안 된다.
    """

    def __init__(self, seed: int = 42, annual_drift: float = 0.07, annual_vol: float = 0.25):
        self.seed = seed
        self.drift = annual_drift
        self.vol = annual_vol

    def fetch_one(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, end=end)
        if len(dates) == 0:
            return pd.DataFrame(columns=OHLCV)

        # 종목마다 다른 경로를 갖도록 종목명을 시드에 섞는다
        seed = (self.seed + int(hashlib.sha1(symbol.encode()).hexdigest()[:8], 16)) % (2**32)
        rng = np.random.default_rng(seed)

        n = len(dates)
        dt = 1 / 252
        shocks = rng.normal(
            (self.drift - 0.5 * self.vol**2) * dt,
            self.vol * np.sqrt(dt),
            n,
        )
        close = 100.0 * np.exp(np.cumsum(shocks))

        intraday = np.abs(rng.normal(0, 0.008, n))
        open_ = close * (1 + rng.normal(0, 0.004, n))
        high = np.maximum(open_, close) * (1 + intraday)
        low = np.minimum(open_, close) * (1 - intraday)

        return pd.DataFrame(
            {
                "open": open_, "high": high, "low": low, "close": close,
                "volume": rng.integers(1e5, 1e7, n).astype(float),
            },
            index=dates,
        )


def get_provider(market: str, **kwargs) -> DataProvider:
    """시장 코드로 공급자를 고른다."""
    key = market.strip().lower()
    if key in ("kr", "krx", "kospi", "kosdaq"):
        return KRXProvider(**kwargs)
    if key == "kis":
        return KISProvider(**kwargs)
    if key in ("us", "nyse", "nasdaq"):
        return USProvider(**kwargs)
    if key in ("synthetic", "sim", "test"):
        return SyntheticProvider()
    raise ValueError(f"알 수 없는 시장: {market} (kr / us / synthetic 중 하나)")
