"""
가격 데이터 컨테이너와 공급자 인터페이스

여러 종목을 하나의 달력에 정렬해서 (날짜 x 종목) 행렬로 들고 있는다.
전략은 항상 "오늘까지"로 잘린 뷰만 볼 수 있어서, 미래 데이터를 훔쳐보는
lookahead bias가 구조적으로 불가능하다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


OHLCV = ["open", "high", "low", "close", "volume"]


class PriceData:
    """여러 종목의 OHLCV를 공통 달력에 정렬해 담는다."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        if not frames:
            raise ValueError("종목 데이터가 비어 있다")

        cleaned: dict[str, pd.DataFrame] = {}
        for symbol, df in frames.items():
            missing = [c for c in OHLCV if c not in df.columns]
            if missing:
                raise ValueError(f"{symbol}: 필수 컬럼 누락 {missing}")
            frame = df[OHLCV].copy()
            frame.index = pd.to_datetime(frame.index)
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
            cleaned[symbol] = frame

        self.symbols: list[str] = sorted(cleaned)

        calendar = pd.DatetimeIndex([])
        for frame in cleaned.values():
            calendar = calendar.union(frame.index)
        self.calendar: pd.DatetimeIndex = calendar

        # 거래 가능 여부 마스크: 원본에 그 날 데이터가 실제로 있었는지.
        # ffill로 채운 값에 주문을 내면 상장 전/거래정지 종목을 사는 셈이 된다.
        self.tradable = pd.DataFrame(
            {s: cleaned[s]["close"].reindex(calendar).notna() for s in self.symbols},
            index=calendar,
        )

        for field in OHLCV:
            matrix = pd.DataFrame(
                {s: cleaned[s][field].reindex(calendar) for s in self.symbols},
                index=calendar,
            )
            # 휴장·결측일은 직전 값으로 이어 붙여 평가만 가능하게 한다.
            # (주문 가능 여부는 위 tradable 마스크가 따로 막는다)
            setattr(self, field, matrix.ffill())

    def __len__(self) -> int:
        return len(self.calendar)

    def __repr__(self) -> str:
        return (
            f"PriceData({len(self.symbols)}종목, {len(self.calendar)}일, "
            f"{self.calendar[0].date()}~{self.calendar[-1].date()})"
        )

    def slice(self, start=None, end=None) -> "PriceData":
        """기간을 잘라낸 새 PriceData."""
        frames = {}
        for s in self.symbols:
            df = pd.DataFrame({f: getattr(self, f)[s] for f in OHLCV})
            df = df[self.tradable[s]]
            frames[s] = df.loc[start:end]
        return PriceData(frames)

    def view(self, upto: int) -> "MarketView":
        """0..upto 까지만 보이는 읽기 전용 뷰."""
        return MarketView(self, upto)


class MarketView:
    """
    전략에 넘겨주는 "오늘까지"의 시장 상태.

    upto는 오늘의 인덱스 위치이고, 오늘 종가까지 포함해서 볼 수 있다.
    (신호는 오늘 종가로 만들고 주문은 다음날 시가에 나가므로 미래참조가 아니다)
    """

    __slots__ = ("_data", "_i")

    def __init__(self, data: PriceData, upto: int):
        self._data = data
        self._i = upto

    @property
    def date(self) -> pd.Timestamp:
        return self._data.calendar[self._i]

    @property
    def index(self) -> int:
        return self._i

    @property
    def symbols(self) -> list[str]:
        return self._data.symbols

    @property
    def bars(self) -> int:
        """지금까지 쌓인 봉 개수."""
        return self._i + 1

    def _field(self, name: str) -> pd.DataFrame:
        return getattr(self._data, name).iloc[: self._i + 1]

    @property
    def open(self) -> pd.DataFrame:
        return self._field("open")

    @property
    def high(self) -> pd.DataFrame:
        return self._field("high")

    @property
    def low(self) -> pd.DataFrame:
        return self._field("low")

    @property
    def close(self) -> pd.DataFrame:
        return self._field("close")

    @property
    def volume(self) -> pd.DataFrame:
        return self._field("volume")

    def last_close(self) -> pd.Series:
        return self._data.close.iloc[self._i]

    def is_tradable(self, symbol: str) -> bool:
        return bool(self._data.tradable.iloc[self._i][symbol])

    def tradable_symbols(self) -> list[str]:
        row = self._data.tradable.iloc[self._i]
        return [s for s in self._data.symbols if row[s]]


class DataProvider(ABC):
    """가격 데이터 공급자. 국내/해외 어댑터가 이걸 구현한다."""

    @abstractmethod
    def fetch_one(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """종목 하나의 OHLCV를 DatetimeIndex + open/high/low/close/volume로 반환."""

    def fetch(self, symbols: list[str], start: str, end: str) -> PriceData:
        """여러 종목을 받아 PriceData로 묶는다. 실패한 종목은 건너뛴다."""
        frames, failed = {}, []
        for symbol in symbols:
            try:
                df = self.fetch_one(symbol, start, end)
            except Exception as exc:  # 종목 하나 때문에 전체가 죽으면 안 된다
                failed.append((symbol, str(exc)))
                continue
            if df is None or df.empty:
                failed.append((symbol, "데이터 없음"))
                continue
            frames[symbol] = df

        if not frames:
            detail = "; ".join(f"{s}: {e}" for s, e in failed) or "요청 종목 없음"
            raise RuntimeError(f"가격 데이터를 하나도 받지 못했다 ({detail})")

        if failed:
            names = ", ".join(s for s, _ in failed)
            print(f"[경고] 다음 종목은 제외됐다: {names}")

        return PriceData(frames)
