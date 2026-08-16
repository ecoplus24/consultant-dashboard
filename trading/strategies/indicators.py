"""
기술적 지표

모두 (날짜 x 종목) DataFrame을 받아 같은 모양으로 돌려준다.
전부 과거 데이터만 쓰는 인과적(causal) 계산이라 미래참조가 섞이지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """단순이동평균."""
    return prices.rolling(window, min_periods=window).mean()


def ema(prices: pd.DataFrame, span: int) -> pd.DataFrame:
    """지수이동평균."""
    return prices.ewm(span=span, adjust=False, min_periods=span).mean()


def roc(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """window일 전 대비 수익률(모멘텀)."""
    return prices.pct_change(window)


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI (Wilder 방식).

    상승분과 하락분의 지수평활 평균 비율. 0~100, 30 이하 과매도 / 70 이상 과매수로 본다.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    # 하락이 전혀 없었던 구간은 rs가 무한대 -> RSI 100
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    """
    ATR (평균 진폭). 손절폭과 포지션 크기를 종목 변동성에 맞춰 정할 때 쓴다.

    갭을 반영하려고 전일 종가와의 차이까지 포함한 True Range를 쓴다.
    """
    prev_close = close.shift(1)
    tr = (
        (high - low)
        .combine((high - prev_close).abs(), np.maximum)
        .combine((low - prev_close).abs(), np.maximum)
    )
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def realized_vol(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """연율화 실현변동성."""
    return prices.pct_change().rolling(window, min_periods=window).std() * (252**0.5)
