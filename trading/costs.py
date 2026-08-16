"""
거래비용 모델

백테스트에서 가장 흔한 자기기만이 "수수료·세금·슬리피지 무시"다.
빈번한 매매 전략은 비용을 넣는 순간 수익이 사라지는 경우가 대부분이므로,
이 모듈은 항상 엔진에 강제로 물려 있다.
"""

from dataclasses import dataclass


BPS = 1e-4  # 1bp = 0.01%


@dataclass(frozen=True)
class CostModel:
    """
    매매 1회당 발생하는 비용.

    commission_bps : 증권사 수수료 (매수/매도 양쪽)
    sell_tax_bps   : 증권거래세 (매도할 때만) — 국내 주식은 무시할 수 없는 크기
    slippage_bps   : 체결 미끄러짐. 주문가 대비 불리하게 체결되는 폭
    """

    commission_bps: float = 1.5
    sell_tax_bps: float = 0.0
    slippage_bps: float = 5.0

    def fill_price(self, price: float, side: str) -> float:
        """슬리피지를 반영한 실제 체결가. 매수는 비싸게, 매도는 싸게 체결된다."""
        slip = price * self.slippage_bps * BPS
        return price + slip if side == "buy" else price - slip

    def transaction_cost(self, notional: float, side: str) -> float:
        """체결금액(notional)에 대해 부과되는 수수료 + 세금."""
        notional = abs(notional)
        cost = notional * self.commission_bps * BPS
        if side == "sell":
            cost += notional * self.sell_tax_bps * BPS
        return cost


# 시장별 기본값 --------------------------------------------------------------
#
# 국내: 증권거래세는 2025년 기준 코스피 0.03% + 농어촌특별세 0.15%,
#       코스닥 0.18%로 매도 시 실질 0.18%. 수수료는 비대면 계좌 기준 대략 0.015%.
#       ※ 세율은 정책에 따라 바뀌므로 실제 운용 전 반드시 최신값 확인할 것.
KR_COSTS = CostModel(commission_bps=1.5, sell_tax_bps=18.0, slippage_bps=10.0)

# 해외(미국): 대형 증권사 기준 수수료 0에 가깝고 거래세 없음.
#             단 환전 스프레드와 SEC 수수료가 있어 0으로 두지는 않는다.
US_COSTS = CostModel(commission_bps=1.0, sell_tax_bps=0.2, slippage_bps=5.0)

# 비용을 완전히 끈 모델. 비용의 영향을 측정할 때 비교군으로만 쓴다.
ZERO_COSTS = CostModel(commission_bps=0.0, sell_tax_bps=0.0, slippage_bps=0.0)


def costs_for(market: str) -> CostModel:
    """시장 코드로 기본 비용모델을 고른다."""
    key = market.strip().upper()
    if key in ("KR", "KRX", "KOSPI", "KOSDAQ"):
        return KR_COSTS
    if key in ("US", "NYSE", "NASDAQ"):
        return US_COSTS
    if key in ("SYNTHETIC", "SIM", "TEST"):
        # 가상 시세용. 비용을 0으로 두면 엔진이 좋아 보이는 착시가 생기므로
        # 실제와 비슷한 수준을 그대로 물린다.
        return US_COSTS
    if key == "ZERO":
        return ZERO_COSTS
    raise ValueError(f"알 수 없는 시장: {market}")
