"""
결과 리포트

성과표 출력과, 목표 금액 도달 가능성을 몬테카를로로 검증하는 기능이 들어 있다.
후자가 이 패키지에서 제일 중요한 부분일 수 있다 — 백테스트 수익률 하나만 보고
"이 전략이면 되겠다"고 판단하는 걸 막아 준다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult
from .metrics import (
    Performance,
    required_cagr,
    required_cagr_with_contributions,
    years_to_target,
)


def _pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "  n/a"
    return f"{x * 100:>7.2f}%"


def _money(x: float) -> str:
    return f"{x:>15,.0f}"


def format_performance(result: BacktestResult) -> str:
    """백테스트 결과 하나를 사람이 읽는 표로."""
    p = result.performance
    lines = [
        "=" * 66,
        f" 전략: {result.strategy_desc}",
        f" 기간: {p.start.date()} ~ {p.end.date()}  ({p.years:.1f}년)",
        f" 종목: {len(result.metadata.get('symbols', []))}개",
        "=" * 66,
        "",
        " [ 수익 ]",
        f"   투자원금            {_money(p.start_equity)} 원",
        f"   최종평가액          {_money(p.end_equity)} 원",
        f"   누적수익률          {_pct(p.total_return)}",
        f"   연평균수익률(CAGR)  {_pct(p.cagr)}",
        "",
        " [ 위험 ]",
        f"   최대낙폭(MDD)       {_pct(p.max_drawdown)}",
        f"   최장 원금회복기간   {p.longest_drawdown_days:>7,}일",
        f"   연변동성            {_pct(p.volatility)}",
        f"   평균 주식노출       {_pct(p.exposure)}",
        "",
        " [ 위험대비수익 ]",
        f"   샤프지수            {p.sharpe:>8.2f}",
        f"   소르티노지수        {p.sortino:>8.2f}",
        f"   칼마지수            {p.calmar:>8.2f}",
        "",
        " [ 매매 ]",
        f"   총 체결             {len(result.trades):>8,}건",
        f"   청산 거래           {p.trades:>8,}건",
        f"   승률                {_pct(p.win_rate)}",
        f"   손익비(PF)          {p.profit_factor:>8.2f}",
        f"   총 비용             {_money(p.total_costs)} 원"
        f"  (수익의 {abs(p.total_costs / max(p.end_equity - p.start_equity, 1)) * 100:.1f}%)",
    ]

    if len(result.stop_events):
        counts = result.stop_events["reason"].value_counts()
        detail = ", ".join(f"{k} {v}회" for k, v in counts.items())
        lines += ["", f" [ 리스크 발동 ] {detail}"]

    lines.append("=" * 66)
    return "\n".join(lines)


def compare(results: list[BacktestResult]) -> str:
    """여러 전략을 나란히 놓고 본다. 반드시 벤치마크를 같이 넣어서 봐야 한다."""
    header = (
        f"{'전략':<28}{'CAGR':>9}{'MDD':>9}{'샤프':>7}"
        f"{'최종금액':>16}{'거래':>7}"
    )
    lines = ["=" * 76, header, "-" * 76]

    for r in sorted(results, key=lambda x: x.performance.cagr, reverse=True):
        p = r.performance
        name = r.strategy_desc[:27]
        lines.append(
            f"{name:<28}{p.cagr * 100:>8.2f}%{p.max_drawdown * 100:>8.1f}%"
            f"{p.sharpe:>7.2f}{p.end_equity:>16,.0f}{len(r.trades):>7,}"
        )

    lines.append("=" * 76)
    return "\n".join(lines)


# --- 목표 도달 가능성 --------------------------------------------------------


def monte_carlo(
    result: BacktestResult,
    years: float,
    target: float,
    monthly_contribution: float = 0.0,
    simulations: int = 10_000,
    block_size: int = 21,
    seed: int = 7,
) -> dict:
    """
    백테스트의 일간 수익률을 블록 부트스트랩으로 재추출해 미래를 시뮬레이션한다.

    블록으로 뽑는 이유는 수익률에 자기상관(변동성 군집)이 있기 때문이다.
    하루씩 무작위로 섞으면 연속 급락이 사라져서 위험을 과소평가하게 된다.

    ※ 이 시뮬레이션은 "과거와 같은 통계적 성질이 유지된다"는 가정 위에 있다.
      그 가정 자체가 자주 틀리므로, 결과는 상한이 아니라 참고치로 봐야 한다.
    """
    returns = result.equity.pct_change().dropna().to_numpy()
    if len(returns) < block_size * 2:
        raise ValueError("수익률 표본이 너무 적어 시뮬레이션할 수 없다")

    rng = np.random.default_rng(seed)
    horizon = int(round(years * 252))
    n_blocks = int(np.ceil(horizon / block_size))
    max_start = len(returns) - block_size

    # (시뮬레이션 x 블록) 시작점을 한 번에 뽑아 붙인다
    starts = rng.integers(0, max_start + 1, size=(simulations, n_blocks))
    offsets = np.arange(block_size)
    paths = returns[(starts[:, :, None] + offsets).reshape(simulations, -1)][:, :horizon]

    start_value = float(result.equity.iloc[0])
    daily_contribution = monthly_contribution * 12 / 252

    equity = np.full(simulations, start_value)
    ruined = np.zeros(simulations, dtype=bool)
    for t in range(horizon):
        equity = equity * (1.0 + paths[:, t]) + daily_contribution
        equity = np.maximum(equity, 0.0)
        ruined |= equity <= start_value * 0.1  # 원금의 10% 밑으로 내려간 적이 있는가

    total_invested = start_value + monthly_contribution * 12 * years
    pcts = np.percentile(equity, [5, 25, 50, 75, 95])

    return {
        "years": years,
        "target": target,
        "start_value": start_value,
        "monthly_contribution": monthly_contribution,
        "total_invested": total_invested,
        "simulations": simulations,
        "p5": pcts[0], "p25": pcts[1], "median": pcts[2], "p75": pcts[3], "p95": pcts[4],
        "mean": float(equity.mean()),
        "prob_target": float((equity >= target).mean()),
        "prob_loss": float((equity < total_invested).mean()),
        "prob_ruin": float(ruined.mean()),
    }


def goal_analysis(
    result: BacktestResult,
    target: float,
    years: float = 10.0,
    monthly_contribution: float = 0.0,
    simulations: int = 10_000,
) -> str:
    """
    "이 전략으로 목표 금액에 도달할 수 있나"에 대한 정직한 답.

    필요 수익률과 실제 백테스트 수익률을 나란히 놓고, 몬테카를로로 확률 분포를 낸다.
    """
    p = result.performance
    start = float(result.equity.iloc[0])
    contributed = monthly_contribution * 12 * years
    need = required_cagr_with_contributions(start, target, years, monthly_contribution)

    lines = [
        "=" * 66,
        " 목표 도달 가능성 분석",
        "=" * 66,
        f"   현재 원금            {_money(start)} 원",
        f"   목표 금액            {_money(target)} 원  ({target / start:,.0f}배)",
        f"   투자 기간            {years:>15.1f} 년",
    ]
    if monthly_contribution > 0:
        lines += [
            f"   월 추가납입          {_money(monthly_contribution)} 원",
            f"   총 투입원금          {_money(start + contributed)} 원",
        ]

    lines += [
        "",
        " [ 필요 수익률 vs 백테스트 수익률 ]",
        f"   목표에 필요한 CAGR   {_pct(need)}",
        f"   백테스트 CAGR        {_pct(p.cagr)}",
    ]

    # 적립액만으로 목표를 넘는 경우, 이건 투자 성과가 아니라 저축의 결과다.
    # 이 구분을 흐리면 "전략이 목표를 달성했다"는 착각이 생긴다.
    savings_only = start + contributed >= target
    if savings_only:
        lines.append(
            f"   ※ 수익률 0%로 저축만 해도 목표를 넘는다 "
            f"({_money(start + contributed).strip()}원). "
        )
        lines.append("     이 목표는 투자 실력이 아니라 납입액으로 달성되는 것이다.")
    elif monthly_contribution == 0:
        if p.cagr > 0:
            actual_years = years_to_target(start, target, p.cagr)
            if np.isfinite(actual_years):
                lines.append(f"   → 이 속도로는        {actual_years:>15.1f} 년 소요")
        else:
            lines.append("   → 백테스트 수익률이 0 이하다. 도달 불가.")

    gap = need - p.cagr
    if np.isfinite(gap) and gap > 0:
        lines.append(f"   → 연 {gap * 100:.1f}%p 부족")

    # 참고용 기준선
    lines += [
        "",
        " [ 참고 : 필요 CAGR의 현실성 ]",
        "   연  7% ─ 미국 S&P500 장기 평균 수준",
        "   연 20% ─ 워런 버핏의 60년 평균 (역사상 최상위권)",
        "   연 30% ─ 지속 사례가 거의 없음. 대부분 몇 년 뒤 반납",
        "   연 50%+ ─ 지속 가능한 전략이 아니라 운 또는 레버리지",
    ]

    if need > 0.30:
        lines.append("")
        lines.append(f"   ※ 필요 CAGR {need * 100:.0f}%는 위 기준으로 현실적인 목표가 아니다.")
        lines.append("     기간을 늘리거나, 목표를 낮추거나, 추가납입을 늘려야 한다.")

    # 몬테카를로
    try:
        mc = monte_carlo(result, years, target, monthly_contribution, simulations)
    except ValueError as exc:
        lines += ["", f" [ 시뮬레이션 생략 ] {exc}", "=" * 66]
        return "\n".join(lines)

    lines += [
        "",
        f" [ 몬테카를로 {mc['simulations']:,}회 — {years:.0f}년 뒤 자산 분포 ]",
        f"   총 투입원금          {_money(mc['total_invested'])} 원",
        f"   상위 5%              {_money(mc['p95'])} 원",
        f"   상위 25%             {_money(mc['p75'])} 원",
        f"   중앙값               {_money(mc['median'])} 원",
        f"   하위 25%             {_money(mc['p25'])} 원",
        f"   하위 5%              {_money(mc['p5'])} 원",
        "",
        f"   목표 달성 확률       {_pct(mc['prob_target'])}",
        f"   원금 손실 확률       {_pct(mc['prob_loss'])}",
        f"   원금 90% 이상 소실   {_pct(mc['prob_ruin'])}",
        "",
    ]

    prob = mc["prob_target"]
    if savings_only:
        verdict = "목표는 달성되지만 그건 납입액 덕분이다. 전략의 성과는 위 CAGR로 판단해라."
    elif prob < 0.05:
        verdict = "사실상 불가능하다. 목표나 기간을 다시 잡아야 한다."
    elif prob < 0.25:
        verdict = "운이 아주 좋아야 한다. 계획으로 삼을 수치가 아니다."
    elif prob < 0.60:
        verdict = "가능은 하지만 절반의 확률로 못 미친다. 여유를 두고 계획해라."
    else:
        verdict = "현실적인 목표다. 규칙을 지키는 게 관건이다."
    lines += [f"   판정: {verdict}", "=" * 66]

    return "\n".join(lines)


def to_csv(result: BacktestResult, prefix: str) -> list[str]:
    """자산곡선과 거래내역을 CSV로 떨어뜨린다."""
    written = []

    equity_path = f"{prefix}_equity.csv"
    result.equity.to_csv(equity_path, header=True)
    written.append(equity_path)

    if len(result.trades):
        trades_path = f"{prefix}_trades.csv"
        result.trades.to_csv(trades_path, index=False)
        written.append(trades_path)

    return written
