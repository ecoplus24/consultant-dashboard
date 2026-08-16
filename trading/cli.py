"""
명령줄 인터페이스

    python -m trading backtest --market kr --symbols 005930,000660 --strategy momentum
    python -m trading compare  --market us --symbols SPY,QQQ,GLD
    python -m trading goal     --market us --symbols SPY --target 100000000 --years 10

한국투자증권 연동 (환경변수 KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT 필요):

    python -m trading account                                    # 잔고 조회
    python -m trading signal  --symbols 005930,000660            # 오늘 주문 계산 (드라이런)
    python -m trading trade   --symbols 005930,000660 --execute  # 모의투자 주문 실행
"""

from __future__ import annotations

import argparse
import sys

from .costs import costs_for
from .engine import run_backtest
from .report import compare, format_performance, goal_analysis, to_csv
from .risk import preset
from .strategies import REGISTRY, build


DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-12-31"


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--market", default="kr", help="kr / us / synthetic (기본: kr)")
    p.add_argument("--symbols", required=True, help="쉼표로 구분한 종목코드")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--capital", type=float, default=1_000_000.0, help="투자원금 (기본: 100만원)")
    p.add_argument("--risk", default="balanced",
                   help="conservative / balanced / aggressive / none")
    p.add_argument("--rebalance", type=int, default=5, help="리밸런싱 최소 간격(거래일)")
    p.add_argument("--fractional", action="store_true", help="소수점 주식 허용")


def _load(args):
    from . import load_data

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("종목을 하나 이상 지정해라")
    print(f"[1/3] 데이터 수집: {len(symbols)}종목 ({args.start} ~ {args.end})", file=sys.stderr)
    return load_data(symbols, market=args.market, start=args.start, end=args.end)


def _run(args, data, strategy_name: str, **strategy_params):
    strategy = build(strategy_name, **strategy_params)
    return run_backtest(
        data,
        strategy=strategy,
        risk=preset(args.risk),
        costs=costs_for(args.market),
        initial_capital=args.capital,
        rebalance_days=args.rebalance,
        fractional_shares=args.fractional,
    )


def cmd_backtest(args) -> int:
    data = _load(args)
    print(f"[2/3] 백테스트: {args.strategy}", file=sys.stderr)
    result = _run(args, data, args.strategy)

    print(f"[3/3] 완료\n", file=sys.stderr)
    print(format_performance(result))

    if args.csv:
        for path in to_csv(result, args.csv):
            print(f"\n저장: {path}")
    return 0


def cmd_compare(args) -> int:
    costs_for(args.market)  # 설정 오류는 전략 루프 안에서 삼키지 말고 먼저 터뜨린다
    data = _load(args)
    results = []
    for name in REGISTRY:
        print(f"[2/3] 백테스트: {name}", file=sys.stderr)
        try:
            results.append(_run(args, data, name))
        except ValueError as exc:
            print(f"  건너뜀 ({name}): {exc}", file=sys.stderr)

    # 리스크 오버레이를 걷어낸 순수 매수 후 보유. "그냥 사서 들고 있었으면"의 기준선이다.
    # 이게 없으면 리스크 관리 비용을 전략 탓으로 오해하게 된다.
    raw = run_backtest(
        data, build("buyhold"), risk=preset("none"), costs=costs_for(args.market),
        initial_capital=args.capital, rebalance_days=args.rebalance,
        fractional_shares=args.fractional,
    )
    raw.strategy_desc = "└ 그냥 사서 보유 (리스크 미적용)"
    results.append(raw)

    if not results:
        print("돌릴 수 있는 전략이 없다. 기간을 늘려라.", file=sys.stderr)
        return 1

    print(f"[3/3] 완료\n", file=sys.stderr)
    print(compare(results))
    print(
        "\n※ 벤치마크(동일비중 매수 후 보유)를 못 이기는 전략은 쓸 이유가 없다."
        "\n  거래 횟수가 많을수록 비용과 실행 리스크도 같이 커진다는 걸 감안해라."
    )
    return 0


def cmd_goal(args) -> int:
    data = _load(args)
    print(f"[2/3] 백테스트: {args.strategy}", file=sys.stderr)
    result = _run(args, data, args.strategy)

    print(f"[3/3] 시뮬레이션\n", file=sys.stderr)
    print(format_performance(result))
    print()
    print(
        goal_analysis(
            result,
            target=args.target,
            years=args.years,
            monthly_contribution=args.monthly,
            simulations=args.simulations,
        )
    )
    return 0


# --- 한국투자증권 연동 -------------------------------------------------------


def _make_broker(args):
    from .broker.kis import KISBroker, KISConfig

    config = KISConfig.from_env()
    if getattr(args, "live", False):
        config.env = "live"
    return KISBroker(config)


def _make_trader(args, broker):
    from .data.providers import KISProvider
    from .live import LiveTrader, SafetyLimits

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("종목을 하나 이상 지정해라")

    limits = SafetyLimits(
        max_order_value=args.max_order,
        max_total_order_value=args.max_total,
        trading_hours_only=not args.ignore_hours,
    )
    return LiveTrader(
        broker=broker,
        strategy=build(args.strategy),
        risk=preset(args.risk),
        provider=KISProvider(client=broker.client),
        symbols=symbols,
        limits=limits,
    )


def cmd_account(args) -> int:
    broker = _make_broker(args)
    account = broker.get_account()

    kind = "실전투자" if broker.is_live else "모의투자"
    print("=" * 66)
    print(f" 계좌 현황 — {kind}")
    print("=" * 66)
    print(f"   주문가능현금   {account.cash:>15,.0f} 원")
    print(f"   주식평가액     {account.holdings_value:>15,.0f} 원")
    print(f"   총평가액       {account.equity:>15,.0f} 원")

    if account.holdings:
        print("\n [ 보유종목 ]")
        for h in account.holdings.values():
            print(
                f"   {h.symbol} {h.name[:12]:<12} {h.quantity:>6,}주  "
                f"평단 {h.avg_price:>10,.0f}  현재 {h.current_price:>10,.0f}  "
                f"손익 {h.pnl_pct * 100:>+7.2f}% ({h.pnl:>+12,.0f}원)"
            )
    else:
        print("\n   보유종목 없음")
    print("=" * 66)
    return 0


def cmd_signal(args) -> int:
    """주문을 계산해서 보여주기만 한다. 절대 주문하지 않는다."""
    from .live import format_plan

    broker = _make_broker(args)
    trader = _make_trader(args, broker)

    print("주문 계획 계산 중...", file=sys.stderr)
    plan = trader.plan()
    print(format_plan(plan, broker))
    print("\n※ 드라이런이다. 실제 주문은 나가지 않았다.")
    print("  주문하려면: python -m trading trade ... --execute")
    return 0


def cmd_trade(args) -> int:
    from .live import format_plan, is_market_open

    broker = _make_broker(args)
    trader = _make_trader(args, broker)

    plan = trader.plan()
    print(format_plan(plan, broker))

    if not args.execute:
        print("\n※ 드라이런이다. 실제 주문은 나가지 않았다. (--execute 를 붙이면 실행)")
        return 0

    if not plan.orders:
        print("\n낼 주문이 없다.")
        return 0

    # 실전 계좌는 사람이 직접 타이핑해서 확인해야 넘어간다
    if broker.is_live:
        print("\n" + "!" * 66)
        print(" 실전 계좌다. 진짜 돈이 움직인다.")
        print(f" 매수 {plan.total_buy:,.0f}원 / 매도 {plan.total_sell:,.0f}원")
        print("!" * 66)
        answer = input(" 계속하려면 '실행'을 입력해라: ").strip()
        if answer != "실행":
            print("취소했다.")
            return 1

    if not is_market_open() and not args.ignore_hours:
        print("\n정규장이 아니다. 주문하지 않는다. (--ignore-hours 로 무시 가능)")
        return 1

    print("\n주문 전송 중...")
    results = trader.execute(plan, confirm=True)

    print()
    for result in results:
        print(f"   {result}")

    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\n{len(failed)}건 실패. 나머지 주문은 중단했다.")
        return 1

    print(f"\n{len(results)}건 접수 완료.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trading",
        description="규칙 기반 주식매매 백테스트 엔진",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="전략 하나를 검증한다")
    _common_args(p_bt)
    p_bt.add_argument("--strategy", default="momentum", choices=list(REGISTRY))
    p_bt.add_argument("--csv", help="결과 CSV 저장 경로 접두사")
    p_bt.set_defaults(func=cmd_backtest)

    p_cmp = sub.add_parser("compare", help="모든 전략을 벤치마크와 비교한다")
    _common_args(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    p_goal = sub.add_parser("goal", help="목표 금액 도달 가능성을 검증한다")
    _common_args(p_goal)
    p_goal.add_argument("--strategy", default="momentum", choices=list(REGISTRY))
    p_goal.add_argument("--target", type=float, default=100_000_000.0, help="목표 금액 (기본: 1억)")
    p_goal.add_argument("--years", type=float, default=10.0)
    p_goal.add_argument("--monthly", type=float, default=0.0, help="월 추가납입액")
    p_goal.add_argument("--simulations", type=int, default=10_000)
    p_goal.set_defaults(func=cmd_goal)

    # --- 한국투자증권 연동 ---
    def _kis_args(p, needs_symbols: bool):
        p.add_argument("--live", action="store_true",
                       help="실전 계좌 사용 (기본은 모의투자)")
        if needs_symbols:
            p.add_argument("--symbols", required=True, help="쉼표로 구분한 종목코드")
            p.add_argument("--strategy", default="momentum", choices=list(REGISTRY))
            p.add_argument("--risk", default="balanced",
                           help="conservative / balanced / aggressive / none")
            p.add_argument("--max-order", type=float, default=1_000_000,
                           dest="max_order", help="주문 1건 최대 금액")
            p.add_argument("--max-total", type=float, default=5_000_000,
                           dest="max_total", help="실행 1회 총 주문 금액 한도")
            p.add_argument("--ignore-hours", action="store_true", dest="ignore_hours",
                           help="정규장 시간 검사를 끈다")

    p_acct = sub.add_parser("account", help="계좌 잔고를 조회한다")
    _kis_args(p_acct, needs_symbols=False)
    p_acct.set_defaults(func=cmd_account)

    p_sig = sub.add_parser("signal", help="오늘 낼 주문을 계산한다 (주문하지 않음)")
    _kis_args(p_sig, needs_symbols=True)
    p_sig.set_defaults(func=cmd_signal)

    p_trade = sub.add_parser("trade", help="주문을 계산하고 --execute 시 실행한다")
    _kis_args(p_trade, needs_symbols=True)
    p_trade.add_argument("--execute", action="store_true",
                         help="실제로 주문을 낸다 (없으면 드라이런)")
    p_trade.set_defaults(func=cmd_trade)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PermissionError as exc:
        print(f"\n차단됨: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"\n오류: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n중단됨", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
