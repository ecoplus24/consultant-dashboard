"""
컨설턴트 공고 스크레이퍼 v1
대상: 소상공인시장진흥공단 (semas.or.kr)

이 스크립트는 GitHub Actions에서 매일 자동 실행됩니다.
결과는 data.json 파일로 저장되어 대시보드가 자동으로 불러갑니다.
"""

import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ============ 설정 ============
KST = timezone(timedelta(hours=9))
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
TIMEOUT = 20

# 컨설턴트 모집과 관련된 키워드 (제목에 이 중 하나라도 있어야 수집)
KEYWORDS = [
    "컨설턴트", "컨설팅", "평가위원", "심사위원", "심사원",
    "멘토", "전문가", "지도사", "코칭",
    "희망리턴", "재도약", "재창업", "강한소상공인",
]

# 대표님 자격
MY_QUALS_DEFAULT = ["경영지도사", "창업보육매니저", "ISO", "MBA"]


# ============ 유틸 ============
def parse_korean_date(s: str) -> str | None:
    """다양한 한국식 날짜 표기를 YYYY-MM-DD로 정규화"""
    if not s:
        return None
    s = s.strip()
    # 2026.05.20 / 2026-05-20 / 2026/05/20
    m = re.search(r"(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    # 2026년 5월 20일
    m = re.search(r"(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return None


def extract_deadline(text: str) -> str | None:
    """본문 텍스트에서 마감일 추출"""
    # "마감", "까지", "~" 주변의 날짜를 우선 찾음
    patterns = [
        r"(?:마감|까지|접수\s*기간|신청\s*기간|모집\s*기간)[^0-9]{0,30}(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})",
        r"~\s*(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})",
        r"(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})\s*까지",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            d = parse_korean_date(m.group(1))
            if d:
                return d
    # 본문 내 마지막 날짜를 마감일로 추정
    dates = re.findall(r"20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}", text)
    if dates:
        return parse_korean_date(dates[-1])
    return None


def extract_qualifications(text: str) -> list[str]:
    """본문에서 자격요건 키워드 추출 (대표님 자격 매칭용)"""
    found = []
    qual_keywords = [
        "경영지도사", "기술지도사", "창업보육매니저", "MBA",
        "ISO 9001", "ISO 14001", "ISO 45001", "ISO 인증심사원",
        "공인노무사", "변호사", "회계사", "세무사",
        "재무·회계 전문가", "재무회계", "마케팅 전문가",
        "제조업 경력", "유통업 경력", "외식업 경력",
        "안전보건 경력", "위험성평가",
    ]
    for kw in qual_keywords:
        if kw in text and kw not in found:
            found.append(kw)
    return found


def matches_keyword(title: str) -> bool:
    """제목에 컨설턴트 관련 키워드가 있는지"""
    return any(kw in title for kw in KEYWORDS)


def make_id(agency: str, board_id: str) -> str:
    """공고 고유 ID 생성"""
    short = re.sub(r"[^a-zA-Z0-9가-힣]", "", agency)[:8]
    return f"{short}-{board_id}"


# ============ 소상공인시장진흥공단 (semas.or.kr) ============
def scrape_semas() -> list[dict]:
    """
    소진공 공지사항/사업공고에서 컨설턴트 모집 공고 수집

    참고: 소진공 사이트 구조는 변경될 수 있습니다.
    아래 URL과 셀렉터는 첫 실행 시 검증이 필요합니다.
    """
    print("[semas] 시작: 소상공인시장진흥공단")
    items = []

    # 후보 URL들 (사이트 구조 변경 대비 여러 경로 시도)
    candidate_urls = [
        "https://www.semas.or.kr/web/board/webBoardList.kmdc?bCd=1&pageId=PG90000001",
        "https://www.semas.or.kr/web/SUP/SUP10/SUP10000.kmdc",  # 사업공고
    ]

    for url in candidate_urls:
        try:
            print(f"[semas] 요청: {url}")
            res = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT,
                verify=True,
            )
            res.raise_for_status()
            res.encoding = res.apparent_encoding or "utf-8"
            soup = BeautifulSoup(res.text, "html.parser")

            # 게시판 목록 행을 찾기 위한 일반적 셀렉터들 시도
            rows = (
                soup.select("table.board_list tbody tr")
                or soup.select("table tbody tr")
                or soup.select(".board_list li")
                or soup.select(".bbs_list tbody tr")
            )

            print(f"[semas] 발견된 행: {len(rows)}개")

            for row in rows:
                # 제목 추출
                title_el = row.select_one("a") or row.select_one(".subject")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                if not title or len(title) < 5:
                    continue

                # 컨설턴트 관련 공고만 필터
                if not matches_keyword(title):
                    continue

                # 링크 추출
                link = title_el.get("href", "")
                if link.startswith("javascript:") or not link:
                    # JS로 만드는 링크는 onclick 속성에서 추출 시도
                    onclick = title_el.get("onclick", "")
                    m = re.search(r"['\"](\d+)['\"]", onclick)
                    if m:
                        board_id = m.group(1)
                        link = f"{url}?boardId={board_id}"
                    else:
                        link = url
                elif not link.startswith("http"):
                    link = "https://www.semas.or.kr" + link

                # 게시일 추출 (행에서 날짜 형식 찾기)
                row_text = row.get_text(" ", strip=True)
                posted = parse_korean_date(row_text)

                # 고유 ID 생성
                id_match = re.search(r"(\d{4,})", link + " " + row_text)
                board_id = id_match.group(1) if id_match else str(hash(title))[:8]

                items.append({
                    "id": make_id("semas", board_id),
                    "agency": "소상공인시장진흥공단",
                    "type": classify_type(title),
                    "title": title,
                    "postedDate": posted,
                    "deadline": None,  # 본문 들어가야 정확함, 1차 수집에서는 비움
                    "region": "전국",
                    "qualifications": [],
                    "fee": "",
                    "url": link,
                })

            if items:
                break  # 첫 번째 성공한 URL에서 끝

        except requests.RequestException as e:
            print(f"[semas] 요청 실패: {e}")
            continue
        except Exception as e:
            print(f"[semas] 파싱 오류: {e}")
            continue

    print(f"[semas] 완료: {len(items)}건 수집")
    return items


def classify_type(title: str) -> str:
    """제목으로 공고 유형 분류"""
    if "평가위원" in title or "심사위원" in title:
        return "평가위원"
    if "심사원" in title:
        return "심사원"
    if "멘토" in title:
        return "멘토"
    if "교육" in title or "양성" in title or "과정" in title:
        return "교육"
    return "컨설턴트"


# ============ 메인 ============
def main():
    print(f"=== 스크레이퍼 실행: {datetime.now(KST).isoformat()} ===")

    all_items = []
    all_items.extend(scrape_semas())
    # 향후 추가:
    # all_items.extend(scrape_kosmes())
    # all_items.extend(scrape_kstartup())

    # 중복 제거 (같은 ID 기준)
    unique = {}
    for item in all_items:
        unique[item["id"]] = item
    items = list(unique.values())

    # 결과 저장
    output = {
        "lastUpdate": datetime.now(KST).isoformat(),
        "totalCount": len(items),
        "items": items,
    }

    out_path = Path("data.json")
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"=== 완료: 총 {len(items)}건 → data.json 저장 ===")

    # 결과 요약 (로그에 표시)
    if items:
        print("\n--- 수집된 공고 미리보기 ---")
        for i, item in enumerate(items[:5], 1):
            print(f"{i}. [{item['agency']}] {item['title'][:50]}")
        if len(items) > 5:
            print(f"... 외 {len(items)-5}건")
    else:
        print("\n[경고] 수집된 공고가 없습니다.")
        print("사이트 구조가 변경되었거나 키워드와 일치하는 공고가 없을 수 있습니다.")
        print("로그를 확인하고 셀렉터를 점검해주세요.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[치명적 오류] {e}", file=sys.stderr)
        # 에러가 나도 빈 결과를 저장 (대시보드가 깨지지 않도록)
        Path("data.json").write_text(
            json.dumps({
                "lastUpdate": datetime.now(KST).isoformat(),
                "error": str(e),
                "items": []
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        sys.exit(1)
