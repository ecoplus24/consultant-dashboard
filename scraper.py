"""
컨설턴트 공고 스크레이퍼 v2
대상: 소상공인시장진흥공단 (semas.or.kr)

v2 개선사항:
- 본문 페이지 자동 진입해서 마감일 정확히 추출
- '26. 4. 30.(목) 같은 한국식 축약 표기 파싱
- "신청기간: A ~ B" 패턴에서 마감일(B) 정확히 추출
- 첨부파일(HWP/PDF) 정보 수집해서 메타데이터로 추가
- 키워드 확장 (진단, 자문, 위촉, 멘토링)
- 사이트 부담 최소화 (본문 진입 사이 3초 대기)
"""

import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ============ 설정 ============
KST = timezone(timedelta(hours=9))
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
TIMEOUT = 20
DETAIL_DELAY = 3  # 본문 페이지 사이 대기 시간(초) — 사이트 부담 최소화

# 컨설턴트 모집과 관련된 키워드 (제목에 이 중 하나라도 있어야 수집)
KEYWORDS = [
    "컨설턴트", "컨설팅", "평가위원", "심사위원", "심사원",
    "멘토", "멘토링", "전문가", "지도사", "코칭",
    "진단", "자문", "위촉",
    "희망리턴", "재도약", "재창업", "강한소상공인",
]


# ============ 유틸 — 한국식 날짜 파싱 ============
def parse_korean_date(s):
    """다양한 한국식 날짜 표기를 YYYY-MM-DD로 정규화"""
    if not s:
        return None
    s = s.strip()

    # ★ '26. 11. 12.(목) 형식 — 소진공의 핵심 패턴
    # 작은따옴표 + 2자리 연도 + 점 + 월 + 점 + 일 + 점 + (요일)
    m = re.search(r"['']?(\d{2,4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?(?:\([^)]+\))?", s)
    if m:
        y, mo, d = m.groups()
        if len(y) == 2:
            y = "20" + y
        try:
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        except ValueError:
            pass

    # 2026.05.20 / 2026-05-20 / 2026/05/20
    m = re.search(r"(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    # 2026년 11월 12일
    m = re.search(r"(20\d{2})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    return None


def extract_application_period(text):
    """본문에서 신청기간/접수기간 → 마감일 추출"""
    # ① "신청기간: A ~ B" 패턴 (소진공 표준 패턴)
    m = re.search(
        r"(?:신청\s*기간|접수\s*기간|모집\s*기간|공모\s*기간)[^~\n]*~\s*(['']?\d{2,4}\.\s*\d{1,2}\.\s*\d{1,2}\.?(?:\([^)]+\))?)",
        text
    )
    if m:
        d = parse_korean_date(m.group(1))
        if d:
            return d

    # ② "~ '26.11.12.(목)" 같이 물결표 뒤 날짜
    m = re.search(r"~\s*(['']?\d{2,4}\.\s*\d{1,2}\.\s*\d{1,2}\.?(?:\([^)]+\))?)", text)
    if m:
        d = parse_korean_date(m.group(1))
        if d:
            return d

    # ③ 일반 "마감", "까지" 패턴
    patterns = [
        r"(?:마감|까지)[^0-9]{0,30}(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})",
        r"(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})\s*까지",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            d = parse_korean_date(m.group(1))
            if d:
                return d

    # ④ 본문 내 모든 날짜 중 미래의 가장 늦은 날짜를 마감일로 추정
    today = datetime.now(KST).date()
    candidates = []
    for m in re.finditer(r"['']?(\d{2,4})\.\s*(\d{1,2})\.\s*(\d{1,2})", text):
        d = parse_korean_date(m.group(0))
        if d:
            try:
                date_obj = datetime.strptime(d, "%Y-%m-%d").date()
                if date_obj >= today:
                    candidates.append(d)
            except ValueError:
                continue
    if candidates:
        return max(candidates)  # 가장 늦은 미래 날짜 = 마감일

    return None


def extract_posted_date(soup):
    """등록일 추출 (소진공 표 형식)"""
    for tr in soup.select("table tr"):
        text = tr.get_text(" ", strip=True)
        if "등록일" in text:
            d = parse_korean_date(text)
            if d:
                return d
    # 백업: 본문에서 "등록일" 찾기
    full_text = soup.get_text(" ", strip=True)
    m = re.search(r"등록일\s*[:\-]?\s*(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})", full_text)
    if m:
        return parse_korean_date(m.group(1))
    return None


def extract_qualifications(text):
    """본문에서 자격요건 키워드 추출"""
    found = []
    qual_keywords = [
        "경영지도사", "기술지도사", "창업보육매니저", "MBA",
        "ISO 9001", "ISO 14001", "ISO 45001", "ISO 인증심사원",
        "공인노무사", "변호사", "회계사", "세무사",
        "재무·회계", "재무회계", "마케팅",
        "제조업", "유통업", "외식업",
        "안전보건", "위험성평가",
    ]
    for kw in qual_keywords:
        if kw in text and kw not in found:
            found.append(kw)
    return found


def matches_keyword(title):
    return any(kw in title for kw in KEYWORDS)


def make_id(agency, board_id):
    short = re.sub(r"[^a-zA-Z0-9가-힣]", "", agency)[:8]
    return f"{short}-{board_id}"


def classify_type(title):
    if "평가위원" in title or "심사위원" in title or "심사원" in title:
        return "평가위원"
    if "멘토" in title:
        return "멘토"
    if "교육" in title or "양성" in title or "과정" in title or "보수" in title:
        return "교육"
    return "컨설턴트"


# ============ 본문 페이지 진입 ============
def fetch_semas_detail(url, session):
    """소진공 본문 페이지에서 상세 정보 추출"""
    try:
        print(f"  [본문 진입] {url[:80]}")
        time.sleep(DETAIL_DELAY)

        res = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        res.raise_for_status()
        res.encoding = res.apparent_encoding or "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")

        body_text = soup.get_text(" ", strip=True)

        # 마감일 추출
        deadline = extract_application_period(body_text)

        # 등록일 추출
        posted = extract_posted_date(soup)

        # 첨부파일 정보 (HWP/PDF/DOCX 등)
        attachments = []
        for a in soup.select("a"):
            href = a.get("href", "")
            text = a.get_text(strip=True)
            combined = (href + " " + text).lower()
            if any(ext in combined for ext in [".hwp", ".pdf", ".docx", ".xlsx", ".zip", ".hwpx"]):
                if text and 5 < len(text) < 120:
                    attachments.append(text)

        # 자격요건
        quals = extract_qualifications(body_text)

        print(f"  → 마감: {deadline}, 등록: {posted}, 첨부: {len(attachments)}개")

        return {
            "deadline": deadline,
            "posted": posted,
            "attachments": attachments[:3],  # 최대 3개만
            "qualifications": quals,
        }
    except Exception as e:
        print(f"  [본문 진입 실패] {e}")
        return {
            "deadline": None,
            "posted": None,
            "attachments": [],
            "qualifications": [],
        }


# ============ 소진공 메인 스크레이퍼 ============
def scrape_semas():
    """소상공인시장진흥공단 공고 수집"""
    print("[semas] 시작: 소상공인시장진흥공단")
    items = []

    list_url = "https://www.semas.or.kr/web/board/webBoardList.kmdc?bCd=1&pageId=PG90000001"
    session = requests.Session()

    try:
        print(f"[semas] 목록 요청: {list_url}")
        res = session.get(list_url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        res.raise_for_status()
        res.encoding = res.apparent_encoding or "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")

        rows = soup.select("table tbody tr") or soup.select("table tr")
        print(f"[semas] 발견된 행: {len(rows)}개")

        # 🔍 진단 로그: 발견된 모든 행의 제목 출력
        print("[semas] --- 게시판 현재 목록 ---")
        for i, row in enumerate(rows[:15], 1):
            title_el = row.select_one("a")
            if title_el:
                row_title = title_el.get_text(strip=True)
                if row_title and len(row_title) > 3:
                    matched = "✓" if matches_keyword(row_title) else "✗"
                    print(f"  {matched} {i:2d}. {row_title[:65]}")
        print("[semas] --- 목록 끝 ---")

        # 1차: 키워드 매칭
        candidates = []
        for row in rows:
            title_el = row.select_one("a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            if not title or len(title) < 5:
                continue

            if not matches_keyword(title):
                continue

            # 본문 URL 추출
            href = title_el.get("href", "")
            onclick = title_el.get("onclick", "")

            detail_url = None
            if href and not href.startswith("javascript:") and href != "#":
                if href.startswith("http"):
                    detail_url = href
                else:
                    detail_url = "https://www.semas.or.kr" + href
            elif onclick:
                # javascript:fnView('12345') 같은 패턴
                m = re.search(r"['\"](\d{4,})['\"]", onclick)
                if m:
                    board_no = m.group(1)
                    detail_url = (
                        f"https://www.semas.or.kr/web/board/webBoardView.kmdc"
                        f"?bCd=1&pageId=PG90000001&b_idx={board_no}"
                    )

            if detail_url:
                candidates.append((title, detail_url))

        print(f"[semas] 키워드 매칭 공고: {len(candidates)}건")

        # 2차: 본문 진입해서 정밀 정보 수집
        for title, detail_url in candidates:
            print(f"[semas] 처리: {title[:50]}")
            detail = fetch_semas_detail(detail_url, session)

            # ID 생성 (URL의 b_idx 또는 첫 번째 큰 숫자 사용)
            id_match = re.search(r"b_idx=(\d+)", detail_url) or re.search(r"(\d{6,})", detail_url)
            board_id = id_match.group(1) if id_match else str(abs(hash(title)))[:8]

            items.append({
                "id": make_id("semas", board_id),
                "agency": "소상공인시장진흥공단",
                "type": classify_type(title),
                "title": title,
                "postedDate": detail["posted"],
                "deadline": detail["deadline"],
                "region": "전국",
                "qualifications": detail["qualifications"],
                "fee": "",
                "attachments": detail["attachments"],
                "url": detail_url,
            })

    except requests.RequestException as e:
        print(f"[semas] 요청 실패: {e}")
    except Exception as e:
        print(f"[semas] 파싱 오류: {e}")
        traceback.print_exc()

    print(f"[semas] 완료: {len(items)}건 수집")
    return items


# ============ 메인 ============
def main():
    print(f"=== 스크레이퍼 v2 실행: {datetime.now(KST).isoformat()} ===")

    all_items = []
    all_items.extend(scrape_semas())
    # 향후 추가:
    # all_items.extend(scrape_kosmes())
    # all_items.extend(scrape_kstartup())

    # 중복 제거
    unique = {}
    for item in all_items:
        unique[item["id"]] = item
    items = list(unique.values())

    output = {
        "lastUpdate": datetime.now(KST).isoformat(),
        "totalCount": len(items),
        "items": items,
    }

    Path("data.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n=== 완료: 총 {len(items)}건 → data.json 저장 ===")

    if items:
        print("\n--- 수집된 공고 미리보기 ---")
        for i, item in enumerate(items[:10], 1):
            print(f"{i}. [{item['agency']}] {item['title'][:50]}")
            print(f"   마감: {item.get('deadline') or '⚠ 미상'}, 등록: {item.get('postedDate') or '⚠ 미상'}")
            if item.get('attachments'):
                print(f"   첨부: {item['attachments'][0][:50]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[치명적 오류] {e}", file=sys.stderr)
        traceback.print_exc()
        Path("data.json").write_text(
            json.dumps({
                "lastUpdate": datetime.now(KST).isoformat(),
                "error": str(e),
                "items": []
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        sys.exit(1)
