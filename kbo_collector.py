"""
KBO 토토 예측 프로그램 - 1단계: 데이터 수집기
=============================================
실행 방법:
  pip install requests beautifulsoup4 pandas playwright
  playwright install chromium
  python kbo_collector.py
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time
import os
import socket
socket.setdefaulttimeout(8)  # DNS 포함 모든 소켓 최대 8초

# ── 공통 헤더 (봇 차단 우회) ─────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/144.0.0.0 Safari/537.36"
    ),
    "Origin": "https://m.sports.naver.com",
    "Referer": "https://m.sports.naver.com/kbaseball/schedule/index",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Accept": "application/json, text/plain, */*",
    "charset": "utf-8",
    "x-sports-backend": "kotlin",
}

# ── 팀 이름 매핑 (KBO 공식 → 짧은 이름) ─────────────────
TEAM_MAP = {
    "KIA": "KIA", "삼성": "삼성", "LG": "LG", "두산": "두산",
    "KT": "KT", "SSG": "SSG", "롯데": "롯데", "한화": "한화",
    "NC": "NC", "키움": "키움",
}


# ════════════════════════════════════════════════════════
# 1. 오늘 경기 일정 + 선발투수 (네이버 스포츠 비공식 API)
# ════════════════════════════════════════════════════════
def get_today_schedule(date: str = None) -> list[dict]:
    """
    네이버 스포츠 내부 API로 오늘 KBO 경기 일정과 선발투수를 가져옵니다.

    반환 예시:
    [
      {
        "game_id": "20260423LGKT0",
        "home_team": "KT",
        "away_team": "LG",
        "home_starter": "고영표",
        "away_starter": "임찬규",
        "stadium": "수원",
        "game_time": "18:30"
      }, ...
    ]
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    # 해당 월 전체 범위로 요청 (네이버 실제 API 방식)
    dt = datetime.strptime(date, "%Y%m%d")
    from_date = dt.strftime("%Y-%m-01")
    if dt.month == 12:
        last_day = dt.replace(year=dt.year+1, month=1, day=1) - timedelta(days=1)
    else:
        last_day = dt.replace(month=dt.month+1, day=1) - timedelta(days=1)
    to_date = last_day.strftime("%Y-%m-%d")
    today_str = dt.strftime("%Y-%m-%d")

    url = (
        "https://api-gw.sports.naver.com/schedule/games"
        "?fields=basic%2Cschedule%2Cbaseball%2CmanualRelayUrl"
        "&upperCategoryId=kbaseball&categoryId=kbo"
        f"&fromDate={from_date}&toDate={to_date}"
        "&roundCodes=&size=500"
    )

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[오류] 네이버 일정 API 실패: {e}")
        return []

    # 오늘 날짜 경기만 필터링
    all_games = data.get("result", {}).get("games", [])
    today_games = [g for g in all_games if g.get("gameDate", "").startswith(today_str)]

    games = []
    for item in today_games:
        # 실제 응답 구조: 최상위에 바로 키가 있음
        game_time_raw = item.get("gameDateTime", "")
        game_time = game_time_raw[11:16] if len(game_time_raw) >= 16 else ""

        game = {
            "game_id":      item.get("gameId", ""),
            "home_team":    item.get("homeTeamName", ""),
            "away_team":    item.get("awayTeamName", ""),
            "home_team_code": item.get("homeTeamCode", ""),
            "away_team_code": item.get("awayTeamCode", ""),
            "home_starter": item.get("homeStarterName") or "미정",
            "away_starter": item.get("awayStarterName") or "미정",
            "stadium":      item.get("stadium", ""),
            "game_time":    game_time,
        }
        games.append(game)

    print(f"[일정] {date} KBO 경기 {len(games)}경기 수집 완료")

    # 디버깅: 처음 실행 시 응답 구조 확인용
    if not games and all_games:
        print(f"[디버그] 전체 {len(all_games)}경기 중 오늘({today_str}) 없음")
        print(f"[디버그] 첫 항목 키: {list(all_games[0].keys())}")
        print(f"[디버그] gameDate 샘플: {all_games[0].get('gameDate','N/A')}")

    return games


# ════════════════════════════════════════════════════════
# 2. 시즌 완료 경기 캐시 (최근 폼 계산용)
# ════════════════════════════════════════════════════════
_SEASON_GAMES_CACHE = []

def _fetch_season_games(date: str = None) -> list:
    """이번 시즌 완료된 경기 전체를 한 번만 요청해 캐싱합니다."""
    global _SEASON_GAMES_CACHE
    if _SEASON_GAMES_CACHE:
        return _SEASON_GAMES_CACHE

    season = datetime.now().year
    from_date = f"{season}-03-01"
    to_dt = datetime.strptime(date, "%Y%m%d") if date else datetime.now()
    to_date = to_dt.strftime("%Y-%m-%d")

    url = (
        "https://api-gw.sports.naver.com/schedule/games"
        "?fields=basic"
        "&upperCategoryId=kbaseball&categoryId=kbo"
        f"&fromDate={from_date}&toDate={to_date}"
        "&roundCodes=&size=1000"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        games = resp.json().get("result", {}).get("games", [])
        _SEASON_GAMES_CACHE = [
            g for g in games
            if g.get("statusCode") == "RESULT"
            and not g.get("cancel")
            and not g.get("suspended")
            and g.get("winner") in ("HOME", "AWAY")
        ]
        print(f"[시즌] 완료 경기 {len(_SEASON_GAMES_CACHE)}경기 캐시")
    except Exception as e:
        print(f"[오류] 시즌 경기 캐시 실패: {e}")
        _SEASON_GAMES_CACHE = []
    return _SEASON_GAMES_CACHE


def get_team_recent_form(team_code: str, date: str = None, n: int = 10) -> dict:
    """팀의 최근 n경기 승률·득실점·연승연패를 계산합니다."""
    games = _fetch_season_games(date)
    today_str = (datetime.strptime(date, "%Y%m%d") if date else datetime.now()).strftime("%Y-%m-%d")

    team_games = [
        g for g in games
        if (g.get("homeTeamCode") == team_code or g.get("awayTeamCode") == team_code)
        and g.get("gameDate", "") < today_str
    ]

    recent = team_games[-n:] if len(team_games) >= n else team_games
    if not recent:
        return {"last10_winrate": 0.5, "last5_scored": 4.5, "last5_allowed": 4.5, "streak": 0}

    wins = 0
    for g in recent:
        is_home = g.get("homeTeamCode") == team_code
        won = (is_home and g.get("winner") == "HOME") or (not is_home and g.get("winner") == "AWAY")
        if won:
            wins += 1

    last5 = recent[-5:] if len(recent) >= 5 else recent
    scored5, allowed5 = [], []
    for g in last5:
        is_home = g.get("homeTeamCode") == team_code
        scored5.append(int(g.get("homeTeamScore" if is_home else "awayTeamScore", 0) or 0))
        allowed5.append(int(g.get("awayTeamScore" if is_home else "homeTeamScore", 0) or 0))

    # 연승/연패 스트릭
    streak = 0
    for g in reversed(recent):
        is_home = g.get("homeTeamCode") == team_code
        won = (is_home and g.get("winner") == "HOME") or (not is_home and g.get("winner") == "AWAY")
        if streak == 0:
            streak = 1 if won else -1
        elif (streak > 0 and won) or (streak < 0 and not won):
            streak += 1 if won else -1
        else:
            break

    return {
        "last10_winrate": round(wins / len(recent), 4),
        "last5_scored":   round(sum(scored5) / len(scored5), 2) if scored5 else 4.5,
        "last5_allowed":  round(sum(allowed5) / len(allowed5), 2) if allowed5 else 4.5,
        "streak":         streak,
    }


# ════════════════════════════════════════════════════════
# 3. 팀별 시즌 성적 (네이버 스포츠)
# ════════════════════════════════════════════════════════
# 팀 코드 → 순위표 조회용 이름 매핑
TEAM_CODE_MAP = {
    "LG": "LG", "HT": "KIA", "HH": "한화", "KT": "KT",
    "NC": "NC", "LT": "롯데", "SS": "삼성", "OB": "두산",
    "SK": "SSG", "WO": "키움",
}

# 시즌 팀 순위/성적 캐시
_STANDINGS_CACHE = {}
# 팀 타격 성적 캐시
_BATTING_CACHE = {}

def _fetch_standings() -> dict:
    """네이버 KBO 팀 순위 API로 시즌 전체 성적을 가져옵니다."""
    global _STANDINGS_CACHE
    if _STANDINGS_CACHE:
        return _STANDINGS_CACHE

    season = datetime.now().year
    url = (
        f"https://api-gw.sports.naver.com/statistics/categories/kbo/seasons"
        f"/{season}/teams?fields=basic&gameType=REGULAR_SEASON&type=rank"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 응답 구조: result.seasonTeamStats 배열
        teams = data.get("result", {}).get("seasonTeamStats", [])
        for t in teams:
            code = t.get("teamId", "")
            wins   = int(t.get("winGameCount", 0))
            losses = int(t.get("loseGameCount", 0))
            draws  = int(t.get("drawnGameCount", 0))
            games  = int(t.get("gameCount", 1)) or 1
            wra    = float(t.get("wra", 0.5))  # 승률 직접 제공
            # 평균 득점/실점 = 시즌 총득점 / 경기수
            avg_score = float(t.get("offenseRun", 0)) / games
            avg_allow = float(t.get("defenseRun", 0)) / games
            _STANDINGS_CACHE[code] = {
                "win_rate":    wra,
                "avg_score":   round(avg_score, 2),
                "avg_allow":   round(avg_allow, 2),
                "recent_win":  wins,
                "recent_lose": losses,
            }
        print(f"[순위] 팀 성적 {len(_STANDINGS_CACHE)}팀 수집 완료")
    except Exception as e:
        print(f"[오류] 팀 순위 API 실패: {e}")
    return _STANDINGS_CACHE


def _fetch_team_batting() -> dict:
    """팀 타격 통계 (타율·OPS) 수집"""
    global _BATTING_CACHE
    if _BATTING_CACHE:
        return _BATTING_CACHE

    season = datetime.now().year
    url = (
        f"https://api-gw.sports.naver.com/statistics/categories/kbo"
        f"/seasons/{season}/teams"
        f"?fields=basic&gameType=REGULAR_SEASON&type=batting"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        teams = resp.json().get("result", {}).get("seasonTeamStats", [])
        for t in teams:
            code = t.get("teamId", "")
            if not code:
                continue
            _BATTING_CACHE[code] = {
                "batting_avg": float(t.get("oba",  t.get("battingAverage", 0.260))),
                "obp":         float(t.get("obp",  t.get("onBasePercentage", 0.330))),
                "slg":         float(t.get("slg",  t.get("sluggingPercentage", 0.390))),
                "ops":         float(t.get("ops",  0.0)) or round(
                    float(t.get("obp", 0.330)) + float(t.get("slg", 0.390)), 4
                ),
            }
        print(f"[타격] 팀 타격 성적 {len(_BATTING_CACHE)}팀 수집 완료")
    except Exception as e:
        print(f"[오류] 팀 타격 API 실패: {e}")
    return _BATTING_CACHE


def get_team_batting(team_code: str) -> dict:
    """팀 타율·OPS 반환, 없으면 리그 평균값"""
    cache = _fetch_team_batting()
    return cache.get(team_code, {"batting_avg": 0.260, "obp": 0.330, "slg": 0.390, "ops": 0.720})


def get_team_recent_record(team_code: str, n: int = 10) -> dict:
    """
    팀의 시즌 성적(승률, 평균 득점/실점)을 반환합니다.
    team_code: 네이버 팀 코드 (LG, HT, HH, KT, NC, LT, SS, OB, SK, WO)
    """
    standings = _fetch_standings()
    if team_code in standings:
        return {"team": team_code, **standings[team_code]}

    print(f"[경고] {team_code} 성적 없음 → 기본값 사용")
    return {"team": team_code, "recent_win": 0, "recent_lose": 0,
            "win_rate": 0.5, "avg_score": 4.5, "avg_allow": 4.5}


# ════════════════════════════════════════════════════════
# 4. 선발투수 세부 기록
# ════════════════════════════════════════════════════════
# 투수 전체 기록 캐시 (한 번만 요청)
_PITCHER_CACHE = {}

def _fetch_all_pitchers() -> dict:
    """
    네이버 투수 시즌 기록을 페이지네이션으로 가져와 이름으로 캐싱합니다.
    pitcherGameCount 내림차순 1가지만 사용해 최대 300명(6페이지)까지 수집.
    """
    global _PITCHER_CACHE
    if _PITCHER_CACHE:
        return _PITCHER_CACHE

    season = datetime.now().year
    MAX_PAGES = 8  # 50명 × 8 = 400명, 구원투수 포함 전수 커버

    for page in range(1, MAX_PAGES + 1):
        url = (
            f"https://api-gw.sports.naver.com/statistics/categories/kbo"
            f"/seasons/{season}/players"
            f"?sortField=pitcherInning&sortDirection=desc"
            f"&playerType=PITCHER&gameType=REGULAR_SEASON"
            f"&page={page}&pageSize=50"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=5)
            resp.raise_for_status()
            players = resp.json().get("result", {}).get("seasonPlayerStats", [])
            if not players:
                break
            for p in players:
                name = p.get("playerName", "")
                if not name or name in _PITCHER_CACHE:
                    continue
                inning_raw = str(p.get("pitcherInning", "0"))
                try:
                    if " " in inning_raw:
                        parts = inning_raw.split()
                        innings = float(parts[0]) + (1/3 if "1/3" in parts[1] else 2/3)
                    else:
                        innings = float(inning_raw)
                except:
                    innings = 0.0
                def _f(val, default):
                    try: return float(val) if val is not None else default
                    except: return default
                _PITCHER_CACHE[name] = {
                    "name":    name,
                    "era":     _f(p.get("pitcherEra"),  4.50),
                    "whip":    _f(p.get("pitcherWhip"), 1.40),
                    "innings": innings,
                    "k9":      _f(p.get("pitcherInningKk"), 7.0),
                    "wins":    int(p.get("pitcherWin",  0) or 0),
                    "losses":  int(p.get("pitcherLose", 0) or 0),
                    "games":   int(p.get("pitcherGameCount", 0) or 0),
                }
            time.sleep(0.15)
            if len(players) < 50:
                break
        except Exception as e:
            print(f"[오류] 투수 기록 API 실패 (p{page}): {e}")
            break

    print(f"[투수] 시즌 기록 {len(_PITCHER_CACHE)}명 수집 완료")
    return _PITCHER_CACHE


def get_pitcher_stats(pitcher_name: str, team_code: str = "") -> dict:
    """선발투수 이름으로 시즌 ERA/WHIP/이닝/K9를 반환합니다."""
    if pitcher_name == "미정":
        return {"name": "미정", "era": 4.50, "whip": 1.40, "innings": 0, "k9": 7.0}

    cache = _fetch_all_pitchers()
    if pitcher_name in cache:
        return cache[pitcher_name]

    # 이름 일부 매칭 (외국인 선수 등 표기 차이 대비)
    for name, stats in cache.items():
        if pitcher_name in name or name in pitcher_name:
            print(f"[투수] '{pitcher_name}' → '{name}' 매칭")
            return stats

    print(f"[경고] {pitcher_name} 기록 없음 → 리그 평균 사용")
    return {"name": pitcher_name, "era": 4.50, "whip": 1.40, "innings": 0, "k9": 7.0}


def get_pitcher_recent_form(pitcher_name: str, date: str, history_df, n: int = 5) -> dict:
    """
    history CSV에서 선발투수의 최근 N경기 폼을 계산합니다.
    - recent_runs_allowed : 최근 N선발에서 팀이 평균 몇 실점했는지 (근사값)
    - recent_win_rate     : 최근 N선발에서 팀 승률
    - starts_count        : 이번 시즌 선발 횟수
    """
    default = {"recent_runs_allowed": 4.50, "recent_win_rate": 0.50, "starts_count": 0}
    if history_df is None or history_df.empty or not pitcher_name or pitcher_name == "미정":
        return default

    try:
        today = str(date)
        home = history_df[history_df["home_starter"] == pitcher_name].copy()
        home["runs_allowed"] = home["away_score"]
        home["won"]          = (home["home_win"] == 1).astype(int)

        away = history_df[history_df["away_starter"] == pitcher_name].copy()
        away["runs_allowed"] = away["home_score"]
        away["won"]          = (away["home_win"] == 0).astype(int)

        all_starts = pd.concat([home[["date","runs_allowed","won"]],
                                 away[["date","runs_allowed","won"]]])
        all_starts = all_starts[all_starts["date"].astype(str).str.replace("-","") < today]
        all_starts = all_starts.sort_values("date", ascending=False)

        total_starts = len(all_starts)
        recent = all_starts.head(n)
        if len(recent) == 0:
            return {**default, "starts_count": total_starts}

        return {
            "recent_runs_allowed": round(recent["runs_allowed"].mean(), 2),
            "recent_win_rate":     round(recent["won"].mean(), 3),
            "starts_count":        total_starts,
        }
    except Exception as e:
        print(f"[경고] 최근 폼 계산 실패 ({pitcher_name}): {e}")
        return default


# ════════════════════════════════════════════════════════
# 5. 날씨 정보 (기상청 공공데이터 API)
# ════════════════════════════════════════════════════════
# 경기장별 격자 좌표 (기상청 단기예보 기준)
STADIUM_GRID = {
    "수원":   (60, 121), "잠실":   (61, 126), "고척":   (58, 126),
    "인천":   (55, 124), "대전":   (67, 100), "광주":   (58,  74),
    "대구":   (89,  90), "부산":   (98,  76), "창원":   (91,  77),
    "청주":   (69, 106),
}

def get_weather(stadium: str, game_date: str = None) -> dict:
    """
    기상청 단기예보 API로 경기 시간대 날씨를 가져옵니다.
    API 키는 환경변수 KMA_API_KEY 로 설정하세요.
    https://www.data.go.kr 에서 무료 발급 가능합니다.
    """
    api_key = os.environ.get("KMA_API_KEY", "8920b18152de2c1a56ae4cf7852828cdae684ecf8c5f4511ef605b2330a8caf0")

    if game_date is None:
        game_date = datetime.now().strftime("%Y%m%d")

    nx, ny = STADIUM_GRID.get(stadium, (61, 126))  # 기본값 잠실

    url = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    params = {
        "serviceKey": api_key,
        "pageNo": 1,
        "numOfRows": 100,
        "dataType": "JSON",
        "base_date": game_date,
        "base_time": "0500",
        "nx": nx,
        "ny": ny,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        items = resp.json()["response"]["body"]["items"]["item"]

        weather = {"stadium": stadium, "temp": None, "rain_prob": None, "humidity": None}
        for item in items:
            if item["fcstTime"] == "1800":  # 오후 6시 (경기 시간대)
                if item["category"] == "TMP":  weather["temp"]      = float(item["fcstValue"])
                if item["category"] == "POP":  weather["rain_prob"] = float(item["fcstValue"])
                if item["category"] == "REH":  weather["humidity"]  = float(item["fcstValue"])
        return weather

    except Exception as e:
        print(f"[오류] 날씨 API 실패: {e}")
        return {"stadium": stadium, "temp": None, "rain_prob": None, "humidity": None}


# ════════════════════════════════════════════════════════
# 6. 상대 전적 (네이버 스포츠)
# ════════════════════════════════════════════════════════
def get_head_to_head(home_code: str, away_code: str) -> dict:
    """
    _fetch_season_games() 캐시를 재사용해 상대 전적을 계산합니다.
    별도 API 요청 없음.
    """
    games = _fetch_season_games()  # 이미 캐싱된 데이터 재사용
    home_wins = away_wins = 0
    for g in games:
        ht = g.get("homeTeamCode", "")
        at = g.get("awayTeamCode", "")
        winner = g.get("winner", "")
        if {ht, at} != {home_code, away_code}:
            continue
        if (ht == home_code and winner == "HOME") or (at == home_code and winner == "AWAY"):
            home_wins += 1
        else:
            away_wins += 1

    total = home_wins + away_wins or 1
    print(f"[상대전적] {home_code} vs {away_code} → {home_wins}승 {away_wins}패")
    return {
        "home_team":     home_code,
        "away_team":     away_code,
        "h2h_home_wins": home_wins,
        "h2h_away_wins": away_wins,
        "h2h_home_rate": round(home_wins / total, 4),
    }


# ════════════════════════════════════════════════════════
# 7. 전체 수집 실행 (메인 함수)
# ════════════════════════════════════════════════════════
def collect_today_data(date: str = None) -> pd.DataFrame:
    """
    오늘 경기에 필요한 모든 데이터를 수집해서
    하나의 DataFrame으로 반환합니다.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    print(f"\n{'='*50}")
    print(f" KBO 데이터 수집 시작 - {date}")
    print(f"{'='*50}")

    # 0) 시즌 history 로드 (투수 최근 폼 계산용)
    _history_df = None
    for hist_path in [f"kbo_history_{datetime.now().year}.csv", "kbo_history_2026.csv"]:
        if os.path.exists(hist_path):
            try:
                _history_df = pd.read_csv(hist_path)
                print(f"[히스토리] {hist_path} 로드 완료 ({len(_history_df)}경기)")
            except Exception:
                pass
            break

    # 1) 오늘 경기 일정
    games = get_today_schedule(date)
    if not games:
        print("[종료] 오늘 경기가 없거나 일정을 가져오지 못했습니다.")
        return pd.DataFrame()

    rows = []
    for g in games:
        print(f"\n▶ {g['away_team']} @ {g['home_team']} ({g['game_time']}, {g['stadium']})")
        print(f"  선발: {g['away_team']} {g['away_starter']} vs {g['home_team']} {g['home_starter']}")

        # 2) 팀 시즌 성적 + 타격 성적
        home_rec = get_team_recent_record(g["home_team_code"])
        away_rec = get_team_recent_record(g["away_team_code"])
        home_bat = get_team_batting(g["home_team_code"])
        away_bat = get_team_batting(g["away_team_code"])

        # 3) 팀 최근 폼 (연승/연패, 최근 득실점)
        home_form = get_team_recent_form(g["home_team_code"], date)
        away_form = get_team_recent_form(g["away_team_code"], date)
        time.sleep(0.3)

        # 4) 선발투수 기록 (ERA, WHIP, K9 포함)
        home_pit = get_pitcher_stats(g["home_starter"], g["home_team_code"])
        away_pit = get_pitcher_stats(g["away_starter"], g["away_team_code"])

        # 4-1) 선발투수 최근 5경기 폼 (history CSV 기반)
        home_recent = get_pitcher_recent_form(g["home_starter"], date, _history_df)
        away_recent = get_pitcher_recent_form(g["away_starter"], date, _history_df)
        time.sleep(0.3)

        # 5) 상대 전적
        h2h = get_head_to_head(g["home_team_code"], g["away_team_code"])

        # 6) 날씨
        weather = get_weather(g["stadium"], date)

        # 하나의 row로 합치기
        row = {
            # 경기 기본 정보
            "date":           date,
            "game_id":        g["game_id"],
            "home_team":      g["home_team"],
            "away_team":      g["away_team"],
            "stadium":        g["stadium"],
            "game_time":      g["game_time"],

            # 선발투수 (K9 추가)
            "home_starter":   g["home_starter"],
            "away_starter":   g["away_starter"],
            "home_era":       home_pit["era"],
            "away_era":       away_pit["era"],
            "home_whip":      home_pit["whip"],
            "away_whip":      away_pit["whip"],
            "home_k9":        home_pit.get("k9", 7.0),
            "away_k9":        away_pit.get("k9", 7.0),

            # 팀 시즌 성적
            "home_win_rate":  home_rec["win_rate"],
            "away_win_rate":  away_rec["win_rate"],
            "home_avg_score": home_rec["avg_score"],
            "away_avg_score": away_rec["avg_score"],
            "home_avg_allow": home_rec["avg_allow"],
            "away_avg_allow": away_rec["avg_allow"],

            # 최근 폼 (NEW)
            "home_last10_winrate": home_form["last10_winrate"],
            "away_last10_winrate": away_form["last10_winrate"],
            "home_last5_scored":   home_form["last5_scored"],
            "away_last5_scored":   away_form["last5_scored"],
            "home_last5_allowed":  home_form["last5_allowed"],
            "away_last5_allowed":  away_form["last5_allowed"],
            "home_streak":         home_form["streak"],
            "away_streak":         away_form["streak"],

            # 타격 성적 (NEW)
            "home_batting_avg": home_bat["batting_avg"],
            "away_batting_avg": away_bat["batting_avg"],
            "home_ops":         home_bat["ops"],
            "away_ops":         away_bat["ops"],

            # 선발투수 최근 폼 (NEW)
            "home_starter_recent_runs":    home_recent["recent_runs_allowed"],
            "away_starter_recent_runs":    away_recent["recent_runs_allowed"],
            "home_starter_recent_winrate": home_recent["recent_win_rate"],
            "away_starter_recent_winrate": away_recent["recent_win_rate"],
            "home_starter_starts":         home_recent["starts_count"],
            "away_starter_starts":         away_recent["starts_count"],

            # 상대 전적
            "h2h_home_rate":  h2h["h2h_home_rate"],

            # 날씨
            "temp":           weather.get("temp"),
            "rain_prob":      weather.get("rain_prob"),
            "humidity":       weather.get("humidity"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # CSV로 저장 (다음 단계 전처리에서 사용)
    save_path = f"kbo_raw_{date}.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n[완료] {len(df)}경기 데이터 저장 → {save_path}")
    print(df[["home_team", "away_team", "home_starter", "away_starter",
              "home_era", "away_era", "home_win_rate", "away_win_rate"]].to_string(index=False))

    return df


# ════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 오늘 날짜 자동 수집
    df = collect_today_data()

    # 특정 날짜 테스트: collect_today_data("20260423")