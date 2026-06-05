"""
KBO 토토 예측 프로그램 - 4단계: 웹 대시보드
"""

from flask import Flask, jsonify, render_template_string
from datetime import datetime, timezone, timedelta
import os, json, pandas as pd

KST = timezone(timedelta(hours=9))

def now_kst():
    return datetime.now(KST)
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_cache = {}

_DB_PATH = os.path.join(BASE_DIR, "kbo_data.db")
_db_conn = None

def get_db_conn():
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
        _db_conn.execute("PRAGMA journal_mode=WAL")
    return _db_conn


def init_db():
    conn = get_db_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS accuracy_log (
            date             TEXT PRIMARY KEY,
            date_str         TEXT,
            results_recorded INTEGER DEFAULT 0,
            hits             INTEGER DEFAULT 0,
            total_games      INTEGER DEFAULT 0,
            accuracy         REAL    DEFAULT 0,
            games            TEXT    DEFAULT '[]'
        )
    """)
    conn.commit()


# ── accuracy log helpers ─────────────────────────────────
def load_accuracy_log():
    conn = get_db_conn()
    try:
        rows = conn.execute("SELECT * FROM accuracy_log").fetchall()
        log = {}
        for row in rows:
            games = row["games"]
            if isinstance(games, str):
                games = json.loads(games)
            log[row["date"]] = {
                "date_str":         row["date_str"],
                "results_recorded": bool(row["results_recorded"]),
                "hits":             row["hits"],
                "total":            row["total_games"],
                "accuracy":         row["accuracy"],
                "games":            games,
            }
        return log
    except Exception as e:
        print(f"[DB] load error: {e}")
        return {}


def save_accuracy_log(log):
    conn = get_db_conn()
    try:
        for date, entry in log.items():
            conn.execute("""
                INSERT OR REPLACE INTO accuracy_log
                    (date, date_str, results_recorded, hits, total_games, accuracy, games)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                entry.get("date_str", ""),
                1 if entry.get("results_recorded") else 0,
                entry.get("hits", 0),
                entry.get("total", 0),
                entry.get("accuracy", 0),
                json.dumps(entry.get("games", []), ensure_ascii=False),
            ))
        conn.commit()
    except Exception as e:
        print(f"[DB] save error: {e}")


# DB 초기화 (앱 시작 시)
try:
    init_db()
except Exception as e:
    print(f"[DB] init error: {e}")



def save_predictions(date: str, games: list):
    if not games:
        return
    log = load_accuracy_log()
    if date not in log:
        log[date] = {
            "date_str": f"{date[:4]}.{date[4:6]}.{date[6:]}",
            "results_recorded": False,
            "games": []
        }
    log[date]["games"] = [
        {
            "home_team":            g["home_team"],
            "away_team":            g["away_team"],
            "predicted_winner":     "HOME" if g["final_home_prob"] >= g["final_away_prob"] else "AWAY",
            "stat_predicted_winner":"HOME" if g["stat_home_prob"]  >= g["stat_away_prob"]  else "AWAY",
            "ml_predicted_winner":  "HOME" if g["ml_home_prob"]    >= g["ml_away_prob"]    else "AWAY",
            "final_home_prob":      g["final_home_prob"],
            "final_away_prob":      g["final_away_prob"],
            "stat_home_prob":       g["stat_home_prob"],
            "ml_home_prob":         g["ml_home_prob"],
            "pick":                 g.get("pick", ""),
            "actual_winner":        None,
            "hit":                  None,
            "stat_hit":             None,
            "ml_hit":               None,
        }
        for g in games
    ]
    save_accuracy_log(log)


def update_actual_results():
    """완료된 경기의 실제 결과를 가져와 accuracy log 업데이트"""
    log = load_accuracy_log()
    if not log:
        return
    today = now_kst().strftime("%Y%m%d")
    pending = [
        d for d, v in log.items()
        if not v.get("results_recorded") and v.get("games") and d < today
    ]
    if not pending:
        return
    try:
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from kbo_model import fetch_season_results
        results_df = fetch_season_results()
        if results_df.empty:
            return
    except Exception as e:
        print(f"[결과업데이트] fetch_season_results 실패: {e}")
        return

    # Naver API는 gameDate를 "YYYYMMDD" 형식으로 반환 → 정규화
    results_df["date_key"] = results_df["date"].str.replace("-", "")

    updated = False
    for date in pending:
        day_df = results_df[results_df["date_key"] == date]
        if day_df.empty:
            print(f"[결과업데이트] {date} 경기 결과 없음 (API 미반영 또는 경기 없음)")
            continue
        hits, total = 0, 0
        for g in log[date]["games"]:
            match = day_df[
                (day_df["home_team"] == g["home_team"]) &
                (day_df["away_team"] == g["away_team"])
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            actual = "HOME" if row["home_win"] == 1 else "AWAY"
            g["actual_winner"]      = actual
            g["actual_home_score"]  = int(row["home_score"])
            g["actual_away_score"]  = int(row["away_score"])
            g["hit"]                = (g["predicted_winner"]      == actual)
            g["stat_hit"]           = (g.get("stat_predicted_winner") == actual)
            g["ml_hit"]             = (g.get("ml_predicted_winner")   == actual)
            total += 1
            if g["hit"]:
                hits += 1
        if total > 0:
            log[date]["results_recorded"] = True
            log[date]["hits"] = hits
            log[date]["total"] = total
            log[date]["accuracy"] = round(hits / total, 4)
            # 모델별 적중 집계
            log[date]["stat_hits"]  = sum(1 for g in log[date]["games"] if g.get("stat_hit"))
            log[date]["ml_hits"]    = sum(1 for g in log[date]["games"] if g.get("ml_hit"))
            updated = True
    if updated:
        save_accuracy_log(log)


# 스케줄러: 매일 23:30 KST 자동 결과 업데이트
try:
    _scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    _scheduler.add_job(update_actual_results, "cron", hour=23, minute=30)
    _scheduler.start()
    print("[스케줄러] 매일 23:30 KST 자동 결과 업데이트 등록 완료")
except Exception as e:
    print(f"[스케줄러] 시작 실패: {e}")


def get_adaptive_weights(recent_n: int = 30):
    """최근 N경기 기반으로 통계/ML 모델 가중치를 계산합니다."""
    log = load_accuracy_log()
    stat_hits = stat_total = ml_hits = ml_total = game_count = 0

    for date in sorted(log.keys(), reverse=True):
        if game_count >= recent_n:
            break
        entry = log[date]
        if not entry.get("results_recorded"):
            continue
        for g in entry.get("games", []):
            if g.get("actual_winner") is None:
                continue
            if g.get("stat_predicted_winner"):
                stat_total += 1
                if g.get("stat_hit"):
                    stat_hits += 1
            if g.get("ml_predicted_winner"):
                ml_total += 1
                if g.get("ml_hit"):
                    ml_hits += 1
            game_count += 1

    # 데이터 부족 시 기본값
    if stat_total < 10 or ml_total < 10:
        return 0.4, 0.6, game_count

    stat_acc = stat_hits / stat_total
    ml_acc   = ml_hits   / ml_total
    total_acc = stat_acc + ml_acc
    stat_w = round(stat_acc / total_acc, 3)
    ml_w   = round(1 - stat_w, 3)

    print(f"[적응형 가중치] 통계:{stat_acc:.1%}→{stat_w:.2f}  ML:{ml_acc:.1%}→{ml_w:.2f}  ({game_count}경기 기반)")
    return stat_w, ml_w, game_count


# ── HTML 템플릿 ──────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KBO PICK</title>
<link href="https://fonts.googleapis.com/css2?family=Black+Han+Sans&family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0a0e1a;
    --bg2:      #111827;
    --bg3:      #1a2236;
    --border:   rgba(255,255,255,0.08);
    --accent:   #3b82f6;
    --accent2:  #f59e0b;
    --green:    #10b981;
    --red:      #ef4444;
    --text:     #f1f5f9;
    --muted:    #64748b;
    --nav-h:    56px;
    --font-display: 'Black Han Sans', sans-serif;
    --font-body:    'Noto Sans KR', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    min-height: 100vh;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── 네비게이션 바 ── */
  .topnav {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(10,14,26,0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    height: var(--nav-h);
  }
  .nav-inner {
    max-width: 960px;
    margin: 0 auto;
    padding: 0 20px;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .nav-logo {
    font-family: var(--font-display);
    font-size: 1.4rem;
    letter-spacing: -0.5px;
    cursor: pointer;
  }
  .nav-logo span { color: var(--accent); }
  .nav-links { display: flex; gap: 4px; }
  .nav-link {
    padding: 6px 16px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--muted);
    cursor: pointer;
    border: none;
    background: transparent;
    transition: all .2s;
    font-family: var(--font-body);
  }
  .nav-link:hover { color: var(--text); background: var(--bg3); }
  .nav-link.active { color: var(--accent); background: rgba(59,130,246,0.1); }

  /* 햄버거 버튼 (모바일) */
  .hamburger {
    display: none;
    flex-direction: column;
    gap: 5px;
    cursor: pointer;
    padding: 8px;
    border: none;
    background: transparent;
  }
  .hamburger span {
    display: block;
    width: 22px;
    height: 2px;
    background: var(--text);
    border-radius: 2px;
    transition: all .3s;
  }
  .hamburger.open span:nth-child(1) { transform: translateY(7px) rotate(45deg); }
  .hamburger.open span:nth-child(2) { opacity: 0; }
  .hamburger.open span:nth-child(3) { transform: translateY(-7px) rotate(-45deg); }

  /* 사이드 드로어 (모바일) */
  .drawer-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    z-index: 200;
  }
  .drawer-overlay.open { display: block; }
  .drawer {
    position: fixed;
    top: 0; left: 0;
    height: 100vh;
    width: 240px;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    z-index: 201;
    transform: translateX(-100%);
    transition: transform .3s ease;
  }
  .drawer.open { transform: translateX(0); }
  .drawer-header {
    font-family: var(--font-display);
    font-size: 1.4rem;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
  }
  .drawer-header span { color: var(--accent); }
  .drawer-link {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 24px;
    color: var(--muted);
    font-size: 0.9rem;
    cursor: pointer;
    border-bottom: 1px solid var(--border);
    transition: all .2s;
    border-left: none;
    border-right: none;
    border-top: none;
    background: transparent;
    width: 100%;
    font-family: var(--font-body);
    text-align: left;
  }
  .drawer-link:hover { color: var(--text); background: var(--bg3); }
  .drawer-link.active { color: var(--accent); background: rgba(59,130,246,0.08); }

  /* ── 페이지 공통 ── */
  .page-wrap {
    position: relative;
    z-index: 1;
    max-width: 960px;
    margin: 0 auto;
    padding: 32px 20px 60px;
  }

  .page-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }
  .page-title {
    font-family: var(--font-display);
    font-size: 1.8rem;
    letter-spacing: -0.5px;
  }
  .page-title small {
    display: block;
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 300;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .header-right { text-align: right; }
  .date-badge {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--muted);
  }
  .refresh-btn {
    margin-top: 8px;
    padding: 7px 16px;
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 6px;
    font-family: var(--font-body);
    font-size: 0.8rem;
    cursor: pointer;
    transition: all .2s;
    display: block;
  }
  .refresh-btn:hover { background: var(--accent); color: #fff; }

  /* 로딩 */
  .loading-box {
    text-align: center;
    padding: 80px 0;
    color: var(--muted);
  }
  .spinner {
    width: 40px; height: 40px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .8s linear infinite;
    margin: 0 auto 20px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes livePulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }

  /* 경기 없음 */
  .no-games {
    text-align: center;
    padding: 80px 20px;
    color: var(--muted);
  }
  .no-games .icon { font-size: 3.5rem; margin-bottom: 16px; }
  .no-games .title { font-size: 1.1rem; color: var(--text); margin-bottom: 10px; font-weight: 500; }
  .no-games .sub { font-size: 0.85rem; line-height: 2; }

  /* 요약 바 */
  .summary {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 32px;
  }
  .summary-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .summary-card .val {
    font-family: var(--font-mono);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
  }
  .summary-card .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 4px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }

  /* 경기 카드 */
  .games { display: flex; flex-direction: column; gap: 16px; }
  .game-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    transition: transform .2s, border-color .2s;
    animation: fadeUp .4s ease both;
  }
  .game-card:hover { transform: translateY(-2px); border-color: rgba(59,130,246,0.3); }
  @keyframes fadeUp {
    from { opacity:0; transform: translateY(16px); }
    to   { opacity:1; transform: translateY(0); }
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: var(--bg3);
    border-bottom: 1px solid var(--border);
  }
  .stadium-info { font-size: 0.78rem; color: var(--muted); font-family: var(--font-mono); }
  .pick-badge {
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(59,130,246,0.15);
    color: var(--accent);
    border: 1px solid rgba(59,130,246,0.3);
  }
  .pick-badge.strong {
    background: rgba(245,158,11,0.15);
    color: var(--accent2);
    border-color: rgba(245,158,11,0.3);
  }
  .pick-badge.draw {
    background: rgba(100,116,139,0.15);
    color: var(--muted);
    border-color: rgba(100,116,139,0.2);
  }

  .card-body { padding: 20px; }

  .matchup {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
  }
  .team { text-align: center; }
  .team-name { font-family: var(--font-display); font-size: 1.6rem; letter-spacing: -0.5px; }
  .team-type { font-size: 0.68rem; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }
  .starter { margin-top: 6px; font-size: 0.8rem; color: var(--muted); }
  .era-tag {
    display: inline-block;
    margin-top: 4px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    background: var(--bg3);
    border: 1px solid var(--border);
  }
  .era-good { color: var(--green); border-color: rgba(16,185,129,0.3); }
  .era-bad  { color: var(--red);   border-color: rgba(239,68,68,0.3); }
  .era-avg  { color: var(--muted); }
  .vs-divider { font-family: var(--font-mono); font-size: 0.9rem; color: var(--border); font-weight: 700; }

  .prob-section { margin-top: 4px; }
  .prob-labels {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    margin-bottom: 6px;
  }
  .prob-home { color: var(--accent); }
  .prob-away { color: var(--accent2); }
  .gauge-wrap {
    height: 10px;
    background: var(--bg3);
    border-radius: 99px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .gauge-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), #60a5fa);
    transition: width 1s cubic-bezier(.4,0,.2,1);
    position: relative;
  }
  .gauge-fill::after {
    content: '';
    position: absolute;
    right: 0; top: 0; bottom: 0;
    width: 3px;
    background: #fff;
    border-radius: 99px;
    opacity: .6;
  }

  .detail-row { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
  .detail-chip {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    padding: 3px 10px;
    border-radius: 4px;
    background: var(--bg3);
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .detail-chip span { color: var(--text); }

  .model-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
  .model-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.75rem;
  }
  .model-box .model-lbl { color: var(--muted); font-size: 0.65rem; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
  .model-box .model-val { font-family: var(--font-mono); font-size: 0.95rem; font-weight: 700; }

  /* ── 적중률 페이지 ── */
  .accuracy-summary {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 32px;
  }
  .acc-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }
  .acc-card .val { font-family: var(--font-mono); font-size: 1.6rem; font-weight: 700; }
  .acc-card .lbl { font-size: 0.7rem; color: var(--muted); margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }
  .acc-blue  { color: var(--accent); }
  .acc-good  { color: var(--green); }
  .acc-warn  { color: var(--accent2); }
  .acc-bad   { color: var(--red); }

  /* 날짜별 기록 */
  .history-list { display: flex; flex-direction: column; gap: 12px; }
  .history-item {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .history-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    cursor: pointer;
    user-select: none;
    transition: background .2s;
  }
  .history-header:hover { background: var(--bg3); }
  .history-date { font-family: var(--font-mono); font-size: 0.85rem; color: var(--text); }
  .history-meta { display: flex; align-items: center; gap: 12px; }
  .accuracy-pill {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
  }
  .accuracy-pill.high    { background: rgba(16,185,129,0.15); color: var(--green);  border: 1px solid rgba(16,185,129,0.3); }
  .accuracy-pill.mid     { background: rgba(245,158,11,0.15); color: var(--accent2); border: 1px solid rgba(245,158,11,0.3); }
  .accuracy-pill.low     { background: rgba(239,68,68,0.15);  color: var(--red);    border: 1px solid rgba(239,68,68,0.3); }
  .accuracy-pill.pending { background: rgba(100,116,139,0.1); color: var(--muted);  border: 1px solid rgba(100,116,139,0.2); }
  .history-chevron { color: var(--muted); font-size: 0.7rem; transition: transform .2s; }
  .history-item.open .history-chevron { transform: rotate(180deg); }

  .history-filters {
    display: flex;
    gap: 8px;
    margin: 14px 0 12px;
    flex-wrap: wrap;
  }
  .hfilter-btn {
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--bg3);
    color: var(--muted);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .hfilter-btn:hover { border-color: var(--accent); color: var(--accent); }
  .hfilter-btn.active { background: rgba(59,130,246,0.15); border-color: var(--accent); color: var(--accent); font-weight: 600; }

  .history-pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-top: 18px;
    padding-bottom: 8px;
  }
  .hpage-btn {
    padding: 6px 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg3);
    color: var(--text);
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  .hpage-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .hpage-btn:disabled { opacity: 0.3; cursor: default; }
  .hpage-info { font-size: 0.8rem; color: var(--muted); font-family: var(--font-mono); min-width: 80px; text-align: center; }

  .history-games {
    display: none;
    padding: 0 16px 16px;
    border-top: 1px solid var(--border);
  }
  .history-item.open .history-games { display: block; }

  /* 경기 상세 카드 */
  .result-game {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-top: 10px;
    padding: 12px 14px;
    font-size: 0.82rem;
  }
  .result-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
  }
  .result-matchup {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text);
  }
  .result-badge {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    white-space: nowrap;
  }
  .result-badge.hit     { background: rgba(16,185,129,0.15); color: var(--green); }
  .result-badge.miss    { background: rgba(239,68,68,0.15);  color: var(--red); }
  .result-badge.pending { background: rgba(100,116,139,0.1); color: var(--muted); }

  .result-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 16px;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 4px;
  }
  .result-row span { white-space: nowrap; }
  .result-row strong { color: var(--text); }
  .result-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 8px 0;
  }
  .result-model-row {
    display: flex;
    gap: 16px;
    font-size: 0.75rem;
    color: var(--muted);
    flex-wrap: wrap;
  }
  .model-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .model-tag .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
  .dot-hit  { background: var(--green); }
  .dot-miss { background: var(--red); }
  .dot-pend { background: var(--muted); }

  /* 빈 상태 */
  .empty {
    text-align: center;
    padding: 60px 0;
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 2;
  }

  /* 푸터 */
  footer {
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 0.72rem;
    color: var(--muted);
    line-height: 2;
  }

  /* ── 가이드 페이지 ── */
  .guide-section {
    margin-bottom: 40px;
  }
  .guide-section-title {
    font-family: var(--font-display);
    font-size: 1.2rem;
    color: var(--text);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* 예측 흐름 */
  .flow-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 20px;
  }
  .flow-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    flex: 1;
    min-width: 120px;
  }
  .flow-box .flow-icon { font-size: 1.5rem; margin-bottom: 6px; }
  .flow-box .flow-label { font-size: 0.78rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
  .flow-box .flow-val { font-size: 0.9rem; font-weight: 600; margin-top: 4px; }
  .flow-arrow { color: var(--muted); font-size: 1.2rem; flex-shrink: 0; }

  /* 가중치 바 */
  .weight-bar-wrap { margin-top: 12px; }
  .weight-bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 6px;
    font-family: var(--font-mono);
  }
  .weight-bar {
    height: 12px;
    border-radius: 99px;
    overflow: hidden;
    background: var(--bg3);
    border: 1px solid var(--border);
    display: flex;
  }
  .weight-stat { background: #a78bfa; height: 100%; transition: width .8s ease; }
  .weight-ml   { background: #34d399; height: 100%; transition: width .8s ease; }

  /* 용어 카드 그리드 */
  .term-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  .term-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
  }
  .term-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 8px;
  }
  .term-name {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
  }
  .term-ko {
    font-size: 0.78rem;
    color: var(--muted);
  }
  .term-desc {
    font-size: 0.82rem;
    color: var(--text);
    line-height: 1.7;
  }
  .term-scale {
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .scale-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
  }
  .scale-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .scale-dot.green  { background: var(--green); }
  .scale-dot.gray   { background: var(--muted); }
  .scale-dot.red    { background: var(--red); }
  .scale-dot.blue   { background: var(--accent); }
  .scale-dot.yellow { background: var(--accent2); }
  .scale-label { color: var(--muted); flex: 1; }
  .scale-value { color: var(--text); font-family: var(--font-mono); font-size: 0.72rem; }

  /* 색상 의미 */
  .color-legend-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }
  .color-legend-item {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .legend-sample {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 700;
    width: fit-content;
  }
  .legend-meaning { font-size: 0.78rem; color: var(--text); line-height: 1.6; }
  .legend-muted { font-size: 0.72rem; color: var(--muted); }

  /* 팁 박스 */
  .tip-box {
    background: rgba(59,130,246,0.06);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: var(--text);
    line-height: 1.8;
    margin-top: 12px;
  }
  .tip-box strong { color: var(--accent); }

  /* ── 반응형 ── */
  @media (max-width: 640px) {
    .hamburger { display: flex; }
    .nav-links { display: none; }
    .summary { grid-template-columns: repeat(2, 1fr); }
    .accuracy-summary { grid-template-columns: repeat(2, 1fr); }
    .team-name { font-size: 1.1rem; }
    .result-matchup { font-size: 0.84rem; }
    .term-grid { grid-template-columns: 1fr; }
    .color-legend-grid { grid-template-columns: 1fr; }
    .flow-wrap { flex-direction: column; }
    .flow-arrow { transform: rotate(90deg); }
    .card-body { padding: 14px 12px; }
    .matchup { gap: 6px; margin-bottom: 14px; }
    .starter { font-size: 0.74rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
    .prob-labels { font-size: 0.85rem; }
    .gauge-wrap { height: 12px; }
    .model-compare { grid-template-columns: 1fr; gap: 5px; }
    .model-box { display: flex; flex-direction: row; align-items: center; justify-content: space-between; padding: 8px 10px; }
    .model-box .model-lbl { margin-bottom: 0; font-size: 0.63rem; }
    .model-box .model-val { font-size: 0.8rem; }
    .detail-row { gap: 6px; }
    .detail-chip { font-size: 0.69rem; padding: 4px 8px; }
    .ev-cards { flex-direction: column; }
    .pick-badge { font-size: 0.68rem; max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  }

  /* ── EV 계산기 ── */
  .ev-section {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }
  .ev-toggle {
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    padding: 6px 2px;
    font-size: 0.74rem;
    color: var(--muted);
    user-select: none;
    transition: color 0.15s;
  }
  .ev-toggle:active { opacity: 0.7; }
  .ev-toggle-icon {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    transition: transform 0.25s;
    line-height: 1;
  }
  .ev-toggle-icon.open { transform: rotate(45deg); }
  .ev-body {
    overflow: hidden;
    max-height: 0;
    opacity: 0;
    transition: max-height 0.3s ease, opacity 0.25s;
  }
  .ev-body.open {
    max-height: 800px;
    opacity: 1;
  }
  .ev-cards {
    display: flex;
    gap: 8px;
  }
  .ev-card {
    flex: 1;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: border-color 0.25s;
  }
  .ev-card.ev-positive { border-color: rgba(16,185,129,0.45); }
  .ev-card.ev-negative { border-color: rgba(239,68,68,0.35); }
  .ev-card-header {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.82rem;
    font-weight: 600;
  }
  .ev-card-badge {
    font-size: 0.62rem;
    color: var(--muted);
    background: var(--bg);
    padding: 1px 5px;
    border-radius: 3px;
  }
  .ev-input {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 1rem;
    padding: 7px 8px;
    text-align: center;
    width: 100%;
    outline: none;
    transition: border-color 0.2s;
  }
  .ev-input:focus { border-color: var(--accent); }
  .ev-input::placeholder { color: var(--muted); font-size: 0.8rem; }
  .ev-card-stats {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 0.8rem;
  }
  .ev-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .ev-stat-label { color: var(--muted); font-size: 0.72rem; }
  .ev-card-ev {
    font-family: var(--font-mono);
    font-size: 1.05rem;
    font-weight: 700;
    text-align: center;
    padding: 7px 4px;
    border-radius: 6px;
    background: var(--bg);
    color: var(--muted);
    transition: all 0.2s;
    margin-top: 2px;
  }
  .ev-card-ev.positive {
    background: rgba(16,185,129,0.12);
    color: var(--green);
    border: 1px solid rgba(16,185,129,0.25);
  }
  .ev-card-ev.negative {
    background: rgba(239,68,68,0.10);
    color: var(--red);
    border: 1px solid rgba(239,68,68,0.2);
  }
  .ev-vig-row {
    font-size: 0.70rem;
    color: var(--muted);
    text-align: center;
    margin-top: 2px;
    display: none;
  }
  .ev-edge-pos { color: var(--green); font-weight: 700; }
  .ev-edge-neg { color: var(--red); }

  /* ── 실시간 탭 ── */
  .live-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
  }
  .live-card.live-active { border-color: rgba(239,68,68,0.4); }
  .live-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .live-card-meta { font-size: 0.8rem; color: var(--muted); }
  .live-card-status { display: flex; align-items: center; gap: 8px; }
  .live-inning { font-size: 0.78rem; color: var(--muted); }
  .live-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 700;
  }
  .live-badge.badge-live { background: #ef4444; color: #fff; animation: livePulse 1.5s infinite; }
  .live-badge.badge-done { background: var(--bg3); color: var(--muted); font-weight: 400; }
  .live-badge.badge-pre  { background: rgba(59,130,246,0.2); color: var(--accent); font-weight: 400; }
  .live-matchup {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .live-team { text-align: center; flex: 1; min-width: 0; }
  .live-team-logo { width: 40px; height: 40px; object-fit: contain; margin-bottom: 6px; display: block; margin: 0 auto 6px; }
  .live-team-name { font-weight: 700; font-size: 1rem; }
  .live-team-starter { font-size: 0.72rem; color: var(--muted); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .live-score-area { text-align: center; min-width: 110px; flex-shrink: 0; }
  .live-vs { font-size: 2rem; font-weight: 800; letter-spacing: 4px; color: var(--muted); }
  .live-scores { display: flex; align-items: center; gap: 14px; justify-content: center; }
  .live-score { font-size: 2.2rem; font-weight: 800; color: var(--text); }
  .live-score.winner { color: var(--green); }
  .live-score-sep { font-size: 1rem; color: var(--muted); }
  .live-pitcher {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }
  .live-pitcher-label { font-size: 0.7rem; color: var(--muted); }
  @media (max-width: 640px) {
    .live-card { padding: 12px 14px; }
    .live-score-area { min-width: 80px; }
    .live-score { font-size: 1.7rem; }
    .live-vs { font-size: 1.5rem; letter-spacing: 2px; }
    .live-scores { gap: 10px; }
    .live-team-logo { width: 30px; height: 30px; }
    .live-pitcher { font-size: 0.7rem; }
  }

  /* ── 팀순위 페이지 ── */
  .standings-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .standings-table th {
    padding: 10px 8px;
    text-align: center;
    color: var(--muted);
    font-weight: 500;
    font-size: 0.78rem;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  .standings-table th.left { text-align: left; }
  .standings-table td {
    padding: 10px 8px;
    text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    white-space: nowrap;
  }
  .standings-table td.left { text-align: left; }
  .standings-table tr:hover td { background: rgba(59,130,246,0.05); }
  .rank-num {
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 1rem;
    color: var(--muted);
  }
  .rank-num.top3 { color: var(--accent2); }
  .team-cell {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .team-logo {
    width: 28px;
    height: 28px;
    object-fit: contain;
    flex-shrink: 0;
  }
  .team-label { font-weight: 600; }
  .wra-val {
    font-family: var(--font-mono);
    font-weight: 700;
    color: var(--accent);
  }
  .gb-val {
    font-family: var(--font-mono);
    color: var(--muted);
  }
  .form-badges { display: flex; gap: 3px; justify-content: center; }
  .form-w {
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--green); color: #fff;
    font-size: 0.65rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
  }
  .form-l {
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--red); color: #fff;
    font-size: 0.65rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
  }
  .form-d {
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--muted); color: #fff;
    font-size: 0.65rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
  }
  .streak-badge {
    padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
  }
  .streak-w { background: rgba(16,185,129,0.15); color: var(--green); }
  .streak-l { background: rgba(239,68,68,0.15);  color: var(--red); }
  .streak-d { background: rgba(100,116,139,0.15); color: var(--muted); }
  .standings-updated {
    text-align: right; font-size: 0.72rem; color: var(--muted);
    margin-bottom: 12px;
  }
  @media (max-width: 600px) {
    .standings-table { font-size: 0.78rem; }
    .standings-table th, .standings-table td { padding: 8px 5px; }
    .team-logo { width: 22px; height: 22px; }
    .col-gb, .col-streak { display: none; }
  }
</style>
</head>
<body>

<!-- ── 네비게이션 바 ── -->
<nav class="topnav">
  <div class="nav-inner">
    <div class="nav-logo" onclick="showPage('today')">KBO<span>PICK</span></div>
    <div class="nav-links">
      <button class="nav-link active" id="nav-today" onclick="showPage('today')">🏟️ 오늘 예측</button>
      <button class="nav-link" id="nav-live" onclick="showPage('live')">⚡ 실시간</button>
      <button class="nav-link" id="nav-history" onclick="showPage('history')">📊 적중률</button>
      <button class="nav-link" id="nav-standings" onclick="showPage('standings')">🏆 팀순위</button>
      <button class="nav-link" id="nav-guide" onclick="showPage('guide')">📖 이용 가이드</button>
    </div>
    <button class="hamburger" id="hamburger" onclick="toggleDrawer()">
      <span></span><span></span><span></span>
    </button>
  </div>
</nav>

<!-- 모바일 드로어 -->
<div class="drawer-overlay" id="drawer-overlay" onclick="closeDrawer()"></div>
<div class="drawer" id="drawer">
  <div class="drawer-header">KBO<span>PICK</span></div>
  <button class="drawer-link active" id="drawer-today"
    onclick="showPage('today'); closeDrawer()">🏟️ 오늘 예측</button>
  <button class="drawer-link" id="drawer-live"
    onclick="showPage('live'); closeDrawer()">⚡ 실시간</button>
  <button class="drawer-link" id="drawer-history"
    onclick="showPage('history'); closeDrawer()">📊 적중률</button>
  <button class="drawer-link" id="drawer-standings"
    onclick="showPage('standings'); closeDrawer()">🏆 팀순위</button>
  <button class="drawer-link" id="drawer-guide"
    onclick="showPage('guide'); closeDrawer()">📖 이용 가이드</button>
</div>

<!-- ── 오늘 예측 페이지 ── -->
<div class="page-wrap" id="page-today">
  <div class="page-header">
    <div class="page-title">오늘 예측<small>Today's Predictions</small></div>
    <div class="header-right">
      <div class="date-badge" id="today-date">—</div>
      <button class="refresh-btn" onclick="refreshData()">⟳ 새로고침</button>
    </div>
  </div>

  <div id="loading" class="loading-box">
    <div class="spinner"></div>
    데이터 불러오는 중...
  </div>

  <div id="content" style="display:none">
    <div class="summary" id="summary"></div>
    <div class="games" id="games"></div>
  </div>

  <footer>
    데이터 출처: 네이버 스포츠 · 예측은 참고용입니다. 투자 판단은 본인 책임입니다.<br>
    KBO PICK — 앙상블 모델 (XGBoost + Random Forest + 통계 기반)
  </footer>
</div>

<!-- ── 적중률 페이지 ── -->
<div class="page-wrap" id="page-history" style="display:none">
  <div class="page-header">
    <div class="page-title">적중률<small>Prediction Accuracy</small></div>
    <div class="header-right">
      <button class="refresh-btn" id="update-btn" onclick="updateResults()">⟳ 결과 업데이트</button>
    </div>
  </div>

  <div id="loading-history" class="loading-box">
    <div class="spinner"></div>
    기록 불러오는 중...
  </div>

  <div id="history-content" style="display:none">
    <div class="accuracy-summary" id="accuracy-summary"></div>
    <div class="history-filters">
      <button class="hfilter-btn active" onclick="setHistoryFilter('all')">전체</button>
      <button class="hfilter-btn" onclick="setHistoryFilter('high')">고적중 ≥70%</button>
      <button class="hfilter-btn" onclick="setHistoryFilter('low')">저적중 &lt;50%</button>
      <button class="hfilter-btn" onclick="setHistoryFilter('pending')">집계 중</button>
    </div>
    <div class="history-list" id="history-list"></div>
    <div class="history-pagination" id="history-pagination"></div>
  </div>

  <footer>
    예측 vs 실제 경기 결과를 비교합니다. 오늘 경기는 경기 종료 후 업데이트됩니다.<br>
    KBO PICK — 앙상블 모델 (XGBoost + Random Forest + 통계 기반)
  </footer>
</div>

<!-- ── 이용 가이드 페이지 ── -->
<!-- ── 실시간 점수 페이지 ── -->
<div class="page-wrap" id="page-live" style="display:none">
  <div class="page-header">
    <div class="page-title">실시간 점수<small>Live Scores</small></div>
    <div style="font-size:0.8rem;color:var(--muted);margin-top:4px" id="live-updated"></div>
  </div>
  <div id="live-container" style="display:flex;flex-direction:column;gap:12px;max-width:700px;margin:0 auto"></div>
</div>

<!-- ── 팀순위 페이지 ── -->
<div class="page-wrap" id="page-standings" style="display:none">
  <div class="page-header">
    <div class="page-title">팀순위<small>2026 KBO Regular Season</small></div>
    <div class="header-right">
      <button class="refresh-btn" onclick="loadStandings(true)">⟳ 새로고침</button>
    </div>
  </div>
  <div class="card">
    <div class="standings-updated" id="standings-updated"></div>
    <div id="standings-loading" class="loading-box" style="height:200px"></div>
    <div id="standings-body" style="display:none; overflow-x:auto">
      <table class="standings-table">
        <thead>
          <tr>
            <th style="width:36px">#</th>
            <th class="left">팀</th>
            <th>경기</th>
            <th>승</th>
            <th>무</th>
            <th>패</th>
            <th>승률</th>
            <th class="col-gb">게차</th>
            <th>최근 5경기</th>
            <th class="col-streak">연속</th>
          </tr>
        </thead>
        <tbody id="standings-rows"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="page-wrap" id="page-guide" style="display:none">
  <div class="page-header">
    <div class="page-title">이용 가이드<small>How It Works</small></div>
  </div>

  <!-- 예측 시스템 -->
  <div class="guide-section">
    <div class="guide-section-title">🔮 예측은 어떻게 만들어지나요?</div>
    <div class="flow-wrap">
      <div class="flow-box">
        <div class="flow-icon">📡</div>
        <div class="flow-label">데이터 수집</div>
        <div class="flow-val" style="color:var(--muted);font-size:0.78rem">선발투수·팀성적<br>최근폼·날씨</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-box">
        <div class="flow-icon">📊</div>
        <div class="flow-label">통계 모델</div>
        <div class="flow-val" style="color:#a78bfa">공식으로 계산</div>
      </div>
      <div class="flow-arrow">+</div>
      <div class="flow-box">
        <div class="flow-icon">🤖</div>
        <div class="flow-label">ML 모델</div>
        <div class="flow-val" style="color:#34d399">패턴 학습</div>
      </div>
      <div class="flow-arrow">→</div>
      <div class="flow-box" style="border-color:rgba(59,130,246,0.4)">
        <div class="flow-icon">🎯</div>
        <div class="flow-label">최종 예측</div>
        <div class="flow-val" style="color:var(--accent)">승률 %</div>
      </div>
    </div>
    <div class="weight-bar-wrap">
      <div class="weight-bar-labels">
        <span style="color:#a78bfa">📊 통계 모델</span>
        <span style="color:#34d399">🤖 ML 모델</span>
      </div>
      <div class="weight-bar">
        <div class="weight-stat" style="width:40%"></div>
        <div class="weight-ml"   style="width:60%"></div>
      </div>
      <div style="font-size:0.72rem;color:var(--muted);margin-top:6px">
        ※ 가중치는 두 모델의 최근 30경기 적중률에 따라 자동으로 조정됩니다
      </div>
    </div>
    <div class="tip-box">
      <strong>📊 통계 모델</strong>은 팀 승률·ERA·OPS 등을 정해진 공식으로 계산합니다.<br>
      <strong>🤖 ML 모델</strong>은 이번 시즌 실제 경기 결과를 학습해 패턴을 스스로 찾아냅니다.<br>
      두 모델이 <strong>같은 팀을 가리킬수록</strong> 신뢰도가 높습니다.
    </div>
  </div>

  <!-- 픽 배지 읽는 법 -->
  <div class="guide-section">
    <div class="guide-section-title">🏷️ 추천 배지 읽는 법</div>
    <div class="color-legend-grid">
      <div class="color-legend-item">
        <span class="legend-sample" style="background:rgba(245,158,11,0.15);color:#f59e0b;border:1px solid rgba(245,158,11,0.3)">★★★ 강추</span>
        <div class="legend-meaning">예측 신뢰도 <strong>65% 이상</strong></div>
        <div class="legend-muted">두 모델이 같은 팀을 강하게 지목</div>
      </div>
      <div class="color-legend-item">
        <span class="legend-sample" style="background:rgba(59,130,246,0.15);color:#3b82f6;border:1px solid rgba(59,130,246,0.3)">★★ 추천</span>
        <div class="legend-meaning">예측 신뢰도 <strong>55~65%</strong></div>
        <div class="legend-muted">한 팀이 유리하지만 변수 있음</div>
      </div>
      <div class="color-legend-item">
        <span class="legend-sample" style="background:rgba(59,130,246,0.15);color:#3b82f6;border:1px solid rgba(59,130,246,0.3)">★ 약추</span>
        <div class="legend-meaning">예측 신뢰도 <strong>50~55%</strong></div>
        <div class="legend-muted">미세하게 유리한 팀이 있음</div>
      </div>
      <div class="color-legend-item">
        <span class="legend-sample" style="background:rgba(100,116,139,0.15);color:#64748b;border:1px solid rgba(100,116,139,0.2)">⚡ 접전</span>
        <div class="legend-meaning">예측 신뢰도 <strong>50% 이하</strong></div>
        <div class="legend-muted">어느 팀이 이길지 불확실한 경기</div>
      </div>
      <div class="color-legend-item">
        <span class="legend-sample era-tag era-good" style="font-size:0.78rem;padding:3px 10px">ERA 2.85</span>
        <div class="legend-meaning">ERA <strong>초록색</strong> = 3.0 미만</div>
        <div class="legend-muted">에이스급 투수, 실점 가능성 낮음</div>
      </div>
      <div class="color-legend-item">
        <span class="legend-sample era-tag era-bad" style="font-size:0.78rem;padding:3px 10px">ERA 6.20</span>
        <div class="legend-meaning">ERA <strong>빨간색</strong> = 5.5 초과</div>
        <div class="legend-muted">부진한 투수, 실점 가능성 높음</div>
      </div>
    </div>
  </div>

  <!-- 용어 사전 -->
  <div class="guide-section">
    <div class="guide-section-title">📚 야구 용어 사전</div>
    <div class="term-grid">

      <div class="term-card">
        <div class="term-header">
          <span class="term-name">ERA</span>
          <span class="term-ko">평균자책점</span>
        </div>
        <div class="term-desc">투수가 9이닝 동안 내줄 것으로 예상되는 평균 실점 수. 야구에서 투수 실력을 나타내는 가장 기본적인 지표예요.</div>
        <div class="term-scale">
          <div class="scale-row"><div class="scale-dot green"></div><span class="scale-label">에이스급</span><span class="scale-value">3.0 미만</span></div>
          <div class="scale-row"><div class="scale-dot gray"></div><span class="scale-label">평균 수준</span><span class="scale-value">3.0 ~ 5.5</span></div>
          <div class="scale-row"><div class="scale-dot red"></div><span class="scale-label">부진</span><span class="scale-value">5.5 초과</span></div>
        </div>
      </div>

      <div class="term-card">
        <div class="term-header">
          <span class="term-name">WHIP</span>
          <span class="term-ko">이닝당 출루 허용</span>
        </div>
        <div class="term-desc">투수가 1이닝에 허용하는 안타와 볼넷의 합계. 주자를 얼마나 자주 내보내는지 나타내요. ERA와 함께 보면 더 정확해요.</div>
        <div class="term-scale">
          <div class="scale-row"><div class="scale-dot green"></div><span class="scale-label">매우 좋음</span><span class="scale-value">1.00 미만</span></div>
          <div class="scale-row"><div class="scale-dot gray"></div><span class="scale-label">평균</span><span class="scale-value">1.20 ~ 1.50</span></div>
          <div class="scale-row"><div class="scale-dot red"></div><span class="scale-label">불안정</span><span class="scale-value">1.60 초과</span></div>
        </div>
      </div>

      <div class="term-card">
        <div class="term-header">
          <span class="term-name">K9</span>
          <span class="term-ko">9이닝당 탈삼진</span>
        </div>
        <div class="term-desc">투수가 9이닝 동안 삼진으로 잡아내는 타자 수. 높을수록 타자를 압도하는 지배적인 투수예요.</div>
        <div class="term-scale">
          <div class="scale-row"><div class="scale-dot green"></div><span class="scale-label">지배적</span><span class="scale-value">9.0 이상</span></div>
          <div class="scale-row"><div class="scale-dot gray"></div><span class="scale-label">평균</span><span class="scale-value">6.0 ~ 9.0</span></div>
          <div class="scale-row"><div class="scale-dot red"></div><span class="scale-label">컨택형</span><span class="scale-value">6.0 미만</span></div>
        </div>
      </div>

      <div class="term-card">
        <div class="term-header">
          <span class="term-name">OPS</span>
          <span class="term-ko">출루율 + 장타율</span>
        </div>
        <div class="term-desc">타자의 종합 공격력 지표. 얼마나 자주 출루하고(OBP), 얼마나 강하게 치는지(SLG)를 합친 값이에요.</div>
        <div class="term-scale">
          <div class="scale-row"><div class="scale-dot green"></div><span class="scale-label">강력한 타선</span><span class="scale-value">0.800 이상</span></div>
          <div class="scale-row"><div class="scale-dot gray"></div><span class="scale-label">평균</span><span class="scale-value">0.700 ~ 0.800</span></div>
          <div class="scale-row"><div class="scale-dot red"></div><span class="scale-label">약한 타선</span><span class="scale-value">0.700 미만</span></div>
        </div>
      </div>

      <div class="term-card">
        <div class="term-header">
          <span class="term-name">승률</span>
          <span class="term-ko">시즌 승리 비율</span>
        </div>
        <div class="term-desc">이번 시즌 전체 경기 중 이긴 경기의 비율. 팀의 전반적인 전력을 나타내는 가장 직관적인 수치예요.</div>
        <div class="term-scale">
          <div class="scale-row"><div class="scale-dot green"></div><span class="scale-label">상위권</span><span class="scale-value">0.600 이상</span></div>
          <div class="scale-row"><div class="scale-dot gray"></div><span class="scale-label">중위권</span><span class="scale-value">0.450 ~ 0.600</span></div>
          <div class="scale-row"><div class="scale-dot red"></div><span class="scale-label">하위권</span><span class="scale-value">0.450 미만</span></div>
        </div>
      </div>

      <div class="term-card">
        <div class="term-header">
          <span class="term-name">상대전적</span>
          <span class="term-ko">올시즌 맞대결 승률</span>
        </div>
        <div class="term-desc">이번 시즌 오늘 두 팀이 붙었던 경기에서 홈팀이 이긴 비율. 같은 팀끼리는 특별한 궁합이 있을 수 있어요.</div>
        <div class="term-scale">
          <div class="scale-row"><div class="scale-dot green"></div><span class="scale-label">홈팀 강세</span><span class="scale-value">0.60 이상</span></div>
          <div class="scale-row"><div class="scale-dot gray"></div><span class="scale-label">팽팽</span><span class="scale-value">0.40 ~ 0.60</span></div>
          <div class="scale-row"><div class="scale-dot red"></div><span class="scale-label">원정팀 강세</span><span class="scale-value">0.40 미만</span></div>
        </div>
      </div>

      <div class="term-card">
        <div class="term-header">
          <span class="term-name">최근 폼</span>
          <span class="term-ko">최근 10경기 승률</span>
        </div>
        <div class="term-desc">시즌 전체가 아닌 최근 10경기만 기준으로 한 승률. 지금 잘 나가고 있는지(Hot) 혹은 부진한지(Cold) 나타내요.</div>
        <div class="term-scale">
          <div class="scale-row"><div class="scale-dot green"></div><span class="scale-label">핫 (상승세)</span><span class="scale-value">7승 이상</span></div>
          <div class="scale-row"><div class="scale-dot gray"></div><span class="scale-label">보통</span><span class="scale-value">4 ~ 6승</span></div>
          <div class="scale-row"><div class="scale-dot red"></div><span class="scale-label">콜드 (하락세)</span><span class="scale-value">3승 이하</span></div>
        </div>
      </div>

      <div class="term-card">
        <div class="term-header">
          <span class="term-name">연승 / 연패</span>
          <span class="term-ko">스트릭</span>
        </div>
        <div class="term-desc">현재 몇 경기 연속으로 이기거나 지고 있는지. 연승 중인 팀은 사기가 올라있고, 연패 중인 팀은 심리적으로 불리할 수 있어요.</div>
        <div class="term-scale">
          <div class="scale-row"><div class="scale-dot green"></div><span class="scale-label">연승</span><span class="scale-value">양수 (+)</span></div>
          <div class="scale-row"><div class="scale-dot red"></div><span class="scale-label">연패</span><span class="scale-value">음수 (−)</span></div>
        </div>
      </div>

    </div>
  </div>

  <!-- 예측 수치 읽는 법 -->
  <div class="guide-section">
    <div class="guide-section-title">📈 예측 수치 읽는 법</div>
    <div class="color-legend-grid" style="grid-template-columns: repeat(2,1fr)">
      <div class="color-legend-item">
        <div style="font-size:0.9rem;font-weight:600;color:var(--text);margin-bottom:4px">게이지 바</div>
        <div class="legend-meaning">파란 부분이 넓을수록 <strong>홈팀</strong> 유리. 오른쪽으로 갈수록 홈팀 승리 확률이 높아요.</div>
      </div>
      <div class="color-legend-item">
        <div style="font-size:0.9rem;font-weight:600;color:var(--text);margin-bottom:4px">두 모델 비교</div>
        <div class="legend-meaning">보라색(통계)·초록색(ML)이 같은 방향이면 <strong>신뢰도 높음</strong>. 다르면 불확실한 경기예요.</div>
      </div>
      <div class="color-legend-item">
        <div style="font-size:0.9rem;font-weight:600;color:var(--text);margin-bottom:4px">홈 어드밴티지</div>
        <div class="legend-meaning">KBO 역대 통계에서 홈팀 승률은 약 <strong>53%</strong>예요. 비슷한 전력이면 홈팀이 유리해요.</div>
      </div>
      <div class="color-legend-item">
        <div style="font-size:0.9rem;font-weight:600;color:var(--text);margin-bottom:4px">주의사항</div>
        <div class="legend-meaning">예측은 참고용입니다. 부상·돌발상황·심리전은 데이터에 반영되지 않아요.</div>
      </div>
    </div>
    <div class="tip-box">
      💡 <strong>가장 믿을 수 있는 픽</strong>은 두 모델이 같은 팀을 가리키고, 신뢰도가 60% 이상이며, 최근 폼도 좋은 경기예요.
    </div>
  </div>

  <footer>
    데이터 출처: 네이버 스포츠 · 예측은 참고용입니다. 투자 판단은 본인 책임입니다.<br>
    KBO PICK — 앙상블 모델 (XGBoost + Random Forest + 통계 기반)
  </footer>
</div>

<script>
// ── 페이지 전환 ──────────────────────────────────────────
function showPage(name) {
  ['today', 'live', 'history', 'standings', 'guide'].forEach(p => {
    document.getElementById('page-' + p).style.display = p === name ? 'block' : 'none';
    document.getElementById('nav-' + p).classList.toggle('active', p === name);
    const dl = document.getElementById('drawer-' + p);
    if (dl) dl.classList.toggle('active', p === name);
  });
  if (name === 'history' && !historyLoaded) loadHistory();
  if (name === 'live') { loadLive(); startLivePolling(); } else { stopLivePolling(); }
  if (name === 'standings') loadStandings();
}

// ── 모바일 드로어 ────────────────────────────────────────
function toggleDrawer() {
  const open = document.getElementById('drawer').classList.toggle('open');
  document.getElementById('drawer-overlay').classList.toggle('open', open);
  document.getElementById('hamburger').classList.toggle('open', open);
}
function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer-overlay').classList.remove('open');
  document.getElementById('hamburger').classList.remove('open');
}

// ── 팀순위 ────────────────────────────────────────────────
let _standingsLoaded = false;

function loadStandings(force) {
  if (_standingsLoaded && !force) return;
  document.getElementById('standings-loading').style.display = 'flex';
  document.getElementById('standings-body').style.display = 'none';

  fetch('/api/standings')
    .then(r => r.json())
    .then(data => {
      if (data.error) { document.getElementById('standings-loading').textContent = '데이터 오류'; return; }
      renderStandings(data);
      _standingsLoaded = true;
    })
    .catch(() => { document.getElementById('standings-loading').textContent = '불러오기 실패'; });
}

function renderStandings(teams) {
  const tbody = document.getElementById('standings-rows');
  tbody.innerHTML = '';
  teams.forEach(t => {
    const rankClass = t.ranking <= 3 ? 'top3' : '';
    const formHtml = t.last_five.split('').map(c => {
      if (c === 'W') return '<div class="form-w">W</div>';
      if (c === 'L') return '<div class="form-l">L</div>';
      return '<div class="form-d">D</div>';
    }).join('');

    const streakText = t.streak || '';
    const streakClass = streakText.includes('승') ? 'streak-w' : streakText.includes('패') ? 'streak-l' : 'streak-d';
    const gb = t.game_behind === 0 ? '-' : t.game_behind.toFixed(1);

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="rank-num ${rankClass}">${t.ranking}</span></td>
      <td class="left">
        <div class="team-cell">
          <img class="team-logo" src="${t.logo}" alt="${t.name}" onerror="this.style.display='none'">
          <span class="team-label">${t.name}</span>
        </div>
      </td>
      <td>${t.games}</td>
      <td style="color:var(--green);font-weight:600">${t.wins}</td>
      <td style="color:var(--muted)">${t.draws}</td>
      <td style="color:var(--red);font-weight:600">${t.losses}</td>
      <td><span class="wra-val">${(t.wra * 100).toFixed(1)}%</span></td>
      <td class="col-gb"><span class="gb-val">${gb}</span></td>
      <td><div class="form-badges">${formHtml}</div></td>
      <td class="col-streak"><span class="streak-badge ${streakClass}">${streakText}</span></td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('standings-loading').style.display = 'none';
  document.getElementById('standings-body').style.display = 'block';
  const now = new Date().toLocaleTimeString('ko-KR', {hour:'2-digit', minute:'2-digit'});
  document.getElementById('standings-updated').textContent = `업데이트: ${now}`;
}

// ── 실시간 점수 ───────────────────────────────────────────
let _liveTimer = null;

function startLivePolling() {
  if (_liveTimer) return;
  _liveTimer = setInterval(loadLive, 30000);
}
function stopLivePolling() {
  clearInterval(_liveTimer);
  _liveTimer = null;
}

function loadLive() {
  fetch('/api/live')
    .then(r => r.json())
    .then(data => renderLive(data))
    .catch(() => {});
}

function renderLive(games) {
  const box = document.getElementById('live-container');
  const upd = document.getElementById('live-updated');
  const now = new Date().toLocaleTimeString('ko-KR', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  upd.textContent = '마지막 갱신: ' + now + ' (30초마다 자동 갱신)';

  if (!games || games.length === 0) {
    box.innerHTML = '<div style="text-align:center;color:var(--muted);padding:60px 0">오늘 경기 정보가 없습니다</div>';
    return;
  }

  box.innerHTML = games.map(g => {
    const isLive = g.status === 'LIVE' || g.status === 'STARTED';
    const isDone = g.status === 'RESULT';
    const isPre  = !isLive && !isDone;

    const statusBadge = isLive
      ? '<span class="live-badge badge-live">● LIVE</span>'
      : isDone
      ? '<span class="live-badge badge-done">종료</span>'
      : '<span class="live-badge badge-pre">예정</span>';

    const inningInfo = g.status_info
      ? `<span class="live-inning">${g.status_info}</span>`
      : '';

    const homeWin = isDone && g.home_score > g.away_score;
    const awayWin = isDone && g.away_score > g.home_score;

    const scoreBlock = isPre
      ? '<div class="live-vs">vs</div>'
      : `<div class="live-scores">
           <div class="live-score${homeWin ? ' winner' : ''}">${g.home_score}</div>
           <div class="live-score-sep">:</div>
           <div class="live-score${awayWin ? ' winner' : ''}">${g.away_score}</div>
         </div>`;

    const homeLogo  = g.home_emblem ? `<img class="live-team-logo" src="${g.home_emblem}" onerror="this.style.display='none'">` : '';
    const awayLogo  = g.away_emblem ? `<img class="live-team-logo" src="${g.away_emblem}" onerror="this.style.display='none'">` : '';

    const pitcherInfo = (g.home_pitcher || g.away_pitcher)
      ? `<div class="live-pitcher">
           <span>${g.home_pitcher || '-'}</span>
           <span class="live-pitcher-label">현재 투수</span>
           <span>${g.away_pitcher || '-'}</span>
         </div>`
      : '';

    return `<div class="live-card${isLive ? ' live-active' : ''}">
      <div class="live-card-header">
        <span class="live-card-meta">${g.stadium} · ${g.game_time}</span>
        <div class="live-card-status">${inningInfo}${statusBadge}</div>
      </div>
      <div class="live-matchup">
        <div class="live-team">
          ${homeLogo}
          <div class="live-team-name">${g.home_team}</div>
          <div class="live-team-starter">선발: ${g.home_starter}</div>
        </div>
        <div class="live-score-area">${scoreBlock}</div>
        <div class="live-team">
          ${awayLogo}
          <div class="live-team-name">${g.away_team}</div>
          <div class="live-team-starter">선발: ${g.away_starter}</div>
        </div>
      </div>
      ${pitcherInfo}
    </div>`;
  }).join('');
}

// ── EV 토글 ───────────────────────────────────────────────
function toggleEV(idx) {
  const body = document.getElementById('ev-body-' + idx);
  const icon = document.getElementById('ev-icon-' + idx);
  body.classList.toggle('open');
  icon.classList.toggle('open');
}

// ── EV 계산기 ─────────────────────────────────────────────
function calcEV(idx, homeProb, awayProb) {
  const homeOdds  = parseFloat(document.getElementById('ev-home-' + idx).value);
  const awayOdds  = parseFloat(document.getElementById('ev-away-' + idx).value);
  const homeEv    = document.getElementById('ev-res-home-' + idx);
  const awayEv    = document.getElementById('ev-res-away-' + idx);
  const homeCard  = document.getElementById('ev-card-home-' + idx);
  const awayCard  = document.getElementById('ev-card-away-' + idx);
  const homeStats = document.getElementById('ev-stats-home-' + idx);
  const awayStats = document.getElementById('ev-stats-away-' + idx);
  const vigRow    = document.getElementById('ev-vig-' + idx);

  function setEv(evEl, cardEl, prob, odds) {
    if (!odds || odds < 1) {
      evEl.textContent = '—';
      evEl.className = 'ev-card-ev';
      cardEl.className = 'ev-card';
      return;
    }
    const ev  = (prob * odds) - 1;
    const pct = (ev * 100).toFixed(1);
    const cls = ev >= 0 ? 'positive' : 'negative';
    evEl.textContent = (ev >= 0 ? '+' : '') + pct + '%';
    evEl.className = 'ev-card-ev ' + cls;
    cardEl.className = 'ev-card ev-' + cls;
  }

  setEv(homeEv, homeCard, homeProb, homeOdds);
  setEv(awayEv, awayCard, awayProb, awayOdds);

  if (homeOdds >= 1 && awayOdds >= 1) {
    const hImpl  = 1 / homeOdds;
    const aImpl  = 1 / awayOdds;
    const over   = hImpl + aImpl;
    const hFair  = hImpl / over;
    const aFair  = aImpl / over;
    const vig    = ((over - 1) * 100).toFixed(1);
    const hEdge  = ((homeProb - hFair) * 100).toFixed(1);
    const aEdge  = ((awayProb - aFair) * 100).toFixed(1);

    function statsHtml(fairPct, modelPct, edge) {
      const cls = edge >= 0 ? 'ev-edge-pos' : 'ev-edge-neg';
      return `<div class="ev-stat-row"><span class="ev-stat-label">북마커 공정확률</span><span>${fairPct}%</span></div>` +
             `<div class="ev-stat-row"><span class="ev-stat-label">모델 예측확률</span><span>${modelPct}%</span></div>` +
             `<div class="ev-stat-row"><span class="ev-stat-label">엣지</span><span class="${cls}">${edge >= 0 ? '+' : ''}${edge}%</span></div>`;
    }

    homeStats.innerHTML = statsHtml((hFair*100).toFixed(1), (homeProb*100).toFixed(1), hEdge);
    awayStats.innerHTML = statsHtml((aFair*100).toFixed(1), (awayProb*100).toFixed(1), aEdge);
    vigRow.textContent  = `북마커 마진 ${vig}%`;
    vigRow.style.display = 'block';
  } else {
    homeStats.innerHTML = '';
    awayStats.innerHTML = '';
    vigRow.style.display = 'none';
  }
}

// ── 유틸 ─────────────────────────────────────────────────
function eraClass(era) {
  if (era < 3.0) return 'era-good';
  if (era > 5.5) return 'era-bad';
  return 'era-avg';
}
function pickClass(pick) {
  if (pick.includes('강추')) return 'strong';
  if (pick.includes('접전')) return 'draw';
  return '';
}

// ── 오늘 예측 ────────────────────────────────────────────
function renderSummary(games, weights) {
  const strong = games.filter(g => Math.max(g.final_home_prob, g.final_away_prob) >= 0.60).length;
  const avg = games.reduce((s, g) => s + Math.max(g.final_home_prob, g.final_away_prob), 0) / games.length;
  const wLabel = weights && weights.based_on >= 10
    ? `통계 ${(weights.stat*100).toFixed(0)}% · ML ${(weights.ml*100).toFixed(0)}%`
    : '기본값';
  const wSub = weights && weights.based_on >= 10
    ? `최근 ${weights.based_on}경기 기반`
    : '데이터 축적 중';
  document.getElementById('summary').innerHTML = `
    <div class="summary-card"><div class="val">${games.length}</div><div class="lbl">오늘 경기 수</div></div>
    <div class="summary-card"><div class="val">${strong}</div><div class="lbl">추천 경기 (60%+)</div></div>
    <div class="summary-card"><div class="val">${(avg*100).toFixed(1)}%</div><div class="lbl">평균 신뢰도</div></div>
    <div class="summary-card"><div class="val" style="font-size:1rem;line-height:1.4">${wLabel}</div><div class="lbl">${wSub}</div></div>
  `;
}

function renderGame(g, idx) {
  const hp = (g.final_home_prob * 100).toFixed(1);
  const ap = (g.final_away_prob * 100).toFixed(1);
  const sh = (g.stat_home_prob  * 100).toFixed(1);
  const mh = (g.ml_home_prob    * 100).toFixed(1);
  const sa = (g.stat_away_prob  * 100).toFixed(1);
  const ma = (g.ml_away_prob    * 100).toFixed(1);
  const pick = g.pick || '';
  return `
  <div class="game-card" style="animation-delay:${idx*0.08}s">
    <div class="card-header">
      <span class="stadium-info">${g.game_time} · ${g.stadium}</span>
      <span class="pick-badge ${pickClass(pick)}">${pick}</span>
    </div>
    <div class="card-body">
      <div class="matchup">
        <div class="team">
          <div class="team-type">원정</div>
          <div class="team-name">${g.away_team}</div>
          <div class="starter">${g.away_starter}</div>
          <div class="era-tag ${eraClass(g.away_era)}">ERA ${g.away_era.toFixed(2)}</div>
        </div>
        <div class="vs-divider">VS</div>
        <div class="team">
          <div class="team-type">홈</div>
          <div class="team-name">${g.home_team}</div>
          <div class="starter">${g.home_starter}</div>
          <div class="era-tag ${eraClass(g.home_era)}">ERA ${g.home_era.toFixed(2)}</div>
        </div>
      </div>
      <div class="prob-section">
        <div class="prob-labels">
          <span class="prob-home">원정 ${ap}%</span>
          <span class="prob-away">홈 ${hp}%</span>
        </div>
        <div class="gauge-wrap"><div class="gauge-fill" style="width:${hp}%"></div></div>
      </div>
      <div class="model-compare">
        <div class="model-box">
          <div class="model-lbl">📊 통계 기반</div>
          <div class="model-val" style="color:#a78bfa">홈 ${sh}% / 원정 ${sa}%</div>
        </div>
        <div class="model-box">
          <div class="model-lbl">🤖 ML 모델</div>
          <div class="model-val" style="color:#34d399">홈 ${mh}% / 원정 ${ma}%</div>
        </div>
      </div>
      <div class="detail-row">
        <div class="detail-chip">홈승률 <span>${(g.home_win_rate*100).toFixed(1)}%</span></div>
        <div class="detail-chip">원정승률 <span>${(g.away_win_rate*100).toFixed(1)}%</span></div>
        <div class="detail-chip">상대전적 <span>${(g.h2h_home_rate*100).toFixed(0)}%</span></div>
        ${g.rain_prob > 0 ? `<div class="detail-chip">강수확률 <span>${(g.rain_prob*100).toFixed(0)}%</span></div>` : ''}
      </div>
      <div class="ev-section">
        <div class="ev-toggle" onclick="toggleEV(${idx})">
          <span>💰 배당률 입력 · EV 계산기</span>
          <span class="ev-toggle-icon" id="ev-icon-${idx}">＋</span>
        </div>
        <div class="ev-body" id="ev-body-${idx}">
          <div class="ev-cards">
            <div class="ev-card" id="ev-card-away-${idx}">
              <div class="ev-card-header">${g.away_team} <span class="ev-card-badge">원정</span></div>
              <input class="ev-input" id="ev-away-${idx}" type="number" step="0.01" min="1" placeholder="예) 1.85"
                oninput="calcEV(${idx}, ${g.final_home_prob}, ${g.final_away_prob})">
              <div class="ev-card-stats" id="ev-stats-away-${idx}"></div>
              <div class="ev-card-ev" id="ev-res-away-${idx}">—</div>
            </div>
            <div class="ev-card" id="ev-card-home-${idx}">
              <div class="ev-card-header">${g.home_team} <span class="ev-card-badge">홈</span></div>
              <input class="ev-input" id="ev-home-${idx}" type="number" step="0.01" min="1" placeholder="예) 2.10"
                oninput="calcEV(${idx}, ${g.final_home_prob}, ${g.final_away_prob})">
              <div class="ev-card-stats" id="ev-stats-home-${idx}"></div>
              <div class="ev-card-ev" id="ev-res-home-${idx}">—</div>
            </div>
          </div>
          <div class="ev-vig-row" id="ev-vig-${idx}"></div>
        </div>
      </div>
    </div>
  </div>`;
}

async function loadData() {
  document.getElementById('loading').className = 'loading-box';
  document.getElementById('loading').innerHTML = '<div class="spinner"></div>데이터 불러오는 중...';
  document.getElementById('loading').style.display = 'block';
  document.getElementById('content').style.display = 'none';
  try {
    const res = await fetch('/api/predictions');
    const data = await res.json();
    if (data.error) {
      document.getElementById('loading').innerHTML =
        `<div class="empty">⚠️ ${data.error}<br><small>${data.detail || ''}</small></div>`;
      return;
    }
    document.getElementById('today-date').textContent = data.date;
    if (data.no_games) {
      document.getElementById('loading').innerHTML = `
        <div class="no-games">
          <div class="icon">🏟️</div>
          <div class="title">오늘은 KBO 경기가 없습니다</div>
          <div class="sub">경기가 없는 날입니다.<br>다음 경기 날 다시 방문해 주세요!</div>
        </div>`;
      return;
    }
    renderSummary(data.games, data.weights);
    document.getElementById('games').innerHTML = data.games.map((g, i) => renderGame(g, i)).join('');
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
  } catch(e) {
    document.getElementById('loading').innerHTML =
      `<div class="empty">⚠️ 서버 연결 실패<br><small>${e.message}</small></div>`;
  }
}

async function refreshData() {
  await fetch('/api/refresh');
  historyLoaded = false;
  loadData();
}

// ── 적중률 페이지 ─────────────────────────────────────────
let historyLoaded  = false;
let _historyAll    = [];
let _historyFilter = 'all';
let _historyPage   = 0;
const HISTORY_PER_PAGE = 7;

function accPillClass(rate) {
  if (rate >= 0.65) return 'high';
  if (rate >= 0.50) return 'mid';
  return 'low';
}

function renderAccuracySummary(stats) {
  const rc = stats.total_games > 0
    ? (stats.accuracy >= 0.65 ? 'acc-good' : stats.accuracy >= 0.50 ? 'acc-warn' : 'acc-bad')
    : 'acc-blue';
  const accText = stats.total_games > 0 ? (stats.accuracy * 100).toFixed(1) + '%' : '—';
  document.getElementById('accuracy-summary').innerHTML = `
    <div class="acc-card"><div class="val acc-blue">${stats.days_recorded}</div><div class="lbl">기록된 날짜</div></div>
    <div class="acc-card"><div class="val acc-blue">${stats.total_games}</div><div class="lbl">총 예측 경기</div></div>
    <div class="acc-card"><div class="val acc-good">${stats.total_hits}</div><div class="lbl">적중</div></div>
    <div class="acc-card"><div class="val ${rc}">${accText}</div><div class="lbl">전체 적중률</div></div>
  `;
}

function renderHistoryItem(entry) {
  const has = entry.results_recorded;
  const pillClass = has ? accPillClass(entry.accuracy) : 'pending';
  const pillText  = has
    ? `${entry.hits}/${entry.total} · ${(entry.accuracy*100).toFixed(0)}%`
    : '집계 중';

  const gamesHtml = (entry.games || []).map(g => {
    const predTeam  = g.predicted_winner === 'HOME' ? g.home_team : g.away_team;
    const predProb  = g.predicted_winner === 'HOME'
      ? (g.final_home_prob * 100).toFixed(1)
      : (g.final_away_prob * 100).toFixed(1);
    const actualTeam = g.actual_winner
      ? (g.actual_winner === 'HOME' ? g.home_team : g.away_team)
      : null;

    let bc = 'pending', bt = '대기 중';
    if (g.hit === true)  { bc = 'hit';  bt = '✓ 적중'; }
    if (g.hit === false) { bc = 'miss'; bt = '✗ 빗나감'; }

    // 점수
    const hasScore = g.actual_home_score !== undefined && g.actual_home_score !== null;
    const scoreHtml = hasScore
      ? `<span>점수: <strong>${g.home_team} ${g.actual_home_score} : ${g.actual_away_score} ${g.away_team}</strong></span>`
      : '';

    // 실제 승자
    const actualHtml = actualTeam
      ? `<span>실제 승리: <strong>${actualTeam}</strong></span>`
      : '';

    // 통계/ML 모델별 적중
    const statTeam = g.stat_predicted_winner === 'HOME' ? g.home_team : g.away_team;
    const mlTeam   = g.ml_predicted_winner   === 'HOME' ? g.home_team : g.away_team;

    const statDot = g.stat_hit === true ? 'dot-hit' : g.stat_hit === false ? 'dot-miss' : 'dot-pend';
    const mlDot   = g.ml_hit   === true ? 'dot-hit' : g.ml_hit   === false ? 'dot-miss' : 'dot-pend';
    const statLabel = g.stat_hit === true ? '적중' : g.stat_hit === false ? '빗나감' : '대기';
    const mlLabel   = g.ml_hit   === true ? '적중' : g.ml_hit   === false ? '빗나감' : '대기';

    const modelHtml = (g.stat_predicted_winner || g.ml_predicted_winner) ? `
      <hr class="result-divider">
      <div class="result-model-row">
        <span class="model-tag"><span class="dot ${statDot}"></span>통계모델: ${statTeam} · ${statLabel}</span>
        <span class="model-tag"><span class="dot ${mlDot}"></span>ML모델: ${mlTeam} · ${mlLabel}</span>
      </div>` : '';

    return `
      <div class="result-game">
        <div class="result-top">
          <div class="result-matchup">${g.away_team} vs ${g.home_team}</div>
          <div class="result-badge ${bc}">${bt}</div>
        </div>
        <div class="result-row">
          <span>예측: <strong>${predTeam}</strong> (${predProb}%)</span>
          ${actualHtml}
          ${scoreHtml}
        </div>
        ${modelHtml}
      </div>`;
  }).join('');

  const gameCount = (entry.games || []).length;
  const hint = gameCount > 0 ? `<span style="font-size:0.72rem;color:var(--muted);">${gameCount}경기 ▼</span>` : '';

  return `
    <div class="history-item" onclick="this.classList.toggle('open')">
      <div class="history-header">
        <div class="history-date">${entry.date_str} ${hint}</div>
        <div class="history-meta">
          <span class="accuracy-pill ${pillClass}">${pillText}</span>
          <span class="history-chevron">▼</span>
        </div>
      </div>
      <div class="history-games">${gamesHtml}</div>
    </div>`;
}

async function updateResults() {
  const btn = document.getElementById('update-btn');
  btn.textContent = '업데이트 중...';
  btn.disabled = true;
  try {
    const res = await fetch('/api/update-results');
    const data = await res.json();
    historyLoaded = false;
    await loadHistory();
    btn.textContent = data.message ? `✓ ${data.message}` : '✓ 완료';
  } catch(e) {
    btn.textContent = '⚠️ 오류 발생';
  } finally {
    setTimeout(() => { btn.textContent = '⟳ 결과 업데이트'; btn.disabled = false; }, 3000);
  }
}

function setHistoryFilter(f) {
  _historyFilter = f;
  _historyPage   = 0;
  document.querySelectorAll('.hfilter-btn').forEach(b => b.classList.remove('active'));
  const map = { all: 0, high: 1, low: 2, pending: 3 };
  const btns = document.querySelectorAll('.hfilter-btn');
  if (btns[map[f]]) btns[map[f]].classList.add('active');
  renderHistoryPage();
}

function renderHistoryPage() {
  const filtered = _historyAll.filter(e => {
    if (_historyFilter === 'all')     return true;
    if (_historyFilter === 'pending') return !e.results_recorded;
    if (_historyFilter === 'high')    return e.results_recorded && e.accuracy >= 0.7;
    if (_historyFilter === 'low')     return e.results_recorded && e.accuracy < 0.5;
    return true;
  });
  const total = filtered.length;
  const pages = Math.max(1, Math.ceil(total / HISTORY_PER_PAGE));
  _historyPage = Math.min(_historyPage, pages - 1);
  const slice = filtered.slice(_historyPage * HISTORY_PER_PAGE, (_historyPage + 1) * HISTORY_PER_PAGE);

  document.getElementById('history-list').innerHTML = slice.length
    ? slice.map(e => renderHistoryItem(e)).join('')
    : '<div class="empty" style="padding:40px 0">해당 조건의 기록이 없습니다</div>';

  document.getElementById('history-pagination').innerHTML = pages <= 1 ? '' : `
    <button class="hpage-btn" onclick="_historyPage--;renderHistoryPage()" ${_historyPage === 0 ? 'disabled' : ''}>‹ 이전</button>
    <span class="hpage-info">${_historyPage + 1} / ${pages}</span>
    <button class="hpage-btn" onclick="_historyPage++;renderHistoryPage()" ${_historyPage >= pages - 1 ? 'disabled' : ''}>다음 ›</button>
  `;
}

async function loadHistory() {
  document.getElementById('loading-history').style.display = 'block';
  document.getElementById('history-content').style.display = 'none';
  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    if (data.error) {
      document.getElementById('loading-history').innerHTML =
        `<div class="empty">⚠️ ${data.error}</div>`;
      return;
    }
    if (!data.entries || data.entries.length === 0) {
      document.getElementById('loading-history').innerHTML =
        `<div class="empty">📭 아직 기록된 예측이 없습니다.<br>오늘 예측을 먼저 확인해 보세요!</div>`;
      return;
    }
    renderAccuracySummary(data.stats);
    _historyAll    = data.entries;
    _historyFilter = 'all';
    _historyPage   = 0;
    document.querySelectorAll('.hfilter-btn').forEach((b, i) => b.classList.toggle('active', i === 0));
    renderHistoryPage();
    document.getElementById('loading-history').style.display = 'none';
    document.getElementById('history-content').style.display = 'block';
    historyLoaded = true;
  } catch(e) {
    document.getElementById('loading-history').innerHTML =
      `<div class="empty">⚠️ 서버 연결 실패<br><small>${e.message}</small></div>`;
  }
}

// 날짜 표시
const now = new Date();
document.getElementById('today-date').textContent =
  `${now.getFullYear()}.${String(now.getMonth()+1).padStart(2,'0')}.${String(now.getDate()).padStart(2,'0')} (${['일','월','화','수','목','금','토'][now.getDay()]})`;

loadData();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


# 데이터 캐시 (날짜별로 메모리에 저장)
def run_pipeline(date: str) -> list:
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    if base not in sys.path:
        sys.path.insert(0, base)
    os.chdir(base)

    raw_path  = os.path.join(base, f"kbo_raw_{date}.csv")
    proc_path = os.path.join(base, f"kbo_processed_{date}.csv")
    pred_path = os.path.join(base, f"kbo_prediction_{date}.csv")

    if not os.path.exists(raw_path):
        from kbo_collector import collect_today_data
        collect_today_data(date)

    if not os.path.exists(proc_path):
        from kbo_preprocessor import preprocess
        preprocess(date)

    from kbo_model import run as model_run
    stat_w, ml_w, based_on = get_adaptive_weights()
    model_run(date, stat_weight=stat_w, ml_weight=ml_w)

    if not os.path.exists(pred_path):
        return []

    df = pd.read_csv(pred_path, encoding="utf-8-sig")
    games = []
    for _, row in df.iterrows():
        games.append({
            "home_team":      str(row.get("home_team", "")),
            "away_team":      str(row.get("away_team", "")),
            "home_starter":   str(row.get("home_starter", "미정")),
            "away_starter":   str(row.get("away_starter", "미정")),
            "stadium":        str(row.get("stadium", "")),
            "game_time":      str(row.get("game_time", "")),
            "home_era":       float(row.get("home_era", 4.5)),
            "away_era":       float(row.get("away_era", 4.5)),
            "home_win_rate":  float(row.get("home_win_rate", 0.5)),
            "away_win_rate":  float(row.get("away_win_rate", 0.5)),
            "h2h_home_rate":  float(row.get("h2h_home_rate", 0.5)),
            "rain_prob":      float(row.get("rain_prob", 0)),
            "stat_home_prob": float(row.get("stat_home_prob", 0.5)),
            "stat_away_prob": float(row.get("stat_away_prob", 0.5)),
            "ml_home_prob":   float(row.get("ml_home_prob", 0.5)),
            "ml_away_prob":   float(row.get("ml_away_prob", 0.5)),
            "final_home_prob":float(row.get("final_home_prob", 0.5)),
            "final_away_prob":float(row.get("final_away_prob", 0.5)),
            "pick":           str(row.get("pick", "")),
        })
    return games


@app.route("/api/predictions")
def predictions():
    date = now_kst().strftime("%Y%m%d")
    date_str = now_kst().strftime("%Y.%m.%d")

    if date in _cache:
        games = _cache[date]
        if not games:
            return jsonify({"date": date_str, "games": [], "no_games": True})
        stat_w, ml_w, based_on = get_adaptive_weights()
        return jsonify({
            "date": date_str,
            "games": games,
            "weights": {"stat": stat_w, "ml": ml_w, "based_on": based_on},
        })

    try:
        stat_w, ml_w, based_on = get_adaptive_weights()
        games = run_pipeline(date)
        _cache[date] = games
        if not games:
            return jsonify({"date": date_str, "games": [], "no_games": True})
        save_predictions(date, games)
        return jsonify({
            "date": date_str,
            "games": games,
            "weights": {"stat": stat_w, "ml": ml_w, "based_on": based_on},
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "detail": traceback.format_exc()[-500:]})


@app.route("/api/history")
def history():
    update_actual_results()
    log = load_accuracy_log()

    entries = []
    total_hits, total_games = 0, 0

    for date in sorted(log.keys(), reverse=True):
        entry = dict(log[date])
        entry["date"] = date
        entries.append(entry)
        if entry.get("results_recorded"):
            total_hits  += entry.get("hits", 0)
            total_games += entry.get("total", 0)

    accuracy = round(total_hits / total_games, 4) if total_games > 0 else 0

    return jsonify({
        "entries": entries,
        "stats": {
            "days_recorded": len(entries),
            "total_games":   total_games,
            "total_hits":    total_hits,
            "accuracy":      accuracy,
        }
    })


@app.route("/api/debug")
def debug():
    info = {"db_type": "sqlite", "db_path": _DB_PATH}
    conn = get_db_conn()
    try:
        info["db_connected"] = True
        info["row_count"] = conn.execute("SELECT COUNT(*) FROM accuracy_log").fetchone()[0]
        rows = conn.execute(
            "SELECT date, results_recorded, total_games FROM accuracy_log ORDER BY date DESC LIMIT 5"
        ).fetchall()
        info["recent_rows"] = [{"date": r[0], "recorded": bool(r[1]), "games": r[2]} for r in rows]
    except Exception as e:
        info["db_connected"] = False
        info["db_error"] = str(e)

    # 네이버 API 결과 진단
    try:
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from kbo_model import fetch_season_results
        results_df = fetch_season_results()
        if results_df.empty:
            info["naver_api"] = "empty - 결과 없음"
        else:
            results_df["date_key"] = results_df["date"].str.replace("-", "")
            dates_in_api = sorted(results_df["date_key"].unique().tolist())
            info["naver_api"] = "ok"
            info["naver_total_games"] = len(results_df)
            info["naver_latest_dates"] = dates_in_api[-5:]
    except Exception as e:
        info["naver_api"] = f"오류: {e}"

    # DB 원본 값 직접 조회
    try:
        raw_rows = conn.execute(
            "SELECT date, results_recorded, total_games, games FROM accuracy_log"
        ).fetchall()
        info["raw_db_rows"] = [
            {
                "date": r[0],
                "results_recorded": bool(r[1]),
                "total_games": r[2],
                "games_preview": (r[3] or "")[:200],
            }
            for r in raw_rows
        ]
    except Exception as e:
        info["raw_db_error"] = str(e)

    # load_accuracy_log 결과 확인
    log = load_accuracy_log()
    info["log_keys"] = list(log.keys())
    for d, v in log.items():
        info[f"log_{d}"] = {
            "results_recorded": v.get("results_recorded"),
            "games_len": len(v.get("games") or []),
            "date_lt_today": d < now_kst().strftime("%Y%m%d"),
        }

    return jsonify(info)


@app.route("/api/update-results")
def api_update_results():
    diag = []
    try:
        # 1단계: DB 로드
        log = load_accuracy_log()
        diag.append(f"1. DB 로드: {len(log)}일치 데이터")

        today = now_kst().strftime("%Y%m%d")
        pending = [d for d, v in log.items()
                   if not v.get("results_recorded") and v.get("games") and d < today]
        diag.append(f"2. 미집계 날짜: {pending}")
        if not pending:
            return jsonify({"status": "ok", "message": "업데이트할 날짜 없음", "diag": diag})

        # 2단계: Naver API
        import sys
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from kbo_model import fetch_season_results
        results_df = fetch_season_results()
        diag.append(f"3. Naver API: {len(results_df)}경기 수집")
        if results_df.empty:
            return jsonify({"status": "error", "message": "Naver API 결과 없음", "diag": diag})

        results_df["date_key"] = results_df["date"].str.replace("-", "")

        # 3단계: 날짜별 매칭
        for date in pending:
            day_df = results_df[results_df["date_key"] == date]
            diag.append(f"4. {date} API경기수: {len(day_df)}")
            if day_df.empty:
                continue

            saved_games = log[date].get("games", [])
            diag.append(f"5. {date} 저장된예측수: {len(saved_games)}")

            hits, total = 0, 0
            for g in saved_games:
                match = day_df[
                    (day_df["home_team"] == g["home_team"]) &
                    (day_df["away_team"] == g["away_team"])
                ]
                diag.append(f"   {g['away_team']}@{g['home_team']} → 매칭: {'O' if not match.empty else 'X'}")
                if match.empty:
                    continue
                row = match.iloc[0]
                actual = "HOME" if row["home_win"] == 1 else "AWAY"
                g["actual_winner"]     = actual
                g["actual_home_score"] = int(row["home_score"])
                g["actual_away_score"] = int(row["away_score"])
                g["hit"]      = (g["predicted_winner"] == actual)
                g["stat_hit"] = (g.get("stat_predicted_winner") == actual)
                g["ml_hit"]   = (g.get("ml_predicted_winner")   == actual)
                total += 1
                if g["hit"]: hits += 1

            diag.append(f"6. total={total}, hits={hits}")
            if total > 0:
                log[date]["results_recorded"] = True
                log[date]["hits"]     = hits
                log[date]["total"]    = total
                log[date]["accuracy"] = round(hits / total, 4)

        # 4단계: DB 저장
        save_accuracy_log(log)
        diag.append("7. DB 저장 완료")

        recorded = sum(1 for v in log.values() if v.get("results_recorded"))
        return jsonify({"status": "ok", "message": f"{recorded}일 기록됨", "diag": diag})

    except Exception as e:
        import traceback
        diag.append(f"오류: {e}")
        return jsonify({"status": "error", "message": str(e),
                        "detail": traceback.format_exc()[-400:], "diag": diag})


@app.route("/api/live")
def live_scores():
    import requests as req
    from datetime import datetime as dt
    today = now_kst()
    date_str = today.strftime("%Y-%m-%d")
    url = (
        "https://api-gw.sports.naver.com/schedule/games"
        "?fields=basic%2Cschedule%2Cbaseball%2CmanualRelayUrl"
        "&upperCategoryId=kbaseball&categoryId=kbo"
        f"&fromDate={date_str}&toDate={date_str}"
        "&roundCodes=&size=50"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 11; SM-G998B) AppleWebKit/537.36",
        "Origin": "https://m.sports.naver.com",
        "Referer": "https://m.sports.naver.com/kbaseball/schedule/index",
    }
    try:
        r = req.get(url, headers=headers, timeout=8)
        games = r.json().get("result", {}).get("games", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = []
    for g in games:
        game_time_raw = g.get("gameDateTime", "")
        game_time = game_time_raw[11:16] if len(game_time_raw) >= 16 else ""
        status = g.get("statusCode", "SCHEDULE")
        result.append({
            "game_id":      g.get("gameId", ""),
            "home_team":    g.get("homeTeamName", ""),
            "away_team":    g.get("awayTeamName", ""),
            "home_score":   g.get("homeTeamScore") if g.get("homeTeamScore") is not None else "-",
            "away_score":   g.get("awayTeamScore") if g.get("awayTeamScore") is not None else "-",
            "status":       status,
            "status_info":  g.get("statusInfo", ""),
            "game_time":    game_time,
            "stadium":      g.get("stadium", ""),
            "home_starter": g.get("homeStarterName") or "미정",
            "away_starter": g.get("awayStarterName") or "미정",
            "home_pitcher": g.get("homeCurrentPitcherName", ""),
            "away_pitcher": g.get("awayCurrentPitcherName", ""),
            "home_emblem":  g.get("homeTeamEmblemUrl", ""),
            "away_emblem":  g.get("awayTeamEmblemUrl", ""),
            "winner":       g.get("winner", ""),
        })
    return jsonify(result)


@app.route("/api/standings")
def standings():
    import requests as req
    season = now_kst().year
    url = (
        f"https://api-gw.sports.naver.com/statistics/categories/kbo"
        f"/seasons/{season}/teams?fields=basic&gameType=REGULAR_SEASON&type=rank"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin": "https://m.sports.naver.com",
        "Referer": "https://m.sports.naver.com/kbaseball/schedule/index",
        "Accept": "application/json",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "x-sports-backend": "kotlin",
    }
    try:
        r = req.get(url, headers=headers, timeout=8)
        r.raise_for_status()
        teams = r.json().get("result", {}).get("seasonTeamStats", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = []
    for t in sorted(teams, key=lambda x: x.get("ranking", 99)):
        result.append({
            "ranking":    int(t.get("ranking", 0)),
            "name":       t.get("teamShortName", t.get("teamName", "")),
            "logo":       t.get("teamImageUrl", ""),
            "games":      int(t.get("gameCount", 0)),
            "wins":       int(t.get("winGameCount", 0)),
            "draws":      int(t.get("drawnGameCount", 0)),
            "losses":     int(t.get("loseGameCount", 0)),
            "wra":        float(t.get("wra", 0)),
            "game_behind": float(t.get("gameBehind", 0)),
            "last_five":  t.get("lastFiveGames", ""),
            "streak":     t.get("continuousGameResult", ""),
        })
    return jsonify(result)


@app.route("/api/refresh")
def refresh():
    date = now_kst().strftime("%Y%m%d")
    _cache.pop(date, None)
    import glob
    for f in glob.glob(os.path.join(BASE_DIR, f"kbo_*_{date}.csv")):
        try: os.remove(f)
        except: pass
    for f in ["kbo_model.pkl", "kbo_scaler.pkl"]:
        fpath = os.path.join(BASE_DIR, f)
        try: os.remove(fpath)
        except: pass
    return jsonify({"status": "ok", "message": "캐시 초기화 완료"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("  KBO PICK 대시보드 시작")
    print(f"  브라우저에서 → http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
