"""
KBO 토토 예측 프로그램 - 3단계: 앙상블 모델
=============================================
① 이번 시즌 과거 경기 결과로 XGBoost + Random Forest 학습
② 오늘 경기 피처에 적용 → ML 예측 승률
③ 통계 기반(2단계) + ML 기반 앙상블 → 최종 승률

실행 방법:
  python kbo_model.py
  python kbo_model.py --date 20260423
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from kbo_preprocessor import (
    fill_missing, engineer_features,
    calc_stat_win_prob, ML_FEATURES
)

# ── 공통 헤더 ────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/144.0.0.0 Safari/537.36"
    ),
    "Origin":         "https://m.sports.naver.com",
    "Referer":        "https://m.sports.naver.com/kbaseball/schedule/index",
    "Accept-Language":"ko-KR,ko;q=0.9",
    "Accept":         "application/json, text/plain, */*",
    "charset":        "utf-8",
    "x-sports-backend": "kotlin",
}

MODEL_PATH   = "kbo_model.pkl"
SCALER_PATH  = "kbo_scaler.pkl"
HISTORY_PATH = "kbo_history.csv"


# ════════════════════════════════════════════════════════
# 1. 과거 경기 결과 수집 (이번 시즌 전체)
# ════════════════════════════════════════════════════════
def fetch_season_results(season: int = None) -> pd.DataFrame:
    """
    이번 시즌 완료된 경기 결과를 수집합니다.
    각 경기에서: 홈팀, 원정팀, 점수, 선발투수, 승패를 가져옵니다.
    """
    if season is None:
        season = datetime.now().year

    print(f"[학습데이터] {season}시즌 경기 결과 수집 중...")

    from_date = f"{season}-03-01"
    to_date   = datetime.now().strftime("%Y-%m-%d")

    url = (
        "https://api-gw.sports.naver.com/schedule/games"
        "?fields=basic%2Cschedule%2Cbaseball%2CmanualRelayUrl"
        "&upperCategoryId=kbaseball&categoryId=kbo"
        f"&fromDate={from_date}&toDate={to_date}"
        "&roundCodes=&size=1000"
    )

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        games = resp.json().get("result", {}).get("games", [])
    except Exception as e:
        print(f"[오류] 시즌 경기 수집 실패: {e}")
        return pd.DataFrame()

    rows = []
    for g in games:
        # 완료된 경기만
        if g.get("statusCode") != "RESULT":
            continue
        # 취소/우천 제외
        if g.get("cancel") or g.get("suspended"):
            continue

        winner = g.get("winner", "")
        if winner not in ("HOME", "AWAY"):
            continue

        rows.append({
            "date":           g.get("gameDate", ""),
            "game_id":        g.get("gameId", ""),
            "home_team":      g.get("homeTeamName", ""),
            "away_team":      g.get("awayTeamName", ""),
            "home_team_code": g.get("homeTeamCode", ""),
            "away_team_code": g.get("awayTeamCode", ""),
            "home_score":     int(g.get("homeTeamScore", 0)),
            "away_score":     int(g.get("awayTeamScore", 0)),
            "home_starter":   g.get("homeStarterName") or "미정",
            "away_starter":   g.get("awayStarterName") or "미정",
            "stadium":        g.get("stadium", ""),
            "home_win":       1 if winner == "HOME" else 0,
        })

    df = pd.DataFrame(rows)
    print(f"[학습데이터] {len(df)}경기 수집 완료")
    return df


# ════════════════════════════════════════════════════════
# 2. 과거 경기에 피처 추가 (누적 통계 기반)
# ════════════════════════════════════════════════════════
def build_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 경기 시점까지의 누적 통계를 계산해서 피처로 만듭니다.
    (미래 데이터 사용 방지 — 해당 경기 이전 기록만 사용)
    투수 ERA는 해당 시점까지의 실제 실점 기반으로 계산합니다.
    """
    print("[피처] 누적 통계 기반 피처 생성 중...")

    df = df.sort_values("date").reset_index(drop=True)

    # 투수별 선발 기록을 미리 인덱싱 (성능 최적화)
    # {pitcher_name: [(date, runs_allowed, won), ...]} — 날짜순 정렬
    pitcher_starts: dict = {}
    for _, row in df.iterrows():
        for role, score_col, win_val in [
            ("home_starter", "away_score", 1),
            ("away_starter", "home_score", 0),
        ]:
            name = row[role]
            if not name or name == "미정":
                continue
            runs = row[score_col]
            won  = int(row["home_win"] == win_val)
            if name not in pitcher_starts:
                pitcher_starts[name] = []
            pitcher_starts[name].append((row["date"], runs, won))

    def pitcher_stats_at(name: str, before_date: str, n_recent: int = 5) -> dict:
        """해당 날짜 이전까지의 투수 누적 실적 반환."""
        default = {"era_proxy": 4.50, "recent_runs": 4.50, "recent_winrate": 0.50}
        if not name or name == "미정":
            return default
        starts = [s for s in pitcher_starts.get(name, []) if s[0] < before_date]
        if not starts:
            return default
        runs_list = [s[1] for s in starts]
        recent    = starts[-n_recent:]
        return {
            "era_proxy":        round(sum(runs_list) / len(runs_list), 2),
            "recent_runs":      round(sum(s[1] for s in recent) / len(recent), 2),
            "recent_winrate":   round(sum(s[2] for s in recent) / len(recent), 3),
        }

    rows = []
    for idx, row in df.iterrows():
        game_date = row["date"]
        home_code = row["home_team_code"]
        away_code = row["away_team_code"]

        # 해당 경기 이전 기록만 사용
        past = df.iloc[:idx]

        def recent_form(code, n=10):
            team_past = past[
                (past["home_team_code"] == code) | (past["away_team_code"] == code)
            ].tail(n)
            if len(team_past) < 3:
                return {"last10_winrate": 0.5, "last5_scored": 4.5, "last5_allowed": 4.5, "streak": 0}

            wins = 0
            for _, tg in team_past.iterrows():
                is_home = tg["home_team_code"] == code
                if (is_home and tg["home_win"] == 1) or (not is_home and tg["home_win"] == 0):
                    wins += 1

            last5 = team_past.tail(5)
            scored5, allowed5 = [], []
            for _, tg in last5.iterrows():
                is_home = tg["home_team_code"] == code
                scored5.append(tg["home_score"] if is_home else tg["away_score"])
                allowed5.append(tg["away_score"] if is_home else tg["home_score"])

            streak = 0
            for _, tg in team_past.iloc[::-1].iterrows():
                is_home = tg["home_team_code"] == code
                won = (is_home and tg["home_win"] == 1) or (not is_home and tg["home_win"] == 0)
                if streak == 0:
                    streak = 1 if won else -1
                elif (streak > 0 and won) or (streak < 0 and not won):
                    streak += 1 if won else -1
                else:
                    break

            return {
                "last10_winrate": round(wins / len(team_past), 4),
                "last5_scored":   round(sum(scored5) / len(scored5), 2) if scored5 else 4.5,
                "last5_allowed":  round(sum(allowed5) / len(allowed5), 2) if allowed5 else 4.5,
                "streak":         streak,
            }

        def team_stats(code):
            home_games = past[past["home_team_code"] == code]
            away_games = past[past["away_team_code"] == code]
            wins = (
                len(home_games[home_games["home_win"] == 1]) +
                len(away_games[away_games["home_win"] == 0])
            )
            total = len(home_games) + len(away_games)
            if total < 3:
                return {"win_rate": 0.5, "avg_score": 4.5, "avg_allow": 4.5}
            avg_score = (home_games["home_score"].sum() + away_games["away_score"].sum()) / total
            avg_allow = (home_games["away_score"].sum() + away_games["home_score"].sum()) / total
            return {
                "win_rate":  round(wins / total, 4),
                "avg_score": round(avg_score, 2),
                "avg_allow": round(avg_allow, 2),
            }

        def h2h_rate(hc, ac):
            h2h = past[
                ((past["home_team_code"] == hc) & (past["away_team_code"] == ac)) |
                ((past["home_team_code"] == ac) & (past["away_team_code"] == hc))
            ]
            if len(h2h) == 0:
                return 0.5
            home_wins = len(h2h[
                ((h2h["home_team_code"] == hc) & (h2h["home_win"] == 1)) |
                ((h2h["away_team_code"] == hc) & (h2h["home_win"] == 0))
            ])
            return round(home_wins / len(h2h), 4)

        home_s    = team_stats(home_code)
        away_s    = team_stats(away_code)
        home_form = recent_form(home_code)
        away_form = recent_form(away_code)

        # 투수 실적 (해당 경기 이전까지 실제 데이터)
        home_pit = pitcher_stats_at(row["home_starter"], game_date)
        away_pit = pitcher_stats_at(row["away_starter"], game_date)

        feat = {
            "date":           game_date,
            "game_id":        row["game_id"],
            "home_team":      row["home_team"],
            "away_team":      row["away_team"],
            "home_team_code": home_code,
            "away_team_code": away_code,
            "home_starter":   row["home_starter"],
            "away_starter":   row["away_starter"],
            "stadium":        row["stadium"],
            "game_time":      "",
            # 팀 성적
            "home_win_rate":  home_s["win_rate"],
            "away_win_rate":  away_s["win_rate"],
            "home_avg_score": home_s["avg_score"],
            "away_avg_score": away_s["avg_score"],
            "home_avg_allow": home_s["avg_allow"],
            "away_avg_allow": away_s["avg_allow"],
            "h2h_home_rate":  h2h_rate(home_code, away_code),
            # 투수 ERA — 실제 누적 실점 기반 (기존 고정값 4.50 대체)
            "home_era":  home_pit["era_proxy"],
            "away_era":  away_pit["era_proxy"],
            "home_whip": 1.40,
            "away_whip": 1.40,
            "home_k9":   7.00,
            "away_k9":   7.00,
            # 선발투수 최근 폼 (새 피처)
            "home_starter_recent_runs":    home_pit["recent_runs"],
            "away_starter_recent_runs":    away_pit["recent_runs"],
            "home_starter_recent_winrate": home_pit["recent_winrate"],
            "away_starter_recent_winrate": away_pit["recent_winrate"],
            "home_starter_starts":         len([s for s in pitcher_starts.get(row["home_starter"], []) if s[0] < game_date]),
            "away_starter_starts":         len([s for s in pitcher_starts.get(row["away_starter"], []) if s[0] < game_date]),
            # 날씨 (학습 데이터엔 없으므로 중립값)
            "rain_prob": 0.0,
            "temp":      18.0,
            "humidity":  60.0,
            # 팀 최근 폼
            "home_last10_winrate": home_form["last10_winrate"],
            "away_last10_winrate": away_form["last10_winrate"],
            "home_last5_scored":   home_form["last5_scored"],
            "away_last5_scored":   away_form["last5_scored"],
            "home_last5_allowed":  home_form["last5_allowed"],
            "away_last5_allowed":  away_form["last5_allowed"],
            "home_streak":         home_form["streak"],
            "away_streak":         away_form["streak"],
            # 정답 레이블
            "home_win": row["home_win"],
        }
        rows.append(feat)

        if (idx + 1) % 30 == 0:
            print(f"  {idx+1}/{len(df)} 경기 처리 중...", end="\r")

    print(f"\n[피처] {len(rows)}경기 피처 생성 완료 (투수 실적 반영)")
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════
# 3. 모델 학습
# ════════════════════════════════════════════════════════
def train_model(df_feat: pd.DataFrame):
    """
    XGBoost + Random Forest 앙상블 모델을 학습합니다.
    """
    # 피처 준비
    df_feat = fill_missing(df_feat)
    df_feat = engineer_features(df_feat)

    # 유효한 행만 사용 (경기 수 부족한 초반 제외)
    df_valid = df_feat.dropna(subset=ML_FEATURES + ["home_win"])

    if len(df_valid) < 20:
        print(f"[경고] 학습 데이터 부족 ({len(df_valid)}경기) → 통계 모델만 사용")
        return None, None

    X = df_valid[ML_FEATURES].values
    y = df_valid["home_win"].values.astype(int)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    # 소프트 보팅 앙상블
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft",
        weights=[0.6, 0.4],   # XGBoost 비중 높게
    )

    # 확률 보정 (Platt scaling)
    calibrated = CalibratedClassifierCV(ensemble, cv=3, method="sigmoid")
    calibrated.fit(X_scaled, y)

    # 교차검증 정확도
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring="accuracy")
    print(f"[모델] 교차검증 정확도: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
    print(f"[모델] 학습 데이터: {len(X)}경기")

    # 피처 중요도 출력
    ensemble.fit(X_scaled, y)
    xgb_imp = ensemble.estimators_[0].feature_importances_
    print("\n[피처 중요도 Top 5]")
    imp_df = pd.DataFrame({"feature": ML_FEATURES, "importance": xgb_imp})
    imp_df = imp_df.sort_values("importance", ascending=False).head(5)
    for _, r in imp_df.iterrows():
        bar = "█" * int(r["importance"] * 50)
        print(f"  {r['feature']:<25} {bar} {r['importance']:.3f}")

    # 모델 저장
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(calibrated, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n[저장] {MODEL_PATH}, {SCALER_PATH}")

    return calibrated, scaler


# ════════════════════════════════════════════════════════
# 4. 오늘 경기 예측
# ════════════════════════════════════════════════════════
def predict_today(date: str = None, stat_weight: float = 0.4, ml_weight: float = 0.6) -> pd.DataFrame:
    """
    전처리된 오늘 경기 데이터에 ML 모델을 적용합니다.
    """
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    processed_path = f"kbo_processed_{date}.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"'{processed_path}' 없음. kbo_preprocessor.py 먼저 실행하세요."
        )

    df = pd.read_csv(processed_path, encoding="utf-8-sig")

    # 통계 기반 승률은 이미 있음 (2단계에서 계산)
    if "stat_home_prob" not in df.columns:
        df = calc_stat_win_prob(df)

    # ML 모델 로드
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        X = df[ML_FEATURES].fillna(0).values
        X_scaled = scaler.transform(X)
        ml_probs = model.predict_proba(X_scaled)[:, 1]

        df["ml_home_prob"]   = ml_probs.round(4)
        df["ml_away_prob"]   = (1 - ml_probs).round(4)

        # 앙상블: 적응형 가중치 적용
        df["final_home_prob"] = (
            df["stat_home_prob"] * stat_weight +
            df["ml_home_prob"]   * ml_weight
        ).round(4)
        df["final_away_prob"] = (1 - df["final_home_prob"]).round(4)
        print(f"[예측] ML 모델 적용 완료 (통계 {stat_weight:.0%} + ML {ml_weight:.0%})")

    else:
        # 모델 없으면 통계 기반만 사용
        df["ml_home_prob"]    = df["stat_home_prob"]
        df["ml_away_prob"]    = df["stat_away_prob"]
        df["final_home_prob"] = df["stat_home_prob"]
        df["final_away_prob"] = df["stat_away_prob"]
        print("[예측] 학습된 모델 없음 → 통계 기반만 사용")

    # 추천 픽 (60% 이상이면 강추천)
    def pick_label(prob):
        if prob >= 0.65: return "★★★ 강추"
        if prob >= 0.55: return "★★  추천"
        if prob >= 0.50: return "★   약추"
        return "⚡  접전"

    df["pick"] = df.apply(
        lambda r: (
            f"홈({r['home_team']}) {pick_label(r['final_home_prob'])}"
            if r["final_home_prob"] >= r["final_away_prob"]
            else f"원정({r['away_team']}) {pick_label(r['final_away_prob'])}"
        ), axis=1
    )

    # 결과 저장
    result_path = f"kbo_prediction_{date}.csv"
    df.to_csv(result_path, index=False, encoding="utf-8-sig")

    return df


# ════════════════════════════════════════════════════════
# 5. 결과 출력
# ════════════════════════════════════════════════════════
def print_predictions(df: pd.DataFrame):
    print(f"\n{'═'*65}")
    print(f"  🏟  KBO 오늘의 예측 — {df['date'].iloc[0]}")
    print(f"{'═'*65}")

    for _, row in df.iterrows():
        print(f"\n  {row['away_team']} @ {row['home_team']}  ({row['game_time']}, {row['stadium']})")
        print(f"  선발: {row['away_team']} {row['away_starter']}(ERA {row['away_era']:.2f})"
              f"  vs  {row['home_team']} {row['home_starter']}(ERA {row['home_era']:.2f})")
        print(f"  {'─'*55}")
        print(f"  홈({row['home_team']:5})  통계:{row['stat_home_prob']:.1%}  ML:{row['ml_home_prob']:.1%}  "
              f"최종: {row['final_home_prob']:.1%}")
        print(f"  원정({row['away_team']:4})  통계:{row['stat_away_prob']:.1%}  ML:{row['ml_away_prob']:.1%}  "
              f"최종: {row['final_away_prob']:.1%}")
        print(f"  📌 {row['pick']}")

    print(f"\n{'═'*65}")


# ════════════════════════════════════════════════════════
# 6. 전체 실행
# ════════════════════════════════════════════════════════
def run(date: str = None, stat_weight: float = 0.4, ml_weight: float = 0.6):
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    print(f"\n{'='*50}")
    print(f" KBO ML 모델 — {date}")
    print(f"{'='*50}")

    # ── 학습 데이터 수집 + 모델 학습 ──────────────────
    # 모델이 없거나 오늘 학습 안 됐으면 새로 학습
    model_fresh = False
    if os.path.exists(MODEL_PATH):
        mtime = os.path.getmtime(MODEL_PATH)
        if datetime.fromtimestamp(mtime).date() == datetime.now().date():
            model_fresh = True
            print("[모델] 오늘 이미 학습된 모델 사용")

    if not model_fresh:
        season = datetime.now().year
        hist_path = f"kbo_history_{season}.csv"

        # 과거 경기 결과 수집 (캐시 활용)
        if os.path.exists(hist_path):
            df_hist = pd.read_csv(hist_path, encoding="utf-8-sig")
            print(f"[캐시] 과거 경기 {len(df_hist)}경기 로드")
        else:
            df_hist = fetch_season_results(season)
            if len(df_hist) > 0:
                df_hist.to_csv(hist_path, index=False, encoding="utf-8-sig")

        if len(df_hist) >= 20:
            df_feat = build_training_features(df_hist)
            train_model(df_feat)
        else:
            print(f"[경고] 학습 데이터 부족 ({len(df_hist)}경기) → 통계 모델만 사용")

    # ── 오늘 예측 ─────────────────────────────────────
    df_pred = predict_today(date, stat_weight=stat_weight, ml_weight=ml_weight)
    print_predictions(df_pred)

    return df_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KBO 승률 예측 모델")
    parser.add_argument("--date", type=str, default=None,
                        help="날짜 (YYYYMMDD, 기본값: 오늘)")
    parser.add_argument("--retrain", action="store_true",
                        help="모델 강제 재학습")
    args = parser.parse_args()

    if args.retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        os.remove(SCALER_PATH)
        print("[모델] 기존 모델 삭제 → 재학습")

    run(args.date)
