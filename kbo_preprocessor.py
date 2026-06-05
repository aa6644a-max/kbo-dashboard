"""
KBO 토토 예측 프로그램 - 2단계: 전처리기
=========================================
1단계에서 만든 kbo_raw_YYYYMMDD.csv를 읽어서
모델이 학습/예측할 수 있는 숫자 피처로 변환합니다.

실행 방법:
  python kbo_preprocessor.py
  python kbo_preprocessor.py --date 20260423  (특정 날짜)
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime


# ════════════════════════════════════════════════════════
# 피처 설명 (모델 입력값)
# ════════════════════════════════════════════════════════
"""
[투수 관련]
- era_diff        : 홈 ERA - 원정 ERA (음수일수록 홈투수 유리)
- home_era        : 홈 선발 ERA
- away_era        : 원정 선발 ERA
- home_whip       : 홈 선발 WHIP
- away_whip       : 원정 선발 WHIP

[팀 성적 관련]
- win_rate_diff   : 홈 승률 - 원정 승률 (양수일수록 홈팀 유리)
- home_win_rate   : 홈팀 시즌 승률
- away_win_rate   : 원정팀 시즌 승률
- score_diff      : 홈 평균득점 - 원정 평균득점
- allow_diff      : 원정 평균실점 - 홈 평균실점 (홈팀 투수력 우위)

[상대전적]
- h2h_home_rate   : 홈팀의 상대전적 승률 (올시즌)

[홈 어드밴티지]
- home_advantage  : 항상 1 (홈팀 기준으로 통계적 홈 이점 반영)

[날씨 - 있을 때만]
- rain_prob       : 강수확률 (높을수록 타자 유리 → 투수 불리)
- temp            : 기온 (너무 춥거나 더우면 영향)
"""


# ════════════════════════════════════════════════════════
# 1. 원시 데이터 로드
# ════════════════════════════════════════════════════════
def load_raw_data(date: str = None) -> pd.DataFrame:
    """1단계에서 저장한 CSV를 읽어옵니다."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    filepath = f"kbo_raw_{date}.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' 파일을 찾을 수 없습니다.\n"
            f"먼저 kbo_collector.py를 실행해주세요."
        )

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    print(f"[로드] {filepath} → {len(df)}경기")
    return df


# ════════════════════════════════════════════════════════
# 2. 결측치 처리
# ════════════════════════════════════════════════════════
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """ERA/WHIP 등 수집 실패한 값을 리그 평균으로 채웁니다."""

    # 리그 평균값 (2026 KBO 기준)
    LEAGUE_AVG = {
        "home_era":       4.50,
        "away_era":       4.50,
        "home_whip":      1.40,
        "away_whip":      1.40,
        "home_k9":        7.00,
        "away_k9":        7.00,
        "home_win_rate":  0.50,
        "away_win_rate":  0.50,
        "home_avg_score": 4.50,
        "away_avg_score": 4.50,
        "home_avg_allow": 4.50,
        "away_avg_allow": 4.50,
        "home_batting_avg": 0.260,
        "away_batting_avg": 0.260,
        "home_ops":         0.720,
        "away_ops":         0.720,
        "home_last10_winrate": 0.50,
        "away_last10_winrate": 0.50,
        "home_last5_scored":   4.50,
        "away_last5_scored":   4.50,
        "home_last5_allowed":  4.50,
        "away_last5_allowed":  4.50,
        "home_streak":    0.0,
        "away_streak":    0.0,
        "h2h_home_rate":  0.50,
        "rain_prob":      0.0,
        "temp":           18.0,
        "humidity":       60.0,
        "home_starter_recent_runs":    4.50,
        "away_starter_recent_runs":    4.50,
        "home_starter_recent_winrate": 0.50,
        "away_starter_recent_winrate": 0.50,
        "home_starter_starts":         0.0,
        "away_starter_starts":         0.0,
    }

    for col, avg in LEAGUE_AVG.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(avg)

    print(f"[결측치] 처리 완료")
    return df


# ════════════════════════════════════════════════════════
# 3. 피처 엔지니어링 (핵심)
# ════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """원시 데이터에서 예측에 필요한 파생 피처를 만듭니다."""

    # ── 투수 관련 피처 ──────────────────────────────────
    # ERA 차이: 음수일수록 홈팀 선발이 더 좋음
    df["era_diff"]  = df["home_era"] - df["away_era"]

    # WHIP 차이
    df["whip_diff"] = df["home_whip"] - df["away_whip"]

    # ERA를 0~1 사이 점수로 변환 (낮을수록 좋으니 역수 스케일)
    # ERA 1.0 → 0.9점, ERA 4.5 → 0.5점, ERA 9.0 → 0.1점
    df["home_pitcher_score"] = 1 - (df["home_era"].clip(0, 9) / 10)
    df["away_pitcher_score"] = 1 - (df["away_era"].clip(0, 9) / 10)
    df["pitcher_score_diff"] = df["home_pitcher_score"] - df["away_pitcher_score"]

    # ── 팀 성적 관련 피처 ───────────────────────────────
    # 승률 차이
    df["win_rate_diff"] = df["home_win_rate"] - df["away_win_rate"]

    # 득점력 차이 (홈팀 공격 - 원정팀 공격)
    df["score_diff"] = df["home_avg_score"] - df["away_avg_score"]

    # 실점 차이 (원정팀 실점 - 홈팀 실점 → 양수면 홈투수 유리)
    df["allow_diff"] = df["away_avg_allow"] - df["home_avg_allow"]

    # 종합 전력 지수 (승률 60% + 득점차 20% + 실점차 20%)
    df["home_power"] = (
        df["home_win_rate"] * 0.6 +
        (df["home_avg_score"] / 10) * 0.2 +
        (1 - df["home_avg_allow"] / 10) * 0.2
    )
    df["away_power"] = (
        df["away_win_rate"] * 0.6 +
        (df["away_avg_score"] / 10) * 0.2 +
        (1 - df["away_avg_allow"] / 10) * 0.2
    )
    df["power_diff"] = df["home_power"] - df["away_power"]

    # ── 상대전적 ────────────────────────────────────────
    # 상대전적이 0.5 기준으로 얼마나 치우쳤는지
    df["h2h_advantage"] = df["h2h_home_rate"] - 0.5

    # ── 홈 어드밴티지 ────────────────────────────────────
    # KBO 홈팀 역대 승률 약 53% → 0.03 보정값
    df["home_advantage"] = 0.03

    # ── 타격 피처 (NEW) ──────────────────────────────────
    df["home_ops"] = df.get("home_ops", pd.Series(0.720, index=df.index)).fillna(0.720)
    df["away_ops"] = df.get("away_ops", pd.Series(0.720, index=df.index)).fillna(0.720)
    # OPS 차이: 양수면 홈 타선이 더 강함
    df["ops_diff"] = (df["home_ops"] - df["away_ops"]).round(4)

    # 타율 차이 (보조 지표)
    df["home_batting_avg"] = df.get("home_batting_avg", pd.Series(0.260, index=df.index)).fillna(0.260)
    df["away_batting_avg"] = df.get("away_batting_avg", pd.Series(0.260, index=df.index)).fillna(0.260)
    df["batting_avg_diff"] = (df["home_batting_avg"] - df["away_batting_avg"]).round(4)

    # ── K9 (탈삼진율) ────────────────────────────────────
    df["home_k9"] = df.get("home_k9", pd.Series(7.0, index=df.index)).fillna(7.0)
    df["away_k9"] = df.get("away_k9", pd.Series(7.0, index=df.index)).fillna(7.0)
    df["k9_diff"] = df["home_k9"] - df["away_k9"]

    # ── 최근 폼 피처 (NEW) ───────────────────────────────
    df["home_last10_winrate"] = df.get("home_last10_winrate", pd.Series(0.5, index=df.index)).fillna(0.5)
    df["away_last10_winrate"] = df.get("away_last10_winrate", pd.Series(0.5, index=df.index)).fillna(0.5)
    df["home_last5_scored"]   = df.get("home_last5_scored",   pd.Series(4.5, index=df.index)).fillna(4.5)
    df["away_last5_scored"]   = df.get("away_last5_scored",   pd.Series(4.5, index=df.index)).fillna(4.5)
    df["home_last5_allowed"]  = df.get("home_last5_allowed",  pd.Series(4.5, index=df.index)).fillna(4.5)
    df["away_last5_allowed"]  = df.get("away_last5_allowed",  pd.Series(4.5, index=df.index)).fillna(4.5)
    df["home_streak"]         = df.get("home_streak",         pd.Series(0.0, index=df.index)).fillna(0.0)
    df["away_streak"]         = df.get("away_streak",         pd.Series(0.0, index=df.index)).fillna(0.0)

    # 최근 10경기 승률 차이 (양수 = 홈팀 최근 폼 좋음)
    df["recent_winrate_diff"] = df["home_last10_winrate"] - df["away_last10_winrate"]

    # 최근 5경기 득점력 차이
    df["recent_score_diff"]   = df["home_last5_scored"]  - df["away_last5_scored"]

    # 최근 5경기 실점 차이 (원정 실점 - 홈 실점 → 양수면 홈 수비 좋음)
    df["recent_allow_diff"]   = df["away_last5_allowed"] - df["home_last5_allowed"]

    # 연승/연패 차이 (홈 스트릭 - 원정 스트릭)
    df["streak_diff"]         = df["home_streak"] - df["away_streak"]

    # 불펜 전력 근사치: 팀 전체 실점 - 선발ERA 기여분
    # 선발이 보통 6이닝 → ERA 기여 = ERA * 6/9 = ERA * 0.67
    df["home_bullpen_proxy"]  = df["home_avg_allow"] - df["home_era"] * 0.67
    df["away_bullpen_proxy"]  = df["away_avg_allow"] - df["away_era"] * 0.67
    # 음수면 불펜 좋음, 양수면 나쁨 → 홈팀 불펜이 원정보다 좋을수록 양수
    df["bullpen_diff"]        = df["away_bullpen_proxy"] - df["home_bullpen_proxy"]

    # ── 선발투수 최근 폼 피처 (NEW) ─────────────────────────
    hr = df.get("home_starter_recent_runs",    pd.Series(4.5, index=df.index)).fillna(4.5)
    ar = df.get("away_starter_recent_runs",    pd.Series(4.5, index=df.index)).fillna(4.5)
    hw = df.get("home_starter_recent_winrate", pd.Series(0.5, index=df.index)).fillna(0.5)
    aw = df.get("away_starter_recent_winrate", pd.Series(0.5, index=df.index)).fillna(0.5)

    # 최근 실점 차이: 음수일수록 홈 선발이 최근 더 적게 실점 (유리)
    df["starter_recent_runs_diff"]    = (hr - ar).round(3)
    # 최근 승률 차이: 양수일수록 홈 선발 최근 폼 좋음
    df["starter_recent_winrate_diff"] = (hw - aw).round(3)
    # 시즌 ERA와 최근 실점 비교로 폼 트렌드 파악 (음수 = 최근 더 잘 던짐)
    df["home_starter_form_trend"] = (hr - df["home_era"]).round(3)
    df["away_starter_form_trend"] = (ar - df["away_era"]).round(3)
    df["starter_form_trend_diff"] = (df["home_starter_form_trend"] - df["away_starter_form_trend"]).round(3)

    # ── 날씨 피처 ────────────────────────────────────────
    if "rain_prob" in df.columns:
        df["rain_prob"] = df["rain_prob"].fillna(0) / 100
    else:
        df["rain_prob"] = 0.0

    if "temp" in df.columns:
        df["temp_effect"] = 1 - abs(df["temp"].fillna(18) - 18) / 20
    else:
        df["temp_effect"] = 1.0

    print(f"[피처] 엔지니어링 완료 → {len(df.columns)}개 컬럼")
    return df


# ════════════════════════════════════════════════════════
# 4. 통계 기반 승률 계산 (모델 없이 바로 예측)
# ════════════════════════════════════════════════════════
def calc_stat_win_prob(df: pd.DataFrame) -> pd.DataFrame:
    """
    통계 기반으로 홈팀 승률을 계산합니다.
    머신러닝 모델의 기준선(baseline)으로 사용됩니다.

    가중치 설계:
    - 팀 승률:     35%
    - 선발 ERA:    30%
    - 상대전적:    20%
    - 득실점차:    10%
    - 홈 어드밴티지: 5%
    """

    # 각 요소별 홈팀 유리도 계산 (0~1, 0.5가 중립)
    # 1) 팀 승률 기반
    total_wr = df["home_win_rate"] + df["away_win_rate"]
    factor_winrate = df["home_win_rate"] / total_wr.replace(0, 1)

    # 2) 선발투수 ERA 기반 (ERA가 낮을수록 좋으니 역수)
    home_pit = 1 / (df["home_era"] + 0.1)
    away_pit = 1 / (df["away_era"] + 0.1)
    factor_era = home_pit / (home_pit + away_pit)

    # 3) 상대전적
    factor_h2h = df["h2h_home_rate"]

    # 4) 득실점차 기반
    home_rs = df["home_avg_score"] + df["away_avg_allow"]
    away_rs = df["away_avg_score"] + df["home_avg_allow"]
    factor_score = home_rs / (home_rs + away_rs).replace(0, 1)

    # 5) 홈 어드밴티지
    factor_home = 0.53

    # 5-1) OPS 기반 타선 강도 (NEW)
    home_ops = df.get("home_ops", pd.Series(0.720, index=df.index)).fillna(0.720)
    away_ops = df.get("away_ops", pd.Series(0.720, index=df.index)).fillna(0.720)
    factor_ops = home_ops / (home_ops + away_ops).replace(0, 1)

    # 6) 최근 폼 (최근 10경기 승률 기반)
    home_last10 = df.get("home_last10_winrate", pd.Series(0.5, index=df.index)).fillna(0.5)
    away_last10 = df.get("away_last10_winrate", pd.Series(0.5, index=df.index)).fillna(0.5)
    total_last10 = home_last10 + away_last10
    factor_recent = home_last10 / total_last10.replace(0, 1)

    # 7) K9 탈삼진율 (높을수록 지배적인 투수)
    home_k9 = df.get("home_k9", pd.Series(7.0, index=df.index)).fillna(7.0)
    away_k9 = df.get("away_k9", pd.Series(7.0, index=df.index)).fillna(7.0)
    factor_k9 = home_k9 / (home_k9 + away_k9).replace(0, 1)

    # 선발투수 최근 폼 반영
    home_recent_runs = df.get("home_starter_recent_runs", pd.Series(4.5, index=df.index)).fillna(4.5)
    away_recent_runs = df.get("away_starter_recent_runs", pd.Series(4.5, index=df.index)).fillna(4.5)
    home_rpit = 1 / (home_recent_runs + 0.1)
    away_rpit = 1 / (away_recent_runs + 0.1)
    factor_recent_pit = home_rpit / (home_rpit + away_rpit)

    # 가중 합산 (선발 최근 폼 10% 추가, 기존 ERA 비중 축소)
    home_prob = (
        factor_winrate   * 0.20 +
        factor_era       * 0.15 +
        factor_recent_pit* 0.12 +   # 선발 최근 폼 (NEW)
        factor_k9        * 0.05 +
        factor_ops       * 0.08 +
        factor_h2h       * 0.12 +
        factor_recent    * 0.13 +
        factor_score     * 0.10 +
        factor_home      * 0.05
    )

    # 0.05 ~ 0.95 사이로 클리핑 (극단값 방지)
    df["stat_home_prob"] = home_prob.clip(0.05, 0.95).round(4)
    df["stat_away_prob"] = (1 - df["stat_home_prob"]).round(4)

    return df


# ════════════════════════════════════════════════════════
# 5. 모델용 피처 선택 및 저장
# ════════════════════════════════════════════════════════
# 머신러닝 모델에 넣을 피처 목록
ML_FEATURES = [
    # 투수
    "era_diff",
    "whip_diff",
    "pitcher_score_diff",
    "k9_diff",
    # 팀 전력
    "win_rate_diff",
    "score_diff",
    "allow_diff",
    "power_diff",
    # 불펜
    "bullpen_diff",
    # 최근 폼 (NEW)
    "recent_winrate_diff",
    "recent_score_diff",
    "recent_allow_diff",
    "streak_diff",
    # 선발투수 최근 폼 (NEW)
    "starter_recent_runs_diff",
    "starter_recent_winrate_diff",
    "starter_form_trend_diff",
    # 기타
    "h2h_advantage",
    "home_advantage",
    "rain_prob",
    "temp_effect",
]


def save_processed(df: pd.DataFrame, date: str = None) -> str:
    """전처리 완료된 데이터를 저장합니다."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    filepath = f"kbo_processed_{date}.csv"
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"[저장] {filepath}")
    return filepath


# ════════════════════════════════════════════════════════
# 6. 전체 실행
# ════════════════════════════════════════════════════════
def preprocess(date: str = None) -> pd.DataFrame:
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    print(f"\n{'='*50}")
    print(f" KBO 전처리 시작 — {date}")
    print(f"{'='*50}")

    # 순서대로 실행
    df = load_raw_data(date)
    df = fill_missing(df)
    df = engineer_features(df)
    df = calc_stat_win_prob(df)
    save_processed(df, date)

    # 결과 출력
    print(f"\n{'─'*60}")
    print(f"{'경기':<20} {'홈승률':>8} {'원정승률':>8} {'선발ERA비교':>12}")
    print(f"{'─'*60}")
    for _, row in df.iterrows():
        matchup = f"{row['away_team']}@{row['home_team']}"
        print(
            f"{matchup:<20} "
            f"{row['stat_home_prob']:>7.1%} "
            f"{row['stat_away_prob']:>8.1%} "
            f"  {row['home_starter']}({row['home_era']:.2f}) vs {row['away_starter']}({row['away_era']:.2f})"
        )
    print(f"{'─'*60}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KBO 데이터 전처리")
    parser.add_argument("--date", type=str, default=None,
                        help="날짜 (YYYYMMDD 형식, 기본값: 오늘)")
    args = parser.parse_args()
    preprocess(args.date)
