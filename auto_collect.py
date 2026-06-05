"""
KBO 자동 수집 스크립트
Windows 작업 스케줄러에서 매일 오전 10시에 실행
: python auto_collect.py
"""

import os, sys, logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

LOG_PATH = os.path.join(BASE_DIR, "auto_collect.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)

def log(msg):
    logging.info(msg)
    print(msg)


def main():
    date = datetime.now().strftime("%Y%m%d")
    log(f"=== 자동 수집 시작: {date} ===")

    raw_path  = os.path.join(BASE_DIR, f"kbo_raw_{date}.csv")
    proc_path = os.path.join(BASE_DIR, f"kbo_processed_{date}.csv")
    pred_path = os.path.join(BASE_DIR, f"kbo_prediction_{date}.csv")

    # 1) 원시 데이터 수집
    if os.path.exists(raw_path):
        log(f"[수집] 이미 존재: {raw_path} → 스킵")
    else:
        try:
            from kbo_collector import collect_today_data
            df = collect_today_data(date)
            if df.empty:
                log("[수집] 오늘 경기 없음 → 종료")
                return
            log(f"[수집] 완료: {len(df)}경기")
        except Exception as e:
            log(f"[수집] 실패: {e}")
            return

    # 2) 전처리
    if os.path.exists(proc_path):
        log(f"[전처리] 이미 존재 → 스킵")
    else:
        try:
            from kbo_preprocessor import preprocess
            preprocess(date)
            log("[전처리] 완료")
        except Exception as e:
            log(f"[전처리] 실패: {e}")
            return

    # 3) 예측 생성
    try:
        from kbo_model import run as model_run
        model_run(date)
        if os.path.exists(pred_path):
            log(f"[예측] 완료: {pred_path}")
        else:
            log("[예측] 결과 파일 없음")
    except Exception as e:
        log(f"[예측] 실패: {e}")
        return

    log(f"=== 자동 수집 완료: {date} ===")


if __name__ == "__main__":
    main()
