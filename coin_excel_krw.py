import os
import time
import datetime
import pandas as pd
import numpy as np
import pyupbit

# ==========================================
# 1. 핵심 지표 계산 함수 모듈
# ==========================================

def get_market_ohlcv(ticker: str, count: int = 100) -> pd.DataFrame:
    """업비트 60분봉 데이터 수집"""
    try:
        df = pyupbit.get_ohlcv(ticker, interval="minute60", count=count)
        time.sleep(0.05)  # API 요청 제한 방지 (초당 10회)
        return df
    except Exception as e:
        return None

def calculate_volume_cliff_ratio(df: pd.DataFrame, window: int = 20) -> float:
    """
    거래량 절벽 비율 = 현재 봉 거래량 / 최근 N봉 평균 거래량
    """
    if df is None or len(df) < window + 1:
        return 1.0
    
    avg_vol = df['volume'].iloc[-(window+1):-1].mean()
    curr_vol = df['volume'].iloc[-1]
    
    if avg_vol == 0:
        return 1.0
        
    return curr_vol / avg_vol

def calculate_accumulation_score(df: pd.DataFrame) -> float:
    """
    T-1 매집점수 산출 (양봉 거래량 비중 + 주가 위치 기준)
    """
    if df is None or len(df) < 20:
        return 50.0
    
    recent_df = df.iloc[-24:].copy()  # 최근 24개 봉(약 1일) 기준
    up_vol = recent_df[recent_df['close'] >= recent_df['open']]['volume'].sum()
    tot_vol = recent_df['volume'].sum()
    
    if tot_vol == 0:
        return 50.0
        
    vol_score = (up_vol / tot_vol) * 100
    return round(vol_score, 2)

def calculate_pattern_similarity(df: pd.DataFrame) -> float:
    """
    패턴 유사도 산출 (단기 이평선 서포트 및 우상향 기울기)
    """
    if df is None or len(df) < 20:
        return 50.0
    
    ma5 = df['close'].rolling(5).mean()
    ma20 = df['close'].rolling(20).mean()
    
    # 5일선이 20일선 위에 있고, 최근 기울기가 양수인지 체크
    is_above = ma5.iloc[-1] > ma20.iloc[-1]
    slope = (ma5.iloc[-1] - ma5.iloc[-3]) / ma5.iloc[-3] * 100
    
    base_score = 70.0 if is_above else 40.0
    slope_score = min(max(slope * 10, -20), 30) # -20 ~ +30 보정
    
    return round(min(max(base_score + slope_score, 0), 100), 2)

def calculate_ma_convergence(df: pd.DataFrame) -> float:
    """
    이평선 수렴도 점수 (5, 10, 20, 60일선 밀집도)
    """
    if df is None or len(df) < 60:
        return 50.0
    
    c = df['close'].iloc[-1]
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma60 = df['close'].rolling(60).mean().iloc[-1]
    
    std_dev = np.std([ma5, ma10, ma20, ma60])
    avg_ma = np.mean([ma5, ma10, ma20, ma60])
    
    dispersion = (std_dev / avg_ma) * 100  # 이격 변동계수 (%)
    convergence_score = max(100 - (dispersion * 20), 0) # 이격이 좁을수록 100점에 수렴
    
    return round(convergence_score, 2)

# ==========================================
# 2. [개정] 제안 수식 반영 종합예측점수 계산
# ==========================================

def calculate_comprehensive_score(
    pattern_similarity: float,
    accumulation_score: float,
    volume_ratio: float,
    ma_convergence_score: float
) -> tuple[float, float]:
    """
    [제안 반영 수식]
    1. 거래량 비율 0.5 이하 시 가중치 1.5배 부여
    2. (T-1 매집점수 * 가중치) 50% + 패턴유사도 30% + 이평선수렴 20%
    """
    # 거래량 절벽가중치 결정
    vol_weight = 1.5 if volume_ratio <= 0.5 else 1.0
    
    # 매집강도 복합 산출 (최대 100점 캡)
    weighted_acc = min(accumulation_score * vol_weight, 100.0)
    
    # 종합 점수 산출
    final_score = (weighted_acc * 0.5) + (pattern_similarity * 0.3) + (ma_convergence_score * 0.2)
    
    return round(final_score, 2), vol_weight

# ==========================================
# 3. 메인 스캐너 및 히스토리 저장 로직
# ==========================================

def run_market_scanner():
    print("🚀 [알고리즘 스캐너] 업비트 전 종목 수급 및 패턴 분석을 시작합니다...")
    
    tickers = pyupbit.get_tickers(fiat="KRW")
    scan_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scan_results = []
    
    for idx, ticker in enumerate(tickers):
        symbol = ticker.replace("KRW-", "")
        df = get_market_ohlcv(ticker)
        
        if df is None or len(df) < 60:
            continue
            
        # 지표 산출
        vol_ratio = calculate_volume_cliff_ratio(df)
        accum_score = calculate_accumulation_score(df)
        pattern_sim = calculate_pattern_similarity(df)
        ma_score = calculate_ma_convergence(df)
        
        # [신규 수식 적용]
        comp_score, applied_weight = calculate_comprehensive_score(
            pattern_similarity=pattern_sim,
            accumulation_score=accum_score,
            volume_ratio=vol_ratio,
            ma_convergence_score=ma_score
        )
        
        scan_results.append({
            "스캔시각": scan_time,
            "심볼": symbol,
            "종목코드": ticker,
            "종합예측점수": comp_score,
            "T-1매집점수": accum_score,
            "거래량절벽비율": vol_ratio,
            "절벽가중치": applied_weight,
            "패턴유사도": pattern_sim,
            "이평선수렴": ma_score,
            "스캔당시가격": df['close'].iloc[-1]
        })
        
        # 진행상황 표시
        if (idx + 1) % 20 == 0 or (idx + 1) == len(tickers):
            print(f"  └ 분석 진행률: {idx + 1}/{len(tickers)} 완료")

    # 종합예측점수 기준 내림차순 정렬 후 상위 10개 추출
    df_results = pd.DataFrame(scan_results)
    df_top10 = df_results.sort_values(by="종합예측점수", ascending=False).head(10).reset_index(drop=True)
    
    print("\n" + "="*70)
    print(f"📊 [{scan_time}] 상승 직전 패턴 TOP 10 추천 종목")
    print("="*70)
    print(df_top10[["심볼", "종합예측점수", "T-1매집점수", "거래량절벽비율", "절벽가중치", "패턴유사도", "스캔당시가격"]].to_string(index=False))
    print("="*70)
    
    # scan_history.csv 자동 히스토리 누적 저장
    history_file = "scan_history.csv"
    save_cols = ["스캔시각", "심볼", "종합예측점수", "패턴유사도", "T-1매집점수", "스캔당시가격", "거래량절벽비율", "이평선수렴"]
    df_save = df_top10[save_cols].rename(columns={"T-1매집점수": "매집점수", "이평선수렴": "이평선수렴점수"})
    
    if not os.path.exists(history_file):
        df_save.to_csv(history_file, index=False, encoding="utf-8-sig")
    else:
        df_save.to_csv(history_file, mode='a', header=False, index=False, encoding="utf-8-sig")
        
    print(f"\n✅ 히스토리가 '{history_file}' 파일에 정상적으로 누적 저장되었습니다.")

if __name__ == "__main__":
    run_market_scanner()
