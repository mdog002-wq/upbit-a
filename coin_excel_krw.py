import os
import time
import datetime
import json
import requests
import numpy as np
import pandas as pd
import pyupbit
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

DATA_DIR = "data"
DOCS_DIR = "./docs"
AI_TRACKER_HISTORY_FILE = os.path.join(DOCS_DIR, "ai_recommend_tracker.json")
REPORT_MD_FILE = os.path.join(DOCS_DIR, "latest_report.md")
WEIGHTS_FILE = os.path.join(DATA_DIR, "weights.json")
GOLDEN_PATTERN_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit/main/data/golden_pattern.json"

MIN_RECOMMEND_SCORE = 75.0

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

class RecommendedCoin(BaseModel):
    coin_name: str = Field(description="코인 한글명")
    symbol: str = Field(description="티커 심볼")
    reason: str = Field(description="추천 사유 (눌림목 및 수급 중심)")

class AIReportResponse(BaseModel):
    report_markdown: str = Field(description="종합 퀀트 분석 리포트 전문")
    recommended_coins: list[RecommendedCoin] = Field(description="AI 최우선 추천 코인 리스트 (미달 시 빈 배열)")

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_golden_pattern():
    try:
        headers = {}
        if GITHUB_TOKEN: headers["Authorization"] = f"token {GITHUB_TOKEN}"
        res = requests.get(f"{GOLDEN_PATTERN_URL}?t={int(time.time())}", headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception: pass
    return None

GLOBAL_GOLDEN_PATTERN_DATA = load_golden_pattern()

def calculate_dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return dtw[n, m]

def calculate_max_pattern_similarity(series):
    if not series or not GLOBAL_GOLDEN_PATTERN_DATA: return 0.0
    s_min, s_max = np.min(series), np.max(series)
    if s_max == s_min: return 0.0
    norm_series = (np.array(series) - s_min) / (s_max - s_min + 1e-8)

    patterns = GLOBAL_GOLDEN_PATTERN_DATA.get("golden_patterns", [])
    if not patterns and "golden_pattern" in GLOBAL_GOLDEN_PATTERN_DATA:
        patterns = [GLOBAL_GOLDEN_PATTERN_DATA["golden_pattern"]]

    max_sim = 0.0
    for p in patterns:
        dist = calculate_dtw_distance(norm_series, np.array(p))
        sim = round(max(0.0, 100.0 * (1.0 - (dist / len(p)))), 1)
        if sim > max_sim: max_sim = sim
    return max_sim

# 🔄 자율 진화 핵심: 과거 추천 결과 복기 및 가중치 자율 학습
def auto_tune_weights_and_evaluate_history():
    history = load_json(AI_TRACKER_HISTORY_FILE, [])
    if not history: return

    weights = load_json(WEIGHTS_FILE, {
        "w_pattern": 0.20, "w_vol_cliff": 0.25, "w_ma_alignment": 0.25,
        "w_vol_surge": 0.15, "w_daily_momentum": 0.10, "w_breakout": 0.05
    })

    updated = False
    now_time = datetime.datetime.now()

    for entry in history:
        # entry가 딕셔너리가 아닌 경우(문자열 등) 예외 처리하여 건너뜀
        if not isinstance(entry, dict):
            continue

        timestamp_str = entry.get("timestamp")
        if not timestamp_str:
            continue

        try:
            entry_time = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        time_diff_hours = (now_time - entry_time).total_seconds() / 3600.0

        # 추천 후 12시간 이상 경과한 미검증 항목 결과 추적
        if time_diff_hours >= 12 and not entry.get("evaluated", False):
            for coin in entry.get("recommended_coins", []):
                if not isinstance(coin, dict): continue
                
                symbol = coin.get("symbol")
                if not symbol: continue

                ticker = f"KRW-{symbol}"
                df = pyupbit.get_ohlcv(ticker, interval="minute60", count=24)
                if df is None or df.empty: continue

                entry_price = coin.get("entry_price", df.iloc[0]["open"])
                max_price = df["high"].max()
                min_price = df["low"].min()

                max_return = (max_price - entry_price) / entry_price * 100.0
                min_return = (min_price - entry_price) / entry_price * 100.0

                is_success = max_return >= 3.0 and min_return >= -2.0

                learning_rate = 0.01
                if is_success:
                    weights["w_vol_cliff"] = round(weights.get("w_vol_cliff", 0.25) + learning_rate, 3)
                    weights["w_pattern"] = round(weights.get("w_pattern", 0.20) + learning_rate, 3)
                else:
                    weights["w_vol_cliff"] = round(max(0.05, weights.get("w_vol_cliff", 0.25) - learning_rate), 3)
                    weights["w_pattern"] = round(max(0.05, weights.get("w_pattern", 0.20) - learning_rate), 3)

                coin["evaluated_result"] = {"max_return": round(max_return, 2), "success": is_success}

            entry["evaluated"] = True
            updated = True

    if updated:
        total_w = sum(weights.values())
        normalized_weights = {k: round(v / total_w, 3) for k, v in weights.items()}
        save_json(WEIGHTS_FILE, normalized_weights)
        save_json(AI_TRACKER_HISTORY_FILE, history)
        print(f"🧬 [자율 진화 완료] 과거 추천 승률 기반 가중치 업데이트: {normalized_weights}")

def get_krw_upbit_tickers():
    url = "https://api.upbit.com/v1/market/all?isDetails=false"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return [{'ticker': c['market'], 'korean_name': c['korean_name'], 'symbol': c['market'].replace("KRW-", "")}
                    for c in res.json() if c['market'].startswith("KRW-")]
    except Exception: pass
    return []

def process_single_coin(item, current_price_map):
    ticker, symbol, korean_name = item['ticker'], item['symbol'], item['korean_name']
    c_price = current_price_map.get(ticker, 0)

    df_1h = pyupbit.get_ohlcv(ticker, interval="minute60", count=100)
    df_5m = pyupbit.get_ohlcv(ticker, interval="minute5", count=24)

    if df_1h is None or df_5m is None or len(df_1h) < 30 or len(df_5m) < 24:
        return None

    acc_price_24h = (df_1h['close'] * df_1h['volume']).iloc[-24:].sum()
    if acc_price_24h < 5_000_000_000:
        return None

    close_1h, vol_1h = df_1h['close'], df_1h['volume']
    vol_recent_3m = df_5m['volume'].iloc[-3:].sum()
    vol_prev_avg = (df_5m['volume'].iloc[-18:-3].mean() * 3) + 1e-8
    vol_vel = float(vol_recent_3m / vol_prev_avg)

    mfv = ((close_1h - df_1h['low']) - (df_1h['high'] - close_1h)) / (df_1h['high'] - df_1h['low'] + 1e-8) * vol_1h
    cmf_1h = float(mfv.iloc[-12:].sum() / (vol_1h.iloc[-12:].sum() + 1e-8))

    delta = close_1h.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_1h = float(100 - (100 / (1 + (gain / (loss + 1e-8)).iloc[-1])))

    pattern_sim = calculate_max_pattern_similarity(df_5m['close'].tolist())

    vol_score = min(25.0, max(0.0, (vol_vel - 0.8) * 15.0))
    cmf_score = max(-5.0, min(20.0, cmf_1h * 25.0))
    pattern_bonus = (pattern_sim - 70.0) * 0.4 if pattern_sim >= 70.0 else 0.0

    raw_score = vol_score + cmf_score + pattern_bonus + 30.0

    recent_1h_change = (df_5m['close'].iloc[-1] - df_5m['close'].iloc[0]) / df_5m['close'].iloc[0] * 100
    if rsi_1h >= 65.0 or recent_1h_change >= 5.0:
        raw_score *= 0.3
    elif rsi_1h <= 35.0:
        raw_score *= 0.5

    acc_score = round(max(0.0, min(100.0, raw_score)), 1)

    return {
        "코인명": korean_name, "심볼": symbol, "현재가(KRW)": c_price,
        "종합예측점수": acc_score, "RSI": round(rsi_1h, 1), "골든패턴유사도(%)": pattern_sim,
        "recent_1h_change": round(recent_1h_change, 2)
    }

def generate_gemini_analysis(df_result):
    if df_result.empty: return "분석 데이터가 존재하지 않습니다.", []

    qualified_df = df_result[df_result['종합예측점수'] >= MIN_RECOMMEND_SCORE].head(3)

    if qualified_df.empty:
        no_rec_md = "## 🛡️ AI Market Alert\n\n현재 시장 상태에서 **눌림목 및 매집 유효 조건(75점 이상)**을 충족하는 종목이 없습니다. 현금 비중을 유지하고 관망하는 것을 권장합니다."
        return no_rec_md, []

    if not GEMINI_API_KEY:
        return "Gemini API 키가 설정되지 않았습니다.", []

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
당신은 리스크 관리를 최우선으로 하는 퀀트 헤지펀드 매니저입니다.
아래 후보 중 **이미 급등한 추격 매수 종목은 전면 제외**하고, **눌림목 자리를 잡은 안전한 매집 종목**만 1~2개 선별하여 리포트를 작성하세요.

[정밀 분석 데이터]
{qualified_df.to_string()}
"""
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AIReportResponse,
                temperature=0.1,
            ),
        )
        parsed = json.loads(response.text)
        raw_recs = parsed.get("recommended_coins", [])
        
        rec_list = []
        for i in raw_recs:
            sym = i.get("symbol", "").upper()
            if sym:
                # 진화형: 추천 시점의 진입 가격 저장
                match_row = qualified_df[qualified_df["심볼"] == sym]
                entry_price = float(match_row["현재가(KRW)"].values[0]) if not match_row.empty else 0.0
                rec_list.append({"symbol": sym, "name": i.get("coin_name", ""), "reason": i.get("reason", ""), "entry_price": entry_price})

        return parsed.get("report_markdown", ""), rec_list
    except Exception as e:
        print(f"⚠️ Gemini 생성 실패: {e}")
        return "AI 리포트 생성 실패", []

def save_tracker_history(rec_coins, df_res=None):
    """
    rec_coins: 추천된 코인 목록 또는 새 엔트리 데이터
    df_res: 추천 시점의 분석 데이터프레임 (선택)
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # rec_coins가 이미 완성된 dict 형태가 아니라 코인 리스트 형태로 들어왔을 경우 구조화
    if isinstance(rec_coins, list):
        formatted_coins = []
        for coin in rec_coins:
            if isinstance(coin, dict):
                formatted_coins.append(coin)
            elif isinstance(coin, str):
                formatted_coins.append({"symbol": coin, "entry_price": 0.0})
        
        new_entry = {
            "timestamp": now_str,
            "recommended_coins": formatted_coins,
            "evaluated": False
        }
    elif isinstance(rec_coins, dict):
        new_entry = rec_coins
    else:
        print("⚠️ 저장할 올바른 추천 데이터 양식이 아닙니다.")
        return

    # 기존 히스토리 로드 및 예외 처리
    history_data = load_json(AI_TRACKER_HISTORY_FILE, [])
    if not isinstance(history_data, list):
        if isinstance(history_data, dict) and history_data:
            history_data = [history_data]
        else:
            history_data = []

    history_data.append(new_entry)
    save_json(AI_TRACKER_HISTORY_FILE, history_data)
    print("✅ AI 추천 이력이 성공적으로 저장되었습니다.")


def main():
    # 1. 이전 추천 백테스트 및 가중치 스스로 학습
    auto_tune_weights_and_evaluate_history()

    # 2. 시장 분석 실행
    tickers_info = get_krw_upbit_tickers()
    if not tickers_info: return

    tickers_list = [item['ticker'] for item in tickers_info]
    current_price_map = pyupbit.get_current_price(tickers_list) or {}

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_coin, item, current_price_map): item for item in tickers_info}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)

    df_res = pd.DataFrame(results).sort_values(by="종합예측점수", ascending=False)
    ai_report_md, rec_coins = generate_gemini_analysis(df_res)

    with open(REPORT_MD_FILE, "w", encoding="utf-8") as f:
        f.write(ai_report_md)

    save_tracker_history(rec_coins, df_res)
    print(f"✅ 정밀 분석 및 자율 저장 완료! (추천 코인: {len(rec_coins)}개)")

if __name__ == "__main__":
    main()
