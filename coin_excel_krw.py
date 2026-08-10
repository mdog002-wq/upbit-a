import os
import io
import time
import datetime
from datetime import timedelta
import json
import smtplib
import requests
import urllib.parse
import feedparser
import numpy as np
import pandas as pd
import pyupbit
import openpyxl
import paramiko
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from google import genai
from google.genai import types

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# [Gemini Output Schema]
class RecommendedCoin(BaseModel):
    coin_name: str = Field(description="코인 한글명 (예: 바빌론, 너보스, 스파크)")
    symbol: str = Field(description="티커 심볼 (예: BABY, CKB, SPK)")
    reason: str = Field(description="추천 핵심 사유 요약")

class AIReportResponse(BaseModel):
    report_markdown: str = Field(description="종합 퀀트 분석 리포트 전문 (마크다운)")
    recommended_coins: List[RecommendedCoin] = Field(description="AI 최우선 추천 코인 리스트")

# [설정 및 환경변수]
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
RECEIVER_EMAILS = [email.strip() for email in os.environ.get("RECEIVER_EMAIL", "").split(",") if email.strip()]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [chat_id.strip() for chat_id in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") if chat_id.strip()]

EXCEL_FILE_PATH = "업비트_원화마켓_매집_패턴분석_리포트.xlsx"
CACHE_DIR = "./cache"
AI_MODELS_DIR = "./ai_models"
DOCS_DIR = "./docs"
EXPERIENCE_FILE = os.path.join(AI_MODELS_DIR, "ai_experience.json")
AI_TRACKER_HISTORY_FILE = os.path.join(DOCS_DIR, "ai_recommend_tracker.json")

# 레포2의 golden_pattern.json URL
GOLDEN_PATTERN_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit/main/data/golden_pattern.json"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AI_MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def load_golden_pattern():
    try:
        headers = {}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        res = requests.get(f"{GOLDEN_PATTERN_URL}?t={int(time.time())}", headers=headers, timeout=10)
        if res.status_code == 200:
            pattern_data = res.json()
            print(f"✅ 레포2 골든 패턴 로드 완료! (업데이트: {pattern_data.get('updated_at', 'N/A')})")
            return pattern_data
    except Exception as e:
        print(f"⚠️ 레포2 골든 패턴 로드 실패: {e}")
    return None

GLOBAL_GOLDEN_PATTERN = load_golden_pattern()

def calculate_dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]

def calculate_pattern_similarity(series, target_pattern):
    if not series or not target_pattern or len(series) == 0:
        return 0.0
    s_min, s_max = np.min(series), np.max(series)
    if s_max == s_min:
        return 0.0
    norm_series = (np.array(series) - s_min) / (s_max - s_min + 1e-8)
    dist = calculate_dtw_distance(norm_series, np.array(target_pattern))
    return round(max(0.0, 100.0 * (1.0 - (dist / len(target_pattern)))), 1)

def format_price(x):
    try:
        val = float(x)
        return f"{int(val):,}" if val >= 100 else (f"{val:,.2f}" if val >= 1 else f"{val:,.5f}")
    except Exception:
        return str(x)

def get_krw_upbit_tickers():
    url = "https://api.upbit.com/v1/market/all?isDetails=false"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return [{'ticker': c['market'], 'korean_name': c['korean_name'], 'symbol': c['market'].replace("KRW-", "")}
                    for c in res.json() if c['market'].startswith("KRW-")]
    except Exception:
        pass
    return []

def calculate_t1_advanced_metrics(ticker):
    try:
        df_1h = pyupbit.get_ohlcv(ticker, interval="minute60", count=100)
        df_5m = pyupbit.get_ohlcv(ticker, interval="minute5", count=24)
        if df_1h is None or len(df_1h) < 30:
            return None

        close_1h, vol_1h = df_1h['close'], df_1h['volume']
        vol_recent_3m = df_5m['volume'].iloc[-3:].sum() if df_5m is not None and len(df_5m) >= 3 else vol_1h.iloc[-1]
        vol_prev_avg = (df_5m['volume'].iloc[-18:-3].mean() * 3) if df_5m is not None and len(df_5m) >= 18 else (vol_1h.iloc[-5:-1].mean() + 1e-8)
        vol_velocity = float(vol_recent_3m / (vol_prev_avg + 1e-8))
        vol_spike_ratio = float(vol_1h.iloc[-1] / (vol_1h.iloc[-25:-1].mean() + 1e-8))

        ma20_1h = close_1h.rolling(20).mean()
        std20_1h = close_1h.rolling(20).std()
        upper_band = ma20_1h + (std20_1h * 2)
        lower_band = ma20_1h - (std20_1h * 2)
        bb_width = float((upper_band.iloc[-1] - lower_band.iloc[-1]) / (ma20_1h.iloc[-1] + 1e-8))
        bb_breakout = float((close_1h.iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1] + 1e-8))

        mfv = ((close_1h - df_1h['low']) - (df_1h['high'] - close_1h)) / (df_1h['high'] - df_1h['low'] + 1e-8) * vol_1h
        cmf_1h = float(mfv.iloc[-20:].sum() / (vol_1h.iloc[-20:].sum() + 1e-8))

        delta = close_1h.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_1h = float(100 - (100 / (1 + (gain / (loss + 1e-8)).iloc[-1])))

        pattern_sim = 0.0
        if GLOBAL_GOLDEN_PATTERN and df_5m is not None and len(df_5m) == 24:
            pattern_sim = calculate_pattern_similarity(df_5m['close'].tolist(), GLOBAL_GOLDEN_PATTERN.get("golden_pattern", []))

        return {
            "last_close": float(close_1h.iloc[-1]),
            "vol_velocity": round(vol_velocity, 2),
            "vol_spike_ratio": round(vol_spike_ratio, 2),
            "bb_width": round(bb_width, 4),
            "bb_breakout": round(bb_breakout, 2),
            "cmf_1h": round(cmf_1h, 2),
            "rsi_1h": round(rsi_1h, 1),
            "pattern_similarity": pattern_sim
        }
    except Exception:
        return None

def process_single_coin(item, current_price_map):
    ticker, symbol, korean_name = item['ticker'], item['symbol'], item['korean_name']
    metrics = calculate_t1_advanced_metrics(ticker)
    c_price = current_price_map.get(ticker, 0)

    if not metrics:
        return {
            "코인명": korean_name, "심볼": symbol, "현재가(KRW)": format_price(c_price),
            "raw_price": float(c_price), "종합예측점수": 0.0, "거래량절벽(배)": 1.0,
            "CMF지표": 0.0, "RSI": 50.0, "골든패턴유사도(%)": 0.0
        }

    rsi = metrics['rsi_1h']
    vol_vel = metrics['vol_velocity']
    cmf = metrics['cmf_1h']
    
    liquidity_penalty = 0.5 if (vol_vel < 0.8 and rsi < 50.0) else 1.0
    vol_score = min(30.0, max(0.0, (vol_vel - 0.9) * 20.0))
    squeeze_score = 15.0 if metrics['bb_width'] < 0.05 else 0.0
    cmf_score = max(-5.0, min(15.0, cmf * 20.0))
    pattern_bonus = (metrics['pattern_similarity'] - 70.0) * 0.33 if metrics['pattern_similarity'] >= 70.0 else 0.0

    raw_score = (vol_score + squeeze_score + cmf_score + pattern_bonus) * liquidity_penalty
    if rsi >= 75.0 or rsi <= 35.0:
        raw_score *= 0.7

    acc_score = round(max(0.0, min(100.0, raw_score)), 1)
    return {
        "코인명": korean_name, "심볼": symbol, "현재가(KRW)": format_price(c_price),
        "raw_price": float(c_price), "종합예측점수": acc_score, "거래량절벽(배)": metrics['vol_spike_ratio'],
        "CMF지표": metrics['cmf_1h'], "RSI": metrics['rsi_1h'], "골든패턴유사도(%)": metrics['pattern_similarity']
    }

def generate_gemini_analysis(df_result):
    if df_result.empty:
        return "분석 데이터 없음", []
    top_coins = df_result.head(5)['코인명'].tolist()
    top_symbols = df_result.head(5)['심볼'].tolist()
    default_rec = [{"symbol": sym, "name": name, "reason": "우량 매집 패턴 선별 종목"} for sym, name in zip(top_symbols, top_coins)]

    if not GEMINI_API_KEY:
        return f"상위 추천 종목: {', '.join(top_coins)}", default_rec

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"다음 업비트 우량 매집 퀀트 데이터 상위 목록을 보고 분석 리포트를 작성하세요:\n{df_result.head(10).to_string()}"
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AIReportResponse,
                temperature=0.2,
            ),
        )
        parsed = json.loads(response.text)
        rec_list = [{"symbol": i.get("symbol", "").upper(), "name": i.get("coin_name", ""), "reason": i.get("reason", "")} for i in parsed.get("recommended_coins", [])]
        return parsed.get("report_markdown", ""), rec_list
    except Exception as e:
        print(f"⚠️ Gemini 리포트 생성 예외: {e}")
        return f"AI 분석 리포트 (상위: {', '.join(top_coins)})", default_rec

def update_ai_tracker(rec_coins, current_price_map):
    history = {}
    if os.path.exists(AI_TRACKER_HISTORY_FILE):
        try:
            with open(AI_TRACKER_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst_tz).strftime("%Y-%m-%d %H:%M:%S")

    for coin in rec_coins:
        sym = coin['symbol']
        c_price = current_price_map.get(f"KRW-{sym}", current_price_map.get(sym, 0.0))
        if sym in history:
            history[sym]['count'] += 1
            history[sym]['current_price'] = c_price
            history[sym]['last_recommended_at'] = now_str
        else:
            history[sym] = {
                "name": coin['name'], "symbol": sym, "count": 1,
                "entry_price": c_price, "current_price": c_price,
                "first_recommended_at": now_str, "last_recommended_at": now_str
            }

    with open(AI_TRACKER_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return history

def generate_repo1_dashboard_html(df_result, ai_report, tracker_data, html_path="docs/index.html"):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    updated_time = datetime.datetime.now(kst_tz).strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    for rank, (_, row) in enumerate(df_result.iterrows(), start=1):
        rows.append(f"<tr><td>{rank}</td><td><b>{row['코인명']}</b> ({row['심볼']})</td><td>{row['현재가(KRW)']}</td><td><b>{row['종합예측점수']}점</b></td></tr>")

    tracker_items = []
    for sym, item in tracker_data.items():
        p_rate = round(((item['current_price'] - item['entry_price']) / (item['entry_price'] + 1e-8)) * 100, 2)
        tracker_items.append(f"<div class='p-2 border mb-1'>🎯 <b>{item['name']}</b> ({sym}) | 추천 {item['count']}회 | 수익률: {p_rate}%</div>")

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>업비트 우량주 매집 분석 대시보드</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light p-4">
    <div class="container bg-white p-4 rounded shadow-sm">
        <h2>📊 업비트 우량주 매집 분석 리포트</h2>
        <p class="text-muted">업데이트: {updated_time}</p>
        <hr>
        <div class="row">
            <div class="col-md-8">
                <h4>📌 매집주 선별 순위</h4>
                <table class="table table-hover">
                    <thead><tr><th>순위</th><th>코인명</th><th>현재가</th><th>매집 점수</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
            <div class="col-md-4">
                <h4>🎯 AI 추천 트래킹</h4>
                <div>{''.join(tracker_items)}</div>
            </div>
        </div>
    </div>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    krw_coins = get_krw_upbit_tickers()
    tickers = [c['ticker'] for c in krw_coins]
    try:
        current_price_map = pyupbit.get_current_price(tickers)
    except Exception:
        current_price_map = {}

    results = []
    print("🚀 우량주 매집 데이터 수집 및 선별 분석 시작...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_coin, item, current_price_map): item for item in krw_coins}
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                results.append(res)

    df_res = pd.DataFrame(results).sort_values(by="종합예측점수", ascending=False)
    ai_report, rec_coins = generate_gemini_analysis(df_res)
    tracker_data = update_ai_tracker(rec_coins, current_price_map)
    
    generate_repo1_dashboard_html(df_res, ai_report, tracker_data, "docs/index.html")
    print("✅ 레포1 매집 분석 및 대시보드 생성 완료!")

if __name__ == "__main__":
    main()
