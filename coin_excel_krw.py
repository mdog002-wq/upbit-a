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


# [추가] R 사이트 연동용 급락위험 종목 JSON 저장 함수
WARNING_TRACKER_FILE = os.path.join(DOCS_DIR, "warning_coins.json")

def update_warning_tracker(df_result):
    warning_coins = []
    if not df_result.empty:
        for _, row in df_result.iterrows():
            dump_risk = float(row.get('STGT_그래프덤핑위험(%)', 0.0))
            rsi = float(row.get('RSI', 50.0))
            
            # 위험/급락 조건 충족 시 R 사이트 표식용으로 저장
            if dump_risk >= 75.0 or rsi >= 80.0:
                warning_coins.append({
                    "symbol": row['심볼'],
                    "name": row['코인명'],
                    "warning_type": "DUMP_RISK" if dump_risk >= 75.0 else "OVERBOUGHT",
                    "reason": row.get('아이스버그역산(고주파)', '급락 위험 포착'),
                    "updated_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
                })
                
    with open(WARNING_TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(warning_coins, f, ensure_ascii=False, indent=2)
        
    return warning_coins

# [수정] 대시보드 HTML 생성 함수
def generate_repo1_dashboard_html(df_result, ai_report, tracking_monitor_data, news_data=None, html_path="docs/index.html"):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    if news_data is None:
        news_data = {}

    # 1. R 사이트 전용 위험 종목 파일(warning_coins.json) 생성/업데이트
    update_warning_tracker(df_result)

    # 2. AI 추천 종목 모니터링 데이터 변환
    tracking_list = []
    if isinstance(tracking_monitor_data, dict):
        for sym, val in tracking_monitor_data.items():
            entry_p = val.get("entry_price", 0.0)
            curr_p = val.get("current_price", 0.0)
            profit_r = round(((curr_p - entry_p) / (entry_p + 1e-8)) * 100, 2) if entry_p > 0 else 0.0
            tracking_list.append({
                "name": val.get("name", sym),
                "symbol": sym,
                "count": val.get("count", 1),
                "entry_price": format_price(entry_p),
                "current_price": format_price(curr_p),
                "profit_rate": profit_r,
                "recommend_time": val.get("last_recommended_at", "")
            })
    else:
        tracking_list = tracking_monitor_data

    monitored_symbols = {item['symbol'] for item in tracking_list}

    # 3. 실시간 속보 데이터만 작성 (경고 메시지 제외)
    news_items = []
    if news_data:
        for coin, items in news_data.items():
            li_tags = ""
            for item in items:
                title = item.get("title", "")
                link = item.get("link", "#")
                li_tags += f'<li class="mb-1"><a href="{link}" target="_blank" rel="noopener noreferrer" class="text-decoration-none text-dark hover-primary">{title}</a></li>'

            card_html = (
                f'<div class="p-2 border rounded bg-light mb-2 shadow-sm">\n'
                f' <div class="fw-bold text-primary mb-1" style="font-size: 0.85rem;"><i class="fa-solid fa-newspaper me-1"></i>{coin}</div>\n'
                f' <ul class="mb-0 ps-3 small text-secondary">{li_tags}</ul>\n'
                f'</div>'
            )
            news_items.append(card_html)
        news_html = "\n".join(news_items)
    else:
        news_html = '<div class="text-muted small text-center py-3">현재 등록된 추천 속보 이슈가 없습니다.</div>'

    # 4. 중앙 전체 코인 AI 예측 순위 테이블
    table_rows_list = []
    if not df_result.empty:
        for rank, (_, row) in enumerate(df_result.iterrows(), start=1):
            symbol = row['심볼']
            name = row['코인명']
            price = row['현재가(KRW)']
            score = float(row['종합예측점수'])
            pattern_sim = float(row.get('골든패턴유사도(%)', 0.0))

            sticker = ' <span class="badge bg-warning text-dark ms-1" style="font-size: 0.7rem;">🎯 AI추천</span>' if symbol in monitored_symbols else ''

            row_html = (
                f'<tr>\n'
                f' <td class="text-center fw-bold text-muted">{rank}</td>\n'
                f' <td class="fw-bold">{name} <span class="text-secondary small">({symbol})</span>{sticker}</td>\n'
                f' <td>{price}</td>\n'
                f' <td class="text-primary fw-bold">{score:.1f}점 <span class="text-muted small">({pattern_sim:.0f}%)</span></td>\n'
                f'</tr>\n'
            )
            table_rows_list.append(row_html)

    all_coins_table_rows = "".join(table_rows_list) if table_rows_list else '<tr><td colspan="4" class="text-center text-muted py-3">분석된 종목이 없습니다.</td></tr>'

    # 5. 우측 AI 추천 종목 카드
    tracking_items = []
    for item in tracking_list:
        p_rate = item.get('profit_rate', 0.0)
        rate_color = "text-danger" if p_rate > 0 else ("text-primary" if p_rate < 0 else "text-dark")
        sign = "+" if p_rate > 0 else ""

        card_html = (
            f'<div class="p-3 border rounded bg-white shadow-sm mb-2">\n'
            f' <div class="d-flex justify-content-between align-items-center mb-1">\n'
            f' <strong class="text-dark fs-6">🎯 {item["name"]} <span class="text-muted small">({item["symbol"]})</span></strong>\n'
            f' <div class="d-flex gap-1 align-items-center">\n'
            f' <span class="badge bg-primary rounded-pill">추천 {item["count"]}회</span>\n'
            f' </div>\n'
            f' </div>\n'
            f' <div class="row g-1 small text-secondary mt-1">\n'
            f' <div class="col-6">추천진입가: <b>{item["entry_price"]}</b></div>\n'
            f' <div class="col-6 text-end">현재가: <b>{item["current_price"]}</b></div>\n'
            f' <div class="col-6">수익률: <b class="{rate_color}">{sign}{p_rate}%</b></div>\n'
            f' <div class="col-6 text-end text-muted" style="font-size:0.75rem;">{item.get("recommend_time", "")}</div>\n'
            f' </div>\n'
            f'</div>\n'
        )
        tracking_items.append(card_html)
    tracking_html = "\n".join(tracking_items) if tracking_items else '<div class="text-muted small text-center py-3">현재 모니터링 중인 AI 추천 종목이 없습니다.</div>'

    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    updated_time = datetime.datetime.now(kst_tz).strftime("%Y-%m-%d %H:%M:%S")

    # 6. HTML 레이아웃 (좌측 패널: 실시간 속보 / AI 분석 리포트만 배치)
    html_template = (
        '<!DOCTYPE html>\n'
        '<html lang="ko">\n'
        '<head>\n'
        ' <meta charset="UTF-8">\n'
        ' <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        ' <meta http-equiv="refresh" content="300">\n'
        ' <title>Upbit AI Quantitative Dashboard</title>\n'
        ' <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">\n'
        ' <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">\n'
        ' <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>\n'
        ' <style>\n'
        ' body { background-color: #f8fafc; color: #1e293b; font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; }\n'
        ' .card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); height: 100%; }\n'
        ' .news-box { max-height: 250px; overflow-y: auto; }\n'
        ' .table-scroll-box { max-height: 650px; overflow-y: auto; }\n'
        ' .tracking-box { max-height: 750px; overflow-y: auto; }\n'
        ' .report-body h1, .report-body h2, .report-body h3 { font-size: 1rem; font-weight: bold; margin-top: 0.5rem; color: #0f172a; }\n'
        ' .report-body ul { padding-left: 1.2rem; margin-bottom: 0.5rem; }\n'
        ' #allCoinsTable tbody tr:hover { background-color: #f1f5f9 !important; transition: background-color 0.15s ease-in-out; }\n'
        ' .btn-render-link { background-color: #093687; color: white; border: none; font-weight: bold; transition: all 0.2s; }\n'
        ' .btn-render-link:hover { background-color: #001f5c; color: white; transform: translateY(-1px); }\n'
        ' </style>\n'
        '</head>\n'
        '<body>\n'
        ' <div id="mainDashboardApp" class="container-fluid my-4 px-4" style="max-width: 1700px;">\n'
        ' <!-- 상단 헤더 -->\n'
        ' <div class="card p-3 shadow-sm mb-4">\n'
        ' <div class="row align-items-center g-3">\n'
        ' <div class="col-md-4 col-lg-3">\n'
        ' <a href="https://upbit-r.onrender.com" target="_blank" rel="noopener noreferrer" class="btn btn-render-link w-100 py-2 shadow-sm d-flex align-items-center justify-content-center gap-2">\n'
        ' <i class="fa-solid fa-paper-plane"></i>\n'
        ' <span>Upbit Realtime Server (Render)</span>\n'
        ' </a>\n'
        ' </div>\n'
        ' <div class="col-md-4 col-lg-5 text-center text-md-start">\n'
        ' <h3 class="fw-bold text-dark mb-0 fs-4"><i class="fa-solid fa-chart-pie text-primary me-2"></i>업비트 AI 분석 대시보드</h3>\n'
        ' <small class="text-muted">최종 업데이트: __UPDATED_TIME__ (총 __TOTAL_COINS__개 종목 분석 완료)</small>\n'
        ' </div>\n'
        ' <div class="col-md-4 col-lg-4">\n'
        ' <div class="input-group">\n'
        ' <span class="input-group-text bg-light border-end-0"><i class="fa-solid fa-magnifying-glass text-muted"></i></span>\n'
        ' <input type="text" id="coinSearchInput" class="form-control border-start-0 bg-light" placeholder="종목명 또는 심볼 검색 (예: 비트코인, BTC)..." oninput="filterCoins()">\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        '\n'
        ' <!-- 메인 레이아웃 -->\n'
        ' <div class="row g-4">\n'
        ' <!-- 좌측 패널: 실시간 속보 / AI 리포트 -->\n'
        ' <div class="col-lg-3">\n'
        ' <div class="d-flex flex-column gap-3">\n'
        ' <div class="card p-3 shadow-sm">\n'
        ' <h6 class="fw-bold text-success mb-3"><i class="fa-solid fa-newspaper me-1"></i> 실시간 속보</h6>\n'
        ' <div class="news-box d-flex flex-column gap-2">\n'
        ' __NEWS_HTML__\n'
        ' </div>\n'
        ' </div>\n'
        ' <div class="card p-3 shadow-sm">\n'
        ' <h6 class="fw-bold text-primary mb-3"><i class="fa-solid fa-brain me-1"></i> AI 분석 리포트</h6>\n'
        ' <div id="reportMarkdownContainer" class="report-body text-secondary small bg-light p-3 rounded" style="max-height: 480px; overflow-y: auto; line-height: 1.5;"></div>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        '\n'
        ' <!-- 중앙 패널: 전체 코인 AI 예측 순위 -->\n'
        ' <div class="col-lg-5">\n'
        ' <div class="card p-4 shadow-sm">\n'
        ' <div class="d-flex justify-content-between align-items-center mb-3">\n'
        ' <h5 class="fw-bold mb-0 text-dark fs-5"><i class="fa-solid fa-trophy text-warning me-1"></i> 전체 코인 AI 예측 순위</h5>\n'
        ' </div>\n'
        ' <div class="table-scroll-box">\n'
        ' <table class="table table-hover align-middle mb-0" id="allCoinsTable">\n'
        ' <thead class="table-light sticky-top">\n'
        ' <tr>\n'
        ' <th class="text-center" style="width: 10%;">순위</th>\n'
        ' <th style="width: 40%;">코인명</th>\n'
        ' <th style="width: 25%;">현재가</th>\n'
        ' <th style="width: 25%;">예측점수 (유사도)</th>\n'
        ' </tr>\n'
        ' </thead>\n'
        ' <tbody>__ALL_COINS_TABLE_ROWS__</tbody>\n'
        ' </table>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        '\n'
        ' <!-- 우측 패널: AI 추천 종목 모니터링 -->\n'
        ' <div class="col-lg-4">\n'
        ' <div class="card p-3 shadow-sm tracking-box">\n'
        ' <h6 class="fw-bold text-primary mb-3"><i class="fa-solid fa-chart-line me-1"></i> AI 추천종목 모니터 (🎯 표시 종목)</h6>\n'
        ' <div class="d-flex flex-column gap-2" id="trackingContainer">\n'
        ' __TRACKING_HTML__\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        '\n'
        ' <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>\n'
        ' <script>\n'
        ' const rawReportText = __AI_REPORT_JSON__;\n'
        ' document.getElementById("reportMarkdownContainer").innerHTML = marked.parse(rawReportText);\n'
        '\n'
        ' function filterCoins() {\n'
        ' const query = document.getElementById(\'coinSearchInput\').value.toLowerCase().trim();\n'
        ' const tableRows = document.querySelectorAll(\'#allCoinsTable tbody tr\');\n'
        ' tableRows.forEach(row => {\n'
        ' const text = row.textContent.toLowerCase();\n'
        ' row.style.display = text.includes(query) ? \'\' : \'none\';\n'
        ' });\n'
        ' const trackingCards = document.querySelectorAll(\'#trackingContainer > div\');\n'
        ' trackingCards.forEach(card => {\n'
        ' const text = card.textContent.toLowerCase();\n'
        ' card.style.display = text.includes(query) ? \'\' : \'none\';\n'
        ' });\n'
        ' }\n'
        ' </script>\n'
        '</body>\n'
        '</html>'
    )

    html_content = html_template.replace("__UPDATED_TIME__", str(updated_time))\
                                .replace("__TOTAL_COINS__", str(len(df_result)))\
                                .replace("__NEWS_HTML__", news_html)\
                                .replace("__AI_REPORT_JSON__", json.dumps(str(ai_report), ensure_ascii=False))\
                                .replace("__ALL_COINS_TABLE_ROWS__", all_coins_table_rows)\
                                .replace("__TRACKING_HTML__", tracking_html)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"🎨 [대시보드] 속보 단일 구성 대시보드 생성 완료 (`{html_path}`)!")
    return html_content

def main():
    print("🚀 우량주 매집 데이터 수집 및 분석 시작...")
    tickers_info = get_krw_upbit_tickers()
    if not tickers_info:
        print("⚠️ 종목 정보를 가져오지 못했습니다.")
        return

    # 현재가 매핑 조회
    tickers_list = [item['ticker'] for item in tickers_info]
    current_price_map = {}
    try:
        prices = pyupbit.get_current_price(tickers_list)
        if isinstance(prices, dict):
            current_price_map = prices
    except Exception as e:
        print(f"⚠️ 현재가 조회 예외: {e}")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_coin, item, current_price_map): item for item in tickers_info}
        for future in tqdm(as_completed(futures), total=len(futures)):
            res = future.result()
            if res:
                results.append(res)

    df_res = pd.DataFrame(results).sort_values(by="종합예측점수", ascending=False)
    ai_report, rec_coins = generate_gemini_analysis(df_res)
    tracker_data = update_ai_tracker(rec_coins, current_price_map)

    generate_repo1_dashboard_html(df_res, ai_report, tracker_data, html_path="docs/index.html")
    print("✅ 레포1 매집 분석 및 대시보드 생성 완료!")


if __name__ == "__main__":
    main()
