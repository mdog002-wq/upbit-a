import os
import time
import datetime
from datetime import timezone, timedelta
import json
import requests
import numpy as np
import pandas as pd
import pyupbit
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

KST = timezone(timedelta(hours=9))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DOCS_DIR = "./docs"
AI_TRACKER_HISTORY_FILE = os.path.join(DOCS_DIR, "ai_recommend_tracker.json")
REPORT_MD_FILE = os.path.join(DOCS_DIR, "latest_report.md")
NEWS_JSON_FILE = os.path.join(DOCS_DIR, "news.json")
WARNING_COINS_FILE = os.path.join(DOCS_DIR, "warning_coins.json")

os.makedirs(DOCS_DIR, exist_ok=True)

class RecommendedCoin(BaseModel):
    coin_name: str = Field(description="코인 한글명")
    symbol: str = Field(description="티커 심볼")
    reason: str = Field(description="추천 사유 및 퀀트적 근거")

class CautionCoin(BaseModel):
    coin_name: str = Field(description="코인 한글명")
    symbol: str = Field(description="티커 심볼")
    reason: str = Field(description="주의/경고 사유 (과열, 급등에 따른 리스크)")

class AIReportResponse(BaseModel):
    report_markdown: str = Field(description="종합 퀀트 분석 리포트 전문")
    recommended_coins: list[RecommendedCoin] = Field(description="AI 최우선 추천 코인 리스트")
    caution_coins: list[CautionCoin] = Field(description="AI 위험/주의 코인 리스트")

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f: return json.load(f)
        except Exception: pass
    return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def fetch_crypto_news():
    news_list = []
    try:
        coinness_url = "https://api.coinness.com/v1/newsflash/list?limit=5"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(coinness_url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            items = data.get("data", {}).get("list", []) if isinstance(data.get("data"), dict) else data.get("data", [])
            for item in items:
                title = item.get("title") or item.get("content", "")[:50]
                news_id = item.get("id")
                link = f"https://coinness.com/newsflash/{news_id}" if news_id else "https://coinness.com"
                pubDate = item.get("created_at") or item.get("published_at", "")
                news_list.append({"title": f"[코인니스] {title}", "link": link, "pubDate": pubDate})
    except Exception as e:
        print(f"⚠️ 코인니스 수집 에러: {e}")

    try:
        upbit_url = "https://api-manager.upbit.com/v1/notices?page=1&per_page=5"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(upbit_url, headers=headers, timeout=5)
        if res.status_code == 200:
            notices = res.json().get("data", {}).get("list", [])
            for notice in notices:
                title = f"[업비트] {notice.get('title', '')}"
                notice_id = notice.get("id")
                link = f"https://upbit.com/service_center/notice?id={notice_id}" if notice_id else "https://upbit.com/service_center/notice"
                pubDate = notice.get("created_at", "")
                news_list.append({"title": title, "link": link, "pubDate": pubDate})
    except Exception as e:
        print(f"⚠️ 업비트 수집 에러: {e}")

    if news_list:
        save_json(NEWS_JSON_FILE, news_list)

def evaluate_and_update_history():
    history = load_json(AI_TRACKER_HISTORY_FILE, [])
    if not history: return

    updated = False
    now_time = datetime.datetime.now(KST)

    for entry in history:
        if not isinstance(entry, dict) or entry.get("evaluated", False): continue
        
        try:
            entry_time = datetime.datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
        except (ValueError, TypeError): continue

        time_diff_hours = (now_time - entry_time).total_seconds() / 3600.0

        if time_diff_hours >= 12:
            for coin in entry.get("recommended_coins", []):
                symbol = coin.get("symbol")
                ticker = f"KRW-{symbol}"
                df = pyupbit.get_ohlcv(ticker, interval="minute60", count=24)
                if df is None or df.empty: continue

                entry_price = coin.get("entry_price", df.iloc[0]["open"])
                max_price = df["high"].max()
                min_price = df["low"].min()

                max_return = (max_price - entry_price) / entry_price * 100.0
                min_return = (min_price - entry_price) / entry_price * 100.0
                
                is_success = max_return >= 3.0 and min_return >= -2.0

                coin["evaluated_result"] = {
                    "max_return": round(max_return, 2),
                    "success": is_success,
                    "status_text": "성공 🎯" if is_success else "실패 ❌"
                }

            entry["evaluated"] = True
            updated = True

    if updated:
        save_json(AI_TRACKER_HISTORY_FILE, history)

def get_enhanced_quant_market_data():
    """
    [강화된 종목 선별 로직]
    업비트 KRW 마켓 전체를 순회하며 거래대금, 변동성, 골든패턴 유사도를 직접 수치화하여 분석합니다.
    """
    tickers = pyupbit.get_tickers(fiat="KRW")
    market_data = []

    print(f"🔍 총 {len(tickers)}개 KRW 마켓 퀀트 지표 분석 시작...")

    for ticker in tickers:
        try:
            df = pyupbit.get_ohlcv(ticker, interval="minute60", count=48)
            if df is None or len(df) < 24: 
                continue

            symbol = ticker.replace("KRW-", "")
            current_price = float(df['close'].iloc[-1])
            volume_24h = float((df['close'] * df['volume']).tail(24).sum())
            
            # 1. 단기 변동성 계산 (최근 24시간 고저 차이 비율)
            high_max = float(df['high'].tail(24).max())
            low_min = float(df['low'].tail(24).min())
            volatility = ((high_max - low_min) / low_min) * 100.0 if low_min > 0 else 0.0

            # 2. 골든패턴 유사도 추정 (저점 다지기 후 거래량 동반 반등 패턴 분석)
            # 최근 12시간 거래량이 이전 12시간 대비 증가했는지 여부 체크
            recent_vol = df['volume'].tail(12).mean()
            prev_vol = df['volume'].iloc[-24:-12].mean()
            vol_growth_ratio = (recent_vol / prev_vol) if prev_vol > 0 else 1.0
            
            # 패턴 점수화 (거래량 증가 + 완만한 우상향 또는 바닥 다지기)
            pattern_score = min(float(vol_growth_ratio * 50.0), 100.0)

            # 3. 종합 예측 점수 산정 (거래대금 가중치 + 패턴 점수)
            market_data.append({
                "symbol": symbol,
                "current_price": current_price,
                "volume_24h": round(volume_24h, 2),
                "volatility_24h": round(volatility, 2),
                "golden_pattern_score": round(pattern_score, 1)
            })
            time.sleep(0.05) # API 부하 방지
        except Exception as e:
            continue

    # 거래대금 상위 및 퀀트 스코어 기준 정렬
    sorted_coins = sorted(market_data, key=lambda x: (x["volume_24h"] * x["golden_pattern_score"]), reverse=True)
    return sorted_coins[:20] # 상위 20개 정예 종목 선별

def generate_ai_analysis(top_coins):
    now_str = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    if not GEMINI_API_KEY or not top_coins:
        return "AI 분석 데이터를 생성할 수 없습니다.", [], []

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
당신은 최고의 암호화폐 퀀트 투자 분석가입니다. 
아래 파이썬 퀀트 엔진이 정밀 선별한 상위 20개 종목의 데이터(거래대금, 24시간 변동성, 골든패턴 유사도 점수)를 바탕으로 엄격하게 검증하여 리포트를 작성하세요.

1. recommended_coins: 바닥 매집 완료 후 반등 확률이 가장 높은 최우선 추천 종목 1~3개 선정
2. caution_coins: 변동성이 지나치게 높거나 단기 과열되어 리스크가 큰 위험/주의 종목 1~2개 선정

[정밀 퀀트 선별 종목 데이터]
{json.dumps(top_coins, ensure_ascii=False, indent=2)}
"""
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AIReportResponse,
                temperature=0.2,
            ),
        )
        parsed = json.loads(response.text)
        
        raw_recs = parsed.get("recommended_coins", [])
        rec_list = []
        for i in raw_recs:
            sym = i.get("symbol", "").upper()
            match_item = next((item for item in top_coins if item["symbol"] == sym), None)
            entry_price = match_item["current_price"] if match_item else 0.0
            rec_list.append({"symbol": sym, "name": i.get("coin_name", ""), "reason": i.get("reason", ""), "entry_price": entry_price})

        raw_cautions = parsed.get("caution_coins", [])
        caution_list = []
        for i in raw_cautions:
            sym = i.get("symbol", "").upper()
            match_item = next((item for item in top_coins if item["symbol"] == sym), None)
            entry_price = match_item["current_price"] if match_item else 0.0
            caution_list.append({"symbol": sym, "name": i.get("coin_name", ""), "reason": i.get("reason", ""), "entry_price": entry_price})

        report_md = f"**최종 분석 시각: {now_str}**\n\n" + parsed.get("report_markdown", "")
        return report_md, rec_list, caution_list
    except Exception as e:
        print(f"⚠️ AI 생성 에러: {e}")
        return f"AI 분석 실패: {e}", [], []

def save_tracker_history(rec_coins):
    if not rec_coins: return
    now_str = datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    history_data = load_json(AI_TRACKER_HISTORY_FILE, [])

    new_entry = {
        "timestamp": now_str,
        "recommended_coins": rec_coins,
        "evaluated": False
    }
    history_data.append(new_entry)
    save_json(AI_TRACKER_HISTORY_FILE, history_data[-50:])

def main():
    fetch_crypto_news()
    evaluate_and_update_history()
    
    # 강화된 종목 선별 퀀트 로직 실행
    top_coins = get_enhanced_quant_market_data()
    report_md, rec_coins, caution_coins = generate_ai_analysis(top_coins)

    with open(REPORT_MD_FILE, "w", encoding="utf-8") as f:
        f.write(report_md)

    if rec_coins:
        save_tracker_history(rec_coins)

    warning_data = {
        "updated_at": datetime.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "warning_coins": caution_coins
    }
    save_json(WARNING_COINS_FILE, warning_data)

    print("✅ 강화된 퀀트 선별 및 AI 검증 프로세스 완료!")

if __name__ == "__main__":
    main()
