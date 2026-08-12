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
WIN_RATE_HISTORY_FILE = os.path.join(DOCS_DIR, "win_rate_history.json")
INDEX_HTML_FILE = os.path.join(DOCS_DIR, "index.html")

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
            entry["evaluated_at"] = now_time.strftime("%Y-%m-%d %H:%M:%S")
            updated = True

    history.sort(
        key=lambda x: x.get("evaluated_at", x.get("timestamp", "")), 
        reverse=True
    )
    save_json(AI_TRACKER_HISTORY_FILE, history)

    total_evals, wins = 0, 0
    for entry in history:
        if entry.get("evaluated", False):
            for coin in entry.get("recommended_coins", []):
                res = coin.get("evaluated_result")
                if res:
                    total_evals += 1
                    if res.get("success"): wins += 1

    win_rate = round((wins / total_evals * 100), 1) if total_evals > 0 else 0.0
    win_record = load_json(WIN_RATE_HISTORY_FILE, [])
    win_record.append({
        "updated_at": now_time.strftime("%Y-%m-%d %H:%M:%S"),
        "win_rate": win_rate,
        "total_trades": total_evals,
        "wins": wins,
        "losses": total_evals - wins
    })
    save_json(WIN_RATE_HISTORY_FILE, win_record)

def get_enhanced_quant_market_data():
    tickers = pyupbit.get_tickers(fiat="KRW")
    market_data = []

    for ticker in tickers:
        try:
            df = pyupbit.get_ohlcv(ticker, interval="minute60", count=48)
            if df is None or len(df) < 24: 
                continue

            symbol = ticker.replace("KRW-", "")
            current_price = float(df['close'].iloc[-1])
            volume_24h = float((df['close'] * df['volume']).tail(24).sum())
            
            high_max = float(df['high'].tail(24).max())
            low_min = float(df['low'].tail(24).min())
            volatility = ((high_max - low_min) / low_min) * 100.0 if low_min > 0 else 0.0

            recent_vol = df['volume'].tail(12).mean()
            prev_vol = df['volume'].iloc[-24:-12].mean()
            vol_growth_ratio = (recent_vol / prev_vol) if prev_vol > 0 else 1.0
            
            pattern_score = min(float(vol_growth_ratio * 50.0), 100.0)

            market_data.append({
                "symbol": symbol,
                "current_price": current_price,
                "volume_24h": round(volume_24h, 2),
                "volatility_24h": round(volatility, 2),
                "golden_pattern_score": round(pattern_score, 1)
            })
            time.sleep(0.05)
        except Exception:
            continue

    sorted_coins = sorted(market_data, key=lambda x: (x["volume_24h"] * x["golden_pattern_score"]), reverse=True)
    return sorted_coins[:20]

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
            model='gemini-3.5-flash-lite',
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
    history_data.insert(0, new_entry)
    save_json(AI_TRACKER_HISTORY_FILE, history_data[:50])

def build_news_html():
    """좌측 상단: 최신 속보 및 공지사항 HTML"""
    news_data = load_json(NEWS_JSON_FILE, [])
    if not news_data:
        return "<div style='color:#888; font-size:12px;'>수집된 속보가 없습니다.</div>"
    
    html = "<ul style='list-style: none; padding-left: 0; margin: 0; font-size: 13px;'>"
    for item in news_data[:6]:
        title = item.get("title", "")
        link = item.get("link", "#")
        html += f"<li style='margin-bottom: 8px; border-bottom: 1px dashed #eee; padding-bottom: 4px;'><a href='{link}' target='_blank' style='text-decoration: none; color: #333;'>{title}</a></li>"
    html += "</ul>"
    return html

def build_tracker_html():
    """중앙 & 우측: AI 추적 및 검증 결과 (최신순)"""
    history = load_json(AI_TRACKER_HISTORY_FILE, [])
    if not history:
        return "<div style='color:#888; font-size:12px;'>검증 기록이 없습니다.</div>"

    html = "<div class='tracker-list'>"
    for item in history:
        ts = item.get("timestamp", "")
        evaluated = item.get("evaluated", False)
        status_badge = "<span class='badge-eval'>검증완료</span>" if evaluated else "<span class='badge-wait'>진행중(12h)</span>"

        html += f"""
        <div class='tracker-item'>
            <div class='tracker-header'>
                <span>📅 {ts}</span>
                {status_badge}
            </div>
            <div class='tracker-coins'>
        """
        for coin in item.get("recommended_coins", []):
            sym = coin.get("symbol", "")
            res = coin.get("evaluated_result")
            if res:
                res_text = f"<b style='color:{'#2b8a3e' if res.get('success') else '#e03131'}'>{res.get('status_text')} ({res.get('max_return')}% )</b>"
            else:
                res_text = "<span style='color:#888;'>측정중</span>"

            html += f"<div class='tracker-coin-row'>• <b>{sym}</b> (진입: {coin.get('entry_price', 0):,}원) → {res_text}</div>"

        html += "</div></div>"
    html += "</div>"
    return html

def generate_full_dashboard_html():
    report_md = ""
    if os.path.exists(REPORT_MD_FILE):
        with open(REPORT_MD_FILE, "r", encoding="utf-8") as f:
            report_md = f.read()

    news_html = build_news_html()
    tracker_html = build_tracker_html()

    html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Upbit AI Quantitative Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8f9fa; margin: 0; padding: 15px; }
        
        /* 2컬럼 메인 레이아웃 (좌측: 속보+리포트 / 중앙&우측 통합: 검증 히스토리) */
        .dashboard-main-container { display: grid; grid-template-columns: 1fr 1.3fr; gap: 15px; }
        .left-column, .right-column { display: flex; flex-direction: column; gap: 15px; }
        
        /* 좌측 상단: 실시간 속보 패널 */
        .news-box { background: #ffffff; border: 1px solid #ced4da; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .box-title-green { font-size: 15px; font-weight: bold; color: #2b8a3e; margin-bottom: 12px; border-bottom: 2px solid #2b8a3e; padding-bottom: 6px; }
        
        /* 좌측 하단: AI 리포트 패널 */
        .report-box { background: #ffffff; border: 1px solid #ced4da; border-radius: 8px; padding: 18px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); font-size: 14px; line-height: 1.6; }
        
        /* 중앙 & 우측 통합: 검증 히스토리 패널 */
        .tracker-box { background: #ffffff; border: 1px solid #ced4da; border-radius: 8px; padding: 18px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        
        .tracker-item { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; margin-bottom: 10px; }
        .tracker-header { display: flex; justify-content: space-between; font-size: 13px; color: #666; margin-bottom: 8px; border-bottom: 1px solid #eee; padding-bottom: 4px; }
        .badge-eval { background: #2b8a3e; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
        .badge-wait { background: #f59f00; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 11px; }
        .tracker-coin-row { font-size: 14px; margin-top: 5px; }
    </style>
</head>
<body>

    <div class="dashboard-main-container">
        <!-- 좌측 영역: (상) 최신 속보 / (하) AI 퀀트 리포트 -->
        <div class="left-column">
            <div class="news-box">
                <div class="box-title-green">📰 실시간 코인 속보 & 공지</div>
                <div>{{NEWS_HTML}}</div>
            </div>

            <div class="report-box">
                <div class="box-title-green">📊 AI 퀀트 종합 분석 리포트</div>
                <div>{{REPORT_MD}}</div>
            </div>
        </div>

        <!-- 중앙 & 우측 통합 영역: AI 추천 검증 히스토리 -->
        <div class="right-column">
            <div class="tracker-box">
                <div class="box-title-green">🔍 AI 추천 종목 실시간 검증 히스토리</div>
                <div>{{TRACKER_HTML}}</div>
            </div>
        </div>
    </div>

</body>
</html>
"""
    full_html = html_template.replace("{{NEWS_HTML}}", news_html)\
                             .replace("{{REPORT_MD}}", report_md.replace("\n", "<br>"))\
                             .replace("{{TRACKER_HTML}}", tracker_html)

    with open(INDEX_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)

def main():
    fetch_crypto_news()
    evaluate_and_update_history()
    
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

    generate_full_dashboard_html()

    print("✅ 강화된 퀀트 선별 및 AI 검증 프로세스 완료!")

if __name__ == "__main__":
    main()
