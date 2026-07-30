import os
import time
import datetime
import json
import smtplib
import requests
import numpy as np
import pandas as pd
import pyupbit
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from google import genai
from google.genai import types

# ==============================================================================
# [설정] GitHub Secrets 환경 변수
# ==============================================================================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
RECEIVER_EMAILS = [
    email.strip() 
    for email in os.environ.get("RECEIVER_EMAIL", "").split(",") 
    if email.strip()
]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

EXCEL_FILE_PATH = "업비트_원화마켓_매집점수_날짜별기록.xlsx"
CACHE_TOKENOMICS_FILE = "cache_tokenomics.json"
CACHE_GITHUB_FILE = "cache_github.json"

# ==============================================================================
# [캐시 유틸] 외부 API Rate Limit 방지용 캐싱 (1일 단위 갱신)
# ==============================================================================
def load_cache(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 당일 캐시인 경우에만 유효
                if data.get("date") == datetime.date.today().isoformat():
                    return data.get("content", {})
        except Exception:
            return {}
    return {}

def save_cache(file_path, content):
    try:
        data = {
            "date": datetime.date.today().isoformat(),
            "content": content
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 캐시 저장 실패: {e}")

# ==============================================================================
# [유틸] 업비트 원화 마켓 조회
# ==============================================================================
def get_krw_upbit_tickers():
    url = "https://api.upbit.com/v1/market/all"
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        return [
            {
                'ticker': coin['market'],
                'korean_name': coin['korean_name'],
                'symbol': coin['market'].replace("KRW-", "")
            }
            for coin in data if coin['market'].startswith("KRW-")
        ]
    except Exception as e:
        print(f"❌ 업비트 원화 코인 목록 조회 실패: {e}")
        return []

# ==============================================================================
# [캐싱 적용 외부 데이터] 유통량 및 GitHub 활동성
# ==============================================================================
def get_cached_coingecko_tokenomics(symbols):
    cache = load_cache(CACHE_TOKENOMICS_FILE)
    if cache:
        return cache

    print("🔄 [CoinGecko] 유통량 데이터 갱신 중...")
    new_cache = {}
    for symbol in symbols:
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"
            res = requests.get(url, timeout=2)
            if res.status_code == 200:
                m_data = res.json().get('market_data', {})
                circ = m_data.get('circulating_supply', 0)
                total = m_data.get('total_supply', 0) or m_data.get('max_supply', 0)
                circ_ratio = (circ / total * 100) if total and total > 0 else 80.0
                new_cache[symbol] = round(circ_ratio, 2)
            else:
                new_cache[symbol] = 75.0
            time.sleep(0.5) # API Rate Limit 예방
        except Exception:
            new_cache[symbol] = 75.0

    save_cache(CACHE_TOKENOMICS_FILE, new_cache)
    return new_cache

def get_cached_github_activity(symbols):
    cache = load_cache(CACHE_GITHUB_FILE)
    if cache:
        return cache

    print("🔄 [GitHub] 개발 활력도 데이터 갱신 중...")
    new_cache = {}
    headers = {"User-Agent": "Crypto-Bot"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    for symbol in symbols:
        try:
            url = f"https://api.github.com/search/repositories?q={symbol}+crypto&sort=updated&order=desc"
            res = requests.get(url, headers=headers, timeout=2)
            if res.status_code == 200:
                items = res.json().get('items', [])
                new_cache[symbol] = len(items) > 0
            else:
                new_cache[symbol] = False
            time.sleep(0.2)
        except Exception:
            new_cache[symbol] = False

    save_cache(CACHE_GITHUB_FILE, new_cache)
    return new_cache

# ==============================================================================
# [확정 봉 기반 지표 연산] 어제 마감 봉(iloc[-2]) 기준
# ==============================================================================
def calculate_stable_metrics(df_daily):
    # 어제까지 마감된 일봉만 사용 (실시간 변동 봉인 iloc[-1] 제외)
    df = df_daily.iloc[:-1].copy()
    if len(df) < 30:
        return None

    close = df['close']
    volume = df['volume']
    
    # 1. 확정 봉 기준 변동률
    chg_1d = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
    chg_7d = ((close.iloc[-1] - close.iloc[-8]) / close.iloc[-8]) * 100 if len(df) >= 8 else 0
    chg_14d = ((close.iloc[-1] - close.iloc[-15]) / close.iloc[-15]) * 100 if len(df) >= 15 else 0

    # 2. 거래대금 가중 이동평균(VWAP)
    vwap = (df['value'].iloc[-14:].sum()) / (volume.iloc[-14:].sum()) if volume.iloc[-14:].sum() > 0 else close.iloc[-1]
    vwap_gap = round(((close.iloc[-1] - vwap) / vwap) * 100, 2)

    # 3. OBV 디버전스
    obv_values = [0.0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv_values.append(obv_values[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv_values.append(obv_values[-1] - volume.iloc[i])
        else:
            obv_values.append(obv_values[-1])
    df['obv'] = obv_values
    
    obv_slope = df['obv'].iloc[-1] - df['obv'].iloc[-10] if len(df) >= 10 else 0
    price_change_10d = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100 if len(df) >= 10 else 0
    is_obv_divergence = (price_change_10d <= 3.0) and (obv_slope > 0)

    # 4. 볼린저 밴드 수축도
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    bandwidth = ((upper - lower) / ma20).iloc[-1] * 100 if ma20.iloc[-1] > 0 else 999.0

    # 5. 확정 봉 기준 매집봉 감지
    recent_7d = df.iloc[-7:]
    prev_vol_avg = volume.iloc[-27:-7].mean() if len(df) >= 27 else volume.mean()
    accum_candles = 0
    if prev_vol_avg > 0:
        for idx, row in recent_7d.iterrows():
            vol_ratio = row['volume'] / prev_vol_avg
            p_change = ((row['close'] - row['open']) / row['open']) * 100 if row['open'] > 0 else 0
            if vol_ratio >= 2.3 and 0.3 <= p_change <= 5.5:
                accum_candles += 1

    # 어제 마감 봉 거래량 급증률
    vol_ratio = (volume.iloc[-1] / prev_vol_avg) if prev_vol_avg > 0 else 1.0

    return {
        "chg_1d": round(chg_1d, 2),
        "chg_7d": round(chg_7d, 2),
        "chg_14d": round(chg_14d, 2),
        "vwap_gap": vwap_gap,
        "is_obv_div": is_obv_divergence,
        "bandwidth": round(bandwidth, 2),
        "accum_candles": accum_candles,
        "vol_ratio": round(vol_ratio, 2),
        "last_close": close.iloc[-1],
        "last_value": df['value'].iloc[-1]
    }

# ==============================================================================
# [정적 점수 연산 알고리즘]
# ==============================================================================
def calculate_stable_score(metrics, surge_from_bottom, circ_ratio, is_dev_active):
    score = 0

    # 1. 고점 추격 배제
    if surge_from_bottom >= 45.0:
        return 0
    elif surge_from_bottom >= 25.0:
        score -= 30

    # 2. 변동률 조건 (고요한 양봉/횡보)
    if 0.0 <= metrics['chg_1d'] <= 3.5 and -5.0 <= metrics['chg_7d'] <= 5.0:
        score += 25
    elif metrics['chg_1d'] > 10.0:
        score -= 25

    # 3. 거래량 & 수급
    if metrics['vol_ratio'] >= 2.5: score += 20
    elif metrics['vol_ratio'] >= 1.5: score += 12

    if -2.0 <= metrics['vwap_gap'] <= 3.0: score += 15

    # 4. 차트 매집 지표
    if metrics['is_obv_div']: score += 20
    if metrics['bandwidth'] <= 8.0: score += 15
    if metrics['accum_candles'] >= 2: score += 25
    elif metrics['accum_candles'] == 1: score += 12

    # 5. 토큰노믹스 & 개발 지표
    if circ_ratio >= 60.0: score += 10
    elif circ_ratio < 30.0: score -= 15

    if is_dev_active: score += 10

    # 어제 마감 거래대금 가점
    trade_value_eow = metrics['last_value'] / 100_000_000
    if trade_value_eow >= 30: score += 10

    return max(0, score)

# ==============================================================================
# [분석 엔진] 전 종목 정적 스캔
# ==============================================================================
def scan_and_rank_coins():
    print("--------------------------------------------------")
    print("🚀 마감 일봉(Determinism) 기반 업비트 원화 마켓 안정 스캔 시작...")
    
    krw_coins = get_krw_upbit_tickers()
    if not krw_coins:
        return pd.DataFrame()

    symbols = [c['symbol'] for c in krw_coins]
    tokenomics_map = get_cached_coingecko_tokenomics(symbols)
    github_map = get_cached_github_activity(symbols)

    results = []
    
    for item in krw_coins:
        ticker = item['ticker']
        korean_name = item['korean_name']
        symbol = item['symbol']
        
        try:
            time.sleep(0.04)
            df_daily = pyupbit.get_ohlcv(ticker, interval="day", count=60)
            if df_daily is None or len(df_daily) < 30:
                continue

            # 확정 봉 기반 지표 계산
            metrics = calculate_stable_metrics(df_daily)
            if not metrics:
                continue

            # 최근 20일 최저가 (마감 봉 기준)
            df_closed = df_daily.iloc[:-1]
            lowest_20d = df_closed['low'].iloc[-20:].min()
            surge_from_bottom = round(((metrics['last_close'] - lowest_20d) / lowest_20d) * 100, 2) if lowest_20d > 0 else 0.0

            circ_ratio = tokenomics_map.get(symbol, 75.0)
            is_dev_active = github_map.get(symbol, False)

            # 정적 매집 점수 연산
            total_score = calculate_stable_score(metrics, surge_from_bottom, circ_ratio, is_dev_active)

            results.append({
                "코인명": korean_name,
                "심볼": symbol,
                "매집점수": total_score,
                "1일 변동률(%)": metrics['chg_1d'],
                "7일 변동률(%)": metrics['chg_7d'],
                "바닥 대비 상승률(%)": surge_from_bottom,
                "거래량 급증(배)": metrics['vol_ratio'],
                "유통량 비율(%)": circ_ratio,
                "거래대금(억원)": round(metrics['last_value'] / 100_000_000, 1),
                "마감가(KRW)": metrics['last_close'],
                "개발활력": "양호" if is_dev_active else "보통"
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="매집점수", ascending=False)
    return df

# ==============================================================================
# [AI 심층 분석] Gemini API (Temperature = 0.0으로 출력 고정)
# ==============================================================================
def generate_gemini_analysis(df):
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY가 설정되지 않았습니다."

    if df.empty:
        return "분석할 데이터가 없습니다."

    try:
        top_10 = df.head(10).to_dict(orient="records")

        prompt = f"""
당신은 가상자산 수급 분석 전문가입니다.
아래 마감 일봉 기준 상위 10개 코인 데이터를 바탕으로 객관적이고 일관성 있는 분석 리포트를 작성해 주세요.

[상위 10개 데이터]
{top_10}

[작성 지침]
- 반말을 금지하고, 반드시 정중한 경어체(~습니다, ~입니다)로 작성하세요.
- 매집 점수 Top 3 코인을 명확히 지정하고, 1일/7일 변동률, 유통량 비율, 거래량 급증 배수를 직접 인용하여 분석하세요.
- 리스크 유의 종목 1개를 선정하고 그 이유를 구체적으로 기술하세요.
"""
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
        
        # Temperature=0.0으로 설정하여 동일 입력에 대한 출력을 고정
        config = types.GenerateContentConfig(temperature=0.0)
        
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=config
        )
        return response.text.strip() if response.text else "AI 응답 없음"
    except Exception as e:
        return f"AI 분석 중 오류 발생: {e}"

# ==============================================================================
# [엑셀 저장 및 메일 전송]
# ==============================================================================
def save_daily_excel_sheet(df):
    if df.empty:
        return None

    now = datetime.datetime.now()
    sheet_name = now.strftime("%Y-%m-%d_%H시")
    
    wb = openpyxl.load_workbook(EXCEL_FILE_PATH) if os.path.exists(EXCEL_FILE_PATH) else openpyxl.Workbook()
    if sheet_name in wb.sheetnames:
        wb.remove(wb[sheet_name])
        
    ws = wb.create_sheet(title=sheet_name)
    if "Sheet" in wb.sheetnames and len(wb.sheetnames) > 1:
        wb.remove(wb["Sheet"])

    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))

    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="맑은 고딕", size=10, bold=True, color="FFFFFF")
    data_font = Font(name="맑은 고딕", size=10)
    high_score_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9')
    )
    
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    for row in range(2, ws.max_row + 1):
        score_cell = ws.cell(row=row, column=3)
        is_high_score = score_cell.value and float(score_cell.value) >= 70
        
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = data_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if is_high_score:
                cell.fill = high_score_fill
                
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 5, 12)

    wb.save(EXCEL_FILE_PATH)
    wb.close()
    return EXCEL_FILE_PATH

def send_email_with_excel(file_path, ai_analysis=""):
    if not file_path or not os.path.exists(file_path) or not SENDER_EMAIL or not RECEIVER_EMAILS:
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = MIMEMultipart()
    msg["Subject"] = f"📊 [안정화 매집분석] 업비트 일봉 마감 기준 리포트 ({now_str})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECEIVER_EMAILS)
    
    body = f"""안녕하세요.

마감 일봉 기준 정적 매집 분석 및 Gemini AI 리포트가 완성되었습니다.

• 분석 시각: {now_str}

==================================================
🤖 [Gemini AI 고정 리포트 (Temperature=0.0)]
==================================================
{ai_analysis}

자세한 데이터는 첨부 엑셀 파일을 참고하세요.
"""
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        part.add_header("Content-Disposition", "attachment", filename=("utf-8", "", os.path.basename(file_path)))
        msg.attach(part)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        server.quit()
        print("📧 이메일 발송 성공!")
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

if __name__ == "__main__":
    df_ranked = scan_and_rank_coins()
    ai_summary = generate_gemini_analysis(df_ranked)
    excel_file = save_daily_excel_sheet(df_ranked)
    send_email_with_excel(excel_file, ai_summary)
