import os
import time
import datetime
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

EXCEL_FILE_PATH = "업비트_원화마켓_매집점수_날짜별기록.xlsx"

# ==============================================================================
# [유틸] 업비트 '원화(KRW) 마켓' 전 종목 및 4시간봉(1주일) 조회
# ==============================================================================
def get_krw_upbit_tickers():
    url = "https://api.upbit.com/v1/market/all"
    headers = {"accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        krw_coins = []
        for coin in data:
            if coin['market'].startswith("KRW-"):
                krw_coins.append({
                    'ticker': coin['market'],
                    'korean_name': coin['korean_name'],
                    'english_name': coin['english_name'],
                    'symbol': coin['market'].replace("KRW-", "")
                })
        return krw_coins
    except Exception as e:
        print(f"❌ 업비트 원화 코인 목록 조회 실패: {e}")
        return []

def get_4h_ohlcv_summary(symbol):
    """특정 종목의 최근 1주일(4시간봉 42개 = 168시간) 수급 및 변동성 분석 요약"""
    ticker = f"KRW-{symbol}"
    try:
        df_4h = pyupbit.get_ohlcv(ticker, interval="minute240", count=42)
        if df_4h is None or df_4h.empty:
            return "4시간봉 데이터 수집 불가"

        recent_vol_avg = df_4h['volume'].iloc[:-1].mean()
        latest_vol = df_4h['volume'].iloc[-1]
        vol_surge_4h = round(latest_vol / recent_vol_avg, 2) if recent_vol_avg > 0 else 1.0

        first_open_7d = df_4h['open'].iloc[0]
        latest_close = df_4h['close'].iloc[-1]
        price_change_7d = round(((latest_close - first_open_7d) / first_open_7d) * 100, 2)

        max_price_7d = df_4h['high'].max()
        min_price_7d = df_4h['low'].min()
        
        return f"최근 1주일(4시간봉 기준) 변동률: {price_change_7d}%, 직전 대비 거래량: {vol_surge_4h}배, 1주일 최고가: {max_price_7d:,.0f}원 / 최저가: {min_price_7d:,.0f}원"
    except Exception as e:
        return f"4시간봉 조회 오류: {e}"

# ==============================================================================
# [고급 보조 지표 계산] OBV, 볼린저밴드 수축도, 매집봉 감지
# ==============================================================================
def calculate_advanced_metrics(df_daily):
    """OBV 우상향 디버전스, 볼린저밴드 수축도, 최근 매집봉 출현 횟수 계산"""
    df = df_daily.copy()
    
    # 1. OBV 누적 계산
    obv_values = [0.0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv_values.append(obv_values[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv_values.append(obv_values[-1] - df['volume'].iloc[i])
        else:
            obv_values.append(obv_values[-1])
    df['obv'] = obv_values
    
    # 최근 10일간 OBV 기울기 및 가격 변동
    obv_slope = df['obv'].iloc[-1] - df['obv'].iloc[-10] if len(df) >= 10 else 0
    price_slope = df['close'].iloc[-1] - df['close'].iloc[-10] if len(df) >= 10 else 0
    price_change_10d = (price_slope / df['close'].iloc[-10]) * 100 if len(df) >= 10 and df['close'].iloc[-10] > 0 else 0
    
    # OBV 디버전스: 주가는 비슷/하락(-5%~+4%) 중인데 OBV 수급은 우상향
    is_obv_divergence = (price_change_10d <= 4.0) and (obv_slope > 0)

    # 2. 볼린저 밴드 수축도 (Band Width %)
    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    bandwidth = ((upper - lower) / ma20).iloc[-1] * 100 if ma20.iloc[-1] > 0 else 999.0

    # 3. 최근 7일 내 '매집봉' 발생 횟수 (거래량 2.5배 이상 & 당일 몸통 상승률 0.5%~6.0%)
    recent_7d = df.iloc[-7:]
    prev_vol_avg = df['volume'].iloc[-27:-7].mean() if len(df) >= 27 else df['volume'].mean()
    
    accumulation_candles = 0
    if prev_vol_avg > 0:
        for idx, row in recent_7d.iterrows():
            vol_ratio = row['volume'] / prev_vol_avg
            p_change = ((row['close'] - row['open']) / row['open']) * 100 if row['open'] > 0 else 0
            if vol_ratio >= 2.5 and 0.5 <= p_change <= 6.0:
                accumulation_candles += 1

    return is_obv_divergence, bandwidth, accumulation_candles

# ==============================================================================
# [알고리즘] 정교화된 매집 점수 산출 (급등주 완벽 필터링 반영)
# ==============================================================================
def calculate_score(vol_ratio, trade_value_krw, price_change, is_yangbong, ma_gap, 
                    is_obv_divergence, bandwidth, accumulation_candles, surge_from_bottom):
    score = 0

    # -------------------------------------------------------------
    # [강력 필터 1] 바닥 대비 이미 너무 많이 상승한 종목 대폭 감점 (고점 잡기 방지)
    # -------------------------------------------------------------
    if surge_from_bottom >= 50.0:
        return 0  # 최근 20일 최저가 대비 50% 이상 급등한 종목은 매집 제외 (0점 처리)
    elif surge_from_bottom >= 30.0:
        score -= 40  # 30%~50% 상승 구간도 강력 감점

    # -------------------------------------------------------------
    # [핵심 1] 당일 과열 페널티 (급등주 배제 및 조용한 양봉 우대)
    # -------------------------------------------------------------
    if price_change > 12.0:
        score -= 30  # 이미 급등한 코인은 매집 종료/설거지 파동으로 판단하여 차감
    elif price_change > 7.0:
        score -= 10  # +7% 이상도 매집 타점으로는 다소 과열
    elif is_yangbong and 0.5 <= price_change <= 4.0:
        score += 25  # 주가가 거의 오르지 않은 고요한 양봉에 최고 점수 부여

    # -------------------------------------------------------------
    # [핵심 2] 매집 흔적 및 거래량 수급 평가
    # -------------------------------------------------------------
    if accumulation_candles >= 2:
        score += 30
    elif accumulation_candles == 1:
        score += 15

    if vol_ratio >= 3.0: score += 20
    elif vol_ratio >= 1.8: score += 12
    elif vol_ratio >= 1.2: score += 5

    # -------------------------------------------------------------
    # [핵심 3] 지표 디버전스 및 차트 에너지 축적(횡보/수축)
    # -------------------------------------------------------------
    if is_obv_divergence:
        score += 25

    if ma_gap <= 3.0: score += 15
    elif ma_gap <= 6.0: score += 10

    if bandwidth <= 8.0: score += 15
    elif bandwidth <= 12.0: score += 8

    trade_value_eow = trade_value_krw / 100_000_000
    if trade_value_eow >= 50: score += 15
    elif trade_value_eow >= 10: score += 10
    elif trade_value_eow >= 5: score += 5

    return max(0, score)  # 음수 방지

# ==============================================================================
# [외부 데이터 수집] 1. 코인니스 속보 | 2. 구글 트렌드 | 3. 업비트 공지 | 4. X(트위터)
# ==============================================================================
def get_coinness_news(coin_name):
    try:
        url = f"https://coinness.com/search?q={coin_name}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers, timeout=3)
        
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            titles = [a.text.strip() for a in soup.find_all('h3')[:2]]
            if titles:
                return " | ".join(titles)
        return "특이 속보 없음 (고요한 상태)"
    except Exception:
        return "속보 조회 불가"

def get_google_search_trend(coin_names):
    try:
        pytrends = TrendReq(hl='ko', tz=540)
        keywords = [name + " 코인" for name in coin_names[:5]]
        
        pytrends.build_payload(keywords, timeframe='now 7-d', geo='KR')
        data = pytrends.interest_over_time()
        
        if not data.empty:
            latest_trend = data.iloc[-1].to_dict()
            return {k.replace(" 코인", ""): int(v) for k, v in latest_trend.items() if k != 'isPartial'}
        return {}
    except Exception as e:
        print(f"⚠️ 구글 트렌드 수집 건너뜀: {e}")
        return {}

def get_upbit_notices(coin_name, symbol):
    try:
        url = "https://api-manager.upbit.com/api/v1/notices?page=1&per_page=15"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers, timeout=3)
        
        if res.status_code == 200:
            notices_list = res.json().get('data', {}).get('list', [])
            matched_notices = []
            
            for notice in notices_list:
                title = notice.get('title', '')
                if coin_name in title or symbol in title:
                    matched_notices.append(title)
            
            if matched_notices:
                return " | ".join(matched_notices[:2])
                
        return "최근 업비트 공식 공지 없음 (안전)"
    except Exception as e:
        return "업비트 공지 조회 불가"

def get_x_twitter_sentiment(symbol, coin_name):
    try:
        nitter_instances = [
            "https://nitter.net",
            "https://nitter.cz",
            "https://nitter.privacydev.net"
        ]
        query = f"${symbol} OR {coin_name}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        for instance in nitter_instances:
            try:
                url = f"{instance}/search?f=tweets&q={query}"
                res = requests.get(url, headers=headers, timeout=2.5)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    tweets = [t.text.strip().replace('\n', ' ') for t in soup.find_all('div', class_='tweet-content')[:3]]
                    if tweets:
                        return " // ".join(tweets)
            except Exception:
                continue
                
        return "X(트위터) 실시간 특이 언급 적음"
    except Exception:
        return "X(트위터) 조회 불가"

# ==============================================================================
# [분석 엔진] 전 종목 조사
# ==============================================================================
def scan_and_rank_coins():
    print("--------------------------------------------------")
    print("🚀 업비트 원화(KRW) 마켓 전 종목 전수 조사 시작...")
    
    krw_coins = get_krw_upbit_tickers()
    if not krw_coins:
        return pd.DataFrame()

    print(f"총 {len(krw_coins)}개 원화 마켓 종목 분석 중...\n")
    results = []
    
    for item in krw_coins:
        ticker = item['ticker']
        korean_name = item['korean_name']
        symbol = item['symbol']
        
        df_daily = None
        for retry in range(3):
            try:
                time.sleep(0.06)
                df_daily = pyupbit.get_ohlcv(ticker, interval="day", count=60)
                if df_daily is not None and not df_daily.empty:
                    break
            except Exception:
                time.sleep(0.1)

        if df_daily is None or df_daily.empty:
            results.append({
                "코인명": korean_name,
                "심볼": symbol,
                "매집점수": 0,
                "당일 변동률(%)": 0.0,
                "바닥 대비 상승률(%)": 0.0,
                "거래량 급증(배)": 0.0,
                "거래대금(억원)": 0.0,
                "현재가(KRW)": 0,
                "이평선 수렴도(%)": 0.0,
                "비고": "OHLCV 데이터 미제공(신규/정지)"
            })
            continue

        try:
            latest = df_daily.iloc[-1]
            
            # 최근 20일 최저가 대비 상승률 계산 (고점 추격 방지용)
            lowest_20d = df_daily['low'].iloc[-20:].min() if len(df_daily) >= 20 else df_daily['low'].min()
            surge_from_bottom = round(((latest['close'] - lowest_20d) / lowest_20d) * 100, 2) if lowest_20d > 0 else 0.0
            
            if len(df_daily) >= 21:
                prev_20_vol_avg = df_daily['volume'].iloc[-21:-1].mean()
            else:
                prev_20_vol_avg = df_daily['volume'].mean()

            vol_ratio = (latest['volume'] / prev_20_vol_avg) if prev_20_vol_avg > 0 else 0
            trade_value_krw = latest['value']
            
            if len(df_daily) >= 2:
                price_change = ((latest['close'] - df_daily['close'].iloc[-2]) / df_daily['close'].iloc[-2]) * 100
            else:
                price_change = 0.0
                
            is_yangbong = latest['close'] >= latest['open']
            
            ma20 = df_daily['close'].rolling(20).mean().iloc[-1] if len(df_daily) >= 20 else latest['close']
            ma60 = df_daily['close'].rolling(60).mean().iloc[-1] if len(df_daily) >= 60 else ma20
            ma_gap = abs(ma20 - ma60) / ma20 * 100 if ma20 > 0 else 0

            # 정교화된 고급 매집 지표 계산
            is_obv_div, bandwidth, accum_candles = calculate_advanced_metrics(df_daily)

            # 수정된 점수 알고리즘 적용 (surge_from_bottom 전달)
            total_score = calculate_score(
                vol_ratio, trade_value_krw, price_change, is_yangbong, ma_gap,
                is_obv_div, bandwidth, accum_candles, surge_from_bottom
            )
            
            results.append({
                "코인명": korean_name,
                "심볼": symbol,
                "매집점수": total_score,
                "당일 변동률(%)": round(price_change, 2),
                "바닥 대비 상승률(%)": surge_from_bottom,
                "거래량 급증(배)": round(vol_ratio, 2),
                "거래대금(억원)": round(trade_value_krw / 100_000_000, 1),
                "현재가(KRW)": latest['close'],
                "이평선 수렴도(%)": round(ma_gap, 2),
                "비고": "정상"
            })
            print(f"  [수집 완료] {korean_name}({symbol}) | 매집점수: {total_score}점 | 바닥대비: +{surge_from_bottom}%")
                
        except Exception as e:
            results.append({
                "코인명": korean_name,
                "심볼": symbol,
                "매집점수": 0,
                "당일 변동률(%)": 0.0,
                "바닥 대비 상승률(%)": 0.0,
                "거래량 급증(배)": 0.0,
                "거래대금(억원)": 0.0,
                "현재가(KRW)": 0,
                "이평선 수렴도(%)": 0.0,
                "비고": f"연산 오류({e})"
            })
            continue
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="매집점수", ascending=False)
    return df

# ==============================================================================
# [AI 심층 분석] Google Gemini AI + 4시간봉(1주일) 수급 + 이슈 + X(트위터) 융합 브리핑
# ==============================================================================
def generate_gemini_analysis(df):
    if not GEMINI_API_KEY:
        print("❌ [경고] GEMINI_API_KEY 환경 변수가 없습니다.")
        return "Gemini API 키가 설정되지 않아 AI 요약이 생성되지 않았습니다."

    if df.empty:
        print("⚠️ 데이터가 없어 AI 분석을 건너뜁니다.")
        return "데이터가 없어 AI 요약을 생성하지 못했습니다."

    try:
        print("\n🤖 일봉 차트 + 4시간봉(1주일) 수급 + 이슈 + 공지 + X(트위터) Google Gemini 분석 시작...")
        top_10 = df.head(10).copy()
        
        top_5_names = top_10['코인명'].tolist()[:5]
        search_trends = get_google_search_trend(top_5_names)
        
        enriched_data = []
        for idx, row in top_10.iterrows():
            coin_name = row['코인명']
            symbol = row['심볼']
            
            info_4h = get_4h_ohlcv_summary(symbol)
            news = get_coinness_news(coin_name)
            upbit_notice = get_upbit_notices(coin_name, symbol)
            x_tweets = get_x_twitter_sentiment(symbol, coin_name)
            trend_score = search_trends.get(coin_name, "수집 미지원")
            
            enriched_data.append({
                "코인명": coin_name,
                "매집점수": row['매집점수'],
                "당일변동률": f"{row['당일 변동률(%)']}%",
                "바닥대비상승률": f"{row['바닥 대비 상승률(%)']}%",
                "거래량급증": f"{row['거래량 급증(배)']}배",
                "거래대금": f"{row['거래대금(억원)']}억원",
                "4시간봉_1주일_수급분석": info_4h,
                "구글검색관심도": trend_score,
                "코인니스속보": news,
                "업비트공지사항": upbit_notice,
                "X(트위터)최근의견": x_tweets
            })

        prompt = f"""
당신은 가상자산 전문 수석 애널리스트입니다.
아래 수집된 상위 10개 코인의 일봉 매집 점수 및 [최근 1주일간의 4시간봉 수급 데이터]를 바탕으로, 정중하고 전문적인 어조(~입니다, ~습니다)로 분석 리포트를 작성해 주세요.

[데이터 종합 표]
{enriched_data}

[❌ 절대 금지 사항]
- 반말 및 명사형 종결(~함, ~임)을 금지합니다. 반드시 정중한 경어체(~습니다, ~입니다)로 작성하세요.
- "조용한 매집 구간으로 해석됩니다", "개미 꼬시기 물량일 수 있습니다" 등 상투적 템플릿 문구를 절대 사용하지 마세요.

[작성 지침 및 차별화 원칙]
1. **[추천 종목 Top 3 및 4시간봉(1주일) 정밀 분석] 필수 포함**:
   - 1위, 2위, 3위 추천 종목을 명확히 지정하고, 일봉 매집 흐름뿐만 아니라 **[최근 1주일간 4시간봉 데이터(1주일 변동률, 최고/최저가, 최근 거래량)]를 직접 인용하여 단/중기 진입 및 지지/저항 구간**을 정밀하게 분석해 주세요.
2. **[주의/과열 유의 종목 1개] 필수 포함**:
   - 매집 점수는 높지만 거래대금이 미미하거나 4시간봉 기준 최근 1주일간 고점 대비 변동폭이 커 위험한 1개 종목을 찍어 경고해 주세요.
3. 데이터의 숫자를 직접 인용하여 객관적이고 날카롭게 서술하세요.
"""
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
        
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
        )
        
        result_text = response.text
        if result_text:
            print("✅ Gemini AI 종합 융합 분석 성공!")
            return result_text.strip()
        else:
            return "Gemini AI 응답이 비어 있습니다."

    except Exception as e:
        print(f"❌ Gemini AI 예외 발생: {e}")
        return f"Gemini AI 분석 생성 중 오류가 발생했습니다: {e}"

# ==============================================================================
# [엑셀 저장]
# ==============================================================================
def save_daily_excel_sheet(df):
    if df.empty:
        return None

    now = datetime.datetime.now()
    sheet_name = now.strftime("%Y-%m-%d_%H시")
    
    if os.path.exists(EXCEL_FILE_PATH):
        wb = openpyxl.load_workbook(EXCEL_FILE_PATH)
    else:
        wb = openpyxl.Workbook()
        
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
            cell.alignment = Alignment(horizontal="center" if col != 8 else "right", vertical="center")
            
            if is_high_score:
                cell.fill = high_score_fill
                
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 5, 12)

    wb.save(EXCEL_FILE_PATH)
    wb.close()
    
    print("--------------------------------------------------")
    print(f"📁 [{sheet_name}] 시트에 총 {len(df)}개 원화 코인 수집 완료!")
    return EXCEL_FILE_PATH

# ==============================================================================
# [메일 전송]
# ==============================================================================
def send_email_with_excel(file_path, ai_analysis=""):
    if not file_path or not os.path.exists(file_path):
        return

    if not SENDER_EMAIL or not EMAIL_PASSWORD or not RECEIVER_EMAILS:
        print("❌ 환경변수(Secrets) 설정에 문제가 있어 이메일을 발송하지 못했습니다.")
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    msg = MIMEMultipart()
    msg["Subject"] = f"📊 [업비트 Gemini AI 심층분석] 원화마켓 매집 점수 리포트 ({now_str})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECEIVER_EMAILS)
    
    body = f"""안녕하세요.

업비트 원화(KRW) 마켓 전체 상장 종목 분석 및 Google Gemini AI 종합 이슈 결합 리포트가 완료되었습니다.

• 분석 시각: {now_str}
• 분석 대상: 원화 마켓 전체 종목

==================================================
🤖 [Gemini AI 차트+4시간봉(1주일)수급+이슈+공지+X(트위터) 종합 브리핑]
==================================================
{ai_analysis}

==================================================
자세한 데이터는 첨부된 엑셀 파일을 확인해 주세요.
"""
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
        file_name = os.path.basename(file_path)
        part.add_header("Content-Disposition", "attachment", filename=("utf-8", "", file_name))
        msg.attach(part)
    except Exception as e:
        print(f"❌ 파일 첨부 실패: {e}")
        return

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        server.quit()
        print(f"📧 엑셀 + Gemini AI 심층 브리핑 이메일 발송 성공! (수신자: {', '.join(RECEIVER_EMAILS)})")
        print("--------------------------------------------------")
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

if __name__ == "__main__":
    df_ranked = scan_and_rank_coins()
    ai_summary = generate_gemini_analysis(df_ranked)
    excel_file = save_daily_excel_sheet(df_ranked)
    send_email_with_excel(excel_file, ai_summary)
