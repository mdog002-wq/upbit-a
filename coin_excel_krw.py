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

from google import genai
from google.genai import types
from tqdm import tqdm

# ==============================================================================
# [설정] GitHub Secrets 및 환경 변수
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

EXCEL_FILE_PATH = "업비트_원화마켓_매집_패턴분석_리포트.xlsx"
CACHE_TOKENOMICS_FILE = "cache_tokenomics.json"
CACHE_GITHUB_FILE = "cache_github.json"

# ==============================================================================
# [유틸] 데이터 포맷팅 및 캐싱 기능
# ==============================================================================
def format_price(x):
    """가격 조건별 소수점 포맷팅 및 지수 표기법 방지"""
    try:
        val = float(x)
        if val >= 100:
            return f"{int(val):,}"
        elif val >= 1:
            return f"{val:,.2f}"
        else:
            return f"{val:,.5f}"
    except Exception:
        return x

def format_number(x, decimals=2):
    """숫자 포맷팅 (콤마 및 소수점 자리수 지정)"""
    try:
        val = float(x)
        return f"{val:,.{decimals}f}"
    except Exception:
        return x

def load_cache(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
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
# [업비트 마켓 및 외부 API 연동]
# ==============================================================================
def get_krw_upbit_tickers():
    url = "https://api.upbit.com/v1/market/all?isDetails=false"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return [
                {
                    'ticker': coin['market'],
                    'korean_name': coin['korean_name'],
                    'symbol': coin['market'].replace("KRW-", "")
                }
                for coin in res.json() if coin['market'].startswith("KRW-")
            ]
    except Exception as e:
        print(f"❌ 업비트 원화 코인 목록 조회 실패: {e}")
    return []

def get_4h_ohlcv_summary(symbol):
    """마감된 4시간봉 기준 최근 1주일 정적 분석"""
    ticker = f"KRW-{symbol}"
    try:
        df_4h = pyupbit.get_ohlcv(ticker, interval="minute240", count=43)
        if df_4h is None or len(df_4h) < 42:
            return "4시간봉 데이터 수집 불가"

        df_closed = df_4h.iloc[:-1].copy()
        recent_vol_avg = df_closed['volume'].iloc[:-1].mean()
        latest_vol = df_closed['volume'].iloc[-1]
        vol_surge_4h = round(latest_vol / recent_vol_avg, 2) if recent_vol_avg > 0 else 1.0

        first_open_7d = df_closed['open'].iloc[0]
        latest_close = df_closed['close'].iloc[-1]
        price_change_7d = round(((latest_close - first_open_7d) / first_open_7d) * 100, 2)

        max_price_7d = df_closed['high'].max()
        min_price_7d = df_closed['low'].min()
        
        return f"최근 1주일(4h 마감) 변동률: {price_change_7d}%, 직전 대비 거래량: {vol_surge_4h}배, 최고: {max_price_7d:,.0f}원 / 최저: {min_price_7d:,.0f}원"
    except Exception as e:
        return f"4시간봉 조회 오류: {e}"

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
            time.sleep(0.3)
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
# [알고리즘] 코사인 유사도 & 매집 지표 산출
# ==============================================================================
def calculate_cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cosine_sim = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return float(max(0, cosine_sim) * 100)

def extract_pre_spike_patterns(df):
    patterns = []
    df["prev_close"] = df["close"].shift(1)
    df["daily_high_rate"] = ((df["high"] - df["prev_close"]) / df["prev_close"]) * 100
    df["vol_ma5"] = df["volume"].rolling(window=5).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].shift(1)
    df["volatility"] = ((df["high"] - df["low"]) / df["close"]) * 100
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["disparity_20"] = ((df["close"] - df["ma20"]) / df["ma20"]) * 100
    df["daily_return"] = ((df["close"] - df["prev_close"]) / df["prev_close"]) * 100

    for i in range(20, len(df) - 1):
        if df["daily_high_rate"].iloc[i] >= 20.0:
            t_minus_1 = df.iloc[i - 1]
            vol_r = t_minus_1["vol_ratio"] if not np.isnan(t_minus_1["vol_ratio"]) else 1.0
            vola = t_minus_1["volatility"] if not np.isnan(t_minus_1["volatility"]) else 0.0
            disp = t_minus_1["disparity_20"] if not np.isnan(t_minus_1["disparity_20"]) else 0.0
            ret = t_minus_1["daily_return"] if not np.isnan(t_minus_1["daily_return"]) else 0.0
            patterns.append([vol_r, vola, disp, ret])

    return patterns, df

def calculate_stable_metrics(df_daily):
    df = df_daily.iloc[:-1].copy()
    if len(df) < 30:
        return None

    close = df['close']
    volume = df['volume']
    
    chg_1d = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
    chg_7d = ((close.iloc[-1] - close.iloc[-8]) / close.iloc[-8]) * 100 if len(df) >= 8 else 0

    vwap = (df['value'].iloc[-14:].sum()) / (volume.iloc[-14:].sum()) if volume.iloc[-14:].sum() > 0 else close.iloc[-1]
    vwap_gap = round(((close.iloc[-1] - vwap) / vwap) * 100, 2)

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

    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    bandwidth = ((upper - lower) / ma20).iloc[-1] * 100 if ma20.iloc[-1] > 0 else 999.0

    recent_7d = df.iloc[-7:]
    prev_vol_avg = volume.iloc[-27:-7].mean() if len(df) >= 27 else volume.mean()
    accum_candles = 0
    if prev_vol_avg > 0:
        for idx, row in recent_7d.iterrows():
            vol_ratio = row['volume'] / prev_vol_avg
            p_change = ((row['close'] - row['open']) / row['open']) * 100 if row['open'] > 0 else 0
            if vol_ratio >= 2.3 and 0.3 <= p_change <= 5.5:
                accum_candles += 1

    vol_ratio = (volume.iloc[-1] / prev_vol_avg) if prev_vol_avg > 0 else 1.0

    return {
        "chg_1d": round(chg_1d, 2),
        "chg_7d": round(chg_7d, 2),
        "vwap_gap": vwap_gap,
        "is_obv_div": is_obv_divergence,
        "bandwidth": round(bandwidth, 2),
        "accum_candles": accum_candles,
        "vol_ratio": round(vol_ratio, 2),
        "last_close": close.iloc[-1],
        "last_value": df['value'].iloc[-1]
    }

def calculate_stable_score(metrics, surge_from_bottom, circ_ratio, is_dev_active):
    score = 0
    if surge_from_bottom >= 45.0:
        return 0
    elif surge_from_bottom >= 25.0:
        score -= 30

    if 0.0 <= metrics['chg_1d'] <= 3.5 and -5.0 <= metrics['chg_7d'] <= 5.0:
        score += 25
    elif metrics['chg_1d'] > 10.0:
        score -= 25

    if metrics['vol_ratio'] >= 2.5: score += 20
    elif metrics['vol_ratio'] >= 1.5: score += 12

    if -2.0 <= metrics['vwap_gap'] <= 3.0: score += 15
    if metrics['is_obv_div']: score += 20
    if metrics['bandwidth'] <= 8.0: score += 15
    if metrics['accum_candles'] >= 2: score += 25
    elif metrics['accum_candles'] == 1: score += 12

    if circ_ratio >= 60.0: score += 10
    elif circ_ratio < 30.0: score -= 15

    if is_dev_active: score += 10
    if (metrics['last_value'] / 100_000_000) >= 30: score += 10

    return max(0, score)

# ==============================================================================
# [스캔 및 분석 엔진]
# ==============================================================================
def analyze_and_scan_market():
    krw_coins = get_krw_upbit_tickers()
    if not krw_coins:
        return pd.DataFrame()

    symbols = [c['symbol'] for c in krw_coins]
    tokenomics_map = get_cached_coingecko_tokenomics(symbols)
    github_map = get_cached_github_activity(symbols)

    print("\n[1/2] 30분봉 수급 집중도(최근 12시간 15위 진입 횟수) 수집 중...")
    hourly_rank_details = {c['ticker']: [] for c in krw_coins}
    market_30m_data = {}
    time_12h_ago = datetime.datetime.now() - datetime.timedelta(hours=12)

    for c in krw_coins:
        try:
            min_df = pyupbit.get_ohlcv(c['ticker'], interval="minute30", count=30)
            if min_df is not None and not min_df.empty:
                filtered_df = min_df[min_df.index >= time_12h_ago]
                if not filtered_df.empty:
                    market_30m_data[c['ticker']] = filtered_df
            time.sleep(0.02)
        except Exception:
            continue

    if market_30m_data:
        sample_ticker = list(market_30m_data.keys())[0]
        timestamps = market_30m_data[sample_ticker].index
        for i in range(len(timestamps)):
            ts_values = []
            for t, m_df in market_30m_data.items():
                if i < len(m_df):
                    ts_values.append((t, m_df["value"].iloc[i]))
            ts_values.sort(key=lambda x: x[1], reverse=True)
            for rank, (t, _) in enumerate(ts_values[:15], start=1):
                if t in hourly_rank_details:
                    hourly_rank_details[t].append(rank)

    print("\n[2/2] 일봉 데이터 종합 수급 및 T-1 패턴 유사도 분석 중...")
    results = []

    for item in tqdm(krw_coins, desc="종합 종목 분석", ncols=100):
        ticker = item['ticker']
        korean_name = item['korean_name']
        symbol = item['symbol']

        try:
            df_daily = pyupbit.get_ohlcv(ticker, interval="day", count=120)
            if df_daily is None or len(df_daily) < 30:
                time.sleep(0.03)
                continue

            metrics = calculate_stable_metrics(df_daily)
            if not metrics:
                continue

            # 1. 매집 점수 계산
            df_closed = df_daily.iloc[:-1]
            lowest_20d = df_closed['low'].iloc[-20:].min()
            surge_from_bottom = round(((metrics['last_close'] - lowest_20d) / lowest_20d) * 100, 2) if lowest_20d > 0 else 0.0
            circ_ratio = tokenomics_map.get(symbol, 75.0)
            is_dev_active = github_map.get(symbol, False)
            accumulation_score = calculate_stable_score(metrics, surge_from_bottom, circ_ratio, is_dev_active)

            # 2. 패턴 코사인 유사도 계산
            patterns, processed_df = extract_pre_spike_patterns(df_daily)
            spike_count = len(patterns)
            curr = processed_df.iloc[-1]
            curr_vector = [
                curr["vol_ratio"] if not np.isnan(curr["vol_ratio"]) else 1.0,
                curr["volatility"] if not np.isnan(curr["volatility"]) else 0.0,
                curr["disparity_20"] if not np.isnan(curr["disparity_20"]) else 0.0,
                curr["daily_return"] if not np.isnan(curr["daily_return"]) else 0.0,
            ]
            similarity_score = calculate_cosine_similarity(curr_vector, np.mean(patterns, axis=0)) if spike_count > 0 else 0.0

            # 3. 30분봉 단기 방향성 분석
            min_df = pyupbit.get_ohlcv(ticker, interval="minute30", count=10)
            recent_buy_sell_trend = "보통"
            net_buy_amount = 0
            if min_df is not None and len(min_df) > 0:
                last_val = min_df["value"].iloc[-1]
                if min_df["close"].iloc[-1] >= min_df["open"].iloc[-1]:
                    net_buy_amount = last_val * 0.6
                    recent_buy_sell_trend = "매수세 우위"
                else:
                    net_buy_amount = -(last_val * 0.6)
                    recent_buy_sell_trend = "매도세 우위"

            rank_count = len(hourly_rank_details.get(ticker, []))
            vol_mean_20 = df_daily["volume"].tail(20).mean()
            vol_mean_total = df_daily["volume"].mean()
            vol_ratio_pattern = vol_mean_20 / vol_mean_total if vol_mean_total > 0 else 0

            # 패턴 예측 종합 점수
            pattern_prediction_score = (similarity_score * 0.5) + (spike_count * 3) + (rank_count * 2) + (vol_ratio_pattern * 5)

            results.append({
                "코인명": f"{korean_name} 🔥" if recent_buy_sell_trend == "매수세 우위" else korean_name,
                "심볼": symbol,
                "매집점수": accumulation_score,
                "패턴유사도(%)": round(similarity_score, 1),
                "종합예측점수": round(pattern_prediction_score, 2),
                "현재가(KRW)": format_price(metrics['last_close']),
                "15위내_진입(회)": rank_count,
                "1일 변동률(%)": metrics['chg_1d'],
                "7일 변동률(%)": metrics['chg_7d'],
                "바닥대비상승(%)": surge_from_bottom,
                "거래량급증(배)": metrics['vol_ratio'],
                "과거급등(회)": spike_count,
                "20일이평괴리도(%)": round(curr["disparity_20"], 2) if not np.isnan(curr["disparity_20"]) else 0.0,
                "순수급금액(KRW)": format_number(net_buy_amount, 0),
                "30분 수급": recent_buy_sell_trend,
                "유통량비율(%)": circ_ratio,
                "거래대금(억원)": round(metrics['last_value'] / 100_000_000, 1),
                "개발활력": "양호" if is_dev_active else "보통"
            })
            time.sleep(0.03)

        except Exception:
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["매집점수", "패턴유사도(%)", "종합예측점수"], ascending=[False, False, False]).reset_index(drop=True)
    return df

# ==============================================================================
# [AI 심층 분석] Gemini 3.1 Flash Lite
# ==============================================================================
def generate_gemini_analysis(df):
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY가 설정되지 않았습니다."
    if df.empty:
        return "분석할 데이터가 없습니다."

    try:
        top_10 = df.head(10).copy()
        enriched_data = []

        for idx, row in top_10.iterrows():
            symbol = row['심볼']
            info_4h = get_4h_ohlcv_summary(symbol)
            
            enriched_data.append({
                "코인명": row['코인명'],
                "매집점수": row['매집점수'],
                "패턴유사도": f"{row['패턴유사도(%)']}%",
                "종합예측점수": row['종합예측점수'],
                "1일/7일변동률": f"{row['1일 변동률(%)']}% / {row['7일 변동률(%)']}%",
                "바닥대비상승률": f"{row['바닥대비상승(%)']}%",
                "거래량급증": f"{row['거래량급증(배)']}배",
                "30분수급": row['30분 수급'],
                "4시간봉_1주일_수급분석": info_4h
            })

        prompt = f"""
당신은 가상자산 수급 및 패턴 분석 전문가입니다.
아래 상위 10개 코인의 [일봉 매집 점수], [T-1 급등 패턴 코사인 유사도], 및 [최근 1주일간 4시간봉 수급 데이터]를 바탕으로 정중한 경어체(~습니다, ~입니다)로 정밀 분석 리포트를 작성해 주세요.

[상위 10개 통합 분석 데이터]
{enriched_data}

[작성 지침 및 차별화 원칙]
1. **[추천 종목 Top 3 및 수급/패턴 정밀 분석] 필수 포함**:
   - 1위, 2위, 3위 추천 종목을 지정하고, 매집 점수와 함께 **[최근 1주일간 4시간봉 데이터(1주일 변동률, 최고/최저가, 직전 대비 거래량)] 및 [30분 수급/패턴 유사도]를 직접 인용**하여 진입 타점과 주요 지지/저항 구간을 제시하세요.
2. **[주의/과열 유의 종목 1개] 필수 포함**:
   - 점수는 높으나 바닥 대비 변동폭이 지나치게 크거나 고점 대비 낙폭/매도세 우위로 위험성이 존재하는 1개 종목을 지정해 주의점 및 대응책을 기술하세요.
"""
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
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
# [엑셀 리포트 저장 및 이메일 전송]
# ==============================================================================
def save_integrated_excel(df):
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
    high_score_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid") # 연한 노란색
    
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9')
    )
    
    # 헤더 스타일링
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    # 종합예측점수 상위 5개 인덱스 계산
    top5_pred_indices = set(df.nlargest(5, "종합예측점수").index)

    # 데이터 행 스타일링
    for row_idx, row in enumerate(range(2, ws.max_row + 1)):
        df_idx = row_idx
        accum_score_cell = ws.cell(row=row, column=3) # 매집점수 (3번째 컬럼)
        
        # 매집점수 >= 70 이거나 종합예측점수 상위 5위인 경우 강조
        is_high_score = (accum_score_cell.value and float(accum_score_cell.value) >= 70) or (df_idx in top5_pred_indices)
        
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = data_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if is_high_score:
                cell.fill = high_score_fill

    # 컬럼 폭 자동 조절
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    wb.save(EXCEL_FILE_PATH)
    wb.close()
    return EXCEL_FILE_PATH

def send_email_report(file_path, ai_analysis):
    if not file_path or not os.path.exists(file_path) or not SENDER_EMAIL or not RECEIVER_EMAILS:
        print("⚠️ 이메일 발송 조건이 충족되지 않았습니다 (환경 변수/파일 확인 필요).")
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = MIMEMultipart()
    msg["Subject"] = f"📊 [매집+급등패턴 통합 리포트] 업비트 종합 수급 브리핑 ({now_str})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECEIVER_EMAILS)
    
    body = f"""안녕하세요.

업비트 원화 마켓 [일봉 매집 지표]와 [과거 T-1 급등 패턴 코사인 유사도]를 종합 산출한 리포트입니다.

• 분석 시각: {now_str}

==================================================
🤖 [Gemini AI 수급 & 패턴 심층 분석 리포트]
==================================================
{ai_analysis}

상세 종목 분석 데이터는 첨부된 엑셀 파일 시트를 확인해 주세요.
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
        print("📧 통합 리포트 이메일 발송 완료!")
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

# ==============================================================================
# [메인 실행부]
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    print("🚀 [업비트 원화 마켓] 매집 분석 + T-1 급등 패턴 통합 분석 프로세스 시작...")
    
    df_result = analyze_and_scan_market()
    
    if not df_result.empty:
        print("\n=== 🎯 상위 5개 종합 추천 종목 미리보기 ===")
        print(df_result[["코인명", "매집점수", "패턴유사도(%)", "종합예측점수", "30분 수급", "거래대금(억원)"]].head(5))

        print("\n🤖 Gemini AI 브리핑 리포트 생성 중...")
        ai_summary = generate_gemini_analysis(df_result)
        
        print("\n📊 엑셀 저장 및 스타일링 적용 중...")
        excel_file = save_integrated_excel(df_result)
        
        print("\n📧 결과 메일 전송 중...")
        send_email_report(excel_file, ai_summary)
    else:
        print("❌ 분석된 결과가 없습니다.")

    print(f"\n✨ 전체 프로세스 완료 (총 소요 시간: {round(time.time() - start_time, 2)}초)")
