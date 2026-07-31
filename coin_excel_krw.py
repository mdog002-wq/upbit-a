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
HISTORY_CSV_PATH = "scan_history.csv"
CACHE_TOKENOMICS_FILE = "cache_tokenomics.json"
CACHE_GITHUB_FILE = "cache_github.json"


# ==============================================================================
# [유틸] 데이터 포맷팅 및 캐싱 기능
# ==============================================================================
def format_price(x):
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
# [백테스팅] 히스토리 자동 누적 및 과거 성과 검증 모듈
# ==============================================================================
def save_scan_history(df_result):
    """현재 스캔된 상위 10개 종목을 CSV에 누적 기록"""
    if df_result.empty:
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_10 = df_result.head(10).copy()
    
    history_data = []
    for _, row in top_10.iterrows():
        history_data.append({
            "스캔시각": now_str,
            "심볼": row["심볼"],
            "코인명": str(row["코인명"]).replace(" 🔥", ""),
            "종합예측점수": row["종합예측점수"],
            "패턴유사도": row["패턴유사도(%)"],
            "매집점수": row["매집점수"],
            "스캔당시가격": float(str(row["현재가(KRW)"]).replace(",", "")),
            "거래량절벽": row["거래량절벽(배)"],
            "이평선수렴": row["이평선수렴(%)"],
            "CMF지표": row["CMF지표"],
            "스프레드비율": row["스프레드(%)"]
        })

    df_new_hist = pd.DataFrame(history_data)
    
    if os.path.exists(HISTORY_CSV_PATH):
        df_new_hist.to_csv(HISTORY_CSV_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df_new_hist.to_csv(HISTORY_CSV_PATH, mode='w', header=True, index=False, encoding='utf-8-sig')
    print("💾 현재 스캔 결과가 히스토리(scan_history.csv)에 자동 누적되었습니다.")


def evaluate_past_performance():
    """과거 히스토리를 불러와 실제 이후 수익률(최대 상승률) 추적 백테스팅"""
    if not os.path.exists(HISTORY_CSV_PATH):
        return "과거 누적 히스토리가 아직 없어 성과 비교를 스킵합니다.", []

    try:
        df_hist = pd.read_csv(HISTORY_CSV_PATH)
        if df_hist.empty:
            return "히스토리 데이터가 비어 있습니다.", []

        df_hist["스캔시각"] = pd.to_datetime(df_hist["스캔시각"])
        now = datetime.datetime.now()
        
        # 12시간 이상 경과하고 72시간 이내의 과거 기록만 추출하여 검증
        past_targets = df_hist[
            (df_hist["스캔시각"] <= now - datetime.timedelta(hours=12)) &
            (df_hist["스캔시각"] >= now - datetime.timedelta(hours=72))
        ].copy()

        if past_targets.empty:
            return "검증 대상(12시간~72시간 전) 히스토리가 아직 누적되지 않았습니다.", []

        results = []
        hit_count = 0  # +5% 이상 상승 성공 건수

        for _, row in past_targets.iterrows():
            ticker = f"KRW-{row['심볼']}"
            scan_time = row["스캔시각"]
            scan_price = row["스캔당시가격"]

            df_ohlcv = pyupbit.get_ohlcv(ticker, interval="minute60", to=now, count=72)
            if df_ohlcv is not None and not df_ohlcv.empty:
                df_after = df_ohlcv[df_ohlcv.index >= scan_time]
                if not df_after.empty:
                    max_price = df_after["high"].max()
                    max_return = round(((max_price - scan_price) / scan_price) * 100, 2)
                    
                    is_hit = max_return >= 5.0
                    if is_hit:
                        hit_count += 1

                    results.append({
                        "스캔시각": scan_time.strftime("%m-%d %H:%M"),
                        "코인명": row["코인명"],
                        "종합점수": row["종합예측점수"],
                        "패턴유사도": row["패턴유사도"],
                        "매집점수": row["매집점수"],
                        "추천당시가": scan_price,
                        "이후최고가": max_price,
                        "최대수익률(%)": max_return,
                        "적중여부": "🎯 성공(+5%↑)" if is_hit else "⚪ 보류/미달"
                    })
            time.sleep(0.02)

        total_eval = len(results)
        hit_rate = round((hit_count / total_eval) * 100, 1) if total_eval > 0 else 0.0
        
        summary_text = f"📊 과거 추천 종목 성과 검증 결과: 총 {total_eval}건 중 {hit_count}건 성공 (적중률: {hit_rate}%)"
        return summary_text, results

    except Exception as e:
        return f"과거 성과 검증 중 오류 발생: {e}", []


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
# 🆕 [호가창 분석] 스프레드 및 호가 잔량 비율 분석 함수
# ==============================================================================
def get_orderbook_metrics(ticker):
    """호가창 스프레드 비율 및 매수/매도 잔량 불균형 비율 산출"""
    try:
        orderbook = pyupbit.get_orderbook(ticker)
        if not orderbook or 'orderbook_units' not in orderbook:
            return {"spread_ratio": 0.0, "bid_ask_ratio": 1.0}

        units = orderbook['orderbook_units']
        if not units:
            return {"spread_ratio": 0.0, "bid_ask_ratio": 1.0}

        best_ask = units[0]['ask_price']  # 최저 매도호가
        best_bid = units[0]['bid_price']  # 최고 매수호가

        # 1. 호가 스프레드 비율 (%)
        spread_ratio = ((best_ask - best_bid) / best_bid) * 100 if best_bid > 0 else 0.0

        # 2. 호가 잔량 불균형 (총 매수 잔량 / 총 매도 잔량)
        total_ask_size = orderbook.get('total_ask_size', 1.0)
        total_bid_size = orderbook.get('total_bid_size', 1.0)
        bid_ask_ratio = (total_bid_size / total_ask_size) if total_ask_size > 0 else 1.0

        return {
            "spread_ratio": round(spread_ratio, 3),
            "bid_ask_ratio": round(bid_ask_ratio, 2)
        }
    except Exception:
        return {"spread_ratio": 0.0, "bid_ask_ratio": 1.0}


# ==============================================================================
# [알고리즘 고도화] 코사인 유사도 & T-1 선행 매집 점수 산출
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


def calculate_t1_advanced_metrics(df_daily, df_30m=None):
    df = df_daily.iloc[:-1].copy()
    if len(df) < 30:
        return None

    close = df['close']
    open_p = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']

    chg_1d = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
    chg_7d = ((close.iloc[-1] - close.iloc[-8]) / close.iloc[-8]) * 100 if len(df) >= 8 else 0

    # 1. 거래량 절벽 비율 산출 (최근 5일 평균 대비)
    vol_ma5 = volume.iloc[-6:-1].mean()
    vol_dry_ratio = (volume.iloc[-1] / vol_ma5) if vol_ma5 > 0 else 1.0
    
    is_volume_dry = (vol_dry_ratio <= 0.50)

    # 2. 이동평균선 수렴도 산출 (5, 10, 20일선)
    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma_max = max(ma5, ma10, ma20)
    ma_min = min(ma5, ma10, ma20)
    ma_compression = ((ma_max - ma_min) / ma20) * 100 if ma20 > 0 else 999.0

    is_volatility_threshold = (ma_compression < 3.0)

    # 3. 매집 봉(윗꼬리 급등봉) 존재 여부
    prev_vol_avg = volume.iloc[-25:-5].mean() if len(df) >= 25 else volume.mean()
    has_spike_candle = False
    if prev_vol_avg > 0:
        for i in range(-10, -1):
            v_ratio = volume.iloc[i] / prev_vol_avg
            candle_body = abs(close.iloc[i] - open_p.iloc[i])
            upper_shadow = high.iloc[i] - max(close.iloc[i], open_p.iloc[i])
            if v_ratio >= 2.2 and upper_shadow >= (candle_body * 0.8):
                has_spike_candle = True
                break

    # 4. OBV 다이버전스 산출
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

    # 5. 마감 직전 30분 수급 유입
    late_volume_surge = False
    if df_30m is not None and len(df_30m) >= 6:
        avg_30m_vol = df_30m['volume'].iloc[:-2].mean()
        recent_30m_vol = df_30m['volume'].iloc[-1]
        if avg_30m_vol > 0 and (recent_30m_vol / avg_30m_vol) >= 1.8:
            late_volume_surge = True

    # 6. CMF (Chaikin Money Flow) & VWAP
    high_low_diff = (high - low).replace(0, np.nan)
    clv = (((close - low) - (high - close)) / high_low_diff).fillna(0)
    money_flow_vol = clv * volume
    
    vol_20_sum = volume.rolling(20).sum()
    cmf_series = money_flow_vol.rolling(20).sum() / vol_20_sum
    latest_cmf = cmf_series.iloc[-1] if not np.isnan(cmf_series.iloc[-1]) else 0.0

    typical_price = (high + low + close) / 3
    tp_vol_sum_14 = (typical_price * volume).tail(14).sum()
    vol_sum_14 = volume.tail(14).sum()
    vwap_14 = (tp_vol_sum_14 / vol_sum_14) if vol_sum_14 > 0 else close.iloc[-1]
    is_above_vwap = (close.iloc[-1] >= vwap_14)

    return {
        "chg_1d": round(chg_1d, 2),
        "chg_7d": round(chg_7d, 2),
        "vol_dry_ratio": round(vol_dry_ratio, 2),
        "is_volume_dry": is_volume_dry,
        "ma_compression": round(ma_compression, 2),
        "is_volatility_threshold": is_volatility_threshold,
        "has_spike_candle": has_spike_candle,
        "is_obv_div": is_obv_divergence,
        "late_volume_surge": late_volume_surge,
        "last_close": close.iloc[-1],
        "last_value": df['value'].iloc[-1],
        "cmf": round(latest_cmf, 3),
        "is_above_vwap": is_above_vwap
    }


def calculate_t1_score(metrics, surge_from_bottom, circ_ratio, is_dev_active, ob_metrics):
    score = 0

    # 바닥 대비 과도한 상승 감점
    if surge_from_bottom >= 35.0:
        return 0
    elif surge_from_bottom <= 15.0:
        score += 20
    elif surge_from_bottom <= 25.0:
        score += 10

    # 거래량 절벽 기본 점수
    if metrics['vol_dry_ratio'] <= 0.50:
        score += 40
    elif metrics['vol_dry_ratio'] <= 0.75:
        score += 20

    # 이평선 수렴 기본 점수
    if metrics['ma_compression'] <= 2.5:
        score += 25
    elif metrics['ma_compression'] <= 4.5:
        score += 15

    # 변동성 임계치 보너스 점수
    if metrics['is_volatility_threshold']:
        score += 15

    # 기타 시그널 점수
    if metrics['has_spike_candle']:
        score += 15
    if metrics['is_obv_div']:
        score += 15

    # CMF 및 VWAP 연동 가점/감점
    cmf_val = metrics['cmf']
    is_above_vwap = metrics['is_above_vwap']

    if cmf_val >= 0.10 and is_above_vwap:
        score += 20
    elif cmf_val >= 0.02:
        score += 10
    elif cmf_val < -0.08:
        score -= 20

    # 🆕 [신규 제안 반영] 호가창 스프레드 및 잔량 불균형 반영
    spread = ob_metrics["spread_ratio"]
    bid_ask = ob_metrics["bid_ask_ratio"]

    if spread <= 0.15 and bid_ask >= 1.8:
        score += 15  # 호가가 촘촘하고 아래로 매수 받침이 둔둔한 진짜 매집 형태
    elif spread > 0.50:
        score -= 15  # 호가 공백이 너무 넓어 거래 자체가 완전히 죽은 무관심 종목 (패널티)

    # 당일 변동률 패널티 및 가점
    if -2.5 <= metrics['chg_1d'] <= 3.0:
        score += 10
    elif metrics['chg_1d'] > 7.0:
        score -= 25

    # 기타 보조 가점
    if metrics['late_volume_surge']:
        score += 10
    if circ_ratio >= 60.0:
        score += 5
    if is_dev_active:
        score += 5
    if (metrics['last_value'] / 100_000_000) >= 30:
        score += 5

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

    print("\n[1/2] 30분봉 수급 및 호가창 데이터 수집 중...")
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

    print("\n[2/2] T-1 선행 매집 패턴 및 호가창 스프레드 종합 분석 중...")
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

            # 🆕 호가창 지표 실시간 수집
            ob_metrics = get_orderbook_metrics(ticker)

            df_30m_recent = market_30m_data.get(ticker, None)
            metrics = calculate_t1_advanced_metrics(df_daily, df_30m_recent)
            if not metrics:
                continue

            df_closed = df_daily.iloc[:-1]
            lowest_20d = df_closed['low'].iloc[-20:].min()
            surge_from_bottom = round(((metrics['last_close'] - lowest_20d) / lowest_20d) * 100, 2) if lowest_20d > 0 else 0.0
            circ_ratio = tokenomics_map.get(symbol, 75.0)
            is_dev_active = github_map.get(symbol, False)
            
            # 매집 점수 계산 (호가창 지표 포함)
            accumulation_score = calculate_t1_score(metrics, surge_from_bottom, circ_ratio, is_dev_active, ob_metrics)

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

            # 거래량 절벽 역수 가중치 & 매집점수 곱셈 연산
            vol_dry_ratio = metrics['vol_dry_ratio']
            inv_vol_dry = 1.0 / (vol_dry_ratio + 0.1)
            compression_score = accumulation_score * inv_vol_dry

            pattern_prediction_score = (
                (compression_score * 0.15) +    # 압축 구간(매집 x 역수절벽)
                (similarity_score * 0.25) +     # 코사인 유사도
                (rank_count * 2.0) +            # 30분 수급 상위 진입 횟수
                (vol_ratio_pattern * 2.5)       # 수급선 형성 비율
            )

            results.append({
                "코인명": f"{korean_name} 🔥" if recent_buy_sell_trend == "매수세 우위" else korean_name,
                "심볼": symbol,
                "종합예측점수": round(pattern_prediction_score, 2),
                "패턴유사도(%)": round(similarity_score, 1),
                "매집점수": accumulation_score,
                "현재가(KRW)": format_price(metrics['last_close']),
                "거래량절벽(배)": metrics['vol_dry_ratio'],
                "이평선수렴(%)": metrics['ma_compression'],
                "CMF지표": metrics['cmf'],
                "스프레드(%)": ob_metrics['spread_ratio'],
                "매수매도비율": ob_metrics['bid_ask_ratio'],
                "VWAP상회": "상회" if metrics['is_above_vwap'] else "하회",
                "1일 변동률(%)": metrics['chg_1d'],
                "7일 변동률(%)": metrics['chg_7d'],
                "바닥대비상승(%)": surge_from_bottom,
                "15위내_진입(회)": rank_count,
                "과거급등(회)": spike_count,
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
        df = df.sort_values(
            by=["종합예측점수", "패턴유사도(%)", "매집점수"], 
            ascending=[False, False, False]
        ).reset_index(drop=True)
    return df


# ==============================================================================
# [AI 심층 분석] Gemini 3.1 Flash Lite - 호가창 밀도 반영 분석
# ==============================================================================
def generate_gemini_analysis(df, eval_summary, eval_details):
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
                "종합예측점수": row['종합예측점수'],
                "패턴유사도": f"{row['패턴유사도(%)']}%",
                "T-1매집점수": row['매집점수'],
                "거래량절벽비율": f"{row['거래량절벽(배)']}배",
                "이평선수렴도": f"{row['이평선수렴(%)']}%",
                "CMF(자금유입)": row['CMF지표'],
                "호가스프레드": f"{row['스프레드(%)']}%",
                "매수매도잔량비율": f"{row['매수매도비율']}배",
                "VWAP위치": row['VWAP상회'],
                "1일/7일변동률": f"{row['1일 변동률(%)']}% / {row['7일 변동률(%)']}%",
                "바닥대비상승률": f"{row['바닥대비상승(%)']}%",
                "30분수급": row['30분 수급'],
                "4시간봉_1주일_수급분석": info_4h
            })

        prompt = f"""
당신은 가상자산 수급 및 T-1 상승 직전 패턴 분석 전문가입니다.
아래 [현재 스캔 상위 10개 데이터]와 [과거 추천 종목의 실제 성과 검증 데이터]를 비교 분석하여 정중한 경어체(~습니다, ~입니다)로 통합 리포트를 작성해 주세요.

이번 알고리즘에는 **[거래량 절벽 역수 가중치]**, **[VWAP + CMF]** 지표에 더해 **[호가창 스프레드 및 매수 잔량 비율]**까지 연동되어 인위적인 호가 공백과 세력의 촘촘한 받침 매집을 동시에 검증합니다.

[1. 현재 스캔 상위 10개 데이터]
{json.dumps(enriched_data, ensure_ascii=False, indent=2)}

[2. 과거 추천 종목 백테스팅 검증 요약 및 세부 내역]
요약: {eval_summary}
세부 성과: {json.dumps(eval_details[:8], ensure_ascii=False, indent=2)} (최대 8개 표기)

[작성 지침]
1. **[과거 리포트 성과 검증 및 호가창 분석 영향]**:
   - 과거 성과와 비교하여, 거래량 절벽 구간에서의 호가창 촘촘함(스프레드)과 매수 잔량 받침이 진짜 매집주를 구분해 내는 데 미치는 영향을 분석해 주세요.
2. **[현재 스캔 T-1 상승 직전 추천 종목 Top 3]**:
   - 상위 1~3위 추천 종목의 **진입 타점, 목표 수익률, 손절 기준**을 호가창 스프레성 및 CMF 수급과 연관지어 설명해 주세요.
3. **[알고리즘 추가 보완 제안]**:
   - 매집 완결성을 더욱 정밀화하기 위한 다음 단계 제안을 1문장으로 남겨주세요.
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
def save_integrated_excel(df, eval_details):
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

    # 1. 현재 스캔 시트 작성
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))

    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="맑은 고딕", size=10, bold=True, color="FFFFFF")
    data_font = Font(name="맑은 고딕", size=10)
    high_score_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
    
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9')
    )
    
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    top5_pred_indices = set(df.nlargest(5, "종합예측점수").index)

    for row_idx, row in enumerate(range(2, ws.max_row + 1)):
        df_idx = row_idx
        accum_score_cell = ws.cell(row=row, column=5)
        
        is_high_score = (accum_score_cell.value and float(accum_score_cell.value) >= 70) or (df_idx in top5_pred_indices)
        
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
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    # 2. 과거 백테스트 검증 시트 생성
    if eval_details:
        eval_sheet_name = "과거검증_백테스트"
        if eval_sheet_name in wb.sheetnames:
            wb.remove(wb[eval_sheet_name])
        ws_eval = wb.create_sheet(title=eval_sheet_name)
        
        df_eval = pd.DataFrame(eval_details)
        ws_eval.append(list(df_eval.columns))
        for row in df_eval.itertuples(index=False):
            ws_eval.append(list(row))

        for col in range(1, ws_eval.max_column + 1):
            cell = ws_eval.cell(row=1, column=col)
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for row in range(2, ws_eval.max_row + 1):
            for col in range(1, ws_eval.max_column + 1):
                cell = ws_eval.cell(row=row, column=col)
                cell.font = data_font
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="center", vertical="center")

    wb.save(EXCEL_FILE_PATH)
    wb.close()
    return EXCEL_FILE_PATH


def send_email_report(file_path, ai_analysis, eval_summary):
    if not file_path or not os.path.exists(file_path) or not SENDER_EMAIL or not RECEIVER_EMAILS:
        print("⚠️ 이메일 발송 조건이 충족되지 않았습니다 (환경 변수/파일 확인 필요).")
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = MIMEMultipart()
    msg["Subject"] = f"📊 [T-1 압축매집+호가스프레드] 실시간 분석 & 백테스트 리포트 ({now_str})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECEIVER_EMAILS)
    
    body = f"""안녕하세요.

업비트 원화 마켓 [T-1 선행 매집 지표] 실시간 스캔 결과입니다.
* 이번 리포트에는 '호가창 스프레드 변화율 & 호가 잔량 비율' 및 'VWAP/CMF 지표'가 함께 반영되었습니다.

• 분석 시각: {now_str}
• 과거 성과: {eval_summary}

==================================================
🤖 [Gemini AI 현재 vs 과거 비교 심층 리포트]
==================================================
{ai_analysis}

상세 종목 분석 데이터 및 성과 검증 내역은 첨부된 엑셀 파일 시트를 확인해 주세요.
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
    print("🚀 [업비트 원화 마켓] T-1 매집 분석 + 호가창 스프레드 통합 연동 시작...")
    
    # 1. 과거 히스토리 성과 검증 (백테스팅)
    print("\n🔍 과거 추천 종목 수익률 자동 검증 중...")
    eval_summary, eval_details = evaluate_past_performance()
    print(f"👉 {eval_summary}")

    # 2. 현재 마켓 실시간 스캔
    df_result = analyze_and_scan_market()
    
    if not df_result.empty:
        # 3. 현재 스캔 결과 히스토리 CSV 자동 저장
        save_scan_history(df_result)

        print("\n=== 🎯 현재 상위 5개 추천 종목 (종합예측점수 순) ===")
        print(df_result[["코인명", "종합예측점수", "패턴유사도(%)", "매집점수", "거래량절벽(배)", "CMF지표", "스프레드(%)", "매수매도비율"]].head(5))

        print("\n🤖 Gemini AI 호가창 밀도 연동 심층 분석 중...")
        ai_summary = generate_gemini_analysis(df_result, eval_summary, eval_details)
        
        print("\n📊 엑셀 저장 중...")
        excel_file = save_integrated_excel(df_result, eval_details)
        
        print("\n📧 통합 리포트 이메일 발송 중...")
        send_email_report(excel_file, ai_summary, eval_summary)
    else:
        print("❌ 분석된 결과가 없습니다.")

    print(f"\n✨ 전체 프로세스 완료 (총 소요 시간: {round(time.time() - start_time, 2)}초)")
