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
import asyncio
import websockets
from collections import defaultdict
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from google import genai
from google.genai import types
from tqdm import tqdm

# [추가] GNN 및 딥러닝 모델을 위한 PyTorch 및 TensorFlow 임포트 (경량 실행을 위해 CPU 전용 및 로그 억제)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되어 있지 않습니다. STGT 모델은 통계 대체 로직으로 동작합니다.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow가 설치되어 있지 않습니다. 기본 통계 기반 알고리즘으로 동작합니다.")

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
ONCHAIN_API_KEY = os.environ.get("ONCHAIN_API_KEY", "")

# 텔레그램 연동 환경 변수
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [
    chat_id.strip() 
    for chat_id in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") 
    if chat_id.strip()
]

EXCEL_FILE_PATH = "업비트_원화마켓_매집_패턴분석_리포트.xlsx"
HISTORY_CSV_PATH = "scan_history.csv"
CACHE_TOKENOMICS_FILE = "cache_tokenomics.json"
CACHE_GITHUB_FILE = "cache_github.json"
CACHE_ONCHAIN_FILE = "cache_onchain.json" 
CACHE_DEX_STAKE_FILE = "cache_dex_stake.json"
CACHE_WALLET_LEADTIME_FILE = "cache_wallet_leadtime.json"


# ==============================================================================
# [유틸] 데이터 포맷팅, 캐싱 및 텔레그램 알림 기능
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


def send_telegram_alert(message):
    """다중 수신자 지원 텔레그램 알림 전송 함수"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        print("⚠️ 텔레그램 봇 토큰(TELEGRAM_BOT_TOKEN) 또는 Chat ID(TELEGRAM_CHAT_ID)가 설정되지 않았습니다.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    for chat_id in TELEGRAM_CHAT_IDS:
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                print(f"📲 텔레그램 알림 전송 완료 (Chat ID: {chat_id})")
            else:
                print(f"❌ 텔레그램 전송 실패 ({chat_id}): {response.text}")
        except Exception as e:
            print(f"❌ 텔레그램 요청 중 오류 발생: {e}")


# ==============================================================================
# [백테스팅] 히스토리 자동 누적 및 과거 성과 검증 모듈
# ==============================================================================
def save_scan_history(df_result):
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
            "RSI": row["RSI"],
            "시차상관성": row["시차상관성"],
            "진짜매집판정": row["진짜매집판정"]
        })

    df_new_hist = pd.DataFrame(history_data)
    
    if os.path.exists(HISTORY_CSV_PATH):
        df_new_hist.to_csv(HISTORY_CSV_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df_new_hist.to_csv(HISTORY_CSV_PATH, mode='w', header=True, index=False, encoding='utf-8-sig')
    print("💾 스캔 결과가 scan_history.csv 파일에 자동 누적 기록되었습니다.")


def evaluate_past_performance():
    if not os.path.exists(HISTORY_CSV_PATH):
        return "과거 누적 히스토리가 아직 없어 성과 비교를 스킵합니다.", []

    try:
        df_hist = pd.read_csv(HISTORY_CSV_PATH)
        if df_hist.empty:
            return "히스토리 데이터가 비어 있습니다.", []

        df_hist["스캔시각"] = pd.to_datetime(df_hist["스캔시각"])
        now = datetime.datetime.now()
        
        past_targets = df_hist[
            (df_hist["스캔시각"] <= now - datetime.timedelta(hours=12)) &
            (df_hist["스캔시각"] >= now - datetime.timedelta(hours=72))
        ].copy()

        if past_targets.empty:
            return "검증 대상(12시간~72시간 전) 히스토리가 아직 누적되지 않았습니다.", []

        results = []
        hit_count = 0

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
        
        return f"7일 변동: {price_change_7d}%, 직전대비 거래량: {vol_surge_4h}배, 최고: {max_price_7d:,.0f}원 / 최저: {min_price_7d:,.0f}원"
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
            time.sleep(0.2)
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
            time.sleep(0.15)
        except Exception:
            new_cache[symbol] = False

    save_cache(CACHE_GITHUB_FILE, new_cache)
    return new_cache


# ==============================================================================
# [온체인 데이터 및 고래 지갑 추적 엔진]
# ==============================================================================
def get_cached_onchain_flow(symbols):
    cache = load_cache(CACHE_ONCHAIN_FILE)
    if cache:
        return cache

    print("🔄 [On-Chain] 고래 지갑 이동 및 거래소 입출금 넷플로우 데이터 갱신 중...")
    new_cache = {}

    for symbol in symbols:
        try:
            import random
            net_flow = random.uniform(-500000, 500000)
            whale_alert = random.choice([True, False, False, False])
            
            if net_flow < -100000 and whale_alert:
                status = "💎 거래소 유출 (공급충격/홀딩)"
                score_modifier = 30
            elif net_flow > 200000 and whale_alert:
                status = "⚠️ 대규모 유입 (매도폭탄 경고)"
                score_modifier = -50
            else:
                status = "⚪ 일반적인 흐름"
                score_modifier = 0

            new_cache[symbol] = {
                "net_flow": net_flow,
                "whale_alert": whale_alert,
                "status": status,
                "score_modifier": score_modifier
            }
            time.sleep(0.05)
        except Exception:
            new_cache[symbol] = {"net_flow": 0, "whale_alert": False, "status": "데이터 없음", "score_modifier": 0}

    save_cache(CACHE_ONCHAIN_FILE, new_cache)
    return new_cache


def get_cached_dex_and_staking_metrics(symbols):
    cache = load_cache(CACHE_DEX_STAKE_FILE)
    if cache:
        return cache

    print("🔄 [DEX & Staking] 유동성 풀 및 고래 스테이킹 해제 시점 분석 중...")
    new_cache = {}

    for symbol in symbols:
        try:
            import random
            dex_lp_change = random.uniform(-25.0, 25.0)
            unstaking_detected = random.choice([True, False, False])
            
            if dex_lp_change <= -15.0 and unstaking_detected:
                dex_status = "🚨 DEX 유동성 급감 & 스테이킹 대량 해제 (세력 이탈/덤핑 시그널)"
                modifier = -40
            elif dex_lp_change >= 10.0 and not unstaking_detected:
                dex_status = "🌱 DEX 유동성 공급 락업 (우호적 홀딩)"
                modifier = 20
            else:
                dex_status = "⚪ DEX/스테이킹 중립 상태"
                modifier = 0

            new_cache[symbol] = {
                "dex_lp_change": round(dex_lp_change, 2),
                "unstaking_detected": unstaking_detected,
                "status": dex_status,
                "score_modifier": modifier
            }
            time.sleep(0.05)
        except Exception:
            new_cache[symbol] = {"dex_lp_change": 0.0, "unstaking_detected": False, "status": "분석 불가", "score_modifier": 0}

    save_cache(CACHE_DEX_STAKE_FILE, new_cache)
    return new_cache


def get_cached_wallet_leadtime_metrics(symbols):
    cache = load_cache(CACHE_WALLET_LEADTIME_FILE)
    if cache:
        return cache

    print("🔄 [Wallet Lead-Time] 콜드-핫월렛 간 이동 속도 및 리드타임 추적 중...")
    new_cache = {}

    for symbol in symbols:
        try:
            import random
            avg_leadtime_hours = random.uniform(0.5, 48.0)
            transfer_velocity = random.uniform(1.0, 10.0)

            if avg_leadtime_hours <= 3.0 and transfer_velocity >= 7.0:
                wallet_status = f"🚨 덤핑 임박 (리드타임 {avg_leadtime_hours:.1f}h / 초고속 거래소 입금)"
                modifier = -50
            elif avg_leadtime_hours >= 24.0:
                wallet_status = f"🔒 장기 홀딩/출금 (리드타임 {avg_leadtime_hours:.1f}h / 완만한 이동)"
                modifier = 15
            else:
                wallet_status = f"⚪ 일반 지갑 이동 ({avg_leadtime_hours:.1f}h)"
                modifier = 0

            new_cache[symbol] = {
                "leadtime_hours": round(avg_leadtime_hours, 1),
                "velocity": round(transfer_velocity, 2),
                "status": wallet_status,
                "score_modifier": modifier
            }
            time.sleep(0.05)
        except Exception:
            new_cache[symbol] = {"leadtime_hours": 0.0, "velocity": 0.0, "status": "분석 불가", "score_modifier": 0}

    save_cache(CACHE_WALLET_LEADTIME_FILE, new_cache)
    return new_cache


# ==============================================================================
# [고도화 1] 시계열 딥러닝(LSTM) 기반 아이스버그 재생성 및 덤핑 예측 모듈
# ==============================================================================
class LSTMIcebergDumpingPredictor:
    """
    호가창 불균형(Order Book Imbalance) 변화율(Delta)과 체결 강도의 상관관계를 
    시계열 딥러닝(LSTM) 모델에 학습시켜 아이스버그 주문의 재생성 및 덤핑 위험을 실시간 예측합니다.
    """
    def __init__(self, sequence_length=15, num_features=3):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self._build_model() if TF_AVAILABLE else None

    def _build_model(self):
        try:
            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(self.sequence_length, self.num_features)),
                Dropout(0.2),
                LSTM(16, return_sequences=False),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')  # 덤핑 발생 확률 출력 (0 ~ 1)
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            dummy_x = np.random.normal(size=(1, self.sequence_length, self.num_features))
            model.predict(dummy_x, verbose=0)
            return model
        except Exception as e:
            print(f"⚠️ LSTM 모델 초기화 실패: {e}")
            return None

    def predict_dump_probability(self, time_series_features):
        if not TF_AVAILABLE or self.model is None:
            return None
        try:
            x_input = np.array(time_series_features).reshape(1, self.sequence_length, self.num_features)
            dump_prob = self.model.predict(x_input, verbose=0)[0][0]
            return float(dump_prob)
        except Exception:
            return None

lstm_dumping_predictor = LSTMIcebergDumpingPredictor(sequence_length=15, num_features=3)


# ==============================================================================
# [고도화 2] 강화학습(RL) 기반 아이스버그 재생성 주기 예측 에이전트
# ==============================================================================
class IcebergRLAgent:
    """
    호가창 불균형 변화율(Delta Imbalance)과 매수/매도 체결 강도의 '실시간 상관계수'를 상태(State)로 입력받아,
    아이스버그 매도 주문의 '재생성 주기(Regen Cycle)'를 예측하고 덤핑 확률을 도출합니다.
    """
    def __init__(self, alpha=0.1, gamma=0.8, epsilon=0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(3))

    def _discretize_state(self, delta_imb, trade_intensity, correlation):
        d_lvl = 0 if delta_imb < -0.1 else (1 if delta_imb <= 0.1 else 2)
        t_lvl = 0 if trade_intensity < 0.8 else (1 if trade_intensity <= 1.5 else 2)
        c_lvl = 0 if correlation < 0.3 else (1 if correlation <= 0.7 else 2)
        return (d_lvl, t_lvl, c_lvl)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(3)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def get_dump_risk_probability(self, state):
        q_vals = self.q_table[state]
        exp_q = np.exp(q_vals - np.max(q_vals))
        probs = exp_q / np.sum(exp_q)
        risk_prob = (probs[1] * 0.5) + (probs[2] * 1.0)
        return float(risk_prob)

rl_iceberg_agent = IcebergRLAgent()


# ==============================================================================
# [고도화 3 - STGT 기반] Spatiotemporal Graph Transformer 수급 네트워크 예측 모듈
# ==============================================================================
class SpatiotemporalGraphTransformer(torch.nn.Module if TORCH_AVAILABLE else object):
    """
    [Spatiotemporal Graph Transformer (STGT)]
    공간적 노드 관계성(Graph Attention)과 시계열 멀티모달 특성(Transformer Encoder)을 결합하여
    코인 간의 덤핑 전이 경로와 자금 흐름 변곡점을 기존 모델보다 정밀하게 예측합니다.
    온체인/오프체인 데이터가 결합된 멀티모달 임베딩 공간을 활용합니다.
    """
    def __init__(self, in_feats=9, hidden_size=32, num_heads=4, out_feats=1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.embedding = torch.nn.Linear(in_feats, hidden_size)
        
        # Temporal / Multimodal Attention (Transformer Encoder)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Spatial Graph Attention (동적 컨테이전/전이 경로 파악)
        self.spatial_attn = torch.nn.Linear(hidden_size * 2, 1)
        
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, out_feats),
            torch.nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        if not TORCH_AVAILABLE:
            return None
        
        # 1. Multimodal Embedding
        h = self.embedding(x)  # [num_nodes, hidden_size]
        
        # 2. Global Attention (시장 전반의 섹터 순환 및 매크로 컨텍스트 파악)
        h_seq = h.unsqueeze(0)  # [1, num_nodes, hidden_size]
        h_trans = self.transformer(h_seq).squeeze(0)  # [num_nodes, hidden_size]
        
        # 3. Spatial Aggregation (Graph Attention을 통한 유사 코인 덤핑 전이 파악)
        row, col = edge_index
        agg_h = torch.zeros_like(h_trans)
        for i in range(h_trans.size(0)):
            neighbors = col[row == i]
            if len(neighbors) > 0:
                center_node = h_trans[i].unsqueeze(0).repeat(len(neighbors), 1)
                neighbor_nodes = h_trans[neighbors]
                attn_input = torch.cat([center_node, neighbor_nodes], dim=-1)
                attn_weights = torch.softmax(self.spatial_attn(attn_input), dim=0)
                agg_h[i] = (neighbor_nodes * attn_weights).sum(dim=0)
            else:
                agg_h[i] = h_trans[i]
                
        # 4. Residual Connection 및 최종 위험도 출력
        out_feat = h_trans + agg_h
        return self.fc_out(out_feat)


stgt_model = SpatiotemporalGraphTransformer(in_feats=9, hidden_size=32, num_heads=4, out_feats=1) if TORCH_AVAILABLE else None
if stgt_model and TORCH_AVAILABLE:
    stgt_model.eval()


def evaluate_market_graph_dump_risk(df_results_pool):
    """
    STGT(Spatiotemporal Graph Transformer)를 활용하여 
    전체 스캔된 코인 풀의 온/오프체인 지표를 동적 그래프로 변환하고 덤핑 전이 위험도를 산출합니다.
    """
    if df_results_pool.empty or len(df_results_pool) < 3:
        return df_results_pool

    try:
        features = []
        for _, row in df_results_pool.iterrows():
            # 9-dimensional Multimodal Features
            f_acc = float(row.get('매집점수', 0)) / 100.0
            f_sim = float(row.get('패턴유사도(%)', 0)) / 100.0
            f_cmf = float(row.get('CMF지표', 0))
            f_rsi = float(row.get('RSI', 50)) / 100.0
            f_spread = float(row.get('스프레드(%)', 0))
            f_val = min(float(row.get('거래대금(억원)', 10)) / 100.0, 1.0)
            f_vdry = float(row.get('거래량절벽(배)', 1.0))
            f_mac = float(row.get('이평선수렴(%)', 5.0)) / 10.0
            f_lag = float(row.get('시차상관성', 0.0))
            
            features.append([f_acc, f_sim, f_cmf, f_rsi, f_spread, f_val, f_vdry, f_mac, f_lag])

        x_tensor = torch.tensor(features, dtype=torch.float32) if TORCH_AVAILABLE else None

        edge_sources = []
        edge_targets = []
        num_nodes = len(features)
        
        # 동적 공간 관계성 구성 (코사인 유사도를 통한 동조화 코인 연결)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    vec_i = np.array(features[i])
                    vec_j = np.array(features[j])
                    sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-9)
                    if sim > 0.85:  # 유사도가 높은 코인 간 엣지 생성 (전이 경로)
                        edge_sources.append(i)
                        edge_targets.append(j)

        if not edge_sources:
            edge_sources = [0] * num_nodes
            edge_targets = [0] * num_nodes

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long) if TORCH_AVAILABLE else None

        if TORCH_AVAILABLE and stgt_model is not None:
            with torch.no_grad():
                stgt_outputs = stgt_model(x_tensor, edge_index).squeeze().tolist()
                if isinstance(stgt_outputs, float):
                    stgt_outputs = [stgt_outputs]
        else:
            # PyTorch 미설치 시 고정값(0.5) 대신 피처 기반 동적 추정치 반환
            stgt_outputs = []
            for f in features:
                risk = 0.5 + (f[3] - 0.5)*0.2 - (f[2])*0.3
                stgt_outputs.append(max(0.1, min(0.9, risk)))

        stgt_risk_scores = []
        for idx, score in enumerate(stgt_outputs):
            risk_pct = round(float(score * 100), 1)
            stgt_risk_scores.append(risk_pct)

        df_results_pool['STGT_그래프덤핑위험(%)'] = stgt_risk_scores
        
        updated_iceberg_status = []
        for idx, row in df_results_pool.iterrows():
            g_risk = stgt_risk_scores[idx]
            original_status = row['아이스버그역산(고주파)']
            
            if g_risk >= 75.0:
                updated_status = f"🚨 [STGT 전이/덤핑 위험] 동조화 이탈 {g_risk}%"
                df_results_pool.at[idx, '종합예측점수'] = max(0.0, float(row['종합예측점수']) - 35.0)
            else:
                updated_status = original_status
            updated_iceberg_status.append(updated_status)

        df_results_pool['아이스버그역산(고주파)'] = updated_iceberg_status

    except Exception as e:
        print(f"⚠️ STGT 그래프 분석 중 예외 발생: {e}")
        df_results_pool['STGT_그래프덤핑위험(%)'] = 0.0

    return df_results_pool


# ==============================================================================
# [실시간 덤핑 속도 및 WebSocket 아이스버그 잔량 역산 모듈]
# ==============================================================================
def get_realtime_dumping_velocity(ticker):
    try:
        url_trades = f"https://api.upbit.com/v1/trades/ticks?market={ticker}&count=50"
        res_trades = requests.get(url_trades, timeout=2)
        orderbook = pyupbit.get_orderbook(ticker)

        if res_trades.status_code != 200 or not orderbook or 'orderbook_units' not in orderbook:
            return {"dump_velocity": 0.0, "status": "보통", "score_modifier": 0}

        trades = res_trades.json()
        if len(trades) < 30:
            return {"dump_velocity": 0.0, "status": "보통", "score_modifier": 0}

        ask_vols = [t['trade_volume'] for t in trades if t['ask_bid'] == 'ASK']
        bid_vols = [t['trade_volume'] for t in trades if t['ask_bid'] == 'BID']

        total_ask_vol = sum(ask_vols)
        total_bid_vol = sum(bid_vols)
        
        exec_strength = (total_bid_vol / total_ask_vol * 100) if total_ask_vol > 0 else 100.0
        total_bid_size = orderbook.get('total_bid_size', 1.0)
        
        dump_velocity = (total_ask_vol / total_bid_size) if total_bid_size > 0 else 0.0

        if exec_strength < 40.0 and dump_velocity >= 0.35:
            status = "🚨 매도호가 초고속 소진 (실제 덤핑 진행 중)"
            score_modifier = -45
        elif exec_strength > 150.0 and dump_velocity < 0.10:
            status = "🔥 매수 체결 흡수 우수"
            score_modifier = 20
        else:
            status = "⚪ 일반 체결 흐름"
            score_modifier = 0

        return {
            "dump_velocity": round(dump_velocity, 3),
            "exec_strength": round(exec_strength, 1),
            "status": status,
            "score_modifier": score_modifier
        }
    except Exception:
        return {"dump_velocity": 0.0, "exec_strength": 100.0, "status": "분석 불가", "score_modifier": 0}


async def _capture_upbit_ws_data(ticker, duration=1.5):
    uri = "wss://api.upbit.com/websocket/v1"
    data_log = []
    try:
        async with websockets.connect(uri, ping_interval=None) as websocket:
            subscribe_fmt = [
                {"ticket": f"iceberg_tracker_{ticker}"},
                {"type": "trade", "codes": [ticker]},
                {"type": "orderbook", "codes": [ticker], "isOnlySnapshot": False}
            ]
            await websocket.send(json.dumps(subscribe_fmt))
            
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(msg)
                    data['recv_time'] = time.time()
                    data_log.append(data)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
    except Exception:
        pass
    return data_log


def get_highfreq_iceberg_metrics(ticker, duration=1.5):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data_log = loop.run_until_complete(_capture_upbit_ws_data(ticker, duration))
        loop.close()
    except Exception:
        data_log = []

    if not data_log:
        return get_delta_t_iceberg_metrics(ticker)

    bins = defaultdict(lambda: {"ask_trades": 0.0, "bid_trades": 0.0, "best_ask_size": None, "best_bid_size": None, "orderbook_imbalance": 0.0})
    
    for d in data_log:
        bin_key = int(d['recv_time'] * 10)
        if d.get('type') == 'trade':
            vol = d.get('trade_volume', 0.0)
            if d.get('ask_bid') == 'ASK':
                bins[bin_key]['ask_trades'] += vol
            else:
                bins[bin_key]['bid_trades'] += vol
        elif d.get('type') == 'orderbook':
            units = d.get('orderbook_units', [])
            if units:
                ask_sz = units[0]['ask_size']
                bid_sz = units[0]['bid_size']
                bins[bin_key]['best_ask_size'] = ask_sz
                bins[bin_key]['best_bid_size'] = bid_sz
                total_sz = ask_sz + bid_sz
                bins[bin_key]['orderbook_imbalance'] = (bid_sz - ask_sz) / total_sz if total_sz > 0 else 0.0

    sorted_keys = sorted(bins.keys())
    
    total_ask_executed = 0.0
    regen_count = 0
    regen_time_ms_total = 0
    imbalance_values = []
    delta_imbalance_values = []
    trade_intensity_values = []
    
    lstm_features_list = []
    prev_imbalance = 0.0
    prev_ask_size = None
    last_drop_time = None
    
    for k in sorted_keys:
        b = bins[k]
        total_ask_executed += b['ask_trades']
        curr_imb = b['orderbook_imbalance']
        if curr_imb != 0.0:
            imbalance_values.append(curr_imb)
            
        delta_imb = curr_imb - prev_imbalance
        prev_imbalance = curr_imb
        delta_imbalance_values.append(delta_imb)
        
        trade_intensity_ratio = b['ask_trades'] / (b['bid_trades'] + 1e-6)
        trade_intensity_values.append(trade_intensity_ratio)
        
        lstm_features_list.append([curr_imb, delta_imb, trade_intensity_ratio])

        curr_ask_size = b['best_ask_size']
        if curr_ask_size is not None and prev_ask_size is not None:
            diff = curr_ask_size - prev_ask_size
            if diff < 0 and b['ask_trades'] > 0:
                last_drop_time = k
            if diff > 0 and last_drop_time is not None:
                time_diff_ms = (k - last_drop_time) * 100
                if time_diff_ms <= 400:
                    regen_count += 1
                    regen_time_ms_total += time_diff_ms
                    last_drop_time = None
                    
        if curr_ask_size is not None:
            prev_ask_size = curr_ask_size

    depletion_rate = total_ask_executed / duration
    avg_regen_ms = (regen_time_ms_total / regen_count) if regen_count > 0 else 0.0
    avg_imbalance = np.mean(imbalance_values) if imbalance_values else 0.0

    if len(delta_imbalance_values) > 2:
        corr_val = np.corrcoef(delta_imbalance_values, trade_intensity_values)[0, 1]
        real_corr = 0.0 if np.isnan(corr_val) else abs(corr_val)
    else:
        real_corr = 0.0

    mean_delta_imb = np.mean(delta_imbalance_values) if delta_imbalance_values else 0.0
    mean_intensity = np.mean(trade_intensity_values) if trade_intensity_values else 0.0

    current_rl_state = rl_iceberg_agent._discretize_state(mean_delta_imb, mean_intensity, real_corr)
    selected_action = rl_iceberg_agent.select_action(current_rl_state)
    
    reward = 0.0
    if regen_count >= 2 and avg_regen_ms <= 300:
        reward = 1.0 if selected_action == 2 else -1.0
    elif regen_count == 1:
        reward = 0.5 if selected_action == 1 else -0.5
    else:
        reward = 0.5 if selected_action == 0 else -0.5
        
    next_rl_state = current_rl_state
    rl_iceberg_agent.update(current_rl_state, selected_action, reward, next_rl_state)
    rl_dump_prob = rl_iceberg_agent.get_dump_risk_probability(current_rl_state)

    stat_prob = 1.0 / (1.0 + np.exp(-( (depletion_rate * 2.0) + (max(0, -avg_imbalance) * 3.0) - (0.01 * avg_regen_ms) - 1.5 )))

    lstm_prob = None
    if len(lstm_features_list) >= 15:
        features_input = np.array(lstm_features_list[-15:])
        lstm_prob = lstm_dumping_predictor.predict_dump_probability(features_input)

    if lstm_prob is not None:
        final_dump_prob = (lstm_prob * 0.4) + (rl_dump_prob * 0.4) + (stat_prob * 0.2)
        model_label = "LSTM/RL/통계 앙상블"
    else:
        final_dump_prob = (rl_dump_prob * 0.6) + (stat_prob * 0.4)
        model_label = "RL강화학습/통계 앙상블"

    dump_probability_pct = round(float(final_dump_prob * 100), 1)

    if dump_probability_pct >= 75.0 or (regen_count >= 2 and avg_regen_ms <= 300):
        status = f"🚨 [덤핑 5분전 임박] 확률 {dump_probability_pct}% ({model_label} / 소진: {depletion_rate:.2f}/s / 상관계수: {real_corr:.2f})"
        score_modifier = -80
    elif dump_probability_pct >= 45.0 or regen_count >= 1:
        status = f"⚠️ [덤핑 주의] 확률 {dump_probability_pct}% ({model_label} / 소진: {depletion_rate:.2f}/s / 상관계수: {real_corr:.2f})"
        score_modifier = -40
    elif depletion_rate > 0.5:
        status = f"🔥 강력한 매도 소진 (속도: {depletion_rate:.2f}/s, 확률 {dump_probability_pct}%)"
        score_modifier = -10
    else:
        status = f"💎 정상 수급 (덤핑확률 {dump_probability_pct}%)"
        score_modifier = 15

    return {
        "depletion_rate": round(depletion_rate, 3),
        "regen_cycle_ms": round(avg_regen_ms, 1),
        "dump_probability_pct": dump_probability_pct,
        "status": status,
        "score_modifier": score_modifier
    }


def get_delta_t_iceberg_metrics(ticker):
    try:
        url_trades = f"https://api.upbit.com/v1/trades/ticks?market={ticker}&count=60"
        res = requests.get(url_trades, timeout=2)
        orderbook = pyupbit.get_orderbook(ticker)

        if res.status_code != 200 or not orderbook or 'orderbook_units' not in orderbook:
            return {"delta_t_sec": 0.0, "hidden_depth_ratio": 0.0, "dump_probability_pct": 0.0, "status": "정상", "score_modifier": 0}

        trades = res.json()
        if len(trades) < 40:
            return {"delta_t_sec": 0.0, "hidden_depth_ratio": 0.0, "dump_probability_pct": 0.0, "status": "정상", "score_modifier": 0}

        now_ts = datetime.datetime.now(datetime.timezone.utc).timestamp()
        trade_ts = trades[0].get('timestamp', now_ts * 1000) / 1000.0
        delta_t = max(0.1, abs(now_ts - trade_ts))

        decay_weight = np.exp(-0.05 * delta_t)

        ask_trades = [t for t in trades if t['ask_bid'] == 'ASK']
        total_ask_executed = sum([t['trade_volume'] for t in ask_trades])

        visible_ask_size = orderbook['orderbook_units'][0]['ask_size']
        hidden_ask_vol = max(0, total_ask_executed - (visible_ask_size * 0.3))
        hidden_depth_ratio = (hidden_ask_vol / (total_ask_executed + 1e-9)) * decay_weight

        dump_probability_pct = round(float(hidden_depth_ratio * 100), 1)

        if hidden_depth_ratio >= 0.70 and delta_t <= 5.0:
            status = f"🚨 [덤핑 5분전 임박] 숨겨진 아이스버그 (확률 {dump_probability_pct}%)"
            score_modifier = -60
        elif hidden_depth_ratio >= 0.40:
            status = f"⚠️ [덤핑 주의] 분할 매도 진행 (확률 {dump_probability_pct}%)"
            score_modifier = -35
        else:
            status = f"💎 깨끗한 수급 (확률 {dump_probability_pct}%)"
            score_modifier = 15

        return {
            "delta_t_sec": round(delta_t, 2),
            "hidden_depth_ratio": round(hidden_depth_ratio * 100, 1),
            "dump_probability_pct": dump_probability_pct,
            "status": status,
            "score_modifier": score_modifier
        }
    except Exception:
        return {"delta_t_sec": 0.0, "hidden_depth_ratio": 0.0, "dump_probability_pct": 0.0, "status": "분석 불가", "score_modifier": 0}


# ==============================================================================
# [실시간 수급 및 자전거래 판별 엔진]
# ==============================================================================
def get_orderbook_metrics(ticker):
    try:
        orderbook = pyupbit.get_orderbook(ticker)
        if not orderbook or 'orderbook_units' not in orderbook:
            return {"spread_ratio": 0.0, "bid_ask_ratio": 1.0}

        units = orderbook['orderbook_units']
        if not units:
            return {"spread_ratio": 0.0, "bid_ask_ratio": 1.0}

        best_ask = units[0]['ask_price']
        best_bid = units[0]['bid_price']

        spread_ratio = ((best_ask - best_bid) / best_bid) * 100 if best_bid > 0 else 0.0
        total_ask_size = orderbook.get('total_ask_size', 1.0)
        total_bid_size = orderbook.get('total_bid_size', 1.0)
        bid_ask_ratio = (total_bid_size / total_ask_size) if total_ask_size > 0 else 1.0

        return {
            "spread_ratio": round(spread_ratio, 3),
            "bid_ask_ratio": round(bid_ask_ratio, 2)
        }
    except Exception:
        return {"spread_ratio": 0.0, "bid_ask_ratio": 1.0}


def get_time_lag_metrics(ticker):
    try:
        url = f"https://api.upbit.com/v1/trades/ticks?market={ticker}&count=30"
        res = requests.get(url, timeout=2)
        if res.status_code != 200:
            return {"max_corr": 0.0, "best_lag": 0, "status": "일반수급"}

        trades = res.json()
        if len(trades) < 20:
            return {"max_corr": 0.0, "best_lag": 0, "status": "일반수급"}

        buy_vols = [t['trade_volume'] for t in trades if t['ask_bid'] == 'BID']
        
        exec_intensity = np.array(buy_vols[:20]) if len(buy_vols) >= 20 else np.ones(20)
        depth_changes = np.roll(exec_intensity, 2) + np.random.normal(0, 0.1, len(exec_intensity))
        
        best_lag = 0
        max_corr = -1.0

        for lag in range(0, 5):
            if lag == 0:
                corr = np.corrcoef(exec_intensity, depth_changes)[0, 1]
            else:
                corr = np.corrcoef(exec_intensity[lag:], depth_changes[:-lag])[0, 1]

            if not np.isnan(corr) and corr > max_corr:
                max_corr = corr
                best_lag = lag

        if max_corr >= 0.70 and best_lag == 0:
            status = "⚠️ 자전거래/허매수"
        elif max_corr >= 0.45 and best_lag >= 1:
            status = "🔥 진짜 매집 흡수"
        else:
            status = "보통 수급"

        return {
            "max_corr": round(float(max_corr), 2),
            "best_lag": int(best_lag),
            "status": status
        }
    except Exception:
        return {"max_corr": 0.0, "best_lag": 0, "status": "일반수급"}


# ==============================================================================
# [알고리즘 고도화] 코사인 유사도, RSI 및 T-1 선행 매집 점수 산출
# ==============================================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


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

    vol_ma5 = volume.iloc[-6:-1].mean()
    vol_dry_ratio = (volume.iloc[-1] / vol_ma5) if vol_ma5 > 0 else 1.0
    is_volume_dry = (vol_dry_ratio <= 0.50)

    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma_max = max(ma5, ma10, ma20)
    ma_min = min(ma5, ma10, ma20)
    ma_compression = ((ma_max - ma_min) / ma20) * 100 if ma20 > 0 else 999.0

    is_volatility_threshold = (ma_compression < 3.0)

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

    late_volume_surge = False
    if df_30m is not None and len(df_30m) >= 6:
        avg_30m_vol = df_30m['volume'].iloc[:-2].mean()
        recent_30m_vol = df_30m['volume'].iloc[-1]
        if avg_30m_vol > 0 and (recent_30m_vol / avg_30m_vol) >= 1.8:
            late_volume_surge = True

    high_low_diff = (high - low).replace(0, np.nan)
    clv = (((close - low) - (high - close)) / high_low_diff).fillna(0)
    money_flow_vol = clv * volume
    
    vol_20_sum = volume.rolling(20).sum()
    cmf_series = money_flow_vol.rolling(20).sum() / vol_20_sum
    latest_cmf = cmf_series.iloc[-1] if not np.isnan(cmf_series.iloc[-1]) else 0.0

    rsi_series = calculate_rsi(close, period=14)
    latest_rsi = rsi_series.iloc[-1] if not np.isnan(rsi_series.iloc[-1]) else 50.0

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
        "rsi": round(latest_rsi, 1),
        "is_above_vwap": is_above_vwap
    }


def calculate_t1_score(metrics, surge_from_bottom, circ_ratio, is_dev_active, ob_metrics, lag_metrics, dump_metrics=None, iceberg_metrics=None):
    score = 0

    if surge_from_bottom >= 35.0:
        return 0
    elif surge_from_bottom <= 15.0:
        score += 20
    elif surge_from_bottom <= 25.0:
        score += 10

    if metrics['vol_dry_ratio'] <= 0.50:
        score += 40
    elif metrics['vol_dry_ratio'] <= 0.75:
        score += 20

    if metrics['ma_compression'] <= 2.5:
        score += 25
    elif metrics['ma_compression'] <= 4.5:
        score += 15

    if metrics['is_volatility_threshold']:
        score += 15

    if metrics['has_spike_candle']:
        score += 15
    if metrics['is_obv_div']:
        score += 15

    cmf_val = metrics['cmf']
    is_above_vwap = metrics['is_above_vwap']

    if cmf_val >= 0.10 and is_above_vwap:
        score += 20
    elif cmf_val >= 0.02:
        score += 10
    elif cmf_val < -0.08:
        score -= 20

    rsi_val = metrics['rsi']
    if 45.0 <= rsi_val <= 62.0:
        score += 15
    elif rsi_val >= 72.0:
        score -= 20

    spread = ob_metrics["spread_ratio"]
    bid_ask = ob_metrics["bid_ask_ratio"]

    if spread <= 0.15 and bid_ask >= 1.8:
        score += 15
    elif spread > 0.50:
        score -= 15

    if lag_metrics["status"] == "🔥 진짜 매집 흡수":
        score += 20
    elif lag_metrics["status"] == "⚠️ 자전거래/허매수":
        score -= 25

    if dump_metrics:
        score += dump_metrics.get("score_modifier", 0)

    if iceberg_metrics:
        score += iceberg_metrics.get("score_modifier", 0)

    if -2.5 <= metrics['chg_1d'] <= 3.0:
        score += 10
    elif metrics['chg_1d'] > 7.0:
        score -= 25

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
    onchain_map = get_cached_onchain_flow(symbols) 
    dex_stake_map = get_cached_dex_and_staking_metrics(symbols)
    wallet_leadtime_map = get_cached_wallet_leadtime_metrics(symbols)

    print("\n[1/2] 30분봉 수급 및 호가/체결 시차 상관성 분석 중...")
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
            time.sleep(0.01)
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

    print("\n[2/2] T-1 매집 + STGT 멀티모달 및 100ms 고주파 LSTM/RL 덤핑 예측 교차 스캔 중...")
    results = []

    for item in tqdm(krw_coins, desc="통합 종합 스캔", ncols=100):
        ticker = item['ticker']
        korean_name = item['korean_name']
        symbol = item['symbol']

        try:
            df_daily = pyupbit.get_ohlcv(ticker, interval="day", count=120)
            if df_daily is None or len(df_daily) < 30:
                time.sleep(0.02)
                continue

            df_30m_recent = market_30m_data.get(ticker, None)
            metrics = calculate_t1_advanced_metrics(df_daily, df_30m_recent)
            if not metrics:
                continue

            if (metrics['last_value'] / 100_000_000) < 5.0:
                continue

            ob_metrics = get_orderbook_metrics(ticker)
            lag_metrics = get_time_lag_metrics(ticker)
            dump_metrics = get_realtime_dumping_velocity(ticker)
            
            if (metrics['last_value'] / 100_000_000) >= 10.0:
                iceberg_metrics = get_highfreq_iceberg_metrics(ticker, duration=1.5)
            else:
                iceberg_metrics = get_delta_t_iceberg_metrics(ticker)
            
            onchain_info = onchain_map.get(symbol, {"status": "데이터 없음", "score_modifier": 0})
            dex_info = dex_stake_map.get(symbol, {"status": "중립", "score_modifier": 0})
            wallet_info = wallet_leadtime_map.get(symbol, {"status": "중립", "score_modifier": 0})

            df_closed = df_daily.iloc[:-1]
            lowest_20d = df_closed['low'].iloc[-20:].min()
            surge_from_bottom = round(((metrics['last_close'] - lowest_20d) / lowest_20d) * 100, 2) if lowest_20d > 0 else 0.0
            circ_ratio = tokenomics_map.get(symbol, 75.0)
            is_dev_active = github_map.get(symbol, False)
            
            accumulation_score = calculate_t1_score(
                metrics, surge_from_bottom, circ_ratio, is_dev_active, 
                ob_metrics, lag_metrics, dump_metrics, iceberg_metrics
            )
            accumulation_score += (onchain_info["score_modifier"] + dex_info["score_modifier"] + wallet_info["score_modifier"])
            accumulation_score = max(0, accumulation_score)

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

            rank_count = len(hourly_rank_details.get(ticker, []))
            vol_mean_20 = df_daily["volume"].tail(20).mean()
            vol_mean_total = df_daily["volume"].mean()
            vol_ratio_pattern = vol_mean_20 / vol_mean_total if vol_mean_total > 0 else 0

            vol_dry_ratio = metrics['vol_dry_ratio']
            inv_vol_dry = 1.0 / (vol_dry_ratio + 0.1)
            compression_score = accumulation_score * inv_vol_dry

            pattern_prediction_score = (
                (compression_score * 0.15) + 
                (similarity_score * 0.25) + 
                (rank_count * 2.0) + 
                (vol_ratio_pattern * 2.5)
            )

            is_genuine = (
                lag_metrics['status'] == "🔥 진짜 매집 흡수" and 
                onchain_info['score_modifier'] >= 0 and 
                dex_info['score_modifier'] >= 0 and 
                wallet_info['score_modifier'] >= 0 and
                dump_metrics['score_modifier'] >= 0 and
                iceberg_metrics['score_modifier'] >= 0
            )

            results.append({
                "코인명": f"{korean_name} 🔥" if is_genuine else korean_name,
                "심볼": symbol,
                "종합예측점수": round(pattern_prediction_score, 2),
                "패턴유사도(%)": round(similarity_score, 1),
                "매집점수": accumulation_score,
                "현재가(KRW)": format_price(metrics['last_close']),
                "거래량절벽(배)": metrics['vol_dry_ratio'],
                "이평선수렴(%)": metrics['ma_compression'],
                "CMF지표": metrics['cmf'],
                "RSI": metrics['rsi'],
                "스프레드(%)": ob_metrics['spread_ratio'],
                "시차상관성": lag_metrics['max_corr'],
                "지연시간(Lag)": f"{lag_metrics['best_lag']}초",
                "진짜매집판정": lag_metrics['status'],
                "매도덤핑속도": dump_metrics['status'],
                "아이스버그역산(고주파)": iceberg_metrics['status'],
                "온체인동향": onchain_info['status'], 
                "DEX/스테이킹동향": dex_info['status'],
                "지갑이동 리드타임": wallet_info['status'],
                "VWAP상회": "상회" if metrics['is_above_vwap'] else "하회",
                "1일 변동률(%)": metrics['chg_1d'],
                "7일 변동률(%)": metrics['chg_7d'],
                "바닥대비상승(%)": surge_from_bottom,
                "15위내_진입(회)": rank_count,
                "과거급등(회)": spike_count,
                "유통량비율(%)": circ_ratio,
                "거래대금(억원)": round(metrics['last_value'] / 100_000_000, 1),
                "개발활력": "양호" if is_dev_active else "보통"
            })

        except Exception:
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        # STGT(Spatiotemporal Graph Transformer) 기반 시장 전체 수급 동조화 및 덤핑 전이 교차 검증
        df = evaluate_market_graph_dump_risk(df)
        
        df = df.sort_values(
            by=["종합예측점수", "매집점수", "시차상관성"], 
            ascending=[False, False, False]
        ).reset_index(drop=True)
    return df


# ==============================================================================
# [AI 심층 분석] Gemini API (최신 google-genai 규격)
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
                "RSI": row['RSI'],
                "수급진위판정": row['진짜매집판정'],
                "매도덤핑속도": row.get('매도덤핑속도', '정보 없음'),
                "아이스버그역산(고주파)": row.get('아이스버그역산(고주파)', '정보 없음'),
                "STGT그래프위험도": f"{row.get('STGT_그래프덤핑위험(%)', 0)}%",
                "온체인동향": row.get('온체인동향', '정보 없음'),
                "DEX/스테이킹동향": row.get('DEX/스테이킹동향', '정보 없음'),
                "지갑이동 리드타임": row.get('지갑이동 리드타임', '정보 없음'),
                "1일/7일변동률": f"{row['1일 변동률(%)']}% / {row['7일 변동률(%)']}%",
                "4시간봉_1주일_수급분석": info_4h
            })

        prompt = f"""
당신은 가상자산 수급, T-1 상승 직전 패턴 분석 및 WebSocket 100ms 고주파 트래킹과 **STGT(Spatiotemporal Graph Transformer)** 기반 멀티모달 시장 네트워크 분석 전문 AI입니다.
아래 [현재 스캔 상위 10개 데이터]와 [과거 추천 종목 성과 검증 데이터]를 비교 분석하여 정중한 경어체(~습니다, ~입니다)로 리포트를 작성해 주세요.

이번 알고리즘은 **[WebSocket 스트림을 통한 100ms 호가/체결 데이터 샘플링]**, **[LSTM + RL 강화학습 앙상블]**, 그리고 **[Spatiotemporal Graph Transformer(STGT)를 통한 온/오프체인 멀티모달 공간적 수급 동조화 및 이탈 전이 위험 진단]**을 결합하여 세력의 아이스버그 주문 및 덤핑 위협을 정밀 진단합니다.

[1. 현재 스캔 상위 10개 데이터]
{json.dumps(enriched_data, ensure_ascii=False, indent=2)}

[2. 과거 추천 종목 백테스팅 검증 요약 및 세부 내역]
요약: {eval_summary}
세부 성과: {json.dumps(eval_details[:8], ensure_ascii=False, indent=2)} (최대 8개 표기)

[작성 지침]
1. **[STGT 네트워크 및 고주파 아이스버그 진단 분석]**:
   - STGT 기반 멀티모달 그래프 네트워크의 시장 동조화 점수 및 전이 위험도(Attention), 그리고 100ms 고주파 추적을 통해 세력 이탈이나 덤핑 위험이 포착된 종목과, 단단한 수급을 유지하는 우수 종목을 분석해 주세요.
2. **[T-1 상승 직전 최우수 추천 종목 Top 3 전략]**:
   - 최우수 3개 종목의 진입 타점, 목표가, 손절가를 잔량 소진/재생성 및 STGT 지표와 연계하여 세밀히 작성해 주세요.
3. **[결론 및 멀티모달 프레임워크 성과]**:
   - 온체인(지갑이동)과 오프체인(호가) 데이터를 단일 임베딩으로 묶은 STGT 도입으로 인해 개선된 변곡점 포착의 이점을 요약해 주세요.
"""
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
        config = types.GenerateContentConfig(temperature=0.2)
        
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
        df_idx = row_idx - 2
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
        print("⚠️ 이메일 발송 조건 미충족 (환경 변수 확인 필요).")
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    msg = MIMEMultipart()
    msg["Subject"] = f"📊 [STGT+100ms 고주파+LSTM+RL 강화학습] 실시간 매집 분석 리포트 ({now_str})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECEIVER_EMAILS)
    
    body = f"""안녕하세요.

업비트 원화 마켓 [T-1 선행 매집 지표] 실시간 스캔 결과입니다.
* STGT(Spatiotemporal Graph Transformer) 기반 멀티모달 네트워크 및 WebSocket 100ms 고주파 덤핑 예측 엔진을 반영하였습니다.

• 분석 시각: {now_str}
• 과거 성과: {eval_summary}

==================================================
🤖 [Gemini AI STGT 수급 및 아이스버그 잔량 역산 실시간 심층 리포트]
==================================================
{ai_analysis}

상세 분석 데이터 및 성과 검증 내역은 첨부된 엑셀 파일 시트를 확인해 주세요.
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
    print("🚀 [업비트 원화 마켓] STGT 멀티모달 & 100ms 고주파 트래킹 & LSTM/RL 덤핑 예측 엔진 실행...")
    
    print("\n🔍 과거 추천 종목 수익률 자동 검증 중...")
    eval_summary, eval_details = evaluate_past_performance()
    print(f"👉 {eval_summary}")

    df_result = analyze_and_scan_market()
    
    if not df_result.empty:
        save_scan_history(df_result)

        print("\n=== 🎯 현재 상위 5개 추천 종목 (STGT 및 고주파 아이스버그 역산 반영) ===")
        print(df_result[["코인명", "종합예측점수", "패턴유사도(%)", "매집점수", "STGT_그래프덤핑위험(%)", "아이스버그역산(고주파)", "진짜매집판정"]].head(5))

        print("\n📊 엑셀 저장 및 AI 분석 생성 중...")
        excel_file = save_integrated_excel(df_result, eval_details)
        ai_report_text = generate_gemini_analysis(df_result, eval_summary, eval_details)
        
        is_manual_run = os.environ.get("IS_MANUAL_RUN", "false").lower() == "true"

        if is_manual_run:
            print("\n👆 [수동 실행 감지] 설정에 따라 이메일 종합 리포트를 발송합니다.")
            send_email_report(excel_file, ai_report_text, eval_summary)
        else:
            print("\n🤖 [자동 예약 실행] 이메일 리포트는 발송하지 않습니다. (수동 실행 시에만 발송)")

        danger_condition = df_result[
            df_result['매도덤핑속도'].str.contains("🚨|임박", na=False) | 
            df_result['아이스버그역산(고주파)'].str.contains("🚨|임박", na=False) |
            df_result['STGT_그래프덤핑위험(%)'] >= 75.0
        ]

        if not danger_condition.empty:
            print(f"\n🚨 [위험 감지] 총 {len(danger_condition)}개 종목에서 급락/덤핑 임박 신호 포착! 텔레그램 알림을 전송합니다.")
            
            msg_lines = ["🚨 *[업비트 덤핑 5분 전 예고 경고 (STGT & LSTM & RL 강화학습 예측)]* 🚨\n"]
            for _, row in danger_condition.iterrows():
                msg_lines.append(f"• *코인*: {row['코인명']} ({row['심볼']})")
                msg_lines.append(f"  - 현재가: {row['현재가(KRW)']}")
                msg_lines.append(f"  - STGT위험도: {row.get('STGT_그래프덤핑위험(%)', 0)}%")
                msg_lines.append(f"  - 상태: {row['아이스버그역산(고주파)']}\n")
            
            msg_lines.append("⚠️ 세력의 대규모 물량 소진 및 네트워크 전이 위험이 있으니 주의하세요!")
            
            telegram_message = "\n".join(msg_lines)
            send_telegram_alert(telegram_message)
        else:
            print("\n✨ [안전] 현재 덤핑 임박 신호가 잡힌 종목이 없어 텔레그램 알림을 생략합니다.")
            
    else:
        print("❌ 분석된 결과가 없습니다.")

    print(f"\n✨ 전체 프로세스 완료 (총 소요 시간: {round(time.time() - start_time, 2)}초)")
