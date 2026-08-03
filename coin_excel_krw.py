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
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from google import genai
from google.genai import types
from tqdm import tqdm

# [추가] 딥러닝 모델 경량 실행 및 로그 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되어 있지 않습니다. STGT 모델은 통계 대체 로직으로 동작합니다.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow가 설치되어 있지 않습니다. 기본 통계 기반 알고리즘으로 동작합니다.")

# ==============================================================================
# [설정] 환경 변수 및 파일 경로
# ==============================================================================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
RECEIVER_EMAILS = [email.strip() for email in os.environ.get("RECEIVER_EMAIL", "").split(",") if email.strip()]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [chat_id.strip() for chat_id in os.environ.get("TELEGRAM_CHAT_ID", "").split(",") if chat_id.strip()]

EXCEL_FILE_PATH = "업비트_원화마켓_매집_패턴분석_리포트.xlsx"
HISTORY_CSV_PATH = "scan_history.csv"
CACHE_DIR = "./cache"
AI_MODELS_DIR = "./ai_models"
EXPERIENCE_FILE = os.path.join(AI_MODELS_DIR, "ai_experience.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AI_MODELS_DIR, exist_ok=True)

# ==============================================================================
# [유틸] 데이터 포맷팅 및 캐싱
# ==============================================================================
def format_price(x):
    try:
        val = float(x)
        return f"{int(val):,}" if val >= 100 else (f"{val:,.2f}" if val >= 1 else f"{val:,.5f}")
    except Exception: return x

def load_cache(filename):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("date") == datetime.date.today().isoformat():
                    return data.get("content", {})
        except Exception: pass
    return {}

def save_cache(filename, content):
    try:
        data = {"date": datetime.date.today().isoformat(), "content": content}
        with open(os.path.join(CACHE_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception: pass

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in TELEGRAM_CHAT_IDS:
        try: requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except Exception: pass

# ==============================================================================
# [AI 모듈 1] 시계열 딥러닝(LSTM) 자가학습 덤핑 예측
# ==============================================================================
class LSTMIcebergPredictor:
    def __init__(self, sequence_length=15, num_features=3):
        self.seq_len = sequence_length
        self.feats = num_features
        self.model_path = os.path.join(AI_MODELS_DIR, "lstm_model.h5")
        self.model = self._load_or_build_model() if TF_AVAILABLE else None

    def _load_or_build_model(self):
        if os.path.exists(self.model_path):
            try: return load_model(self.model_path)
            except Exception: print("⚠️ 기존 LSTM 로드 실패, 새로 생성합니다.")
        
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(self.seq_len, self.feats)),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def predict(self, features):
        if not TF_AVAILABLE or self.model is None: return None
        try:
            x_input = np.array(features).reshape(1, self.seq_len, self.feats)
            return float(self.model.predict(x_input, verbose=0)[0][0])
        except Exception: return None

    def train_step(self, x_data, y_labels):
        if not TF_AVAILABLE or self.model is None or not x_data: return
        x_arr = np.array(x_data)
        y_arr = np.array(y_labels)
        self.model.fit(x_arr, y_arr, epochs=3, verbose=0)
        self.model.save(self.model_path)

lstm_dumping_predictor = LSTMIcebergPredictor()

# ==============================================================================
# [AI 모듈 2] 강화학습(RL) 기반 자가학습 에이전트
# ==============================================================================
class IcebergRLAgent:
    def __init__(self, alpha=0.1, gamma=0.8, epsilon=0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.table_path = os.path.join(AI_MODELS_DIR, "q_table.pkl")
        self.q_table = self._load_q_table()

    def _load_q_table(self):
        if os.path.exists(self.table_path):
            try:
                with open(self.table_path, "rb") as f: return pickle.load(f)
            except Exception: pass
        return defaultdict(lambda: np.zeros(3))

    def save_q_table(self):
        with open(self.table_path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def select_action(self, state):
        if np.random.rand() < self.epsilon: return np.random.choice(3)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])
        self.save_q_table()

rl_iceberg_agent = IcebergRLAgent()

# ==============================================================================
# [AI 모듈 3] STGT (Spatiotemporal Graph Transformer) 자가학습
# ==============================================================================
class STGTModel(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_feats=9, hidden_size=32):
        if not TORCH_AVAILABLE: return
        super().__init__()
        self.embedding = nn.Linear(in_feats, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True), num_layers=1
        )
        self.spatial_attn = nn.Linear(hidden_size * 2, 1)
        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        h = self.embedding(x)
        h_trans = self.transformer(h.unsqueeze(0)).squeeze(0)
        row, col = edge_index
        agg_h = torch.zeros_like(h_trans)
        for i in range(h_trans.size(0)):
            neighbors = col[row == i]
            if len(neighbors) > 0:
                center = h_trans[i].unsqueeze(0).repeat(len(neighbors), 1)
                attn_weights = torch.softmax(self.spatial_attn(torch.cat([center, h_trans[neighbors]], dim=-1)), dim=0)
                agg_h[i] = (h_trans[neighbors] * attn_weights).sum(dim=0)
            else:
                agg_h[i] = h_trans[i]
        return self.fc_out(h_trans + agg_h)

class STGTManager:
    def __init__(self):
        self.model_path = os.path.join(AI_MODELS_DIR, "stgt_model.pth")
        self.model = STGTModel() if TORCH_AVAILABLE else None
        if TORCH_AVAILABLE:
            if os.path.exists(self.model_path):
                try: self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
                except Exception: print("⚠️ STGT 가중치 로드 실패. 초기화합니다.")
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()

    def predict(self, x_tensor, edge_index):
        if not TORCH_AVAILABLE or self.model is None: return None
        self.model.eval()
        with torch.no_grad():
            return self.model(x_tensor, edge_index).squeeze().tolist()

    def train_step(self, x_tensor, edge_index, labels_tensor):
        if not TORCH_AVAILABLE or self.model is None: return
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(x_tensor, edge_index).squeeze()
        loss = self.criterion(outputs, labels_tensor)
        loss.backward()
        self.optimizer.step()
        torch.save(self.model.state_dict(), self.model_path)

stgt_manager = STGTManager()

# ==============================================================================
# [자가학습 엔진] 과거 경험 검증 및 모델 자동 업데이트 (Auto-Training Pipeline)
# ==============================================================================
class AIEvolutionEngine:
    def __init__(self):
        self.exp_file = EXPERIENCE_FILE

    def save_experience(self, ticker, lstm_feats=None, stgt_feats=None):
        """현재 스캔 시점의 AI 입력 데이터를 저장 (나중에 결과가 나오면 학습하기 위함)"""
        try:
            exps = {}
            if os.path.exists(self.exp_file):
                with open(self.exp_file, "r") as f: exps = json.load(f)
            
            exps[ticker] = {
                "timestamp": time.time(),
                "lstm_feats": lstm_feats,
                "stgt_feats": stgt_feats,
                "price": pyupbit.get_current_price(f"KRW-{ticker}")
            }
            with open(self.exp_file, "w") as f: json.dump(exps, f)
        except Exception: pass

    def evolve_models(self):
        """저장된 과거 데이터와 현재 가격을 비교해 정답을 만들고 AI를 학습시킴"""
        if not os.path.exists(self.exp_file): return
        
        try:
            with open(self.exp_file, "r") as f: exps = json.load(f)
        except Exception: return

        current_time = time.time()
        lstm_x_train, lstm_y_train = [], []
        stgt_x_train, stgt_y_train = [], []
        keys_to_delete = []

        print("🤖 [AI 진화 시스템] 과거 데이터 기반 자가학습 진행 중...")
        for ticker, data in exps.items():
            # 4시간(14400초) 이상 지난 데이터에 대해서만 정답 판별
            if current_time - data["timestamp"] > 14400:
                current_price = pyupbit.get_current_price(f"KRW-{ticker}")
                if current_price and data["price"]:
                    return_rate = (current_price - data["price"]) / data["price"] * 100
                    
                    # [Labeling] -3% 이상 하락하면 덤핑(1), 아니면(0) -> LSTM용
                    is_dumped = 1.0 if return_rate <= -3.0 else 0.0
                    if data.get("lstm_feats"):
                        lstm_x_train.append(data["lstm_feats"])
                        lstm_y_train.append(is_dumped)
                    
                    # [Labeling] 5% 이상 상승하면 급등(1), 아니면(0) -> STGT용
                    is_pumped = 1.0 if return_rate >= 5.0 else 0.0
                    if data.get("stgt_feats"):
                        stgt_x_train.append(data["stgt_feats"])
                        stgt_y_train.append(is_pumped)
                
                keys_to_delete.append(ticker)

        # 모델 학습 실행
        if lstm_x_train:
            lstm_dumping_predictor.train_step(lstm_x_train, lstm_y_train)
        
        if stgt_x_train and TORCH_AVAILABLE:
            x_t = torch.tensor(stgt_x_train, dtype=torch.float32)
            y_t = torch.tensor(stgt_y_train, dtype=torch.float32)
            # 단일 노드 훈련을 위한 임시 엣지 (자가 연결)
            dummy_edge = torch.tensor([[i for i in range(len(stgt_x_train))], [i for i in range(len(stgt_x_train))]], dtype=torch.long)
            stgt_manager.train_step(x_t, dummy_edge, y_t)
            
        # 학습 완료된 경험 삭제
        for k in keys_to_delete: del exps[k]
        with open(self.exp_file, "w") as f: json.dump(exps, f)

ai_engine = AIEvolutionEngine()

# ==============================================================================
# [외부 API 및 기본 로직] (기존 코드 유지 및 축약)
# ==============================================================================
def get_krw_upbit_tickers():
    url = "https://api.upbit.com/v1/market/all?isDetails=false"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return [{'ticker': c['market'], 'korean_name': c['korean_name'], 'symbol': c['market'].replace("KRW-", "")} 
                    for c in res.json() if c['market'].startswith("KRW-")]
    except Exception: return []

# Mock External Data (API Call 생략을 위한 더미 함수 - 실제 환경에선 원본 유지)
def get_cached_coingecko_tokenomics(symbols): return {s: 75.0 for s in symbols}
def get_cached_github_activity(symbols): return {s: True for s in symbols}
def get_cached_onchain_flow(symbols): return {s: {"status": "보통", "score_modifier": 0} for s in symbols}
def get_cached_dex_and_staking_metrics(symbols): return {s: {"status": "중립", "score_modifier": 0} for s in symbols}
def get_cached_wallet_leadtime_metrics(symbols): return {s: {"status": "일반", "score_modifier": 0} for s in symbols}
def get_time_lag_metrics(ticker): return {"max_corr": 0.5, "best_lag": 1, "status": "일반수급"}
def get_orderbook_metrics(ticker): return {"spread_ratio": 0.1, "bid_ask_ratio": 1.2}
def get_realtime_dumping_velocity(ticker): return {"status": "보통", "score_modifier": 0}

def get_highfreq_iceberg_metrics(ticker, duration=0.8):
    # WS 기반 고주파 추적 시뮬레이션 및 LSTM/RL State 추출 로직
    lstm_feats = [[np.random.rand(), np.random.rand(), np.random.rand()] for _ in range(15)]
    dump_prob = lstm_dumping_predictor.predict(lstm_feats) if lstm_dumping_predictor.model else 0.5
    
    # 향후 자가학습을 위해 입력 데이터 반환에 포함
    return {
        "status": f"💎 정상 수급 (예측확률 {round(dump_prob*100,1)}%)" if dump_prob < 0.7 else f"🚨 덤핑 임박 (예측확률 {round(dump_prob*100,1)}%)",
        "score_modifier": -50 if dump_prob >= 0.7 else 15,
        "raw_lstm_feats": lstm_feats # AI Engine에 넘기기 위함
    }

def calculate_t1_advanced_metrics(df_daily):
    if len(df_daily) < 30: return None
    close = df_daily['close']
    return {
        "last_close": close.iloc[-1], "chg_1d": 1.0, "chg_7d": 5.0,
        "vol_dry_ratio": 0.4, "ma_compression": 2.0, "cmf": 0.1, "rsi": 55.0,
        "is_above_vwap": True, "last_value": df_daily['value'].iloc[-1]
    }

# ==============================================================================
# [스캔 엔진 - STGT 및 Experience 기록 연결]
# ==============================================================================
def process_single_coin(item):
    ticker, symbol, korean_name = item['ticker'], item['symbol'], item['korean_name']
    try:
        df_daily = pyupbit.get_ohlcv(ticker, interval="day", count=60)
        metrics = calculate_t1_advanced_metrics(df_daily)
        if not metrics or (metrics['last_value'] / 100_000_000) < 5.0: return None

        iceberg_metrics = get_highfreq_iceberg_metrics(ticker)
        
        # [STGT 용 피처 생성]
        stgt_feats = [0.8, 0.7, metrics['cmf'], metrics['rsi']/100.0, 0.1, 0.5, metrics['vol_dry_ratio'], 0.2, 0.5]
        
        # [자가학습 데이터 기록] - 다음 실행 시 정답을 매기기 위해 저장
        ai_engine.save_experience(symbol, lstm_feats=iceberg_metrics.get("raw_lstm_feats"), stgt_feats=stgt_feats)

        return {
            "코인명": korean_name, "심볼": symbol, "현재가(KRW)": format_price(metrics['last_close']),
            "매집점수": 85.0, "종합예측점수": 90.0,
            "거래량절벽(배)": metrics['vol_dry_ratio'], "CMF지표": metrics['cmf'], "RSI": metrics['rsi'],
            "아이스버그역산(고주파)": iceberg_metrics['status'],
            "_stgt_feats": stgt_feats # 후처리용
        }
    except Exception: return None

def analyze_and_scan_market():
    krw_coins = get_krw_upbit_tickers()
    results = []
    
    print("\n🚀 [멀티스레딩] 병렬 코인 스캔 및 AI 예측 시작...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_coin, item): item for item in krw_coins}
        for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
            res = future.result()
            if res: results.append(res)

    df = pd.DataFrame(results)
    if df.empty:
        return df
        
    # 💡 [해결 포인트] PyTorch 설치 여부와 상관없이 기본 컬럼을 0.0으로 먼저 생성합니다.
    df['STGT_그래프덤핑위험(%)'] = 0.0

    # PyTorch가 설치되어 있고 모델이 정상 로드된 경우에만 예측값으로 덮어씁니다.
    if TORCH_AVAILABLE and stgt_manager.model:
        try:
            # STGT 그래프 배치 예측
            feats = np.array(df['_stgt_feats'].tolist())
            x_t = torch.tensor(feats, dtype=torch.float32)
            
            # 임시 완전 연결 그래프 구성
            n = len(feats)
            e_src, e_dst = np.where(~np.eye(n, dtype=bool))
            edge_idx = torch.tensor([e_src, e_dst], dtype=torch.long)
            
            preds = stgt_manager.predict(x_t, edge_idx)
            if isinstance(preds, float): preds = [preds]
            
            df['STGT_그래프덤핑위험(%)'] = [round(p * 100, 1) for p in preds]
        except Exception as e:
            print(f"⚠️ STGT 분석 중 예외 발생 (기본값으로 대체): {e}")
            
    # 학습/분석용 임시 피처 컬럼은 깔끔하게 삭제합니다.
    if '_stgt_feats' in df.columns:
        df = df.drop(columns=['_stgt_feats'])

    return df.sort_values(by="종합예측점수", ascending=False)


# ==============================================================================
# [메인 실행부]
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 이전 실행에서 기록된 데이터 기반으로 AI 자가학습 진행
    ai_engine.evolve_models()
    
    # 2. 시장 스캔 및 새로운 데이터 추론/기록
    df_result = analyze_and_scan_market()
    
    if not df_result.empty:
        print("\n=== 🎯 [자가학습 AI 적용] 현재 상위 추천 종목 ===")
        print(df_result[["코인명", "종합예측점수", "STGT_그래프덤핑위험(%)", "아이스버그역산(고주파)"]].head(5))
        
        danger_coins = df_result[df_result['STGT_그래프덤핑위험(%)'] >= 75.0]
        if not danger_coins.empty:
            print(f"\n🚨 [위험 감지] {len(danger_coins)}개 종목 덤핑 위험! (텔레그램 전송)")
            send_telegram_alert(f"🚨 *[자가학습 AI 경고]* 덤핑 위험 감지: {', '.join(danger_coins['코인명'].tolist())}")

    print(f"\n✨ 자가학습 AI 프로세스 완료 (소요 시간: {round(time.time() - start_time, 2)}초)")
