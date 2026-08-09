import os
import io
import time
import datetime
import json
import requests
import urllib.parse
import feedparser
import numpy as np
import pandas as pd
import pyupbit
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 딥러닝 모델 경량 실행 및 로그 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 미설치: STGT 모델은 대체 로직으로 동작합니다.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow 미설치: 기본 통계 기반 로직으로 동작합니다.")


# ==============================================================================
# [설정] 파일 경로 및 환경 설정
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
AI_MODELS_DIR = "./ai_models"
CACHE_DIR = "./cache"
EXPERIENCE_FILE = os.path.join(AI_MODELS_DIR, "ai_experience.json")
SCAN_RESULT_JSON = os.path.join(DATA_DIR, "market_scan_result.json")
GOLDEN_PATTERN_URL = "https://raw.githubusercontent.com/mdog002-wq/upbit/main/data/golden_pattern.json"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AI_MODELS_DIR, exist_ok=True)


# ==============================================================================
# [골든 패턴 및 DTW 수집 모듈]
# ==============================================================================
def load_golden_pattern():
    try:
        headers = {}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        res = requests.get(GOLDEN_PATTERN_URL, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"⚠️ 골든 패턴 불러오기 중 오류: {e}")
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
    max_possible_dist = len(target_pattern)
    return round(max(0.0, 100.0 * (1.0 - (dist / max_possible_dist))), 1)


# ==============================================================================
# [AI 딥러닝 엔진] LSTM & STGT
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
            except Exception: pass
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
        if not TF_AVAILABLE or self.model is None: return 0.3
        try:
            x_input = np.array(features).reshape(1, self.seq_len, self.feats)
            return float(self.model.predict(x_input, verbose=0)[0][0])
        except Exception: return 0.3

    def train_step(self, x_data, y_labels):
        if not TF_AVAILABLE or self.model is None or not x_data: return
        try:
            self.model.fit(np.array(x_data), np.array(y_labels), epochs=3, verbose=0)
            self.model.save(self.model_path)
        except Exception: pass

lstm_predictor = LSTMIcebergPredictor()

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
                except Exception: pass
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()

    def predict(self, x_tensor, edge_index):
        if not TORCH_AVAILABLE or self.model is None: return []
        self.model.eval()
        with torch.no_grad():
            res = self.model(x_tensor, edge_index).squeeze()
            return res.tolist() if res.dim() > 0 else [res.item()]

    def train_step(self, x_tensor, edge_index, labels_tensor):
        if not TORCH_AVAILABLE or self.model is None: return
        try:
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(x_tensor, edge_index).squeeze()
            loss = self.criterion(outputs, labels_tensor)
            loss.backward()
            self.optimizer.step()
            torch.save(self.model.state_dict(), self.model_path)
        except Exception: pass

stgt_manager = STGTManager()


# ==============================================================================
# [자가학습 엔진]
# ==============================================================================
class AIEvolutionEngine:
    def __init__(self):
        self.exp_file = EXPERIENCE_FILE

    def save_experience(self, ticker, price, lstm_feats=None, stgt_feats=None):
        try:
            exps = {}
            if os.path.exists(self.exp_file):
                with open(self.exp_file, "r", encoding="utf-8") as f: exps = json.load(f)
            exps[ticker] = {"timestamp": time.time(), "lstm_feats": lstm_feats, "stgt_feats": stgt_feats, "price": price}
            with open(self.exp_file, "w", encoding="utf-8") as f: json.dump(exps, f, ensure_ascii=False)
        except Exception: pass

    def evolve_models(self):
        if not os.path.exists(self.exp_file): return
        try:
            with open(self.exp_file, "r", encoding="utf-8") as f: exps = json.load(f)
        except Exception: return

        current_time = time.time()
        expired_tickers = [f"KRW-{t}" for t, d in exps.items() if current_time - d.get("timestamp", 0) > 14400]
        if not expired_tickers: return

        print("🤖 [백엔드] 자가학습 진행 중...")
        try:
            prices_now = pyupbit.get_current_price(expired_tickers)
            if isinstance(prices_now, float): prices_now = {expired_tickers[0]: prices_now}
        except Exception: prices_now = {}

        lstm_x_train, lstm_y_train, stgt_x_train, stgt_y_train, keys_to_delete = [], [], [], [], []
        for ticker, data in exps.items():
            if current_time - data.get("timestamp", 0) > 14400:
                c_price = prices_now.get(f"KRW-{ticker}")
                if c_price and data.get("price"):
                    rate = (c_price - data["price"]) / data["price"] * 100
                    if data.get("lstm_feats"):
                        lstm_x_train.append(data["lstm_feats"])
                        lstm_y_train.append(1.0 if rate <= -3.0 else 0.0)
                    if data.get("stgt_feats"):
                        stgt_x_train.append(data["stgt_feats"])
                        stgt_y_train.append(1.0 if rate >= 5.0 else 0.0)
                keys_to_delete.append(ticker)

        if lstm_x_train: lstm_predictor.train_step(lstm_x_train, lstm_y_train)
        if stgt_x_train and TORCH_AVAILABLE:
            x_t = torch.tensor(stgt_x_train, dtype=torch.float32)
            y_t = torch.tensor(stgt_y_train, dtype=torch.float32)
            edge = torch.tensor([[i for i in range(len(stgt_x_train))], [i for i in range(len(stgt_x_train))]], dtype=torch.long)
            stgt_manager.train_step(x_t, edge, y_t)

        for k in keys_to_delete: exps.pop(k, None)
        with open(self.exp_file, "w", encoding="utf-8") as f: json.dump(exps, f, ensure_ascii=False)

ai_engine = AIEvolutionEngine()


# ==============================================================================
# [수집 & 정밀 메트릭 수집]
# ==============================================================================
def calculate_metrics(ticker):
    try:
        df_1h = pyupbit.get_ohlcv(ticker, interval="minute60", count=100)
        df_5m = pyupbit.get_ohlcv(ticker, interval="minute5", count=24)
        if df_1h is None or len(df_1h) < 30: return None

        close_1h, vol_1h = df_1h['close'], df_1h['volume']
        vol_recent_3m = df_5m['volume'].iloc[-3:].sum() if df_5m is not None and len(df_5m) >= 3 else vol_1h.iloc[-1]
        vol_prev_avg = (df_5m['volume'].iloc[-18:-3].mean() * 3) if df_5m is not None and len(df_5m) >= 18 else (vol_1h.iloc[-5:-1].mean() + 1e-8)
        
        vol_velocity = float(vol_recent_3m / (vol_prev_avg + 1e-8))
        vol_spike_ratio = float(vol_1h.iloc[-1] / (vol_1h.iloc[-25:-1].mean() + 1e-8))

        ma20 = close_1h.rolling(20).mean()
        std20 = close_1h.rolling(20).std()
        upper, lower = ma20 + (std20 * 2), ma20 - (std20 * 2)
        bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / (ma20.iloc[-1] + 1e-8))
        bb_breakout = float((close_1h.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-8))

        mfv = ((close_1h - df_1h['low']) - (df_1h['high'] - close_1h)) / (df_1h['high'] - df_1h['low'] + 1e-8) * vol_1h
        cmf_1h = float(mfv.iloc[-20:].sum() / (vol_1h.iloc[-20:].sum() + 1e-8))

        delta = close_1h.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_1h = float(100 - (100 / (1 + (gain / (loss + 1e-8)).iloc[-1])))

        vol_pct, price_pct = vol_1h.pct_change().fillna(0), close_1h.pct_change().fillna(0)
        lstm_sequence = [[float(vol_pct.iloc[i]), float(price_pct.iloc[i]), float(cmf_1h)] for i in range(-15, 0)]

        pattern_sim = 0.0
        if GLOBAL_GOLDEN_PATTERN and df_5m is not None and len(df_5m) == 24:
            golden_p = GLOBAL_GOLDEN_PATTERN.get("golden_pattern", [])
            if golden_p: pattern_sim = calculate_pattern_similarity(df_5m['close'].tolist(), golden_p)

        return {
            "last_close": float(close_1h.iloc[-1]),
            "vol_velocity": round(vol_velocity, 2),
            "vol_spike_ratio": round(vol_spike_ratio, 2),
            "bb_width": round(bb_width, 4),
            "bb_breakout": round(bb_breakout, 2),
            "cmf_1h": round(cmf_1h, 2),
            "rsi_1h": round(rsi_1h, 1),
            "lstm_sequence": lstm_sequence,
            "pattern_similarity": pattern_sim
        }
    except Exception: return None

def process_coin(item, price_map):
    ticker, symbol, name = item['market'], item['market'].replace("KRW-", ""), item['korean_name']
    metrics = calculate_metrics(ticker)
    if not metrics: return None

    dump_prob = lstm_predictor.predict(metrics["lstm_sequence"])
    dump_risk_flag = dump_prob >= 0.6

    score = round((metrics['vol_velocity'] * 20) + (metrics['cmf_1h'] * 15) + (metrics['pattern_similarity'] * 0.3) - (30 if dump_risk_flag else -5), 1)
    score = max(0.0, min(100.0, score))
    c_price = price_map.get(ticker, metrics['last_close'])

    stgt_feats = [metrics['vol_velocity'] / 10.0, metrics['bb_width'], metrics['cmf_1h'], metrics['rsi_1h'] / 100.0, metrics['bb_breakout'], metrics['pattern_similarity'] / 100.0, 1.0, 0.2, 0.5]
    ai_engine.save_experience(symbol, price=c_price, lstm_feats=metrics["lstm_sequence"], stgt_feats=stgt_feats)

    return {
        "market": ticker, "symbol": symbol, "name": name, "price": c_price,
        "quant_score": score, "dump_risk_pct": round(dump_prob * 100, 1),
        "pattern_similarity": metrics['pattern_similarity'], "rsi": metrics['rsi_1h'],
        "vol_velocity": metrics['vol_velocity'], "stgt_feats": stgt_feats
    }

def main():
    print("🚀 [Backend Data Collector] 매집 정보 수집 및 자가학습 실행 중...")
    ai_engine.evolve_models()

    res = requests.get("https://api.upbit.com/v1/market/all?isDetails=false").json()
    krw_coins = [c for c in res if c['market'].startswith("KRW-")]
    
    tickers = [c['market'] for c in krw_coins]
    try: price_map = pyupbit.get_current_price(tickers)
    except Exception: price_map = {}

    scanned_data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_coin, c, price_map): c for c in krw_coins}
        for f in tqdm(as_completed(futures), total=len(futures), ncols=80):
            r = f.result()
            if r: scanned_data.append(r)

    # STGT 그래프 네트워크 덤핑 확률 재계산
    if TORCH_AVAILABLE and stgt_manager.model and scanned_data:
        try:
            feats = np.array([d['stgt_feats'] for d in scanned_data])
            n = len(feats)
            if n > 1:
                x_t = torch.tensor(feats, dtype=torch.float32)
                e_src, e_dst = np.where(~np.eye(n, dtype=bool))
                edge_idx = torch.tensor([e_src, e_dst], dtype=torch.long)
                preds = stgt_manager.predict(x_t, edge_idx)
                for idx, p in enumerate(preds): scanned_data[idx]['stgt_dump_risk'] = round(p * 100, 1)
        except Exception: pass

    # 결과 데이터를 JSON으로 연동 저장 (Realtime 파이프라인으로 넘김)
    output_payload = {
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(scanned_data),
        "data": scanned_data
    }
    with open(SCAN_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"✅ 백엔드 매집 데이터 수집 완료! -> `{SCAN_RESULT_JSON}` 저장 됨.")

if __name__ == "__main__":
    main()
