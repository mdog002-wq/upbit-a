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
import asyncio
import pickle
import redis
import paramiko
from typing import List, Dict, Any
from pydantic import BaseModel, Field
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

# 딥러닝 모델 경량 실행 및 로그 억제
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
# [Gemini Structured Output 스키마 정의]
# ==============================================================================
class RecommendedCoin(BaseModel):
    coin_name: str = Field(description="코인 한글명 (예: 바빌론, 너보스, 스파크, 바운드리스, 시빅)")
    symbol: str = Field(description="티커 심볼 (예: BABY, CKB, SPK, ZKC, CVC)")
    reason: str = Field(description="추천 핵심 사유 요약")

class AIReportResponse(BaseModel):
    report_markdown: str = Field(description="좌측 패널용 종합 퀀트 분석 리포트 전문 (마크다운 형식)")
    recommended_coins: List[RecommendedCoin] = Field(description="AI가 최우선 추천하는 코인 종목 리스트")


def upload_html_to_oracle_server(local_file_path):
    """
    GitHub Secrets에 등록된 ORACLE_SSH_KEY(.key 내용)를 이용해 
    오라클 서버로 대시보드 HTML 파일을 자동 전송하는 함수
    """
    hostname = os.environ.get("ORACLE_DSN")          
    username = os.environ.get("ORACLE_USER", "ubuntu") 
    ssh_key_content = os.environ.get("ORACLE_SSH_KEY") 

    if not hostname or not ssh_key_content:
        print("⚠️ 오라클 접속 정보(IP 또는 SSH 키)가 설정되지 않아 서버 전송을 스킵합니다.")
        return

    remote_file_path = "templates/dashboard.html"

    try:
        key_file_like = io.StringIO(ssh_key_content)
        pkey = paramiko.RSAKey.from_private_key(key_file_like)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port=22, username=username, pkey=pkey, timeout=10)

        sftp = ssh.open_sftp()
        sftp.put(local_file_path, remote_file_path)
        print(f"🚀 오라클 서버로 HTML 대시보드 전송 완료! ({remote_file_path})")
        
        sftp.close()
        ssh.close()
        
    except Exception as e:
        print(f"❌ 오라클 서버 전송 실패: {e}")

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

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

EXCEL_FILE_PATH = "업비트_원화마켓_매집_패턴분석_리포트.xlsx"
HISTORY_CSV_PATH = "scan_history.csv"
CACHE_DIR = "./cache"
AI_MODELS_DIR = "./ai_models"
EXPERIENCE_FILE = os.path.join(AI_MODELS_DIR, "ai_experience.json")
AI_TRACKER_HISTORY_FILE = os.path.join(AI_MODELS_DIR, "ai_recommend_tracker.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AI_MODELS_DIR, exist_ok=True)

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True, socket_timeout=3)
    redis_client.ping()
except Exception as e:
    redis_client = None
    print(f"⚠️ Redis 연결 설정 실패: {e}")

# ==============================================================================
# [유틸] 데이터 포맷팅 및 캐싱
# ==============================================================================
def format_price(x):
    try:
        val = float(x)
        return f"{int(val):,}" if val >= 100 else (f"{val:,.2f}" if val >= 1 else f"{val:,.5f}")
    except Exception: return str(x)

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in TELEGRAM_CHAT_IDS:
        try: 
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except Exception: 
            pass

# ==============================================================================
# [신규 모듈] 추천 종목 실시간 속보/이슈 수집기 (Google RSS 기반)
# ==============================================================================
def fetch_news_for_recommended_coins(target_coins, max_news_per_coin=2):
    coin_news_dict = {}
    for coin in target_coins:
        query = urllib.parse.quote(f"{coin} 코인 이슈 when:7d")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            feed = feedparser.parse(rss_url)
            news_items = []
            for entry in feed.entries[:max_news_per_coin]:
                news_items.append({"title": entry.title, "link": entry.link})
            if news_items:
                coin_news_dict[coin] = news_items
        except Exception as e:
            print(f"⚠️ {coin} 속보 수집 스킵: {e}")
    return coin_news_dict

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
        try:
            x_arr = np.array(x_data)
            y_arr = np.array(y_labels)
            self.model.fit(x_arr, y_arr, epochs=3, verbose=0)
            self.model.save(self.model_path)
        except Exception as e:
            print(f"⚠️ LSTM 학습 스킵: {e}")

lstm_dumping_predictor = LSTMIcebergPredictor()

# ==============================================================================
# [AI 모듈 2] STGT (Spatiotemporal Graph Transformer) 자가학습
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
                try: 
                    self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
                except Exception: 
                    print("⚠️ STGT 가중치 로드 실패. 초기화합니다.")
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()

    def predict(self, x_tensor, edge_index):
        if not TORCH_AVAILABLE or self.model is None: return None
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
        except Exception as e:
            print(f"⚠️ STGT 학습 스킵: {e}")

stgt_manager = STGTManager()

# ==============================================================================
# [자가학습 엔진] 과거 경험 검증 및 모델 자동 업데이트
# ==============================================================================
class AIEvolutionEngine:
    def __init__(self):
        self.exp_file = EXPERIENCE_FILE

    def save_experience(self, ticker, price, lstm_feats=None, stgt_feats=None):
        try:
            exps = {}
            if os.path.exists(self.exp_file):
                with open(self.exp_file, "r", encoding="utf-8") as f: exps = json.load(f)
            
            exps[ticker] = {
                "timestamp": time.time(),
                "lstm_feats": lstm_feats,
                "stgt_feats": stgt_feats,
                "price": price
            }
            with open(self.exp_file, "w", encoding="utf-8") as f: json.dump(exps, f, ensure_ascii=False)
        except Exception: pass

    def evolve_models(self):
        if not os.path.exists(self.exp_file): return
        
        try:
            with open(self.exp_file, "r", encoding="utf-8") as f: exps = json.load(f)
        except Exception: return

        current_time = time.time()
        lstm_x_train, lstm_y_train = [], []
        stgt_x_train, stgt_y_train = [], []
        keys_to_delete = []

        expired_tickers = [f"KRW-{t}" for t, d in exps.items() if current_time - d.get("timestamp", 0) > 14400]
        if not expired_tickers: return

        print("🤖 [AI 진화 시스템] 과거 데이터 기반 자가학습 진행 중...")
        try:
            prices_now = pyupbit.get_current_price(expired_tickers)
            if isinstance(prices_now, float): prices_now = {expired_tickers[0]: prices_now}
        except Exception: prices_now = {}

        for ticker, data in list(exps.items()):
            if current_time - data.get("timestamp", 0) > 14400:
                market_symbol = f"KRW-{ticker}"
                current_price = prices_now.get(market_symbol) if prices_now else None
                
                if current_price and data.get("price"):
                    return_rate = (current_price - data["price"]) / data["price"] * 100
                    
                    is_dumped = 1.0 if return_rate <= -3.0 else 0.0
                    if data.get("lstm_feats"):
                        lstm_x_train.append(data["lstm_feats"])
                        lstm_y_train.append(is_dumped)
                    
                    is_pumped = 1.0 if return_rate >= 5.0 else 0.0
                    if data.get("stgt_feats"):
                        stgt_x_train.append(data["stgt_feats"])
                        stgt_y_train.append(is_pumped)
                
                keys_to_delete.append(ticker)

        if lstm_x_train:
            lstm_dumping_predictor.train_step(lstm_x_train, lstm_y_train)
        
        if stgt_x_train and TORCH_AVAILABLE:
            x_t = torch.tensor(stgt_x_train, dtype=torch.float32)
            y_t = torch.tensor(stgt_y_train, dtype=torch.float32)
            dummy_edge = torch.tensor([[i for i in range(len(stgt_x_train))], [i for i in range(len(stgt_x_train))]], dtype=torch.long)
            stgt_manager.train_step(x_t, dummy_edge, y_t)
            
        for k in keys_to_delete:
            if k in exps: del exps[k]
        with open(self.exp_file, "w", encoding="utf-8") as f: json.dump(exps, f, ensure_ascii=False)

ai_engine = AIEvolutionEngine()

# ==============================================================================
# [스캔 분석 유틸 및 점수 보정 함수]
# ==============================================================================
def get_krw_upbit_tickers():
    url = "https://api.upbit.com/v1/market/all?isDetails=false"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return [{'ticker': c['market'], 'korean_name': c['korean_name'], 'symbol': c['market'].replace("KRW-", "")} 
                    for c in res.json() if c['market'].startswith("KRW-")]
    except Exception: pass
    return []

def calculate_t1_advanced_metrics(ticker):
    for attempt in range(2):
        try:
            df_1h = pyupbit.get_ohlcv(ticker, interval="minute60", count=100)
            df_daily = pyupbit.get_ohlcv(ticker, interval="day", count=30)
            
            if df_1h is None or len(df_1h) < 30 or df_daily is None or len(df_daily) < 10:
                return None

            close_1h = df_1h['close']
            vol_1h = df_1h['volume']
            
            vol_mean_24h = vol_1h.iloc[-25:-1].mean() + 1e-8
            vol_spike_ratio = float(vol_1h.iloc[-1] / vol_mean_24h)
            
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
            rs = gain / (loss + 1e-8)
            rsi_1h = float(100 - (100 / (1 + rs.iloc[-1])))

            vol_pct = df_1h['volume'].pct_change().fillna(0)
            price_pct = df_1h['close'].pct_change().fillna(0)
            
            lstm_sequence = []
            for i in range(-15, 0):
                lstm_sequence.append([
                    float(vol_pct.iloc[i]),
                    float(price_pct.iloc[i]),
                    float(cmf_1h)
                ])

            return {
                "last_close": float(close_1h.iloc[-1]),
                "vol_spike_ratio": round(vol_spike_ratio, 2),
                "bb_width": round(bb_width, 4),
                "bb_breakout": round(bb_breakout, 2),
                "cmf_1h": round(cmf_1h, 2),
                "rsi_1h": round(rsi_1h, 1),
                "lstm_sequence": lstm_sequence
            }
        except Exception:
            time.sleep(0.1)
    return None

def get_highfreq_iceberg_metrics(ticker, real_lstm_sequence=None):
    if real_lstm_sequence and len(real_lstm_sequence) == 15:
        lstm_feats = real_lstm_sequence
    else:
        lstm_feats = [[0.0, 0.0, 0.0] for _ in range(15)]
        
    dump_prob = lstm_dumping_predictor.predict(lstm_feats) if (lstm_dumping_predictor and lstm_dumping_predictor.model) else 0.3
    if dump_prob is None: dump_prob = 0.3
    
    return {
        "status": f"💎 정상 수급 (덤핑확률 {round(dump_prob*100,1)}%)" if dump_prob < 0.6 else f"🚨 덤핑 위험 (덤핑확률 {round(dump_prob*100,1)}%)",
        "score_modifier": -30 if dump_prob >= 0.6 else 5,
        "raw_lstm_feats": lstm_feats
    }

def process_single_coin(item, current_price_map):
    ticker, symbol, korean_name = item['ticker'], item['symbol'], item['korean_name']
    try:
        time.sleep(0.1)
        metrics = calculate_t1_advanced_metrics(ticker)
        
        if not metrics: 
            c_price = current_price_map.get(ticker, 0)
            return {
                "코인명": korean_name, "심볼": symbol, "현재가(KRW)": format_price(c_price),
                "raw_price": float(c_price), "종합예측점수": 50.0, "거래량절벽(배)": 1.0,
                "CMF지표": 0.0, "RSI": 50.0, "아이스버그역산(고주파)": "💎 정상 수급",
                "_stgt_feats": [0.5]*9
            }

        iceberg_metrics = get_highfreq_iceberg_metrics(ticker, metrics.get("lstm_sequence"))
        
        score = 40.0
        vol_score = min(25.0, (metrics['vol_spike_ratio'] - 1.0) * 10.0) if metrics['vol_spike_ratio'] > 1.0 else 0
        
        squeeze_bonus = 0
        if metrics['bb_width'] < 0.08:
            squeeze_bonus += 10.0
            if metrics['bb_breakout'] >= 0.8:
                squeeze_bonus += 10.0
                
        cmf_score = max(-10.0, min(15.0, metrics['cmf_1h'] * 20.0))
        rsi_penalty = -10.0 if metrics['rsi_1h'] >= 75.0 else 0.0

        total_score = score + vol_score + squeeze_bonus + cmf_score + rsi_penalty + iceberg_metrics['score_modifier']
        acc_score = round(max(0.0, min(100.0, total_score)), 1)

        c_price = current_price_map.get(ticker, metrics['last_close'])
        
        stgt_feats = [
            metrics['vol_spike_ratio'] / 10.0, 
            metrics['bb_width'], 
            metrics['cmf_1h'], 
            metrics['rsi_1h'] / 100.0, 
            metrics['bb_breakout'], 
            0.5, 1.0, 0.2, 0.5
        ]
        
        ai_engine.save_experience(symbol, price=c_price, lstm_feats=iceberg_metrics.get("raw_lstm_feats"), stgt_feats=stgt_feats)

        return {
            "코인명": korean_name,
            "심볼": symbol,
            "현재가(KRW)": format_price(c_price),
            "raw_price": float(c_price),
            "종합예측점수": acc_score,
            "거래량절벽(배)": metrics['vol_spike_ratio'],
            "CMF지표": metrics['cmf_1h'],
            "RSI": metrics['rsi_1h'],
            "아이스버그역산(고주파)": iceberg_metrics['status'],
            "_stgt_feats": stgt_feats
        }
    except Exception: 
        c_price = current_price_map.get(ticker, 0)
        return {
            "코인명": korean_name, "심볼": symbol, "현재가(KRW)": format_price(c_price),
            "raw_price": float(c_price), "종합예측점수": 50.0, "거래량절벽(배)": 1.0,
            "CMF지표": 0.0, "RSI": 50.0, "아이스버그역산(고주파)": "💎 정상 수급",
            "_stgt_feats": [0.5]*9
        }

def analyze_and_scan_market():
    krw_coins = get_krw_upbit_tickers()
    if not krw_coins: return pd.DataFrame()

    print("\n🚀 [시세 일괄 조회] 배치 처리 중...")
    tickers_list = [c['ticker'] for c in krw_coins]
    try:
        current_price_map = pyupbit.get_current_price(tickers_list)
        if isinstance(current_price_map, float): current_price_map = {tickers_list[0]: current_price_map}
    except Exception: current_price_map = {}

    results = []
    print(f"\n🚀 [멀티스레딩] 전체 원화마켓 코인({len(krw_coins)}개) 병렬 스캔 및 AI 예측 시작...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_coin, item, current_price_map): item for item in krw_coins}
        for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
            res = future.result()
            if res: results.append(res)

    df = pd.DataFrame(results)
    if df.empty: return df

    df['STGT_그래프덤핑위험(%)'] = 0.0

    if TORCH_AVAILABLE and stgt_manager.model and '_stgt_feats' in df.columns:
        try:
            feats = np.array(df['_stgt_feats'].tolist())
            n = len(feats)
            if n > 1:
                x_t = torch.tensor(feats, dtype=torch.float32)
                e_src, e_dst = np.where(~np.eye(n, dtype=bool))
                edge_idx = torch.tensor([e_src, e_dst], dtype=torch.long)
                preds = stgt_manager.predict(x_t, edge_idx)
                if isinstance(preds, (float, int)): preds = [preds]
                df['STGT_그래프덤핑위험(%)'] = [round(p * 100, 1) for p in preds]
        except Exception as e:
            print(f"⚠️ STGT 분석 스킵: {e}")

    if '_stgt_feats' in df.columns:
        df = df.drop(columns=['_stgt_feats'])

    return df.sort_values(by="종합예측점수", ascending=False)

def update_ai_recommendation_tracker(ai_report_coins, current_price_map, coin_status_map, top10_symbols=set()):
    history = {}
    
    if os.path.exists(AI_TRACKER_HISTORY_FILE):
        try:
            with open(AI_TRACKER_HISTORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    history = json.loads(content)
        except Exception as e:
            print(f"⚠️ 기존 트래킹 파일 로드 실패 (초기화 후 진행): {e}")
            history = {}

    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst_tz).strftime("%Y-%m-%d %H:%M:%S")

    for coin in ai_report_coins:
        symbol = coin['symbol']
        name = coin['name']
        c_price = current_price_map.get(symbol, coin.get('raw_price', 0.0))

        if symbol in history:
            history[symbol]['count'] += 1
            history[symbol]['current_price'] = c_price
            history[symbol]['last_recommended_at'] = now_str
        else:
            history[symbol] = {
                "name": name,
                "symbol": symbol,
                "count": 1,
                "top10_count": 0,
                "entry_price": c_price,
                "current_price": c_price,
                "first_recommended_at": now_str,
                "last_recommended_at": now_str
            }

    for symbol, item in history.items():
        if 'top10_count' not in item:
            item['top10_count'] = 0
            
        if symbol in top10_symbols:
            item['top10_count'] += 1
    
    to_remove = []
    for symbol, item in history.items():
        if symbol in current_price_map:
            item['current_price'] = current_price_map[symbol]
        
        entry_p = item['entry_price']
        curr_p = item['current_price']
        profit_rate = ((curr_p - entry_p) / entry_p * 100) if entry_p > 0 else 0.0

        status = coin_status_map.get(symbol, {})
        dump_risk = status.get('dump_risk', 0.0)

        is_target_reached = profit_rate >= 20.0
        is_value_lost = (dump_risk >= 70.0) or (profit_rate <= -10.0)

        if is_target_reached or is_value_lost:
            to_remove.append(symbol)

    for s in to_remove:
        if s in history:
            del history[s]

    try:
        with open(AI_TRACKER_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 트래킹 데이터 저장 실패: {e}")

    tracker_list = []
    for symbol, item in history.items():
        entry_p = item['entry_price']
        curr_p = item['current_price']
        profit_rate = ((curr_p - entry_p) / entry_p * 100) if entry_p > 0 else 0.0
        
        tracker_list.append({
            "name": item['name'],
            "symbol": item['symbol'],
            "count": item['count'],
            "top10_count": item.get('top10_count', 0),
            "entry_price": format_price(entry_p),
            "current_price": format_price(curr_p),
            "profit_rate": round(profit_rate, 2),
            "recommend_time": item['last_recommended_at']
        })

    tracker_list.sort(key=lambda x: (x['count'], x['recommend_time']), reverse=True)
    return tracker_list

def update_redis_for_dashboard(df_result, ai_report, tracking_monitor_data):
    if not redis_client or df_result.empty: return

    try:
        coin_grades = []
        for rank, (_, row) in enumerate(df_result.iterrows(), start=1):
            score = row['종합예측점수']
            dump_risk = row['STGT_그래프덤핑위험(%)']

            coin_grades.append({
                "rank": rank,
                "name": row['코인명'],
                "symbol": row['심볼'],
                "price": row['현재가(KRW)'],
                "score": score,
                "dump_risk": dump_risk,
                "iceberg": row['아이스버그역산(고주파)']
            })

        dashboard_payload = {
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_report": ai_report,
            "summary": {
                "total_scanned": len(coin_grades)
            },
            "ai_recommended_monitor": tracking_monitor_data,
            "all_coins": coin_grades
        }

        redis_client.set("upbit_ai_dashboard_data", json.dumps(dashboard_payload, ensure_ascii=False))
        print("⚡ [Redis] 전체 종목 페이로드 업데이트 완료!")
    except Exception as e:
        print(f"❌ [Redis] 데이터 업로드 실패: {e}")

# ==============================================================================
# [Gemini Structured Output 기반 분석 리포트]
# ==============================================================================
def generate_gemini_analysis(df_result):
    if df_result.empty:
        return "분석할 종목 데이터가 없습니다.", []
    
    top_coins = df_result.head(5)['코인명'].tolist()
    top_symbols = df_result.head(5)['심볼'].tolist()
    
    default_recommended = [
        {"symbol": sym, "name": name, "reason": "퀀트 예측 점수 상위 종목"} 
        for sym, name in zip(top_symbols, top_coins)
    ]

    if not GEMINI_API_KEY:
        report = f"AI 리포트: 현재 상위 모니터링 종목은 {', '.join(top_coins)} 입니다."
        return report, default_recommended

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            "당신은 암호화폐 퀀트 투자 전문가입니다. 아래 업비트 원화마켓 AI 퀀트 분석 상위 종목 데이터를 바탕으로 "
            "작성 원칙에 맞춰 종합 시장 분석 리포트를 작성하세요.\n\n"
            "작성 원칙:\n"
            "1. report_markdown 필드에는 마크다운 형식으로 작성된 종합 퀀트 분석 리포트 전문을 넣으세요.\n"
            "2. 가장 강력하게 추천하는 코인 3~5개를 선정하여 recommended_coins 배열에 한글 코인명, 심볼, 핵심 추천 사유를 명시하세요.\n\n"
            f"분석 데이터:\n{df_result.head(10).to_string()}"
        )

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AIReportResponse,
                temperature=0.2,
            ),
        )

        parsed_data = json.loads(response.text)
        ai_report = parsed_data.get("report_markdown", f"상위 모니터링 추천 종목: {', '.join(top_coins)}")
        rec_coins_data = parsed_data.get("recommended_coins", [])

        recommended_list = []
        for item in rec_coins_data:
            recommended_list.append({
                "symbol": item.get("symbol", "").strip().upper(),
                "name": item.get("coin_name", "").strip(),
                "reason": item.get("reason", "").strip()
            })

        return ai_report, recommended_list

    except Exception as e:
        print(f"⚠️ Gemini Structured Output 생성 스킵 (대체 로직): {e}")
        return f"AI 분석 리포트 (상위 추천 종목: {', '.join(top_coins)})", default_recommended

def export_to_excel_and_email(df_result, ai_report):
    try:
        df_result.to_excel(EXCEL_FILE_PATH, index=False)
        print(f"📊 엑셀 리포트 저장 완료: {EXCEL_FILE_PATH}")
        
        if SENDER_EMAIL and EMAIL_PASSWORD and RECEIVER_EMAILS:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = ", ".join(RECEIVER_EMAILS)
            msg['Subject'] = f"[업비트 AI Quant] 시장 분석 리포트 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')})"
            msg.attach(MIMEText(ai_report, 'plain', 'utf-8'))
            
            if os.path.exists(EXCEL_FILE_PATH):
                with open(EXCEL_FILE_PATH, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(EXCEL_FILE_PATH))
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(EXCEL_FILE_PATH)}"'
                    msg.attach(part)
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(SENDER_EMAIL, EMAIL_PASSWORD)
                server.send_message(msg)
            print("📧 이메일 리포트 발송 완료!")
    except Exception as e:
        print(f"❌ 엑셀/이메일 발송 작업 중 오류: {e}")

# ==============================================================================
# [리포트 생성 및 대시보드 HTML 출력]
# ==============================================================================import os
def generate_dashboard_html(df_result, ai_report, tracking_monitor_data, news_data, html_path="docs/index.html"):
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    
    # AI 모니터링 종목 심볼 집합
    monitored_symbols = {item['symbol'] for item in tracking_monitor_data}
    
    alerts = []
    
    if not df_result.empty:
        for _, row in df_result.iterrows():
            dump_risk = float(row['STGT_그래프덤핑위험(%)'])
            if dump_risk >= 75.0:
                alerts.append({"text": f"⚠️ {row['코인명']}({row['심볼']}) - {row['아이스버그역산(고주파)']}"})

    kst_tz = datetime.timezone(datetime.timedelta(hours=9))
    updated_time = datetime.datetime.now(kst_tz).strftime("%Y-%m-%d %H:%M:%S")

    # 모든 코인의 세부 정보를 JS 연동용 딕셔너리로 저장
    coins_dict = {}

    # 순위별 전체 종목 테이블 행 생성
    table_rows_list = []
    if not df_result.empty:
        for rank, (_, row) in enumerate(df_result.iterrows(), start=1):
            symbol = row['심볼']
            market_code = f"KRW-{symbol}"
            name = row['코인명']
            price = row['현재가(KRW)']
            score = float(row['종합예측점수'])
            vol_ratio = float(row.get('거래량절벽(배)', 1.0))
            cmf_val = float(row.get('CMF지표', 0.0))
            rsi_val = float(row.get('RSI', 50.0))
            dump_risk = float(row.get('STGT_그래프덤핑위험(%)', 0.0))
            iceberg_status = str(row.get('아이스버그역산(고주파)', '💎 정상 수급'))

            coins_dict[market_code] = {
                "market": market_code,
                "symbol": symbol,
                "name": name,
                "price": price,
                "score": score,
                "rank": rank,
                "vol_ratio": vol_ratio,
                "cmf": cmf_val,
                "rsi": rsi_val,
                "dump_risk": dump_risk,
                "iceberg": iceberg_status,
                "is_monitored": symbol in monitored_symbols
            }
            
            sticker = ' <span class="badge bg-warning text-dark ms-1" style="font-size: 0.7rem;">🎯 AI추천</span>' if symbol in monitored_symbols else ''
            
            row_html = (
                f'<tr onclick="openModal(\'{market_code}\')" style="cursor: pointer;">\n'
                f' <td class="text-center fw-bold text-muted">{rank}</td>\n'
                f' <td class="fw-bold">{name} <span class="text-secondary small">({symbol})</span>{sticker}</td>\n'
                f' <td>{price}</td>\n'
                f' <td class="text-primary fw-bold">{score:.1f}점</td>\n'
                f"</tr>\n"
            )
            table_rows_list.append(row_html)
   
    all_coins_table_rows = "".join(table_rows_list) if table_rows_list else '<tr><td colspan="4" class="text-center text-muted py-3">분석된 종목이 없습니다.</td></tr>'

    alert_items = []
    for alert in alerts[:15]:
        alert_text = alert.get('text', '')
        alert_items.append(f'<div class="p-2 rounded bg-danger bg-opacity-10 border border-danger text-danger small fw-bold">{alert_text}</div>')
    alerts_html = "\n".join(alert_items) if alert_items else '<div class="text-muted small text-center py-3">현재 주의/위험 종목이 없습니다.</div>'

    news_items = []
    if news_data:
        for coin, items in news_data.items():
            li_tags = "".join([f'<li><a href="{item.get("link", "#")}" target="_blank" class="text-decoration-none text-dark">{item.get("title", "")}</a></li>' for item in items])
            news_items.append(f'<div class="p-2 border rounded bg-light"><strong class="text-primary">{coin}</strong><ul class="mb-0 ps-3 small">{li_tags}</ul></div>')
        news_html = "\n".join(news_items)
    else:
        news_html = '<div class="text-muted small text-center py-3">현재 등록된 추천 속보 이슈가 없습니다.</div>'

    tracking_items = []
    for item in tracking_monitor_data:
        p_rate = item['profit_rate']
        rate_color = "text-danger" if p_rate > 0 else ("text-primary" if p_rate < 0 else "text-dark")
        sign = "+" if p_rate > 0 else ""
        top10_cnt = item.get('top10_count', 0)
        market_code = f"KRW-{item['symbol']}"

        card_html = (
            f'<div class="p-3 border rounded bg-white shadow-sm mb-2" onclick="openModal(\'{market_code}\')" style="cursor: pointer;">\n'
            f' <div class="d-flex justify-content-between align-items-center mb-1">\n'
            f' <strong class="text-dark fs-6">🎯 {item["name"]} <span class="text-muted small">({item["symbol"]})</span></strong>\n'
            f' <div class="d-flex gap-1 align-items-center">\n'
            f' <span class="badge bg-primary rounded-pill">추천 {item["count"]}회</span>\n'
            f' <span class="badge bg-primary rounded-pill">TOP10 {top10_cnt}회</span>\n'
            f' </div>\n'
            f' </div>\n'
            f' <div class="row g-1 small text-secondary mt-1">\n'
            f' <div class="col-6">추천진입가: <b>{item["entry_price"]}</b></div>\n'
            f' <div class="col-6 text-end">현재가: <b>{item["current_price"]}</b></div>\n'
            f' <div class="col-6">수익률: <b class="{rate_color}">{sign}{p_rate}%</b></div>\n'
            f' <div class="col-6 text-end text-muted" style="font-size:0.75rem;">{item["recommend_time"]}</div>\n'
            f' </div>\n'
            f'</div>\n'
        )
        tracking_items.append(card_html)
    tracking_html = "\n".join(tracking_items) if tracking_items else '<div class="text-muted small text-center py-3">현재 모니터링 중인 AI 추천 종목이 없습니다.</div>'

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
        ' .alert-box { max-height: 140px; overflow-y: auto; }\n'
        ' .news-box { max-height: 140px; overflow-y: auto; }\n'
        ' .table-scroll-box { max-height: 650px; overflow-y: auto; }\n'
        ' .tracking-box { max-height: 750px; overflow-y: auto; }\n'
        ' .report-body h1, .report-body h2, .report-body h3 { font-size: 1rem; font-weight: bold; margin-top: 0.5rem; color: #0f172a; }\n'
        ' .report-body ul { padding-left: 1.2rem; margin-bottom: 0.5rem; }\n'
        ' #allCoinsTable tbody tr:hover { background-color: #f1f5f9 !important; transition: background-color 0.15s ease-in-out; }\n'
        '\n'
        ' .modal-overlay {\n'
        ' display: none;\n'
        ' position: fixed;\n'
        ' top: 0; left: 0; width: 100%; height: 100%;\n'
        ' background-color: rgba(15, 23, 42, 0.55);\n'
        ' backdrop-filter: blur(4px);\n'
        ' z-index: 9999;\n'
        ' align-items: center;\n'
        ' justify-content: center;\n'
        ' }\n'
        ' .modal-memo {\n'
        ' width: 950px;\n'
        ' height: 650px;\n'
        ' background-color: #ffffff;\n'
        ' border-radius: 16px;\n'
        ' box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.08);\n'
        ' border: 1px solid #e2e8f0;\n'
        ' display: flex;\n'
        ' flex-direction: column;\n'
        ' overflow: hidden;\n'
        ' position: relative;\n'
        ' animation: modalPop 0.2s ease-out;\n'
        ' }\n'
        ' @keyframes modalPop {\n'
        ' from { transform: scale(0.92); opacity: 0; }\n'
        ' to { transform: scale(1); opacity: 1; }\n'
        ' }\n'
        ' .modal-header-memo {\n'
        ' background-color: #f8fafc;\n'
        ' padding: 12px 16px;\n'
        ' border-bottom: 1px solid #e2e8f0;\n'
        ' display: flex;\n'
        ' justify-content: space-between;\n'
        ' align-items: center;\n'
        ' }\n'
        ' .modal-frames-container {\n'
        ' display: flex;\n'
        ' width: 100%;\n'
        ' height: 100%;\n'
        ' flex: 1;\n'
        ' }\n'
        ' .modal-frame-pane {\n'
        ' width: 50%;\n'
        ' height: 100%;\n'
        ' border: none;\n'
        ' }\n'
        ' .modal-frame-pane:first-child {\n'
        ' border-right: 1px solid #e2e8f0;\n'
        ' }\n'
        ' .close-memo-btn {\n'
        ' border: none;\n'
        ' background: transparent;\n'
        ' color: #94a3b8;\n'
        ' font-size: 1.25rem;\n'
        ' cursor: pointer;\n'
        ' line-height: 1;\n'
        ' transition: color 0.15s;\n'
        ' }\n'
        ' .close-memo-btn:hover { color: #0f172a; }\n'
        ' </style>\n'
        '</head>\n'
        '<body>\n'
        ' <div id="mainDashboardApp" class="container-fluid my-4 px-4" style="max-width: 1700px;">\n'
        ' <div class="row mb-4 align-items-center">\n'
        ' <div class="col-md-3 text-start">\n'
        ' <a href="https://upbit-r.onrender.com" class="btn btn-primary fw-bold px-3 py-2 shadow-sm">\n'
        ' <i class="fa-solid fa-robot me-1"></i> AI 실시간\n'
        ' </a>\n'
        ' </div>\n'
        ' <div class="col-md-6 text-center">\n'
        ' <h2 class="fw-bold text-dark mb-0 fs-4"><i class="fa-solid fa-chart-pie text-primary me-2"></i>업비트 AI 분석 대시보드</h2>\n'
        ' <small class="text-muted">최종 업데이트: __UPDATED_TIME__ (총 __TOTAL_COINS__개 종목 분석 완료)</small>\n'
        ' </div>\n'
        ' <div class="col-md-3"></div>\n'
        ' </div>\n'
        ' <div class="row g-4">\n'
        ' <div class="col-lg-3">\n'
        ' <div class="d-flex flex-column gap-3">\n'
        ' <div class="card p-3 shadow-sm">\n'
        ' <h6 class="fw-bold text-danger mb-3"><i class="fa-solid fa-triangle-exclamation me-1"></i> 실시간 급락/위험 경고</h6>\n'
        ' <div class="alert-box d-flex flex-column gap-2">\n'
        ' __ALERTS_HTML__\n'
        ' </div>\n'
        ' </div>\n'
        ' <div class="card p-3 shadow-sm">\n'
        ' <h6 class="fw-bold text-success mb-3"><i class="fa-solid fa-newspaper me-1"></i> 실시간 속보</h6>\n'
        ' <div class="news-box d-flex flex-column gap-2">\n'
        ' __NEWS_HTML__\n'
        ' </div>\n'
        ' </div>\n'
        ' <div class="card p-3 shadow-sm">\n'
        ' <h6 class="fw-bold text-primary mb-3"><i class="fa-solid fa-brain me-1"></i> AI 분석 리포트</h6>\n'
        ' <div id="reportMarkdownContainer" class="report-body text-secondary small bg-light p-3 rounded" style="max-height: 450px; overflow-y: auto; line-height: 1.5;"></div>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        ' <div class="col-lg-5">\n'
        ' <div class="card p-4 shadow-sm">\n'
        ' <div class="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2">\n'
        ' <h5 class="fw-bold mb-0 text-dark fs-5"><i class="fa-solid fa-trophy text-warning me-1"></i> 전체 코인 AI 예측 순위</h5>\n'
        ' <div class="input-group" style="max-width: 200px;">\n'
        ' <span class="input-group-text bg-white"><i class="fa-solid fa-search text-muted"></i></span>\n'
        ' <input type="text" id="coinSearchInput" class="form-control form-control-sm" placeholder="코인명/심볼 검색..." onkeyup="filterCoins()">\n'
        ' </div>\n'
        ' </div>\n'
        ' <div class="table-scroll-box">\n'
        ' <table class="table table-hover align-middle mb-0" id="allCoinsTable">\n'
        ' <thead class="table-light sticky-top"><tr><th class="text-center" style="width: 10%;">순위</th><th style="width: 40%;">코인명</th><th style="width: 25%;">현재가</th><th style="width: 25%;">예측점수</th></tr></thead>\n'
        ' <tbody>__ALL_COINS_TABLE_ROWS__</tbody>\n'
        ' </table>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'      
        ' <div class="col-lg-4">\n'
        ' <div class="card p-3 shadow-sm tracking-box">\n'
        ' <h6 class="fw-bold text-primary mb-3"><i class="fa-solid fa-chart-line me-1"></i> AI 추천종목 모니터 (🎯 표시 종목)</h6>\n'
        ' <div class="d-flex flex-column gap-2">\n'
        ' __TRACKING_HTML__\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        '\n'
        ' <div id="coinDetailModal" class="modal-overlay" onclick="closeModal()">\n'
        ' <div class="modal-memo" onclick="event.stopPropagation()">\n'
        ' <div class="modal-header-memo">\n'
        ' <div>\n'
        ' <h6 class="fw-bold mb-0 text-dark" id="modalCoinTitle">종목 상세 정보 (1번 & 2번 비교)</h6>\n'
        ' <small class="text-muted" id="modalCoinSub" style="font-size: 0.75rem;">upbit-r & upbit-a 양방향 비교</small>\n'
        ' </div>\n'
        ' <button type="button" class="close-memo-btn" onclick="closeModal()"><i class="fa-solid fa-xmark"></i></button>\n'
        ' </div>\n'
        ' <div class="modal-frames-container">\n'
        ' <iframe id="modalIframeLeft" class="modal-frame-pane" title="1번 사이트 상세"></iframe>\n'
        ' <iframe id="modalIframeRight" class="modal-frame-pane" title="2번 사이트 상세"></iframe>\n'
        ' </div>\n'
        ' </div>\n'
        ' </div>\n'
        '\n'
        ' <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>\n'
        ' <script>\n'
        ' const rawReportText = __AI_REPORT_JSON__;\n'
        ' const allCoinsMap = __ALL_COINS_MAP_JSON__;\n'
        ' document.getElementById("reportMarkdownContainer").innerHTML = marked.parse(rawReportText);\n'
        '\n'
        ' function filterCoins() {\n'
        ' let input = document.getElementById(\'coinSearchInput\').value.toLowerCase().trim();\n'
        ' let allRows = document.querySelectorAll(\'#allCoinsTable tbody tr\');\n'
        '\n'
        ' allRows.forEach(row => {\n'
        ' let coinText = row.cells[1]?.innerText.toLowerCase() || \'\';\n'
        ' if (input === \'\') {\n'
        ' row.style.display = \'\';\n'
        ' } else {\n'
        ' row.style.display = coinText.includes(input) ? \'\' : \'none\';\n'
        ' }\n'
        ' });\n'
        ' }\n'
        '\n'
        ' function openModal(marketCode) {\n'
        ' const modalTitle = document.getElementById(\'modalCoinTitle\');\n'
        ' const modalSub = document.getElementById(\'modalCoinSub\');\n'
        '\n'
        ' const coinData = allCoinsMap[marketCode];\n'
        ' if (coinData) {\n'
        ' modalTitle.innerText = `${coinData.name} (${coinData.symbol}) - 양방향 상세 비교`;\n'
        ' modalSub.innerText = `Market: ${marketCode}`;\n'
        ' } else {\n'
        ' modalTitle.innerText = "종목 상세 비교";\n'
        ' modalSub.innerText = marketCode;\n'
        ' }\n'
        '\n'
        ' const leftUrl = `https://upbit-r.onrender.com/?symbol=${marketCode}`;\n'
        ' const rightUrl = `https://upbit-a.onrender.com/?symbol=${marketCode}`;\n'
        '\n'
        ' document.getElementById(\'modalIframeLeft\').src = leftUrl;\n'
        ' document.getElementById(\'modalIframeRight\').src = rightUrl;\n'
        '\n'
        ' document.getElementById(\'coinDetailModal\').style.display = \'flex\';\n'
        ' }\n'
        '\n'
        ' function closeModal() {\n'
        ' document.getElementById(\'coinDetailModal\').style.display = \'none\';\n'
        ' document.getElementById(\'modalIframeLeft\').src = \'\';\n'
        ' document.getElementById(\'modalIframeRight\').src = \'\';\n'
        ' }\n'
        '\n'
        ' window.addEventListener(\'DOMContentLoaded\', () => {\n'
        ' const urlParams = new URLSearchParams(window.location.search);\n'
        ' const symbolParam = urlParams.get(\'symbol\');\n'
        '\n'
        ' if (symbolParam) {\n'
        ' const targetMarket = symbolParam.toUpperCase().startsWith(\'KRW-\') ? symbolParam.toUpperCase() : `KRW-${symbolParam.toUpperCase()}`;\n'
        ' openModal(targetMarket);\n'
        ' }\n'
        ' });\n'
        ' </script>\n'
        '</body>\n'
        '</html>'
    )

    html_content = html_template.replace("__UPDATED_TIME__", str(updated_time))\
                        .replace("__TOTAL_COINS__", str(len(df_result)))\
                        .replace("__ALERTS_HTML__", alerts_html)\
                        .replace("__NEWS_HTML__", news_html)\
                        .replace("__AI_REPORT_JSON__", json.dumps(str(ai_report), ensure_ascii=False))\
                        .replace("__ALL_COINS_MAP_JSON__", json.dumps(coins_dict, ensure_ascii=False))\
                        .replace("__ALL_COINS_TABLE_ROWS__", all_coins_table_rows)\
                        .replace("__TRACKING_HTML__", tracking_html)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"🎨 [대시보드] HTML 생성 완료 (`{html_path}`)!")
    return html_content

# ==============================================================================
# [메인 실행부]
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    ai_engine.evolve_models()
    
    df_result = analyze_and_scan_market()
    
    if not df_result.empty:
        print(f"\n=== 🎯 [자가학습 AI 적용] 전체 분석 종목 수: {len(df_result)}개 ===")
        print(df_result[["코인명", "종합예측점수", "STGT_그래프덤핑위험(%)", "아이스버그역산(고주파)"]].head(5))
        
        top10_symbols = set(df_result.head(10)['심볼'].tolist())

        danger_coins = df_result[df_result['STGT_그래프덤핑위험(%)'] >= 85.0]
        if not danger_coins.empty:
            print(f"\n🚨 [위험 감지] {len(danger_coins)}개 종목 덤핑 위험! (텔레그램 전송)")
            send_telegram_alert(f"🚨 *[자가학습 AI 경고]* 덤핑 위험 감지: {', '.join(danger_coins['코인명'].tolist())}")

        ai_report, recommended_coins_ai = generate_gemini_analysis(df_result)

        symbol_to_raw_price = dict(zip(df_result['심볼'], df_result['raw_price']))
        
        ai_report_coins = []
        for coin_info in recommended_coins_ai:
            sym = coin_info['symbol']
            name = coin_info['name']
            matching_rows = df_result[df_result['심볼'] == sym]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                ai_report_coins.append({
                    "symbol": sym,
                    "name": row['코인명'],
                    "raw_price": row['raw_price']
                })
            else:
                c_price = symbol_to_raw_price.get(sym, 0.0)
                ai_report_coins.append({
                    "symbol": sym,
                    "name": name if name else sym,
                    "raw_price": c_price
                })

        coin_status_map = {}
        for _, r in df_result.iterrows():
            coin_status_map[r['심볼']] = {
                "score": float(r['종합예측점수']),
                "dump_risk": float(r['STGT_그래프덤핑위험(%)'])
            }

        tracking_monitor_data = update_ai_recommendation_tracker(
            ai_report_coins, 
            symbol_to_raw_price, 
            coin_status_map,
            top10_symbols=top10_symbols
        )

        all_target_coins = [item['name'] for item in ai_report_coins]
        news_data = fetch_news_for_recommended_coins(all_target_coins)

        try:
            fastapi_payload = {
                "generation": 1,
                "recommendations": [
                    {
                        "market": f"KRW-{coin['symbol']}", 
                        "ai_grade": "추천", 
                        "score": float(df_result[df_result['심볼'] == coin['symbol']]['종합예측점수'].values[0]) if not df_result[df_result['심볼'] == coin['symbol']].empty else 95.0
                    }
                    for coin in ai_report_coins
                ]
            }
            response = requests.post("http://140.245.99.254:8000/api/ai-recommendations", json=fastapi_payload, timeout=5)
            if response.status_code == 200:
                print("🔥 [연동 성공] Gemini AI 추천 종목이 FastAPI 대시보드 서버로 성공적으로 전송되었습니다!")
            else:
                print(f"⚠️ FastAPI 서버 전송 응답 오류: {response.status_code}")
        except Exception as e:
            print(f"⚠️ FastAPI 서버로 AI 추천 종목 전송 실패: {e}")
            
        update_redis_for_dashboard(df_result, ai_report, tracking_monitor_data)

        generate_dashboard_html(df_result, ai_report, tracking_monitor_data, news_data, html_path="docs/index.html")

        upload_html_to_oracle_server("docs/index.html")

        kst_tz = datetime.timezone(datetime.timedelta(hours=9))
        kst_now = datetime.datetime.now(kst_tz)
        current_hour = kst_now.hour
        target_hours = [9, 13, 17, 21] 

        if current_hour in target_hours:
            export_to_excel_and_email(df_result, ai_report)
        else:
            print(f"⏰ 현재 시각(KST {current_hour}시)은 이메일 발송 시간이 아니므로 대시보드만 갱신합니다.")
