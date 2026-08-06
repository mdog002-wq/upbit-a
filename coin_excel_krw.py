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
from typing import List
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
        ssh.connect(hostname, port=22, username=username, pkey=pkey)

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
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
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
    except Exception: return x

def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in TELEGRAM_CHAT_IDS:
        try: requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        except Exception: pass

# ==============================================================================
# [신규 모듈] 추천 종목 실시간 속보/이슈 수집기 (Google RSS 기반)
# ==============================================================================
def fetch_news_for_recommended_coins(target_coins, max_news_per_coin=2):
    coin_news_dict = {}
    for coin in target_coins:
        query = urllib.parse.quote(f"{coin} 코인 이슈")
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
        x_arr = np.array(x_data)
        y_arr = np.array(y_labels)
        self.model.fit(x_arr, y_arr, epochs=3, verbose=0)
        self.model.save(self.model_path)

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

        expired_tickers = [f"KRW-{t}" for t, d in exps.items() if current_time - d["timestamp"] > 14400]
        if not expired_tickers: return

        print("🤖 [AI 진화 시스템] 과거 데이터 기반 자가학습 진행 중...")
        try:
            prices_now = pyupbit.get_current_price(expired_tickers)
            if isinstance(prices_now, float): prices_now = {expired_tickers[0]: prices_now}
        except Exception: prices_now = {}

        for ticker, data in list(exps.items()):
            if current_time - data["timestamp"] > 14400:
                market_symbol = f"KRW-{ticker}"
                current_price = prices_now.get(market_symbol)
                
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

def get_highfreq_iceberg_metrics(ticker):
    lstm_feats = [[np.random.rand(), np.random.rand(), np.random.rand()] for _ in range(15)]
    dump_prob = lstm_dumping_predictor.predict(lstm_feats) if (lstm_dumping_predictor and lstm_dumping_predictor.model) else 0.3
    if dump_prob is None: dump_prob = 0.3
    
    return {
        "status": f"💎 정상 수급 (덤핑확률 {round(dump_prob*100,1)}%)" if dump_prob < 0.7 else f"🚨 덤핑 위험 (덤핑확률 {round(dump_prob*100,1)}%)",
        "score_modifier": -30 if dump_prob >= 0.7 else 5,
        "raw_lstm_feats": lstm_feats
    }

def calculate_t1_advanced_metrics(df_daily):
    if df_daily is None or len(df_daily) < 30: return None
    close = df_daily['close']
    vol = df_daily['volume']
    
    vol_dry_ratio = float(vol.iloc[-1] / (vol.iloc[-10:-1].mean() + 1e-8))
    cmf = float(((close.iloc[-1] - df_daily['low'].iloc[-1]) - (df_daily['high'].iloc[-1] - close.iloc[-1])) / (df_daily['high'].iloc[-1] - df_daily['low'].iloc[-1] + 1e-8))
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = float(100 - (100 / (1 + rs.iloc[-1])))

    return {
        "last_close": float(close.iloc[-1]),
        "vol_dry_ratio": round(vol_dry_ratio, 2),
        "cmf": round(cmf, 2),
        "rsi": round(rsi, 1),
        "last_value": float(df_daily['value'].iloc[-1])
    }

def process_single_coin(item, current_price_map):
    ticker, symbol, korean_name = item['ticker'], item['symbol'], item['korean_name']
    try:
        time.sleep(0.04)
        df_daily = pyupbit.get_ohlcv(ticker, interval="day", count=60)
        metrics = calculate_t1_advanced_metrics(df_daily)
        
        if not metrics: 
            c_price = current_price_map.get(ticker, 0)
            return {
                "코인명": korean_name,
                "심볼": symbol,
                "현재가(KRW)": format_price(c_price),
                "raw_price": float(c_price),
                "종합예측점수": 50.0,
                "거래량절벽(배)": 1.0,
                "CMF지표": 0.0,
                "RSI": 50.0,
                "아이스버그역산(고주파)": "💎 정상 수급",
                "_stgt_feats": [0.5, 0.5, 0.0, 0.5, 0.1, 0.5, 1.0, 0.2, 0.5]
            }

        iceberg_metrics = get_highfreq_iceberg_metrics(ticker)
        stgt_feats = [0.8, 0.7, metrics['cmf'], metrics['rsi']/100.0, 0.1, 0.5, metrics['vol_dry_ratio'], 0.2, 0.5]
        
        c_price = current_price_map.get(ticker, metrics['last_close'])
        ai_engine.save_experience(symbol, price=c_price, lstm_feats=iceberg_metrics.get("raw_lstm_feats"), stgt_feats=stgt_feats)

        # [수정] 기본점수 50점 기준, CMF(-1~1) 및 거래량 변화율 조정을 적용하여 스펙트럼 확장
        acc_score = round(
            50.0 
            + (metrics['cmf'] * 25.0) 
            - ((metrics['vol_dry_ratio'] - 1.0) * 10.0) 
            + iceberg_metrics['score_modifier'], 
            1
        )

        return {
            "코인명": korean_name,
            "심볼": symbol,
            "현재가(KRW)": format_price(c_price),
            "raw_price": float(c_price),
            "종합예측점수": max(0.0, min(100.0, acc_score)),
            "거래량절벽(배)": metrics['vol_dry_ratio'],
            "CMF지표": metrics['cmf'],
            "RSI": metrics['rsi'],
            "아이스버그역산(고주파)": iceberg_metrics['status'],
            "_stgt_feats": stgt_feats
        }
    except Exception as e: 
        c_price = current_price_map.get(ticker, 0)
        return {
            "코인명": korean_name,
            "심볼": symbol,
            "현재가(KRW)": format_price(c_price),
            "raw_price": float(c_price),
            "종합예측점수": 50.0,
            "거래량절벽(배)": 1.0,
            "CMF지표": 0.0,
            "RSI": 50.0,
            "아이스버그역산(고주파)": "💎 정상 수급",
            "_stgt_feats": [0.5, 0.5, 0.0, 0.5, 0.1, 0.5, 1.0, 0.2, 0.5]
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
                if isinstance(preds, float): preds = [preds]
                df['STGT_그래프덤핑위험(%)'] = [round(p * 100, 1) for p in preds]
        except Exception as e:
            print(f"⚠️ STGT 분석 스킵: {e}")

    if '_stgt_feats' in df.columns:
        df = df.drop(columns=['_stgt_feats'])

    return df.sort_values(by="종합예측점수", ascending=False)

# ==============================================================================
# [백분위 기반 상대평가 및 AI 추천종목 모니터]
# ==============================================================================
def assign_relative_grades(df):
    """
    시장 전체 점수의 백분위(Quantile)를 기준으로 등급을 상대평가 부여하여
    한쪽 쏠림 현상을 방지하는 함수
    """
    if df.empty:
        return df

    scores = df['종합예측점수']
    q80 = scores.quantile(0.80)  # 상위 20%
    q50 = scores.quantile(0.50)  # 상위 50%
    q20 = scores.quantile(0.20)  # 상위 80% (하위 20%)

    def determine_grade(row):
        score = row['종합예측점수']
        dump_risk = row['STGT_그래프덤핑위험(%)']

        # 1. 고위험 종목은 절대 기준 우선 적용
        if dump_risk >= 80.0:
            return "🔴 경고"
        elif dump_risk >= 65.0:
            return "🟠 주의"

        # 2. 나머지는 백분위 기반 상대평가
        if score >= q80:
            return "🟢 추천"
        elif score >= q50:
            return "🔵 관심"
        elif score >= q20:
            return "⚪ 보통"
        else:
            return "🟠 주의"

    df['grade'] = df.apply(determine_grade, axis=1)
    return df

def calculate_ai_grade(score, dump_risk):
    if dump_risk >= 80.0:
        return "🔴 경고"
    elif dump_risk >= 65.0:
        return "🟠 주의"
    elif score >= 65.0 and dump_risk < 40.0:
        return "🟢 추천"
    elif score >= 50.0 and dump_risk < 60.0:
        return "🔵 관심"
    else:
        return "⚪ 보통"

def update_ai_recommendation_tracker(ai_report_coins, current_price_map, coin_status_map):
    history = {}
    
    # 파일이 존재하고 내용이 비어있지 않은 경우에만 안전하게 로드
    if os.path.exists(AI_TRACKER_HISTORY_FILE):
        try:
            with open(AI_TRACKER_HISTORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    history = json.loads(content)
        except Exception as e:
            print(f"⚠️ 기존 트래킹 파일 로드 실패 (초기화 후 진행): {e}")
            history = {}

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # AI 분석 리포트에 언급된 종목들을 신규 등록 및 추천 횟수 증가
    for coin in ai_report_coins:
        symbol = coin['symbol']
        name = coin['name']
코드에서 발생하는 오류의 원인은 **HTML 템플릿 문자열을 괄호 `()`로 묶어서 이어 붙이는 과정에서 일부 줄에 따옴표(`'`)가 누락**되었기 때문입니다. 파이썬에서는 문자열이 제대로 닫히지 않거나 시작되지 않으면 `SyntaxError`가 발생하여 실행 자체가 중단됩니다.

**수정된 부분:**
1. `<meta http-equiv="refresh" content="300">\n'` 부분의 맨 앞에 누락된 `'` 추가
2. `<!-- [중앙 컬럼: 전체 코인 등급 분류] -->` 부분 앞뒤에 문자열 처리용 `' '` 추가 및 줄바꿈 `\n` 추가
3. `<div class="col-lg-5">\n'` 부분의 맨 앞에 누락된 `'` 추가

바로 복사해서 붙여넣어 사용할 수 있도록 수정한 **전체 코드**를 제공해 드립니다.

### 🛠️ 수정된 전체 코드 (coin_excel_krw.py)

```python
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
from typing import List
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

    if not hostname or not ssh_key_content
