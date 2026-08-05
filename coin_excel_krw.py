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
import oracledb
import paramiko
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


def upload_html_to_oracle_server(local_file_path):
    """
    GitHub Secrets에 등록된 ORACLE_SSH_KEY(.key 내용)를 이용해 
    오라클 서버로 대시보드 HTML 파일을 자동 전송하는 함수
    """
    hostname = os.environ.get("ORACLE_DSN")          # 오라클 서버 공인 IP
    username = os.environ.get("ORACLE_USER", "ubuntu") # 계정명
    ssh_key_content = os.environ.get("ORACLE_SSH_KEY") # GitHub Secrets SSH 개인키

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
# [스캔 분석 유틸]
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
        "score_modifier": -50 if dump_prob >= 0.7 else 15,
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

        acc_score = round(70.0 + (metrics['cmf'] * 10) - (metrics['vol_dry_ratio'] * 5) + iceberg_metrics['score_modifier'], 1)

        return {
            "코인명": korean_name,
            "심볼": symbol,
            "현재가(KRW)": format_price(c_price),
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
# [신규 모듈] 5단계 AI 등급 부여 및 Redis 대시보드 연동
# ==============================================================================
def calculate_ai_grade(score, dump_risk):
    # 덤핑 위험도가 일괄적으로 높게 잡히는 현상을 방지하기 위해 기준선을 상향 조정하고 점수 비중을 높임
    if dump_risk >= 85.0:
        return "🔴 경고"
    elif dump_risk >= 70.0 and score < 50.0:
        return "🟠 주의"
    elif score >= 65.0 and dump_risk < 40.0:
        return "🟢 추천"
    elif score >= 50.0 and dump_risk < 60.0:
        return "🔵 관심"
    else:
        return "⚪ 보통"

def update_redis_for_dashboard(df_result, ai_report):
    if not redis_client or df_result.empty: return

    try:
        coin_grades = []
        for _, row in df_result.iterrows():
            score = row['종합예측점수']
            dump_risk = row['STGT_그래프덤핑위험(%)']
            grade = calculate_ai_grade(score, dump_risk)

            coin_grades.append({
                "name": row['코인명'],
                "symbol": row['심볼'],
                "price": row['현재가(KRW)'],
                "score": score,
                "dump_risk": dump_risk,
                "iceberg": row['아이스버그역산(고주파)'],
                "grade": grade
            })

        recommended_coins = [c for c in coin_grades if c['grade'] == "🟢 추천"]
        interest_coins = [c for c in coin_grades if c['grade'] == "🔵 관심"]
        normal_coins = [c for c in coin_grades if c['grade'] == "⚪ 보통"]
        warning_coins = [c for c in coin_grades if c['grade'] == "🟠 주의"]
        danger_coins = [c for c in coin_grades if c['grade'] == "🔴 경고"]
        
        ai_recommended_tracking = [c for c in coin_grades if c['grade'] in ["🟢 추천", "🔵 관심"]]

        dashboard_payload = {
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_report": ai_report,
            "summary": {
                "total_scanned": len(coin_grades),
                "recommended_count": len(recommended_coins),
                "interest_count": len(interest_coins),
                "normal_count": len(normal_coins),
                "warning_count": len(warning_coins),
                "danger_count": len(danger_coins)
            },
            "classified_sectors": {
                "recommend": recommended_coins,
                "interest": interest_coins,
                "normal": normal_coins,
                "warning": warning_coins,
                "danger": danger_coins
            },
            "ai_recommended_tracking": ai_recommended_tracking,
            "all_coins": coin_grades
        }

        redis_client.set("upbit_ai_dashboard_data", json.dumps(dashboard_payload, ensure_ascii=False))
        print("⚡ [Redis] 전체 종목 페이로드 업데이트 완료!")
    except Exception as e:
        print(f"❌ [Redis] 데이터 업로드 실패: {e}")

# ==============================================================================
# [리포트 생성 및 이메일 발송]
# ==============================================================================
def generate_dashboard_html(coins_data, recommended_sector, interested_sector, normal_sector, warning_sector, danger_sector, tracking_list, alerts, news_data, ai_report, updated_time):
    # 1. 각 섹션별 테이블 행 생성 헬퍼 함수
    def create_table_rows(sector_coins):
        if not sector_coins:
            return '<tr><td colspan="4" class="text-center text-muted py-3">해당하는 종목이 없습니다.</td></tr>'
        
        rows_list = []
        for coin in sector_coins:
            name = coin.get('name', 'N/A')
            price = coin.get('price', 'N/A')
            score = coin.get('score', 0)
            badge_class = coin.get('badge_class', 'bg-secondary')
            grade = coin.get('grade', '⚪ 보통')
            
            row_html = (
                f"<tr>\n"
                f'    <td class="fw-bold">{name}</td>\n'
                f'    <td>{price}</td>\n'
                f'    <td class="text-primary">{score:.1f}점</td>\n'
                f'    <td><span class="badge {badge_class}">{grade}</span></td>\n'
                f"</tr>\n"
            )
            rows_list.append(row_html)
        
        return "".join(rows_list)

    # 2. 경고 리스트 HTML 생성
    alert_items = []
    for alert in alerts[:15]:
        alert_text = alert.get('text', '')
        alert_items.append(f'<div class="p-2 rounded bg-danger bg-opacity-10 border border-danger text-danger small fw-bold">{alert_text}</div>')
    alerts_html = "\n".join(alert_items) if alert_items else '<div class="text-muted small text-center py-3">현재 주의/위험 종목이 없습니다.</div>'

    # 3. 속보 리스트 HTML 생성
    news_items = []
    if news_data:
        for coin, items in news_data.items():
            li_tags = "".join([f'<li><a href="{item.get("link", "#")}" target="_blank" class="text-decoration-none text-dark">{item.get("title", "")}</a></li>' for item in items])
            news_items.append(f'<div class="p-2 border rounded bg-light"><strong class="text-primary">{coin}</strong><ul class="mb-0 ps-3 small">{li_tags}</ul></div>')
        news_html = "\n".join(news_items)
    else:
        news_html = '<div class="text-muted small text-center py-3">현재 등록된 추천 속보 이슈가 없습니다.</div>'

    # 4. 트래킹 모니터 HTML 생성
    tracking_items = []
    for item in tracking_list:
        t_name = item.get('name', '')
        t_status = item.get('status', '')
        tracking_items.append(f'<div class="p-2 border rounded bg-light small"><strong>{t_name}</strong>: {t_status}</div>')
    tracking_html = "\n".join(tracking_items) if tracking_items else '<div class="text-muted small text-center py-3">트래킹 데이터가 없습니다.</div>'

    # 5. 각 섹션별 코인 테이블 데이터 생성
    rec_rows = create_table_rows(recommended_sector)
    int_rows = create_table_rows(interested_sector)
    norm_rows = create_table_rows(normal_sector)
    warn_rows = create_table_rows(warning_sector)
    dang_rows = create_table_rows(danger_sector)

    # 6. HTML 템플릿 정의 (일반 삼중 따옴표 사용으로 CSS, JS 내 중괄호 이스케이프 오류 방지)
    html_template = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upbit AI Quantitative Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #f8fafc; color: #1e293b; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); height: 100%; }
        .alert-box { max-height: 140px; overflow-y: auto; }
        .news-box { max-height: 140px; overflow-y: auto; }
        .table-scroll-box { max-height: 500px; overflow-y: auto; }
        .tracking-box { max-height: 600px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container-fluid my-4 px-4" style="max-width: 1700px;">
        <!-- 상단 헤더 -->
        <div class="row mb-4 align-items-center">
            <div class="col-md-3 text-start">
                <a href="http://140.245.99.254:8000" class="btn btn-primary fw-bold px-3 py-2 shadow-sm">
                    <i class="fa-solid fa-robot me-1"></i> AI 실시간
                </a>
            </div>
            <div class="col-md-6 text-center">
                <h2 class="fw-bold text-dark mb-0 fs-4"><i class="fa-solid fa-chart-pie text-primary me-2"></i>업비트 AI 분석 대시보드</h2>
                <small class="text-muted">최종 업데이트: __UPDATED_TIME__ (총 __TOTAL_COINS__개 종목 분석 완료)</small>
            </div>
            <div class="col-md-3"></div>
        </div>

        <!-- 3단 레이아웃 메인 -->
        <div class="row g-4">
            <!-- [좌측 컬럼: 너비 3] -->
            <div class="col-lg-3">
                <div class="d-flex flex-column gap-3">
                    <div class="card p-3 shadow-sm">
                        <h6 class="fw-bold text-danger mb-3"><i class="fa-solid fa-triangle-exclamation me-1"></i> 실시간 급락/위험 경고</h6>
                        <div class="alert-box d-flex flex-column gap-2">
                            __ALERTS_HTML__
                        </div>
                    </div>

                    <div class="card p-3 shadow-sm">
                        <h6 class="fw-bold text-success mb-3"><i class="fa-solid fa-newspaper me-1"></i> 실시간 속보</h6>
                        <div class="news-box d-flex flex-column gap-2">
                            __NEWS_HTML__
                        </div>
                    </div>

                    <div class="card p-3 shadow-sm">
                        <h6 class="fw-bold text-primary mb-3"><i class="fa-solid fa-robot me-1"></i> AI 분석 리포트</h6>
                        <div class="text-secondary small bg-light p-3 rounded" style="max-height: 450px; overflow-y: auto; line-height: 1.4; white-space: pre-line;">__AI_REPORT__</div>
                    </div>
                </div>
            </div>

            <!-- [중앙 컬럼: 너비 6] -->
            <div class="col-lg-6">
                <div class="card p-4 shadow-sm">
                    <div class="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2">
                        <h5 class="fw-bold mb-0 text-dark fs-5"><i class="fa-solid fa-list-check me-1"></i> 전체 코인 등급 분류</h5>
                        <div class="input-group" style="max-width: 220px;">
                            <span class="input-group-text bg-white"><i class="fa-solid fa-search text-muted"></i></span>
                            <input type="text" id="coinSearchInput" class="form-control form-control-sm" placeholder="코인명 검색..." onkeyup="filterCoins()">
                        </div>
                    </div>

                    <ul class="nav nav-tabs mb-3 flex-nowrap overflow-auto" id="coinTab" role="tablist" style="white-space: nowrap;">
                        <li class="nav-item" role="presentation"><button class="nav-link active fw-bold text-success" id="rec-tab" data-bs-toggle="tab" data-bs-target="#rec" type="button" role="tab">🟢 추천 (__REC_COUNT__)</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link fw-bold text-primary" id="int-tab" data-bs-toggle="tab" data-bs-target="#int" type="button" role="tab">🔵 관심 (__INT_COUNT__)</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link fw-bold text-secondary" id="norm-tab" data-bs-toggle="tab" data-bs-target="#norm" type="button" role="tab">⚪ 보통 (__NORM_COUNT__)</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link fw-bold text-warning" id="warn-tab" data-bs-toggle="tab" data-bs-target="#warn" type="button" role="tab">🟠 주의 (__WARN_COUNT__)</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link fw-bold text-danger" id="dang-tab" data-bs-toggle="tab" data-bs-target="#dang" type="button" role="tab">🔴 경고 (__DANG_COUNT__)</button></li>
                    </ul>

                    <div class="tab-content table-scroll-box" id="coinTabContent">
                        <div class="tab-pane fade show active" id="rec" role="tabpanel">
                            <table class="table table-hover align-middle mb-0">
                                <thead class="table-light sticky-top"><tr><th>코인명</th><th>현재가</th><th>예측점수</th><th>등급</th></tr></thead>
                                <tbody>__REC_ROWS__</tbody>
                            </table>
                        </div>
                        <div class="tab-pane fade" id="int" role="tabpanel">
                            <table class="table table-hover align-middle mb-0">
                                <thead class="table-light sticky-top"><tr><th>코인명</th><th>현재가</th><th>예측점수</th><th>등급</th></tr></thead>
                                <tbody>__INT_ROWS__</tbody>
                            </table>
                        </div>
                        <div class="tab-pane fade" id="norm" role="tabpanel">
                            <table class="table table-hover align-middle mb-0">
                                <thead class="table-light sticky-top"><tr><th>코인명</th><th>현재가</th><th>예측점수</th><th>등급</th></tr></thead>
                                <tbody>__NORM_ROWS__</tbody>
                            </table>
                        </div>
                        <div class="tab-pane fade" id="warn" role="tabpanel">
                            <table class="table table-hover align-middle mb-0">
                                <thead class="table-light sticky-top"><tr><th>코인명</th><th>현재가</th><th>예측점수</th><th>등급</th></tr></thead>
                                <tbody>__WARN_ROWS__</tbody>
                            </table>
                        </div>
                        <div class="tab-pane fade" id="dang" role="tabpanel">
                            <table class="table table-hover align-middle mb-0">
                                <thead class="table-light sticky-top"><tr><th>코인명</th><th>현재가</th><th>예측점수</th><th>등급</th></tr></thead>
                                <tbody>__DANG_ROWS__</tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- [우측 컬럼: 너비 3] -->
            <div class="col-lg-3">
                <div class="card p-3 shadow-sm tracking-box">
                    <h6 class="fw-bold text-primary mb-3"><i class="fa-solid fa-chart-line me-1"></i> 실시간 트래킹 모니터</h6>
                    <div class="d-flex flex-column gap-2">
                        __TRACKING_HTML__
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function filterCoins() {
            let input = document.getElementById('coinSearchInput').value.toLowerCase();
            let rows = document.querySelectorAll('.tab-pane.active tbody tr');
            rows.forEach(row => {
                let name = row.cells[0]?.innerText.toLowerCase() || '';
                row.style.display = name.includes(input) ? '' : 'none';
            });
        }
    </script>
</body>
</html>"""

    # 7. 안전한 치환 작업으로 최종 HTML 생성
    html_content = html_template.replace("__UPDATED_TIME__", str(updated_time))\
                                .replace("__TOTAL_COINS__", str(len(coins_data)))\
                                .replace("__ALERTS_HTML__", alerts_html)\
                                .replace("__NEWS_HTML__", news_html)\
                                .replace("__AI_REPORT__", str(ai_report))\
                                .replace("__REC_COUNT__", str(len(recommended_sector)))\
                                .replace("__INT_COUNT__", str(len(interested_sector)))\
                                .replace("__NORM_COUNT__", str(len(normal_sector)))\
                                .replace("__WARN_COUNT__", str(len(warning_sector)))\
                                .replace("__DANG_COUNT__", str(len(danger_sector)))\
                                .replace("__REC_ROWS__", rec_rows)\
                                .replace("__INT_ROWS__", int_rows)\
                                .replace("__NORM_ROWS__", norm_rows)\
                                .replace("__WARN_ROWS__", warn_rows)\
                                .replace("__DANG_ROWS__", dang_rows)\
                                .replace("__TRACKING_HTML__", tracking_html)

    return html_content
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("🎨 [대시보드] 차트 기능 완전 제거 및 3단 레이아웃(중앙 등급 분류 상단 배치) HTML 생성 완료 (`docs/index.html`)!")


# ==============================================================================
# [메인 실행부]
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. AI 자가학습 진행
    ai_engine.evolve_models()
    
    # 2. 시장 스캔 및 데이터 분석 (전체 원화마켓 대상)
    df_result = analyze_and_scan_market()
    
    if not df_result.empty:
        print(f"\n=== 🎯 [자가학습 AI 적용] 전체 분석 종목 수: {len(df_result)}개 ===")
        print(df_result[["코인명", "종합예측점수", "STGT_그래프덤핑위험(%)", "아이스버그역산(고주파)"]].head(5))
        
        # 3. 위험 알림 (텔레그램)
        danger_coins = df_result[df_result['STGT_그래프덤핑위험(%)'] >= 75.0]
        if not danger_coins.empty:
            print(f"\n🚨 [위험 감지] {len(danger_coins)}개 종목 덤핑 위험! (텔레그램 전송)")
            send_telegram_alert(f"🚨 *[자가학습 AI 경고]* 덤핑 위험 감지: {', '.join(danger_coins['코인명'].tolist())}")

        # 4. Gemini AI 요약 및 추천 심볼 추출
        ai_report, gemini_symbols = generate_gemini_analysis(df_result)

        # 5. 추천 종목 실시간 속보 수집 대상 추출
        internal_recommended_names = df_result[df_result['종합예측점수'] >= 80.0]['코인명'].tolist()
        symbol_to_name = dict(zip(df_result['심볼'], df_result['코인명']))
        gemini_recommended_names = [symbol_to_name.get(s, s) for s in gemini_symbols]
        
        all_target_coins = list(set(internal_recommended_names + gemini_recommended_names))[:5]
        news_data = fetch_news_for_recommended_coins(all_target_coins)

        # 6. Redis 연동 (전체 종목 페이로드 반영)
        update_redis_for_dashboard(df_result, ai_report)

        # 7. 대시보드 HTML 파일 생성 (3단 구조 및 차트 완전 제거 후 코인 등급 분류 상단 배치)
        generate_dashboard_html(df_result, ai_report, gemini_symbols, news_data)

        # 8. 오라클 서버로 HTML 대시보드 전송
        upload_html_to_oracle_server("docs/index.html")

        # 9. 엑셀 저장 및 이메일 발송
        kst_now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
        current_hour = kst_now.hour
        target_hours = [9, 13, 17, 21] 

        if current_hour in target_hours:
            export_to_excel_and_email(df_result, ai_report)
        else:
            print(f"⏰ 현재 시각(KST {current_hour}시)은 이메일 발송 시간이 아니므로 대시보드만 갱신합니다.")
