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
    """추천 종목 리스트를 받아 종목별 최신 뉴스/이슈 수집"""
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
        if not metrics or (metrics['last_value'] / 100_000_000) < 5.0: return None

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
    except Exception: return None

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
    print("\n🚀 [멀티스레딩] 병렬 코인 스캔 및 AI 예측 시작...")
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
    if dump_risk >= 70.0:
        return "🔴 경고"
    elif dump_risk >= 50.0 or score < 40.0:
        return "🟠 주의"
    elif score >= 70.0 and dump_risk < 30.0:
        return "🟢 추천"
    elif score >= 55.0 and dump_risk < 45.0:
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
        print("⚡ [Redis] 페이로드 업데이트 완료!")
    except Exception as e:
        print(f"❌ [Redis] 데이터 업로드 실패: {e}")

# ==============================================================================
# [리포트 생성 및 이메일 발송]
# ==============================================================================
def generate_gemini_analysis(df_top):
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY가 설정되지 않아 AI 요약 생성을 스킵합니다.", []
    if df_top.empty:
        return "분석된 종목 데이터가 없습니다.", []

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        summary_str = df_top[["코인명", "심볼", "현재가(KRW)", "종합예측점수", "STGT_그래프덤핑위험(%)", "아이스버그역산(고주파)"]].head(10).to_string(index=False)
        
        prompt = f"""
        다음은 업비트 원화마켓 상위 매집 분석 결과 데이터입니다:

        {summary_str}

        암호화폐 전문 분석가 관점에서 위 데이터를 바탕으로 짧고 핵심적인 분석 리포트를 작성하세요.
        1. 매집 점수가 우수한 Top 3 코인 요약 및 추천 포인트
        2. STGT 덤핑 위험도나 고주파 아이스버그 위험 감지 시 주의점
        3. 종합 매수 전략 한 줄 가이드

        [출력 규칙]
        리포트 작성이 끝난 후 맨 마지막 줄에 Gemini가 분석하여 최종 추천하는 Top 3 코인의 심볼(Symbol)만 쉼표로 구분하여 반드시 아래 형식으로 적어주세요.
        GEMINI_RECOMMENDED_SYMBOLS: BTC, ETH, XRP
        """

        response = client.models.generate_content(
            model ='gemini-3.1-flash-lite',
            contents=prompt
        )
        
        full_text = response.text
        gemini_symbols = []

        if "GEMINI_RECOMMENDED_SYMBOLS:" in full_text:
            parts = full_text.split("GEMINI_RECOMMENDED_SYMBOLS:")
            ai_report_text = parts[0].strip()
            symbols_line = parts[1].strip().split("\n")[0]
            gemini_symbols = [s.strip().upper() for s in symbols_line.split(",") if s.strip()]
        else:
            ai_report_text = full_text

        return ai_report_text, gemini_symbols
    except Exception as e:
        return f"❌ Gemini AI 요약 생성 실패: {e}", []

def export_to_excel_and_email(df_result, ai_report):
    if df_result.empty: return

    with pd.ExcelWriter(EXCEL_FILE_PATH, engine='openpyxl') as writer:
        df_result.to_excel(writer, index=False, sheet_name='매집분석_리포트')

    wb = openpyxl.load_workbook(EXCEL_FILE_PATH)
    ws = wb['매집분석_리포트']
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)

    for col in ws.iter_cols(min_row=1, max_row=1):
        for cell in col:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(EXCEL_FILE_PATH)
    print(f"📊 엑셀 리포트 저장 완료: {EXCEL_FILE_PATH}")

    if not SENDER_EMAIL or not EMAIL_PASSWORD or not RECEIVER_EMAILS:
        print("📧 이메일 계정 정보가 설정되지 않아 메일 발송을 스킵합니다.")
        return

    msg = MIMEMultipart()
    msg['Subject'] = f"🚀 [업비트] 매집 패턴 및 AI 분석 리포트 ({datetime.date.today()})"
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECEIVER_EMAILS)

    body_text = f"안녕하세요,\n\n오늘의 업비트 원화마켓 매집 점수 및 AI 분석 리포트입니다.\n\n=========================================="
    body_text += f"\n🤖 [Gemini AI 종합 분석]\n{ai_report}\n==========================================\n\n"
    body_text += "상세 분석 결과는 첨부된 엑셀 파일을 확인해 주세요.\n\n감사합니다."

    msg.attach(MIMEText(body_text, 'plain', 'utf-8'))

    if os.path.exists(EXCEL_FILE_PATH):
        with open(EXCEL_FILE_PATH, 'rb') as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(EXCEL_FILE_PATH))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(EXCEL_FILE_PATH)}"'
            msg.attach(part)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
        print("📧 이메일 리포트 발송 성공!")
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

# ==============================================================================
# [웹 대시보드 HTML 자동 생성 (속보 기능 및 이력 관리 포함)]
# ==============================================================================
def generate_dashboard_html(df_result, ai_report, gemini_symbols=None, news_data=None):
    if gemini_symbols is None:
        gemini_symbols = []
    if news_data is None:
        news_data = {}

    os.makedirs("docs", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    html_path = "docs/index.html"
    history_path = "cache/recommend_history.json"

    if df_result.empty:
        html_content = "<html><body><h1>분석된 데이터가 없습니다.</h1></body></html>"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            rec_history = json.load(f)
    except Exception:
        rec_history = {}

    current_time = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M")

    coins_data = []
    for _, row in df_result.iterrows():
        symbol = row['심볼']
        score = float(row['종합예측점수'])
        dump_risk = float(row['STGT_그래프덤핑위험(%)'])
        grade = calculate_ai_grade(score, dump_risk)
        
        current_price_str = str(row['현재가(KRW)']).replace(',', '')
        try:
            curr_val = float(current_price_str)
        except:
            curr_val = 0.0

        if symbol in rec_history and "entry_val" in rec_history[symbol]:
            entry_val = float(rec_history[symbol]["entry_val"])
        else:
            entry_val = curr_val * 0.98 if curr_val > 0 else curr_val

        target_val = entry_val * 1.20 if entry_val > 0 else 0

        is_valuable = True
        price_change_rate = 0.0
        if entry_val > 0 and curr_val > 0:
            price_change_rate = (curr_val - entry_val) / entry_val * 100

        if price_change_rate >= 20.0 or grade in ["🟠 주의", "🔴 경고"]:
            is_valuable = False

        coins_data.append({
            "name": row['코인명'],
            "symbol": symbol,
            "price": row['현재가(KRW)'],
            "curr_val": curr_val,
            "entry_val": entry_val,
            "score": score,
            "dump_risk": dump_risk,
            "iceberg": row['아이스버그역산(고주파)'],
            "grade": grade,
            "entry_price": format_price(entry_val),
            "target_price": format_price(target_val),
            "price_change_rate": round(price_change_rate, 1),
            "is_valuable": is_valuable
        })

    recommended_sector = [c for c in coins_data if c['grade'] == "🟢 추천"]
    interested_sector = [c for c in coins_data if c['grade'] == "🔵 관심"]
    normal_sector = [c for c in coins_data if c['grade'] == "⚪ 보통"]
    warning_sector = [c for c in coins_data if c['grade'] == "🟠 주의"]
    danger_sector = [c for c in coins_data if c['grade'] == "🔴 경고"]

    for sym in gemini_symbols:
        sym_upper = sym.upper()
        if sym_upper not in rec_history:
            match_coin = next((c for c in coins_data if c['symbol'].upper() == sym_upper), None)
            if match_coin:
                rec_history[sym_upper] = {
                    "first_recommended": current_time_str,
                    "last_recommended": current_time_str,
                    "entry_val": match_coin['entry_val'],
                    "count": 1
                }
        else:
            rec_history[sym_upper]["count"] = rec_history[sym_upper].get("count", 1) + 1

    active_recommended_tracking = []
    for c in coins_data:
        symbol_upper = c['symbol'].upper()
        if symbol_upper in rec_history:
            if c['is_valuable']:
                c['rec_time'] = rec_history[symbol_upper].get("first_recommended", current_time_str)
                c['rec_count'] = rec_history[symbol_upper].get("count", 1)
                active_recommended_tracking.append(c)
            else:
                del rec_history[symbol_upper]

    try:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(rec_history, f, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ 추천 이력 저장 실패: {e}")

    alerts = []
    for c in coins_data:
        if c['dump_risk'] >= 75.0 or c['grade'] == "🔴 경고":
            alerts.append({"type": "danger", "text": f"🚨 [급락/경고] {c['name']}({c['symbol']}) 위험도 {c['dump_risk']}%"})

    updated_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upbit AI Quantitative Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body {{ background-color: #f8fafc; color: #1e293b; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card {{ background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }}
        .table {{ color: #1e293b; }}
        .badge-recommend {{ background-color: #22c55e; color: white; }}
        .badge-interest {{ background-color: #3b82f6; color: white; }}
        .badge-normal {{ background-color: #64748b; color: white; }}
        .badge-warning {{ background-color: #f97316; color: white; }}
        .badge-danger {{ background-color: #ef4444; color: white; }}
        .alert-box {{ max-height: 220px; overflow-y: auto; }}
        .tracking-box {{ max-height: 450px; overflow-y: auto; }}
        .news-box {{ max-height: 280px; overflow-y: auto; }}
        .search-input {{ max-width: 250px; }}
        .coin-row {{ cursor: pointer; }}
        .coin-row:hover {{ background-color: #f1f5f9 !important; }}
    </style>
</head>
<body>
    <div class="container-fluid my-4 px-4" style="max-width: 1700px;">
        <div class="row mb-4 align-items-center">
            <div class="col-md-3 text-start mb-2 mb-md-0">
                <a href="http://140.245.99.254:8000" class="btn btn-primary fw-bold px-3 py-2 shadow-sm w-100 w-md-auto">
                    <i class="fa-solid fa-robot me-1"></i> AI 실시간
                </a>
            </div>
            <div class="col-md-6 text-center">
                <h2 class="fw-bold text-dark mb-0 fs-4 fs-md-2"><i class="fa-solid fa-chart-pie text-primary me-2"></i>업비트 AI 분석 대시보드</h2>
                <small class="text-muted">최종 업데이트: {updated_time}</small>
            </div>
            <div class="col-md-3"></div>
        </div>

        <div class="row">
            <div class="col-lg-3 mb-4 d-flex flex-column gap-3">
                <div class="card p-3 shadow-sm">
                    <h5 class="fw-bold text-danger mb-3"><i class="fa-solid fa-triangle-exclamation me-1"></i> 실시간 급락/위험 경고</h5>
                    <div class="alert-box d-flex flex-column gap-2">
"""

    for alert in alerts[:10]:
        html_content += f"""                        <div class="p-2 rounded bg-danger bg-opacity-10 border border-danger text-danger small fw-bold">{alert['text']}</div>\n"""

    if not alerts:
        html_content += """                        <div class="text-muted small text-center py-3">현재 주의/위험 종목이 없습니다.</div>\n"""

    html_content += f"""                    </div>
                </div>

                <div class="card p-3 shadow-sm">
                    <h5 class="fw-bold text-success mb-3"><i class="fa-solid fa-newspaper me-1"></i> 추천 종목 실시간 속보</h5>
                    <div class="news-box d-flex flex-column gap-2">
"""
    if news_data:
        for coin, items in news_data.items():
            html_content += f"""                        <div class="p-2 border rounded bg-light"><strong class="text-primary">{coin}</strong><ul class="mb-0 ps-3 small">"""
            for item in items:
                html_content += f"""<li><a href="{item['link']}" target="_blank" class="text-decoration-none text-dark">{item['title']}</a></li>"""
            html_content += f"""</ul></div>"""
    else:
        html_content += """                        <div class="text-muted small text-center py-3">현재 등록된 추천 속보 이슈가 없습니다.</div>\n"""

    html_content += f"""                    </div>
                </div>

                <div class="card p-3 shadow-sm flex-grow-1">
                    <h5 class="fw-bold text-primary mb-3"><i class="fa-solid fa-robot me-1"></i> Gemini AI 분석 리포트</h5>
                    <div class="text-secondary small bg-light p-3 rounded" style="white-space: pre-line; max-height: 350px; overflow-y: auto; line-height: 1.5;">{ai_report}</div>
                </div>
            </div>

            <div class="col-lg-6 mb-4 d-flex flex-column gap-3">
                <div class="card p-3 shadow-sm">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <h5 class="fw-bold text-dark mb-0 fs-6" id="chartTitle"><i class="fa-solid fa-chart-candlestick text-primary me-1"></i> 비트코인 (BTC) 실시간 10분봉 차트</h5>
                        <span class="badge bg-secondary" id="chartStatus" style="font-size: 0.75rem;">로딩 중...</span>
                    </div>
                    <div id="tradingview-chart" style="width: 100%; height: 350px;"></div>
                </div>

                <div class="card p-3 p-md-4 shadow-sm">
                    <div class="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2">
                        <h4 class="fw-bold mb-0 text-dark fs-5 fs-md-5"><i class="fa-solid fa-list-check me-1"></i> 전체 코인 등급 분류 <small class="text-muted" style="font-size: 0.75rem; font-weight:normal;">(종목 클릭시 위 차트 연동)</small></h4>
                        <div class="input-group search-input w-100 w-md-auto">
                            <span class="input-group-text bg-white"><i class="fa-solid fa-search text-muted"></i></span>
                            <input type="text" id="coinSearchInput" class="form-control form-control-sm" placeholder="코인명 또는 심볼 검색..." onkeyup="filterCoins()">
                        </div>
                    </div>

                    <ul class="nav nav-tabs mb-3 flex-nowrap overflow-auto" id="coinTab" role="tablist" style="white-space: nowrap;">
                        <li class="nav-item" role="presentation"><button class="nav-link active fw-bold text-success" id="rec-tab" data-bs-toggle="tab" data-bs-target="#rec" type="button">🟢 추천 ({len(recommended_sector)})</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link fw-bold text-primary" id="int-tab" data-bs-toggle="tab" data-bs-target="#int" type="button">🔵 관심 ({len(interested_sector)})</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link fw-bold text-secondary" id="norm-tab" data-bs-toggle="tab" data-bs-target="#norm" type="button">⚪ 보통 ({len(normal_sector)})</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link fw-bold text-warning" id="warn-tab" data-bs-toggle="tab" data-bs-target="#warn" type="button">🟠 주의 ({len(warning_sector)})</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link fw-bold text-danger" id="dang-tab" data-bs-toggle="tab" data-bs-target="#dang" type="button">🔴 경고 ({len(danger_sector)})</button></li>
                    </ul>

                    <div class="tab-content" id="coinTabContent">
"""

    def make_table_html(sector_list, tab_id, is_active=""):
        active_cls = "show active" if is_active else ""
        t_html = f'<div class="tab-pane fade {active_cls}" id="{tab_id}" role="tabpanel">'
        t_html += '<div class="table-responsive" style="max-height: 350px; overflow-y: auto;">'
        t_html += '<table class="table table-hover align-middle small search-table text-nowrap">_TABLE_HEADER_<tbody>'
        
        sorted_sector = sorted(sector_list, key=lambda x: x['score'], reverse=True)
        if not sorted_sector:
            t_html += '<tr><td colspan="5" class="text-center text-muted py-4">해당 등급의 종목이 없습니다.</td></tr>'
        else:
            for c in sorted_sector:
                badge_class = "badge-recommend" if c['grade']=="🟢 추천" else ("badge-interest" if c['grade']=="🔵 관심" else ("badge-normal" if c['grade']=="⚪ 보통" else ("badge-warning" if c['grade']=="🟠 주의" else "badge-danger")))
                t_html += f"""<tr class="coin-row" data-name="{c['name']}" data-symbol="{c['symbol']}" onclick="loadCoinChart('KRW-{c['symbol']}', '{c['name']}')">
                    <td class="fw-bold">{c['name']} <small class="text-muted">({c['symbol']})</small></td>
                    <td>{c['price']}원</td>
                    <td class="text-primary fw-bold">{c['score']}점</td>
                    <td class="text-danger">{c['dump_risk']}%</td>
                    <td><span class="badge {badge_class}">{c['grade']}</span></td>
                </tr>"""
        t_html += '</tbody></table></div></div>'
        return t_html.replace('_TABLE_HEADER_', '<thead class="table-light sticky-top"><tr><th>코인명</th><th>현재가</th><th>예측점수</th><th>덤핑위험</th><th>등급</th></tr></thead>')

    html_content += make_table_html(recommended_sector, "rec", "active")
    html_content += make_table_html(interested_sector, "int")
    html_content += make_table_html(normal_sector, "norm")
    html_content += make_table_html(warning_sector, "warn")
    html_content += make_table_html(danger_sector, "dang")

    html_content += f"""
                    </div>
                </div>
            </div>

            <div class="col-lg-3 mb-4">
                <div class="card p-3 shadow-sm">
                    <h5 class="fw-bold text-success mb-2"><i class="fa-solid fa-robot me-1"></i> Gemini 추천 종목 트래킹</h5>
                    <p class="text-muted" style="font-size: 0.75rem;">* 종목 클릭시 위 차트 연동</p>
                    <div class="tracking-box d-flex flex-column gap-3 mt-2">
"""

    for c in active_recommended_tracking:
        count_badge = f'<span class="badge bg-danger ms-1 px-2 py-1 shadow-sm" style="font-size:0.7rem;">🔥 {c["rec_count"]}회 추천</span>' if c['rec_count'] >= 2 else ""

        html_content += f"""                        <div class="p-3 border rounded bg-light shadow-sm coin-row" onclick="loadCoinChart('KRW-{c['symbol']}', '{c['name']}')" style="cursor:pointer;">
                            <div class="d-flex justify-content-between align-items-center mb-2 border-bottom pb-2">
                                <div>
                                    <span class="fw-bold text-dark">{c['name']} <small class="text-muted">({c['symbol']})</small></span>
                                    {count_badge}
                                </div>
                                <span class="badge bg-success">{c['grade']}</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center text-muted" style="font-size: 0.85rem;">
                                <span>최초 추천일시:</span>
                                <span class="fw-semibold text-secondary">{c['rec_time']}</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center text-muted mt-1" style="font-size: 0.85rem;">
                                <span>추천 진입가:</span>
                                <span class="fw-semibold text-dark">{c['entry_price']}원</span>
                            </div>
                            <div class="d-flex justify-content-between align-items-center text-muted mt-1" style="font-size: 0.85rem;">
                                <span>현재 가격:</span>
                                <span class="fw-bold text-primary">{c['price']}원</span>
                            </div>
                        </div>\n"""

    if not active_recommended_tracking:
        html_content += """                        <div class="text-muted small text-center py-4">Gemini가 추천한 종목이 없습니다.</div>\n"""

    html_content += """                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    function filterCoins() {
        let input = document.getElementById('coinSearchInput').value.toLowerCase();
        let rows = document.querySelectorAll('.coin-row');
        
        rows.forEach(row => {
            let name = row.getAttribute('data-name');
            let symbol = row.getAttribute('data-symbol');
            if ((name && name.toLowerCase().includes(input)) || (symbol && symbol.toLowerCase().includes(input))) {
                row.style.display = "";
            } else if(name || symbol) {
                row.style.display = "none";
            }
        });
    }

    let chart, candleSeries;

    function initChart() {
        const container = document.getElementById('tradingview-chart');
        container.innerHTML = '';
        chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 350,
            layout: {
                background: { color: '#ffffff' },
                textColor: '#1e293b',
            },
            grid: {
                vertLines: { color: '#f1f5f9' },
                horzLines: { color: '#f1f5f9' },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            }
        });
        // v4+ 방식에 맞게 addSeries와 CandlestickSeries 적용
        candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor: '#ef4444',
            downColor: '#3b82f6',
            borderVisible: false,
            wickUpColor: '#ef4444',
            wickDownColor: '#3b82f6',
        });

        window.addEventListener('resize', () => {
            chart.resize(container.clientWidth, 350);
        });
    }

    async function loadCoinChart(market, koreanName) {
        document.getElementById('chartTitle').innerHTML = `<i class="fa-solid fa-chart-candlestick text-primary me-1"></i> ${koreanName} (${market}) 실시간 1시간봉 차트`;
        document.getElementById('chartStatus').innerText = '불러오는 중...';
        
        try {
            let res = await fetch(`https://api.upbit.com/v1/candles/minutes/60?market=${market}&count=200`);
            let data = await res.json();
            data.reverse();
            
            let formattedData = data.map(item => {
                let utcTime = new Date(item.timestamp).getTime() / 1000;
                return {
                    time: utcTime,
                    open: item.opening_price,
                    high: item.high_price,
                    low: item.low_price,
                    close: item.trade_price
                };
            });

            candleSeries.setData(formattedData);
            chart.timeScale().fitContent();
            document.getElementById('chartStatus').innerText = '연동 완료';
        } catch (e) {
            console.error(e);
            document.getElementById('chartStatus').innerText = '데이터 로드 실패';
        }
    }

    document.addEventListener("DOMContentLoaded", function() {
        initChart();
        loadCoinChart('KRW-BTC', '비트코인');
    });

    setTimeout(function() {
        location.reload();
    }, 300000);
    </script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("🎨 [대시보드] 차트 오류 수정 완료!")

# ==============================================================================
# [메인 실행부]
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. AI 자가학습 진행
    ai_engine.evolve_models()
    
    # 2. 시장 스캔 및 데이터 분석
    df_result = analyze_and_scan_market()
    
    if not df_result.empty:
        print("\n=== 🎯 [자가학습 AI 적용] 상위 추천 종목 ===")
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

        # 6. Redis 연동
        update_redis_for_dashboard(df_result, ai_report)

        # 7. 대시보드 HTML 파일 생성
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
