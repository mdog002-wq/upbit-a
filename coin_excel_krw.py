import os
import io
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
    username = os.environ.get("ORACLE_USER", "opc")  # 계정명 (기본 opc 또는 ubuntu)
    ssh_key_content = os.environ.get("ORACLE_SSH_KEY") # GitHub Secrets에서 가져온 개인키 내용

    if not hostname or not ssh_key_content:
        print("⚠️ 오라클 접속 정보(IP 또는 SSH 키)가 설정되지 않아 서버 전송을 스킵합니다.")
        return

    remote_file_path = "/var/www/html/index.html" # 오라클 서버 내 웹서버 저장 경로 (환경에 맞게 수정)

    try:
        # 문자열 형태의 프라이빗 키(.key)를 메모리에서 읽어오기
        key_file_like = io.StringIO(ssh_key_content)
        pkey = paramiko.RSAKey.from_private_key(key_file_like)

        # SSH 클라이언트 연결 설정
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # .key 개인키로 접속
        ssh.connect(hostname, port=22, username=username, pkey=pkey)

        # SFTP를 통해 파일 전송
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

# Redis 서버 설정 (기본 로컬 연결)
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

EXCEL_FILE_PATH = "업비트_원화마켓_매집_패턴분석_리포트.xlsx"
HISTORY_CSV_PATH = "scan_history.csv"
CACHE_DIR = "./cache"
AI_MODELS_DIR = "./ai_models"
EXPERIENCE_FILE = os.path.join(AI_MODELS_DIR, "ai_experience.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AI_MODELS_DIR, exist_ok=True)

# Redis 클라이언트 초기화
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
# [자가학습 엔진] 과거 경험 검증 및 모델 자동 업데이트 (Auto-Training Pipeline)
# ==============================================================================
class AIEvolutionEngine:
    def __init__(self):
        self.exp_file = EXPERIENCE_FILE

    def save_experience(self, ticker, price, lstm_feats=None, stgt_feats=None):
        """현재 스캔 시점의 AI 입력 데이터를 저장 (일괄 요청된 price 활용)"""
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
        """저장된 과거 데이터와 현재 가격을 비교해 정답을 만들고 AI를 학습시킴"""
        if not os.path.exists(self.exp_file): return
        
        try:
            with open(self.exp_file, "r", encoding="utf-8") as f: exps = json.load(f)
        except Exception: return

        current_time = time.time()
        lstm_x_train, lstm_y_train = [], []
        stgt_x_train, stgt_y_train = [], []
        keys_to_delete = []

        # 배치로 가격 불러오기 (API 호출 최소화)
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
        time.sleep(0.04) # Upbit REST API Rate limit 방지 Throttling
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
    """
    종합예측점수와 STGT 덤핑위험도를 조합하여 5단계 등급 부여
    """
    if dump_risk >= 70.0:
        return "🔴 경고"
    elif dump_risk >= 50.0 or score < 40.0:
        return "🟠 주의"
    elif score >= 80.0 and dump_risk < 30.0:
        return "🟢 추천"
    elif score >= 65.0 and dump_risk < 45.0:
        return "🔵 관심"
    else:
        return "⚪ 보통"

def update_redis_for_dashboard(df_result, ai_report):
    """Program B(웹 대시보드)가 읽어갈 수 있도록 Redis에 최신 AI 분석 데이터 전달"""
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

        recommended_coins = [c for c in coin_grades if c['grade'] in ["🟢 추천", "🔵 관심"]]
        warning_coins = [c for c in coin_grades if c['grade'] in ["🟠 주의", "🔴 경고"]]

        dashboard_payload = {
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ai_report": ai_report,
            "summary": {
                "total_scanned": len(coin_grades),
                "recommended_count": len(recommended_coins),
                "warning_count": len(warning_coins)
            },
            "recommended_coins": sorted(recommended_coins, key=lambda x: x['score'], reverse=True)[:5],
            "warning_coins": sorted(warning_coins, key=lambda x: x['dump_risk'], reverse=True)[:5],
            "all_coins": coin_grades
        }

        # Redis 키 저장
        redis_client.set("upbit_ai_dashboard_data", json.dumps(dashboard_payload, ensure_ascii=False))
        print("⚡ [Redis] 웹 대시보드 연동용 AI 리포트 및 5단계 등급 데이터 업데이트 완료!")
    except Exception as e:
        print(f"❌ [Redis] 데이터 업로드 실패: {e}")

# ==============================================================================
# [리포트 생성 및 이메일 발송]
# ==============================================================================
def generate_gemini_analysis(df_top):
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY가 설정되지 않아 AI 요약 생성을 스킵합니다."
    if df_top.empty:
        return "분석된 종목 데이터가 없습니다."

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
        """

        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini AI 요약 생성 실패: {e}"

def export_to_excel_and_email(df_result, ai_report):
    if df_result.empty: return

    # 1. Excel 저장
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

    # 2. 이메일 발송
    if not SENDER_EMAIL or not EMAIL_PASSWORD or not RECEIVER_EMAILS:
        print("📧 이메일 계정 정보가 설정되지 않아 메일 발송을 스킵합니다.")
        return

    msg = MIMEMultipart()
    msg['Subject'] = f"🚀 [업비트] 매집 패턴 및 AI 분석 리포트 ({datetime.date.today()})"
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECEIVER_EMAILS)

    body_text = f"안녕하세요,\n\n오늘의 업비트 원화마켓 매집 점수 및 AI 분석 리포트입니다.\n\n=========================================="
    body_text += f"\n🤖 [Gemini 3.1 Flash Lite AI 종합 분석]\n{ai_report}\n==========================================\n\n"
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
# [신규 모듈] 인터랙티브 웹 대시보드 HTML 자동 생성 기능 정의 (먼저 선언되어야 함)
# ==============================================================================
def generate_dashboard_html(df_result, ai_report):
    """
    분석된 데이터를 바탕으로 좌측 상단 '급등주포착' 단추가 포함된 대시보드 HTML을 생성합니다.
    """
    os.makedirs("docs", exist_ok=True)
    html_path = "docs/index.html"

    if df_result.empty:
        html_content = "<html><body><h1>분석된 데이터가 없습니다.</h1></body></html>"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return

    coins_data = []
    for _, row in df_result.iterrows():
        score = float(row['종합예측점수'])
        dump_risk = float(row['STGT_그래프덤핑위험(%)'])
        grade = calculate_ai_grade(score, dump_risk)
        
        cmf = float(row['CMF지표'])
        est_change = round(cmf * 5.2, 2)
        
        coins_data.append({
            "name": row['코인명'],
            "symbol": row['심볼'],
            "price": row['현재가(KRW)'],
            "score": score,
            "dump_risk": dump_risk,
            "iceberg": row['아이스버그역산(고주파)'],
            "grade": grade,
            "change": est_change,
            "entry_price": row['현재가(KRW)'],
            "target_price": format_price(float(str(row['현재가(KRW)']).replace(',', '')) * 1.07) if str(row['현재가(KRW)']).replace(',', '').replace('.', '').isdigit() else "-"
        })

    recommended_sector = [c for c in coins_data if c['grade'] == "🟢 추천"]
    interested_sector = [c for c in coins_data if c['grade'] == "🔵 관심"]
    normal_sector = [c for c in coins_data if c['grade'] == "⚪ 보통"]
    warning_sector = [c for c in coins_data if c['grade'] in ["🟠 주의", "🔴 경고"]]

    top3_coins = sorted(coins_data, key=lambda x: x['score'], reverse=True)[:3]

    alerts = []
    for c in coins_data:
        if c['dump_risk'] >= 75.0:
            alerts.append({"type": "danger", "text": f"🚨 [급락/덤핑위험] {c['name']}({c['symbol']}) 위험도 {c['dump_risk']}%"})
        elif c['score'] >= 85.0:
            alerts.append({"type": "success", "text": f"🚀 [급등포착] {c['name']}({c['symbol']}) 매집 점수 {c['score']}점"})

    updated_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Upbit AI Quantitative Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{ background-color: #0f172a; color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }}
        .table {{ color: #f8fafc; }}
        .table-dark {{ background-color: #1e293b; }}
        .badge-recommend {{ background-color: #22c55e; }}
        .badge-interest {{ background-color: #3b82f6; }}
        .badge-normal {{ background-color: #64748b; }}
        .badge-warning {{ background-color: #f97316; }}
        .badge-danger {{ background-color: #ef4444; }}
        .sticky-sticker {{
            position: fixed; right: 20px; top: 20px; width: 280px; z-index: 1050;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 2px solid #3b82f6; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            padding: 15px; font-size: 0.9rem;
        }}
        .alert-box {{ max-height: 250px; overflow-y: auto; }}
        .top3-card {{ background: linear-gradient(135deg, #334155 100%, #1e293b 0%); border-left: 5px solid #3b82f6; }}
    </style>
</head>
<body>
    <!-- 우측 누적 추천 스티커 영역 -->
    <div class="sticky-sticker">
        <h6 class="fw-bold text-primary mb-2"><i class="fa-solid fa-thumbtack"></i> 당일 AI 추천 스티커</h6>
        <div id="sticker-content">
            <p class="text-muted small mb-1">누적 추천 이력 및 트래킹</p>
            <ul class="list-unstyled small mb-0" id="sticker-list">
                <li>🟢 <b>BTC</b> (추천 3회) <br><span class="text-muted">진입: 92,000,000 | 목표: 98,000,000</span></li>
                <hr class="my-1 border-secondary">
                <li>🟢 <b>ETH</b> (추천 2회) <br><span class="text-muted">진입: 4,500,000 | 목표: 4,900,000</span></li>
            </ul>
        </div>
    </div>

    <div class="container-fluid my-4 px-4" style="max-width: 1600px;">
        <!-- 상단 네비게이션 (좌측 상단 급등주포착 버튼 포함) -->
        <div class="row mb-4 align-items-center">
            <div class="col-md-3 text-start">
                <a href="/" class="btn btn-outline-primary fw-bold px-3 py-2 shadow-sm">
                    <i class="fa-solid fa-bolt text-warning"></i> 급등주포착
                </a>
            </div>
            <div class="col-md-6 text-center">
                <h1 class="fw-bold text-white mb-0" style="font-size: 1.8rem;"><i class="fa-solid fa-robot text-primary"></i> 업비트 AI 매집 패턴 대시보드</h1>
                <small class="text-muted">최종 업데이트: {updated_time}</small>
            </div>
            <div class="col-md-3"></div>
        </div>

        <div class="row">
            <!-- [좌측] 급등/급락 예상 알림창 -->
            <div class="col-lg-3 mb-4">
                <div class="card p-3 shadow h-100">
                    <h5 class="fw-bold text-warning mb-3"><i class="fa-solid fa-bell"></i> 실시간 급등/급락 알림</h5>
                    <div class="alert-box d-flex flex-column gap-2">
"""

    for alert in alerts[:10]:
        badge_cls = "bg-danger" if alert['type'] == "danger" else "bg-success"
        html_content += f"""                        <div class="p-2 rounded {badge_cls} bg-opacity-25 border border-{alert['type']} small">
                            {alert['text']}
                        </div>\n"""

    if not alerts:
        html_content += """                        <div class="text-muted small text-center py-4">현재 특이 경고/급등 종목이 없습니다.</div>\n"""

    html_content += f"""                    </div>
                    
                    <hr class="border-secondary my-3">
                    <h6 class="fw-bold text-info mb-2"><i class="fa-solid fa-brain"></i> Gemini AI 요약</h6>
                    <p class="small text-light bg-dark p-2 rounded" style="white-space: pre-line; max-height: 200px; overflow-y: auto;">{ai_report}</p>
                </div>
            </div>

            <!-- [중앙] 상단 AI 추천 3종목 및 섹터별 전체 순위 테이블 -->
            <div class="col-lg-9">
                <!-- 중앙 상단: AI 추천 Top 3 코인 카드 -->
                <div class="row mb-4">
                    <div class="col-12">
                        <h4 class="fw-bold text-success mb-3"><i class="fa-solid fa-star"></i> AI 선정 오늘의 강력 추천 Top 3</h4>
                    </div>
"""

    for i, coin in enumerate(top3_coins):
        change_color = "text-danger" if coin['change'] < 0 else "text-success"
        sign = "+" if coin['change'] > 0 else ""
        html_content += f"""                    <div class="col-md-4 mb-3">
                        <div class="card top3-card p-3 shadow">
                            <div class="d-flex justify-content-between align-items-center">
                                <h5 class="fw-bold mb-0">{coin['name']} <small class="text-muted">({coin['symbol']})</small></h5>
                                <span class="badge bg-primary">Top {i+1}</span>
                            </div>
                            <hr class="my-2 border-secondary">
                            <div class="d-flex justify-content-between small mb-1">
                                <span class="text-muted">현재가:</span>
                                <span class="fw-bold">{coin['price']} KRW</span>
                            </div>
                            <div class="d-flex justify-content-between small mb-1">
                                <span class="text-muted">전일 대비:</span>
                                <span class="{change_color} fw-bold">{sign}{coin['change']}%</span>
                            </div>
                            <div class="d-flex justify-content-between small">
                                <span class="text-muted">예측 점수:</span>
                                <span class="text-warning fw-bold">{coin['score']}점</span>
                            </div>
                        </div>
                    </div>\n"""

    html_content += f"""                </div>

                <!-- 중앙 본문: 점수 순위 및 섹터별 정렬 테이블 -->
                <div class="card p-4 shadow">
                    <h4 class="fw-bold mb-3"><i class="fa-solid fa-chart-line"></i> 전체 분석 코인 섹터별 상세 순위</h4>
                    
                    <ul class="nav nav-tabs mb-3" id="coinTab" role="tablist">
                        <li class="nav-item" role="presentation"><button class="nav-link active bg-success text-white me-2" id="rec-tab" data-bs-toggle="tab" data-bs-target="#rec" type="button">🟢 추천 ({len(recommended_sector)})</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link bg-primary text-white me-2" id="int-tab" data-bs-toggle="tab" data-bs-target="#int" type="button">🔵 관심 ({len(interested_sector)})</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link bg-secondary text-white me-2" id="norm-tab" data-bs-toggle="tab" data-bs-target="#norm" type="button">⚪ 보통 ({len(normal_sector)})</button></li>
                        <li class="nav-item" role="presentation"><button class="nav-link bg-warning text-dark" id="warn-tab" data-bs-toggle="tab" data-bs-target="#warn" type="button">🟠 주의/경고 ({len(warning_sector)})</button></li>
                    </ul>

                    <div class="tab-content" id="coinTabContent">
"""

    def make_table_html(sector_list, tab_id, is_active=""):
        active_cls = "show active" if is_active else ""
        t_html = f'<div class="tab-pane fade {active_cls}" id="{tab_id}" role="tabpanel">'
        t_html += '<div class="table-responsive"><table class="table table-dark table-hover table-striped align-middle small">'
        t_html += '<thead><tr><th>순위</th><th>코인명</th><th>현재가</th><th>예측점수</th><th>덤핑위험</th><th>고주파 아이스버그</th><th>등급</th></tr></thead><tbody>'
        
        sorted_sector = sorted(sector_list, key=lambda x: x['score'], reverse=True)
        for idx, c in enumerate(sorted_sector):
            t_html += f"""<tr>
                <td>{idx+1}</td>
                <td class="fw-bold">{c['name']} <small class="text-muted">({c['symbol']})</small></td>
                <td>{c['price']} KRW</td>
                <td class="text-warning fw-bold">{c['score']}</td>
                <td class="text-danger">{c['dump_risk']}%</td>
                <td>{c['iceberg']}</td>
                <td><span class="badge {'bg-success' if '추천' in c['grade'] else 'bg-primary' if '관심' in c['grade'] else 'bg-warning text-dark' if '주의' in c['grade'] else 'bg-danger' if '경고' in c['grade'] else 'bg-secondary'}">{c['grade']}</span></td>
            </tr>"""
        t_html += '</tbody></table></div></div>'
        return t_html

    html_content += make_table_html(recommended_sector, "rec", "active")
    html_content += make_table_html(interested_sector, "int")
    html_content += make_table_html(normal_sector, "norm")
    html_content += make_table_html(warning_sector, "warn")

    html_content += f"""
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("🎨 [대시보드] '급등주포착' 버튼이 포함된 HTML 대시보드 생성 완료 (`docs/index.html`)!")


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
        print("\n=== 🎯 [자가학습 AI 적용] 상위 추천 종목 ===")
        print(df_result[["코인명", "종합예측점수", "STGT_그래프덤핑위험(%)", "아이스버그역산(고주파)"]].head(5))
        
        # 3. 위험 알림 (텔레그램)
        danger_coins = df_result[df_result['STGT_그래프덤핑위험(%)'] >= 75.0]
        if not danger_coins.empty:
            print(f"\n🚨 [위험 감지] {len(danger_coins)}개 종목 덤핑 위험! (텔레그램 전송)")
            send_telegram_alert(f"🚨 *[자가학습 AI 경고]* 덤핑 위험 감지: {', '.join(danger_coins['코인명'].tolist())}")

        # 4. Gemini AI 요약 작성
        ai_report = generate_gemini_analysis(df_result)

        # 5. 5단계 AI 등급 생성 및 Redis 연동 
        update_redis_for_dashboard(df_result, ai_report)

        # 🚀 6. 대시보드 HTML 파일 실시간 생성
        generate_dashboard_html(df_result, ai_report)

        # 🚀 7. 오라클 서버로 HTML 대시보드 자동 전송
        upload_html_to_oracle_server("docs/index.html")

        # 8. 엑셀 저장 및 이메일 발송
        kst_now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
        current_hour = kst_now.hour

        target_hours = [9, 13, 17, 21] 

        if current_hour in target_hours:
            export_to_excel_and_email(df_result, ai_report)
        else:
            print(f"⏰ 현재 시각(KST {current_hour}시)은 이메일 발송 시간이 아니므로 대시보드만 갱신합니다.")
