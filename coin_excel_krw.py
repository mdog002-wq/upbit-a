import os
import time
import datetime
import smtplib
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

import pandas as pd
import pyupbit
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Google 최신 GenAI SDK
from google import genai

# ==============================================================================
# [설정] GitHub Secrets 환경 변수
# ==============================================================================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

EXCEL_FILE_PATH = "업비트_원화마켓_매집점수_날짜별기록.xlsx"

# ==============================================================================
# [유틸] 업비트 '원화(KRW) 마켓' 전 종목 조회
# ==============================================================================
def get_krw_upbit_tickers():
    url = "https://api.upbit.com/v1/market/all"
    headers = {"accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        krw_coins = []
        for coin in data:
            if coin['market'].startswith("KRW-"):
                krw_coins.append({
                    'ticker': coin['market'],
                    'korean_name': coin['korean_name'],
                    'english_name': coin['english_name'],
                    'symbol': coin['market'].replace("KRW-", "")
                })
        return krw_coins
    except Exception as e:
        print(f"❌ 업비트 원화 코인 목록 조회 실패: {e}")
        return []

# ==============================================================================
# [알고리즘] 매집 점수 산출
# ==============================================================================
def calculate_score(vol_ratio, trade_value_krw, price_change, is_yangbong, ma_gap):
    score = 0
    
    if vol_ratio >= 4.0: score += 35
    elif vol_ratio >= 2.5: score += 28
    elif vol_ratio >= 1.8: score += 20
    elif vol_ratio >= 1.3: score += 12
    else: score += 5

    trade_value_eow = trade_value_krw / 100_000_000
    if trade_value_eow >= 50: score += 25
    elif trade_value_eow >= 20: score += 20
    elif trade_value_eow >= 10: score += 15
    elif trade_value_eow >= 5: score += 10
    else: score += 5

    if is_yangbong and 0.3 <= price_change <= 7.0: score += 20
    elif is_yangbong and price_change > 7.0: score += 12
    elif not is_yangbong: score += 0

    if ma_gap <= 3.0: score += 20
    elif ma_gap <= 6.0: score += 15
    elif ma_gap <= 10.0: score += 10
    else: score += 3

    return score

# ==============================================================================
# [분석 엔진] 전 종목 조사
# ==============================================================================
def scan_and_rank_coins():
    print("--------------------------------------------------")
    print("🚀 업비트 원화(KRW) 마켓 전 종목 전수 조사 시작...")
    
    krw_coins = get_krw_upbit_tickers()
    if not krw_coins:
        return pd.DataFrame()

    print(f"총 {len(krw_coins)}개 원화 마켓 종목 분석 중...\n")
    results = []
    
    for item in krw_coins:
        ticker = item['ticker']
        korean_name = item['korean_name']
        symbol = item['symbol']
        
        df_daily = None
        for retry in range(3):
            try:
                time.sleep(0.06)
                df_daily = pyupbit.get_ohlcv(ticker, interval="day", count=60)
                if df_daily is not None and not df_daily.empty:
                    break
            except Exception:
                time.sleep(0.1)

        if df_daily is None or df_daily.empty:
            results.append({
                "코인명": korean_name,
                "심볼": symbol,
                "매집점수": 0,
                "당일 변동률(%)": 0.0,
                "거래량 급증(배)": 0.0,
                "거래대금(억원)": 0.0,
                "현재가(KRW)": 0,
                "이평선 수렴도(%)": 0.0,
                "비고": "OHLCV 데이터 미제공(신규/정지)"
            })
            continue

        try:
            latest = df_daily.iloc[-1]
            
            if len(df_daily) >= 21:
                prev_20_vol_avg = df_daily['volume'].iloc[-21:-1].mean()
            else:
                prev_20_vol_avg = df_daily['volume'].mean()

            vol_ratio = (latest['volume'] / prev_20_vol_avg) if prev_20_vol_avg > 0 else 0
            trade_value_krw = latest['value']
            
            if len(df_daily) >= 2:
                price_change = ((latest['close'] - df_daily['close'].iloc[-2]) / df_daily['close'].iloc[-2]) * 100
            else:
                price_change = 0.0
                
            is_yangbong = latest['close'] >= latest['open']
            
            ma20 = df_daily['close'].rolling(20).mean().iloc[-1] if len(df_daily) >= 20 else latest['close']
            ma60 = df_daily['close'].rolling(60).mean().iloc[-1] if len(df_daily) >= 60 else ma20
            ma_gap = abs(ma20 - ma60) / ma20 * 100 if ma20 > 0 else 0

            total_score = calculate_score(vol_ratio, trade_value_krw, price_change, is_yangbong, ma_gap)
            
            results.append({
                "코인명": korean_name,
                "심볼": symbol,
                "매집점수": total_score,
                "당일 변동률(%)": round(price_change, 2),
                "거래량 급증(배)": round(vol_ratio, 2),
                "거래대금(억원)": round(trade_value_krw / 100_000_000, 1),
                "현재가(KRW)": latest['close'],
                "이평선 수렴도(%)": round(ma_gap, 2),
                "비고": "정상"
            })
            print(f"  [수집 완료] {korean_name}({symbol}) | 매집점수: {total_score}점")
                
        except Exception as e:
            results.append({
                "코인명": korean_name,
                "심볼": symbol,
                "매집점수": 0,
                "당일 변동률(%)": 0.0,
                "거래량 급증(배)": 0.0,
                "거래대금(억원)": 0.0,
                "현재가(KRW)": 0,
                "이평선 수렴도(%)": 0.0,
                "비고": f"연산 오류({e})"
            })
            continue
            
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="매집점수", ascending=False)
    return df

# ==============================================================================
# [AI 분석] Google Gemini 상위 10개 코인 분석 (100% 한국어 전용 + 자동 재시도)
# ==============================================================================
def generate_gemini_analysis(df):
    if not GEMINI_API_KEY:
        print("❌ [경고] GEMINI_API_KEY 환경 변수가 없습니다.")
        return "Gemini API 키가 설정되지 않아 AI 요약이 생성되지 않았습니다."

    if df.empty:
        print("⚠️ 데이터가 없어 AI 분석을 건너뜁니다.")
        return "데이터가 없어 AI 요약을 생성하지 못했습니다."

    try:
        print("\n🤖 Google Gemini AI 분석 시작...")
        top_coins = df.head(10).to_dict(orient="records")
        
        prompt = f"""
당신은 암호화폐 전업 트레이더 겸 데이터 분석 전문가입니다.
아래는 오늘 업비트 원화 마켓에서 매집 점수가 가장 높은 상위 10개 코인 데이터입니다.

[상위 10개 매집 코인 데이터]
{top_coins}

위 데이터를 바탕으로 수신자가 한눈에 파악할 수 있도록 4~5줄 내외의 깔끔한 AI 분석 요약 보고서를 작성해 주세요.

[작성 지침 - 엄격 적용]
1. 반드시 100% 한국어로만 작성해 주세요. (영단어, 영어 알파벳 사용 금지. 코인 이름도 한글명 위주로 작성)
2. 매집 점수 상위권 코인들의 주요 공통점과 특징(거래량 급증, 이동평균선 수렴 등)을 종합적으로 분석해 주세요.
3. 상위 10개 중 가장 주목할 만한 특이 종목 2~3개를 콕 집어 한글 이름으로 언급해 주세요.
4. 정중하고 친절한 경어체(~합니다, ~입니다)를 사용해 주세요.
"""
        client = genai.Client(api_key=GEMINI_API_KEY.strip())
        
        # 💡 현재 정상 연결이 보장된 단 하나의 모델: gemini-2.0-flash
        target_model = "gemini-2.0-flash"
        
        # 429 한도 초과(Rate Limit) 방지를 위한 자동 3회 재시도 로직
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                print(f"🔄 [{target_model}] 모델 연결 시도 중... (시도 {attempt}/{max_retries})")
                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt
                )
                
                if response and response.text:
                    print(f"✅ Gemini AI 분석 성공! (사용된 모델: {target_model})")
                    return response.text.strip()
            except Exception as err:
                err_msg = str(err)
                print(f"❌ 시도 {attempt} 실패: {err_msg}")
                # 429 한도 초과 에러 시 15초 대기 후 재시도
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    print("⏳ 분당 요청 한도(429) 대기 중... 15초 후 재시도합니다.")
                    time.sleep(15)
                else:
                    time.sleep(3)
                
        return "Gemini AI 모델 호출 한도 초과로 생성 실패했습니다. 잠시 후 다시 시도해 주세요."

    except Exception as e:
        print(f"❌ Gemini AI 예외 발생: {e}")
        return f"AI 분석 생성 중 오류가 발생했습니다: {e}"

# ==============================================================================
# [엑셀 저장]
# ==============================================================================
def save_daily_excel_sheet(df):
    if df.empty:
        return None

    now = datetime.datetime.now()
    sheet_name = now.strftime("%Y-%m-%d_%H시")
    
    if os.path.exists(EXCEL_FILE_PATH):
        wb = openpyxl.load_workbook(EXCEL_FILE_PATH)
    else:
        wb = openpyxl.Workbook()
        
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
    high_score_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9')
    )
    
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    for row in range(2, ws.max_row + 1):
        score_cell = ws.cell(row=row, column=3)
        is_high_score = score_cell.value and float(score_cell.value) >= 70
        
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = data_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center" if col != 7 else "right", vertical="center")
            
            if is_high_score:
                cell.fill = high_score_fill
                
    for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 5, 12)

    wb.save(EXCEL_FILE_PATH)
    wb.close()
    
    print("--------------------------------------------------")
    print(f"📁 [{sheet_name}] 시트에 총 {len(df)}개 원화 코인 수집 완료!")
    return EXCEL_FILE_PATH

# ==============================================================================
# [메일 전송]
# ==============================================================================
def send_email_with_excel(file_path, ai_analysis=""):
    if not file_path or not os.path.exists(file_path):
        return

    if not SENDER_EMAIL or not EMAIL_PASSWORD or not RECEIVER_EMAIL:
        print("❌ 환경변수(Secrets) 설정에 문제가 있어 이메일을 발송하지 못했습니다.")
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    msg = MIMEMultipart()
    msg["Subject"] = f"📊 [업비트 AI 분석] 원화마켓 매집 점수 리포트 ({now_str})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    
    body = f"""안녕하세요.

업비트 원화(KRW) 마켓 전체 상장 종목 분석 및 Gemini AI 리포트가 완료되었습니다.

• 분석 시각: {now_str}
• 분석 대상: 원화 마켓 전체 종목

==================================================
🤖 [Google Gemini AI 상위 코인 분석 브리핑]
==================================================
{ai_analysis}

==================================================
자세한 데이터는 첨부된 엑셀 파일을 확인해 주세요.
"""
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
        file_name = os.path.basename(file_path)
        part.add_header("Content-Disposition", "attachment", filename=("utf-8", "", file_name))
        msg.attach(part)
    except Exception as e:
        print(f"❌ 파일 첨부 실패: {e}")
        return

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"📧 엑셀 + AI 브리핑 이메일 발송 성공! (수신: {RECEIVER_EMAIL})")
        print("--------------------------------------------------")
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

if __name__ == "__main__":
    df_ranked = scan_and_rank_coins()
    ai_summary = generate_gemini_analysis(df_ranked)
    excel_file = save_daily_excel_sheet(df_ranked)
    send_email_with_excel(excel_file, ai_summary)
