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

from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from openai import OpenAI

# ==============================================================================
# [설정] GitHub Secrets 환경 변수
# ==============================================================================
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
# 수신자 이메일을 쉼표(,)로 구분하여 다중 수신 지원 (예: "user1@gmail.com, user2@naver.com")
RECEIVER_EMAILS = [
    email.strip() 
    for email in os.environ.get("RECEIVER_EMAIL", "").split(",") 
    if email.strip()
]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

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
# [외부 데이터 수집] 1. 코인니스 속보 | 2. 구글 트렌드 | 3. 업비트 공지 | 4. X(트위터)
# ==============================================================================
def get_coinness_news(coin_name):
    """코인니스에서 특정 코인 관련 최근 속보 수집"""
    try:
        url = f"https://coinness.com/search?q={coin_name}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers, timeout=3)
        
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            titles = [a.text.strip() for a in soup.find_all('h3')[:2]]
            if titles:
                return " | ".join(titles)
        return "특이 속보 없음 (고요한 상태)"
    except Exception:
        return "속보 조회 불가"

def get_google_search_trend(coin_names):
    """상위 코인들의 구글 상대적 검색량 지수 수집 (0~100)"""
    try:
        pytrends = TrendReq(hl='ko', tz=540)
        keywords = [name + " 코인" for name in coin_names[:5]] # 상위 5개 비교
        
        pytrends.build_payload(keywords, timeframe='now 7-d', geo='KR')
        data = pytrends.interest_over_time()
        
        if not data.empty:
            latest_trend = data.iloc[-1].to_dict()
            return {k.replace(" 코인", ""): int(v) for k, v in latest_trend.items() if k != 'isPartial'}
        return {}
    except Exception as e:
        print(f"⚠️ 구글 트렌드 수집 건너뜀: {e}")
        return {}

def get_upbit_notices(coin_name, symbol):
    """업비트 공식 공지사항 중 해당 코인 관련 이슈 검색 (투자유의, 입출금, 상장 등)"""
    try:
        url = "https://api-manager.upbit.com/api/v1/notices?page=1&per_page=15"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers, timeout=3)
        
        if res.status_code == 200:
            notices_list = res.json().get('data', {}).get('list', [])
            matched_notices = []
            
            for notice in notices_list:
                title = notice.get('title', '')
                if coin_name in title or symbol in title:
                    matched_notices.append(title)
            
            if matched_notices:
                return " | ".join(matched_notices[:2])
                
        return "최근 업비트 공식 공지 없음 (안전)"
    except Exception as e:
        return "업비트 공지 조회 불가"

def get_x_twitter_sentiment(symbol, coin_name):
    """X(트위터) 오픈소스 Nitter 인스턴스/검색을 통해 최근 트윗 반응 수집"""
    try:
        nitter_instances = [
            "https://nitter.net",
            "https://nitter.cz",
            "https://nitter.privacydev.net"
        ]
        
        query = f"${symbol} OR {coin_name}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        for instance in nitter_instances:
            try:
                url = f"{instance}/search?f=tweets&q={query}"
                res = requests.get(url, headers=headers, timeout=2.5)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, 'html.parser')
                    tweets = [t.text.strip().replace('\n', ' ') for t in soup.find_all('div', class_='tweet-content')[:3]]
                    if tweets:
                        return " // ".join(tweets)
            except Exception:
                continue
                
        return "X(트위터) 실시간 특이 언급 적음"
    except Exception:
        return "X(트위터) 조회 불가"

# ==============================================================================
# [알고리즘] 매집 점수 산출
# ==============================================================================
def calculate_score(vol_ratio, trade_value_krw, price_change, is_yangbong, ma_gap):
    score = 0
    
    # 1. 거래량 급증 점수
    if vol_ratio >= 4.0: score += 35
    elif vol_ratio >= 2.5: score += 28
    elif vol_ratio >= 1.8: score += 20
    elif vol_ratio >= 1.3: score += 12
    else: score += 5

    # 2. 거래대금 점수
    trade_value_eow = trade_value_krw / 100_000_000
    if trade_value_eow >= 50: score += 25
    elif trade_value_eow >= 20: score += 20
    elif trade_value_eow >= 10: score += 15
    elif trade_value_eow >= 5: score += 10
    else: score += 5

    # 3. 주가 변동 및 양봉 여부
    if is_yangbong and 0.3 <= price_change <= 7.0: score += 20
    elif is_yangbong and price_change > 7.0: score += 12
    elif not is_yangbong: score += 0

    # 4. 이동평균선(20일-60일) 수렴도
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
# [AI 심층 분석] Groq AI + 종합 이슈 + X(트위터) 융합 브리핑 (다양성/직설적 분석)
# ==============================================================================
def generate_groq_analysis(df):
    if not GROQ_API_KEY:
        print("❌ [경고] GROQ_API_KEY 환경 변수가 없습니다.")
        return "Groq API 키가 설정되지 않아 AI 요약이 생성되지 않았습니다."

    if df.empty:
        print("⚠️ 데이터가 없어 AI 분석을 건너뜁니다.")
        return "데이터가 없어 AI 요약을 생성하지 못했습니다."

    try:
        print("\n🤖 차트 수급 + 이슈 + 업비트 공지 + X(트위터) 반응 융합 AI 분석 시작...")
        top_10 = df.head(10).copy()
        
        # 1. 상위 5개 종목 구글 검색량 수집
        top_5_names = top_10['코인명'].tolist()[:5]
        search_trends = get_google_search_trend(top_5_names)
        
        # 2. 상위 10개 종목 외부 이슈, 공지, X(트위터) 반응 데이터 구축
        enriched_data = []
        for idx, row in top_10.iterrows():
            coin_name = row['코인명']
            symbol = row['심볼']
            
            news = get_coinness_news(coin_name)
            upbit_notice = get_upbit_notices(coin_name, symbol)
            x_tweets = get_x_twitter_sentiment(symbol, coin_name)
            trend_score = search_trends.get(coin_name, "수집 미지원")
            
            enriched_data.append({
                "코인명": coin_name,
                "매집점수": row['매집점수'],
                "당일변동률": f"{row['당일 변동률(%)']}%",
                "거래량급증": f"{row['거래량 급증(배)']}배",
                "거래대금": f"{row['거래대금(억원)']}억원",
                "구글검색관심도(0~100)": trend_score,
                "코인니스속보": news,
                "업비트공지사항": upbit_notice,
                "X(트위터)최근의견": x_tweets
            })

        prompt = f"""
당신은 가상자산 헤지펀드에서 가장 날카롭고 직설적인 수석 트레이더입니다.
아래 수집된 상위 10개 코인의 종합 데이터를 바탕으로, 매일 똑같은 서술 방식을 완전히 배제하고 종목별 개별 특성에 집중한 정밀 리포트를 작성하세요.

[데이터 종합 표]
{enriched_data}

[❌ 절대 금지 사항 (상투적/정형화 표현 배제)]
- "조용한 매집 구간으로 해석됩니다", "개미 꼬시기 물량일 수 있습니다", "실전 트레이딩에 바로 활용할 수 있는" 같은 틀에 박힌 템플릿 문구를 절대 사용하지 마세요.
- 모든 종목에 동일한 형태의 문장 구조(예: ~이 높으므로 ~을 주의해야 합니다)를 복사-붙여넣기하듯 되풀이하지 마세요.

[작성 지침 및 차별화 원칙]
1. **종목별 독자적 시각 부여**: 각 종목마다 가장 눈에 띄는 '하나의 변수'(예: 엄청난 거래대금, 트위터 반응 폭발, 공지사항 이슈, 검색량 기괴한 증가 등)를 핵심 축으로 삼아 개별적으로 다르게 해석하세요.
2. **구체적 인용**: 데이터에 나와있는 거래대금 액수, 구글 검색 지수 숫자, 트위터 언급 내용의 키워드를 직접 인용하여 서술하세요.
3. **[추천 종목 Top 3] 필수 포함**:
   - 수급과 트윗/이슈 민심이 가장 강력한 3개 종목을 1위, 2위, 3위로 명확히 선정하세요.
   - 각 추천 종목이 '왜 다른 추천 종목과 차별화되는지' 개별적 이유를 기술하세요.
4. **[주의/과열 유의 종목 1개] 필수 포함**:
   - 매집 점수가 높더라도 거래대금이 쥐꼬리만 하거나, 과도하게 검색량이 폭증해 '설거지/상투' 가능성이 의심되는 1개 종목을 찍어 경고하세요.
5. 어조는 전문적이되 뜬구름 잡지 말고, 트레이딩 현장에서 쓰는 듯한 직설적이고 명확한 어조(~함, ~임 또는 정중한 ~합니다체)를 사용하세요.
"""
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY.strip()
        )
        
        # Temperature를 0.7로 끌어올려 창의성 및 문장 다양성 확보
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.9
        )
        
        result_text = response.choices[0].message.content
        if result_text:
            print("✅ Groq AI 종합 융합 분석 성공!")
            return result_text.strip()
        else:
            return "Groq AI 응답이 비어 있습니다."

    except Exception as e:
        print(f"❌ Groq AI 예외 발생: {e}")
        return f"Groq AI 분석 생성 중 오류가 발생했습니다: {e}"

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
# [메일 전송] (다중 수신자 지원)
# ==============================================================================
def send_email_with_excel(file_path, ai_analysis=""):
    if not file_path or not os.path.exists(file_path):
        return

    if not SENDER_EMAIL or not EMAIL_PASSWORD or not RECEIVER_EMAILS:
        print("❌ 환경변수(Secrets) 설정에 문제가 있어 이메일을 발송하지 못했습니다.")
        return

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    msg = MIMEMultipart()
    msg["Subject"] = f"📊 [업비트 AI 심층분석] 원화마켓 매집 점수 리포트 ({now_str})"
    msg["From"] = SENDER_EMAIL
    msg["To"] = ", ".join(RECEIVER_EMAILS)
    
    body = f"""안녕하세요.

업비트 원화(KRW) 마켓 전체 상장 종목 분석 및 Groq AI 종합 이슈 결합 리포트가 완료되었습니다.

• 분석 시각: {now_str}
• 분석 대상: 원화 마켓 전체 종목

==================================================
🤖 [Groq AI 차트+이슈+공지+X(트위터) 종합 심층 분석 브리핑]
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
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        server.quit()
        print(f"📧 엑셀 + AI 심층 브리핑 이메일 발송 성공! (수신자: {', '.join(RECEIVER_EMAILS)})")
        print("--------------------------------------------------")
    except Exception as e:
        print(f"❌ 이메일 발송 실패: {e}")

if __name__ == "__main__":
    df_ranked = scan_and_rank_coins()
    ai_summary = generate_groq_analysis(df_ranked)
    excel_file = save_daily_excel_sheet(df_ranked)
    send_email_with_excel(excel_file, ai_summary)
