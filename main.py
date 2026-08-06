from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

# 깃허브가 보내준 HTML을 임시로 담아둘 공간
latest_html = "<h1>아직 데이터가 없습니다.</h1>"


# 1. 깃허브 액션이 HTML을 쏘는 곳 (POST)
@app.post("/upload")
async def receive_html(request: Request):
  global latest_html
  # 깃허브가 보낸 HTML 내용을 읽어서 저장
  body = await request.body()
  latest_html = body.decode("utf-8")
  return {"status": "success"}


# 2. 사용자가 웹으로 접속해서 보는 곳 (GET)
@app.get("/", response_class=HTMLResponse)
async def view_page():
  return latest_html
