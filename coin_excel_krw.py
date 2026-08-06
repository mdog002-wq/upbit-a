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
