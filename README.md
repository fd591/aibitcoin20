# 2번 코드드 3.30. 09:55분 수정함
# 1. 전역 상수 정의
SYMBOL = "BTCUSDT"
FEE_RATE = 0.0006
MIN_QTY = 0.001
REINVESTMENT_RATIO = 0.5 
MAX_ACCOUNT_RISK_PCT = 20.0  # 총자산 대비 최대 손실 비율 (20%)

# 전략 파라미터 수정된 기본값
RSI_HIGH_THRESHOLD = 60    # 완화됨
RSI_LOW_THRESHOLD = 40     # 완화됨
ATR_MULTIPLIER = 3.0  # 2.0에서 3.0으로 증가
TP_LEVELS = [1.0, 1.5, 2.0]  # 현실적인 수준으로 조정
VOLATILITY_THRESHOLD = 0.8    # 완화됨
ADX_THRESHOLD = 15           # 완화됨

# 전역 상수 부분에 추가할 내용
H1_INTERVAL = "60"  # 1시간봉
H4_INTERVAL = "240"  # 4시간봉
DB_PATH = "bitcoin_trades.db"

# 2. 필요한 모듈 import
import os
import json
import time
import uuid
import sqlite3
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from typing import Optional, Dict

import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import hmac
import hashlib

from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from datetime import timedelta
from dotenv import load_dotenv
from openai import OpenAI
from pybit.unified_trading import HTTP
from pydantic import BaseModel, Field
from dataclasses import dataclass  # <- dataclass 임포트 추가

# TA 라이브러리
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# 3. 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 4. 환경변수 로드
load_dotenv()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = "https://api.bybit.com"

# 5. API 클라이언트 초기화
bybit_client = HTTP(
    api_key=os.getenv("BYBIT_API_KEY"),
    api_secret=os.getenv("BYBIT_API_SECRET")
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# bybit_client = HTTP(
#     api_key=os.getenv("BYBIT_TESTNET_API_KEY"),
#     api_secret=os.getenv("BYBIT_TESTNET_API_SECRET"),
#     testnet=True  # 테스트넷 사용 설정
# )

# ---- 여기에 추가 ----
API_KEY = os.getenv("BYBIT_API_KEY")

# session 객체 선언 및 헤더 설정
session = requests.Session()
session.headers.update({
    "Content-Type": "application/json",
    "X-BAPI-API-KEY": API_KEY
})

# 6. 기본 클래스 정의
class TradingDecision(BaseModel):
    decision: str = Field(..., description="Trading decision: Long/Short/Close Long Position/Close Short Position/Hold")
    reason: str = Field(..., description="Decision reasoning")
    position_size: int = Field(..., ge=0, le=100, description="Position size percentage (0-100)")
    leverage: Optional[int] = Field(None, ge=1, le=20, description="Leverage (1-20)")
    stop_loss_pct: Optional[float] = Field(None, ge=0.1, le=5.0, description="Stop loss percentage (0.1-5.0)")    
        # 새로 추가할 필드들
    take_profit_targets: Optional[List[float]] = Field(None, description="Take profit target percentages")
    take_profit_distribution: Optional[List[float]] = Field(None, description="Distribution of position size across TP levels")

@dataclass
class TradingMetrics:
    total_trades: int
    successful_trades: int
    avg_profit: float
    avg_leverage: float
    avg_confidence: float
    win_rate: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    consecutive_losses: int

@dataclass
class TradingPatterns:
    quick_losses: int
    consecutive_wins: int
    high_leverage_fails: int
    sentiment_correlation: float
    technical_accuracy: float
    emotional_trades: int
    overtrading: bool
    risk_adjustment_needed: bool


# 1. 데이터베이스 연결 컨텍스트 매니저 구현
@contextmanager
def get_db_connection():
    """
    데이터베이스 연결을 위한 컨텍스트 매니저
    - 타임아웃 설정
    - WAL 모드 활성화
    - 행 객체를 딕셔너리로 반환
    - 예외 처리 및 자동 연결 해제
    
    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    conn = None
    try:
        conn = sqlite3.connect('bitcoin_trades.db', timeout=60.0)
        conn.row_factory = sqlite3.Row  # 첫 번째 함수에서 가져온 설정
        conn.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging 모드로 변경
        conn.execute('PRAGMA busy_timeout = 30000')  # 30초 타임아웃 설정
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"[오류] 데이터베이스 연결 오류: {e}")
        raise e
    finally:
        if conn:
            conn.commit()
            conn.close()            

# ✅ 바이비트 서명된 요청 함수
def send_signed_request(method, endpoint, params=None):
    if params is None:
        params = {}

    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"

    # 서명용 파라미터 구성
    params_for_sign = {
        **params,
        "apiKey": BYBIT_API_KEY,
        "timestamp": timestamp,
        "recvWindow": recv_window,
    }

    # 정렬 + 쿼리 스트링 구성
    sorted_params = dict(sorted(params_for_sign.items()))
    query_string = "&".join([f"{k}={v}" for k, v in sorted_params.items()])

    # 시그니처 생성
    signature = hmac.new(
        BYBIT_API_SECRET.encode(), 
        query_string.encode(), 
        hashlib.sha256
    ).hexdigest()

    # 최종 URL
    url = f"{BASE_URL}{endpoint}?{query_string}&sign={signature}"
    headers = {"Content-Type": "application/json"}

    # 요청 실행
    if method.upper() == "GET":
        response = requests.get(url, headers=headers)
    else:
        response = requests.post(url, headers=headers, json=params)

    return response.json()


def ensure_table_exists(conn, table_name, create_sql):
    """
    지정한 테이블이 존재하지 않으면 create_sql을 실행하여 테이블을 생성하는 함수.
    테이블 생성에 성공하면 True, 오류 발생 시 False를 반환합니다.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            cursor.execute(create_sql)
            conn.commit()
            print(f"[정보] {table_name} 테이블 생성 완료")
        return True
    except Exception as e:
        print(f"[오류] {table_name} 테이블 확인/생성 중 문제 발생: {e}")
        return False

if __name__ == '__main__':
    with get_db_connection() as conn:
        stop_loss_create_sql = '''
            CREATE TABLE IF NOT EXISTS stop_loss_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_type TEXT,
                entry_price REAL,
                stop_price REAL,
                set_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                liquidated INTEGER DEFAULT 0,
                liquidation_price REAL DEFAULT NULL,
                liquidation_time DATETIME DEFAULT NULL,
                profit_loss REAL DEFAULT NULL
            );
        '''
        # stop_loss_records 테이블이 존재하는지 확인하고 없으면 생성
        if ensure_table_exists(conn, 'stop_loss_records', stop_loss_create_sql):
            # 스탑로스 관련 작업 수행 (예시: 간단한 로그 출력)
            print("[정보] 스탑로스 관련 작업 수행")
            # 여기에 스탑로스 관련 작업 코드를 추가하세요.
        else:
            print("[오류] stop_loss_records 테이블 생성 실패")


# 2. 재시도 로직을 포함한 데이터베이스 작업 함수
def execute_with_retry(operation, max_retries=5, initial_delay=1.0):
    """
    데이터베이스 작업을 재시도하는 함수
    - 지수 백오프 방식 적용
    - database is locked 오류에 대한 특별 처리
    """
    retries = 0
    last_error = None
    
    while retries < max_retries:
        try:
            with get_db_connection() as conn:
                return operation(conn)
        except sqlite3.OperationalError as e:
            last_error = e
            if "database is locked" in str(e):
                retries += 1
                wait_time = initial_delay * (2 ** retries)  # 지수 백오프
                print(f"[정보] 데이터베이스 잠금 감지, {wait_time:.2f}초 후 재시도 ({retries}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            raise e
            
    print(f"[오류] 최대 재시도 횟수 초과: {last_error}")
    raise last_error

# === 추가: 누락된 함수들 ===
def get_orderbook_data(client: HTTP, symbol: str = SYMBOL, limit: int = 100) -> Optional[Dict]:
    """
    향상된 오더북 데이터 조회 함수
    - 더 많은 주문 데이터 요청 (기본값 100개)
    - 요청 실패 시 재시도 메커니즘 추가
    - 데이터 정제 및 구조화 강화
    - 주문량 집계 및 가격대별 분석 포함
    
    Args:
        client: Bybit API 클라이언트
        symbol: 거래 심볼 (기본값: BTCUSDT)
        limit: 최대 주문 수 (기본값: 100)
        
    Returns:
        Dict 또는 None: 구조화된 오더북 데이터
    """
    try:
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                response = client.get_orderbook(category="linear", symbol=symbol, limit=limit)
                
                if response["retCode"] == 0:
                    raw_data = response["result"]
                    
                    # 원시 데이터 추출 및 구조화
                    bids = [(float(bid[0]), float(bid[1])) for bid in raw_data["b"]]
                    asks = [(float(ask[0]), float(ask[1])) for ask in raw_data["a"]]
                    
                    # 추가: 가격대별 주문량 집계
                    bid_clusters = {}
                    ask_clusters = {}
                    price_step = 100  # $100 단위로 집계 (BTC 기준)
                    
                    for price, volume in bids:
                        cluster_price = int(price / price_step) * price_step
                        if cluster_price not in bid_clusters:
                            bid_clusters[cluster_price] = 0
                        bid_clusters[cluster_price] += volume
                    
                    for price, volume in asks:
                        cluster_price = int(price / price_step) * price_step
                        if cluster_price not in ask_clusters:
                            ask_clusters[cluster_price] = 0
                        ask_clusters[cluster_price] += volume
                    
                    # 주요 가격대 식별 (상위 5개)
                    top_bid_clusters = sorted(bid_clusters.items(), key=lambda x: x[1], reverse=True)[:5]
                    top_ask_clusters = sorted(ask_clusters.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    # 현재 스프레드 계산
                    best_bid = max([price for price, _ in bids]) if bids else 0
                    best_ask = min([price for price, _ in asks]) if asks else 0
                    spread = best_ask - best_bid if best_bid and best_ask else 0
                    spread_pct = (spread / best_bid * 100) if best_bid else 0
                    
                    # 오더북 뎁스 분석
                    bid_depth = {}
                    ask_depth = {}
                    depth_levels = [1, 2, 5, 10, 20, 50]  # 다양한 뎁스 레벨
                    
                    for level in depth_levels:
                        bid_depth[level] = sum(vol for _, vol in bids[:level]) if len(bids) >= level else 0
                        ask_depth[level] = sum(vol for _, vol in asks[:level]) if len(asks) >= level else 0
                    
                    # 결과 데이터 구성
                    result = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "bids": bids,
                        "asks": asks,
                        "bid_clusters": bid_clusters,
                        "ask_clusters": ask_clusters,
                        "top_bid_clusters": top_bid_clusters,
                        "top_ask_clusters": top_ask_clusters,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": spread,
                        "spread_pct": spread_pct,
                        "bid_depth": bid_depth,
                        "ask_depth": ask_depth
                    }
                    
                    # 리밋 정보 기록
                    print(f"[정보] 오더북 데이터 조회 완료 (Bids: {len(bids)}, Asks: {len(asks)})")
                    return result
                
                # 오류 시 로그 출력 및 재시도
                print(f"[경고] 오더북 조회 실패 (시도 {retry+1}/{max_retries}): {response.get('retMsg', '알 수 없는 오류')}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay * (retry + 1))  # 지수 백오프
                    continue
                return None
                
            except Exception as req_error:
                print(f"[경고] 오더북 요청 중 오류 발생 (시도 {retry+1}/{max_retries}): {req_error}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay * (retry + 1))
                    continue
                raise
                
    except Exception as e:
        print(f"[오류] 오더북 데이터 조회 중 문제 발생: {e}")
        print(traceback.format_exc())
        return None

def store_bybit_execution_data(symbol: str = SYMBOL) -> bool:
    """
    바이비트 API에서 체결 내역을 가져와 데이터베이스에 저장하는 함수.
    - 만약 'bybit_executions' 테이블이 없으면 생성합니다.
    - 각 체결 내역의 execution_id, price, quantity, fee, 실행 시간을 저장합니다.
    
    Args:
        symbol (str): 거래 심볼 (기본값: SYMBOL).
        
    Returns:
        bool: 저장 성공 여부.
    """
    try:
        executions = get_bybit_executions(bybit_client, symbol=symbol)
        if executions is None:
            logger.warning(f"[경고] 체결 내역 조회 실패 - DB 저장을 건너뜁니다. (심볼: {symbol})")
            return False

        with get_db_connection() as conn:
            cursor = conn.cursor()
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS bybit_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                execution_id TEXT,
                price REAL,
                quantity REAL,
                fee REAL,
                exec_time TEXT,
                inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            conn.commit()

            insert_sql = """
            INSERT INTO bybit_executions (symbol, execution_id, price, quantity, fee, exec_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            for exec_data in executions:
                execution_id = exec_data.get("orderId") or str(uuid.uuid4())
                price = float(exec_data.get("price", 0))
                quantity = float(exec_data.get("qty", 0))
                fee = float(exec_data.get("fee", 0))
                exec_time = exec_data.get("execTime")
                cursor.execute(insert_sql, (symbol, execution_id, price, quantity, fee, exec_time))
            conn.commit()
            logger.info(f"[정보] 바이비트 체결 내역 {len(executions)}건 DB 저장 완료 (심볼: {symbol}).")
        return True
    except Exception as e:
        logger.error(f"[오류] 바이비트 체결 데이터 DB 저장 중 문제 발생: {e}")
        return False



def calculate_actual_roi(executions: List[Dict], entry_price: float, position_type: str, leverage: int) -> Optional[float]:
    """
    바이비트 체결 내역 데이터를 기반으로 실제 수익률(ROI)을 계산하는 함수.
    각 체결 내역은 get_bybit_executions() 함수로 가져온 결과를 따르며, 
    'price', 'qty', 'fee' 등의 정보를 포함합니다.
    
    계산 방식:
      - 전체 체결 수량과 금액을 이용하여 가중 평균 체결 가격을 계산합니다.
      - 롱 포지션: ((평균 체결 가격 - 진입가) / 진입가 * 100 * 레버리지) - 총 수수료 비율
      - 숏 포지션: ((진입가 - 평균 체결 가격) / 진입가 * 100 * 레버리지) - 총 수수료 비율
      - 총 수수료 비율은 전체 수수료 합계를 (진입가 * 총 체결 수량)으로 나눈 후 100을 곱하여 계산합니다.
    
    Args:
        executions (List[Dict]): 체결 내역 리스트.
        entry_price (float): 거래 진입 가격.
        position_type (str): 포지션 유형 ("Long" 또는 "Short").
        leverage (int): 레버리지 배수.
    
    Returns:
        Optional[float]: 계산된 실제 수익률(%) 또는 None.
    """
    try:
        if not executions:
            logger.warning("[경고] 체결 내역이 없습니다. ROI 계산을 건너뜁니다.")
            return None
        
        total_qty = 0.0
        total_amount = 0.0
        total_fee = 0.0
        for exec_data in executions:
            qty = float(exec_data.get("qty", 0))
            price = float(exec_data.get("price", 0))
            fee = float(exec_data.get("fee", 0))
            total_qty += qty
            total_amount += price * qty
            total_fee += fee
        
        if total_qty == 0:
            logger.warning("[경고] 총 체결 수량이 0입니다. ROI 계산을 건너뜁니다.")
            return None
        
        avg_price = total_amount / total_qty
        fee_pct = (total_fee / (entry_price * total_qty)) * 100 if (entry_price * total_qty) != 0 else 0
        
        if position_type.lower() == "long":
            roi = ((avg_price - entry_price) / entry_price) * 100 * leverage
        elif position_type.lower() == "short":
            roi = ((entry_price - avg_price) / entry_price) * 100 * leverage
        else:
            logger.error(f"[오류] 알 수 없는 포지션 유형: {position_type}")
            return None
        
        net_roi = roi - fee_pct
        logger.info(f"[정보] ROI 계산 완료: 평균 체결가 = {avg_price:.2f}, 총 수수료율 = {fee_pct:.2f}%, 순수익률 = {net_roi:.2f}%")
        return net_roi
    except Exception as e:
        logger.error(f"[오류] ROI 계산 중 문제 발생: {e}")
        return None



def get_bybit_executions(client: HTTP, symbol: str = SYMBOL, limit: int = 50) -> Optional[List[Dict]]:
    """
    바이비트 체결 내역을 가져오는 함수.
    - 바이비트 API의 체결 내역(execution history)을 조회하여, 
      실제 체결 가격, 체결 수량, 발생 수수료 등의 정보를 반환합니다.
    - 요청 실패 시 None을 반환합니다.
    
    Args:
        client (HTTP): Bybit API 클라이언트.
        symbol (str): 조회할 거래 심볼 (기본값: SYMBOL 전역 변수 사용).
        limit (int): 조회할 최대 체결 내역 건수 (기본값: 50).
    
    Returns:
        Optional[List[Dict]]: 체결 내역 리스트 또는 None.
    """
    try:
        response = client.get_execution_history(category="linear", symbol=symbol, limit=limit)
        if response.get("retCode") == 0:
            executions = response["result"]["list"]
            logger.info(f"[정보] {len(executions)}개의 체결 내역 조회 완료 (심볼: {symbol}).")
            return executions
        else:
            logger.warning(f"[경고] 체결 내역 조회 실패 (심볼: {symbol}): {response.get('retMsg', '알 수 없는 오류')}.")
            return None
    except Exception as e:
        logger.error(f"[오류] 체결 내역 조회 중 문제 발생: {e}")
        return None



def sync_closed_trades_from_bybit(symbol: str = SYMBOL) -> bool:
    """
    바이비트 API에서 종료된 거래(Closed Trade) 정보를 조회하여,
    trades 테이블에 해당 거래의 종료 정보를 업데이트하는 함수.
    
    - 바이비트 API의 거래 내역(예: get_order_history)을 통해 종료된 거래 데이터를 조회합니다.
    - 각 거래에 대해 trade_id가 존재하면, 해당 거래의 상태를 'Closed'로, exit_price, profit_loss, 종료 시각을 업데이트합니다.
    
    Args:
        symbol (str): 거래 심볼 (기본값: SYMBOL).
        
    Returns:
        bool: 동기화 성공 여부.
    """
    try:
        response = bybit_client.get_order_history(category="linear", symbol=symbol)
        if response.get("retCode") != 0:
            logger.warning(f"[경고] 종료된 거래 내역 조회 실패 (심볼: {symbol}): {response.get('retMsg', '알 수 없는 오류')}")
            return False
        
        closed_trades = response["result"]["list"]
        if not closed_trades:
            logger.info("[정보] 종료된 거래 내역이 없습니다.")
            return True
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for trade in closed_trades:
                trade_id = trade.get("tradeId")
                if not trade_id:
                    logger.warning("[경고] tradeId가 없는 거래 내역 건 발견, 건너뜁니다.")
                    continue
                order_status = trade.get("orderStatus", "").lower()
                if order_status not in ["filled", "partiallyfilled"]:
                    continue
                exit_price = float(trade.get("avgPrice", trade.get("price", 0)))
                if exit_price == 0:
                    continue
                cursor.execute("SELECT entry_price FROM trades WHERE trade_id = ?", (trade_id,))
                record = cursor.fetchone()
                if not record:
                    logger.warning(f"[경고] DB에 존재하지 않는 거래 ID: {trade_id}, 건너뜁니다.")
                    continue
                entry_price = float(record["entry_price"])
                position_type = trade.get("side", "Long")
                if position_type.lower() == "long":
                    profit_loss = ((exit_price - entry_price) / entry_price) * 100
                else:
                    profit_loss = ((entry_price - exit_price) / entry_price) * 100
                update_sql = """
                    UPDATE trades
                    SET trade_status = 'Closed',
                        exit_price = ?,
                        profit_loss = ?,
                        timestamp = ?
                    WHERE trade_id = ?
                """
                closed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(update_sql, (exit_price, profit_loss, closed_time, trade_id))
            conn.commit()
            logger.info("[정보] 종료된 거래 내역과 trades 테이블 동기화 완료.")
        return True
    except Exception as e:
        logger.error(f"[오류] 종료된 거래 내역 동기화 중 문제 발생: {e}")
        return False



def cancel_all_open_orders(symbol: str = SYMBOL, position_idx: int = 1) -> bool:
    """
    바이비트 API를 이용해 지정된 심볼과 포지션 인덱스의 모든 오픈 주문을 취소하는 함수.
    
    - 현재 활성화된 모든 주문을 조회한 후, 각 주문에 대해 cancel_order()를 호출하여 취소합니다.
    
    Args:
        symbol (str): 거래 심볼 (기본값: SYMBOL).
        position_idx (int): 포지션 인덱스 (1: Long, 2: Short).
    
    Returns:
        bool: 모든 오픈 주문 취소 성공 여부.
    """
    try:
        response = bybit_client.get_open_orders(category="linear", symbol=symbol, positionIdx=position_idx)
        if response.get("retCode") != 0:
            logger.warning(f"[경고] 오픈 주문 조회 실패 (심볼: {symbol}): {response.get('retMsg', '알 수 없는 오류')}")
            return False
        
        orders = response["result"]["list"]
        if not orders:
            logger.info("[정보] 취소할 오픈 주문이 없습니다.")
            return True
        
        success = True
        for order in orders:
            order_id = order.get("orderId")
            if not order_id:
                continue
            cancel_response = bybit_client.cancel_order(category="linear", symbol=symbol, orderId=order_id)
            if cancel_response.get("retCode") != 0:
                logger.warning(f"[경고] 주문 {order_id} 취소 실패: {cancel_response.get('retMsg', '알 수 없는 오류')}")
                success = False
            else:
                logger.info(f"[정보] 주문 {order_id} 취소 성공.")
        return success
    except Exception as e:
        logger.error(f"[오류] 오픈 주문 취소 중 문제 발생: {e}")
        return False



def analyze_market_pressure(orderbook_data: Dict) -> Optional[Dict]:
    """
    향상된 오더북 기반 매수/매도 압력 분석 함수
    - 다양한 뎁스 레벨에서의 압력 분석
    - 가격대별 주문량 집중도 평가
    - 주문 불균형 지표 계산
    - 시장 흐름 예측 지표 제공
    - 저항/지지 레벨 강도 평가
    
    Args:
        orderbook_data: get_orderbook_data 함수에서 반환된 오더북 데이터
        
    Returns:
        Dict 또는 None: 시장 압력 분석 결과
    """
    try:
        if not orderbook_data:
            print("[경고] 오더북 데이터가 없습니다.")
            return None
        
        # 기본 매수/매도 압력 계산 (전체 볼륨 기준)
        total_bid_volume = sum(float(bid[1]) for bid in orderbook_data["bids"])
        total_ask_volume = sum(float(ask[1]) for ask in orderbook_data["asks"])
        
        # 무한대 방지를 위한 안전 장치
        pressure_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else float('inf')
        
        # 가중치 적용된 매수/매도 압력 계산 (가격에 가까울수록 더 중요)
        weighted_bids = sum(float(bid[1]) * (1.0 - (0.01 * idx)) for idx, bid in enumerate(orderbook_data["bids"][:20]))
        weighted_asks = sum(float(ask[1]) * (1.0 - (0.01 * idx)) for idx, ask in enumerate(orderbook_data["asks"][:20]))
        weighted_pressure_ratio = weighted_bids / weighted_asks if weighted_asks > 0 else float('inf')
        
        # 다양한 뎁스 레벨에서의 압력 비율
        depth_pressure = {}
        for level in orderbook_data["bid_depth"].keys():
            bid_vol = orderbook_data["bid_depth"][level]
            ask_vol = orderbook_data["ask_depth"][level]
            ratio = bid_vol / ask_vol if ask_vol > 0 else float('inf')
            depth_pressure[level] = ratio
        
        # 클러스터 분석 - 주요 저항/지지 레벨 식별
        resistance_levels = []
        support_levels = []
        
        # 상위 매도 클러스터를 저항 레벨로 해석
        for price, volume in orderbook_data["top_ask_clusters"]:
            volume_ratio = volume / total_ask_volume if total_ask_volume > 0 else 0
            strength = min(1.0, volume_ratio * 3)  # 0-1 범위로 정규화
            resistance_levels.append({"price": price, "strength": strength})
        
        # 상위 매수 클러스터를 지지 레벨로 해석
        for price, volume in orderbook_data["top_bid_clusters"]:
            volume_ratio = volume / total_bid_volume if total_bid_volume > 0 else 0
            strength = min(1.0, volume_ratio * 3)
            support_levels.append({"price": price, "strength": strength})
        
        # 불균형 지표 - 매수/매도 사이드 압력 불균형 정도
        imbalance = abs(total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
        
        # 스프레드 분석 - 넓은 스프레드는 시장 불확실성 의미
        spread = orderbook_data["spread"]
        spread_pct = orderbook_data["spread_pct"]
        
        # 매수/매도벽 분석 - 상위 5개 주문의 집중도
        top5_bid_concentration = sum(vol for _, vol in orderbook_data["bids"][:5]) / total_bid_volume if total_bid_volume > 0 else 0
        top5_ask_concentration = sum(vol for _, vol in orderbook_data["asks"][:5]) / total_ask_volume if total_ask_volume > 0 else 0
        
        # 시장 흐름 예측 지표
        bullish_signals = 0
        bearish_signals = 0
        
        # 매수 압력이 매도 압력보다 1.2배 이상 크면 강세 신호
        if pressure_ratio > 1.2:
            bullish_signals += 1
        # 매도 압력이 매수 압력보다 1.2배 이상 크면 약세 신호
        elif pressure_ratio < 0.8:
            bearish_signals += 1
            
        # 가중 압력 비율도 분석
        if weighted_pressure_ratio > 1.2:
            bullish_signals += 1
        elif weighted_pressure_ratio < 0.8:
            bearish_signals += 1
            
        # 깊은 뎁스에서도 확인 (20레벨)
        if depth_pressure.get(20, 1.0) > 1.2:
            bullish_signals += 1
        elif depth_pressure.get(20, 1.0) < 0.8:
            bearish_signals += 1
        
        # 불균형 정도가 높을 때 신호 강화
        if imbalance > 0.3:  # 30% 이상 불균형
            if total_bid_volume > total_ask_volume:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # 결론 도출
        market_bias = "neutral"
        confidence = 0.5  # 기본 확신도
        
        if bullish_signals >= 3:
            market_bias = "strongly_bullish"
            confidence = 0.9
        elif bullish_signals >= 2:
            market_bias = "bullish"
            confidence = 0.7
        elif bearish_signals >= 3:
            market_bias = "strongly_bearish"
            confidence = 0.9
        elif bearish_signals >= 2:
            market_bias = "bearish"
            confidence = 0.7
        
        # 결과 구성
        result = {
            "buy_pressure": total_bid_volume,
            "sell_pressure": total_ask_volume,
            "pressure_ratio": pressure_ratio,
            "weighted_pressure_ratio": weighted_pressure_ratio,
            "depth_pressure": depth_pressure,
            "imbalance": imbalance,
            "spread": spread,
            "spread_pct": spread_pct,
            "resistance_levels": resistance_levels,
            "support_levels": support_levels,
            "top_bid_concentration": top5_bid_concentration,
            "top_ask_concentration": top5_ask_concentration,
            "market_bias": market_bias,
            "confidence": confidence,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals
        }
        
        # 디버깅을 위한 로그 출력
        print(f"[정보] 시장 압력 분석 완료: {market_bias} (확신도: {confidence:.2f})")
        print(f"[정보] 매수/매도 비율: {pressure_ratio:.2f}, 불균형: {imbalance:.2f}")
        
        return result
        
    except Exception as e:
        print(f"[오류] 시장 압력 분석 중 문제 발생: {e}")
        print(traceback.format_exc())
        return None

def analyze_resistance_strength(market_data: Dict, orderbook_data: Dict = None, position_type: str = None) -> Dict:
    """
    저항/지지 레벨의 강도를 종합적으로 분석하는 함수
    - 오더북 데이터와 기술적 지표를 결합한 분석
    - 가격대별 저항/지지 강도 점수화
    - 거래량 프로필 기반 분석
    - 역사적 가격 반응 고려
    - 심리적 가격대 분석
    
    Args:
        market_data: 시장 기술적 지표 데이터
        orderbook_data: 오더북 분석 데이터 (선택적)
        position_type: 분석 방향 지정 ("Long" 또는 "Short") (선택적)
        
    Returns:
        Dict: 저항/지지 레벨 강도 분석 결과
    """
    try:
        current_price = float(market_data['timeframes']["15"].iloc[-1]['Close'])
        
        # 기술적 지표 기반 저항/지지 레벨 식별
        df_15m = market_data['timeframes']["15"]
        df_1h = market_data['timeframes'].get("60", df_15m)
        df_4h = market_data['timeframes'].get("240", df_15m)
        
        resistance_levels = []
        support_levels = []
        
        # 1. 이동평균선 기반 레벨
        ema_levels = []
        for tf, df, weight in [("15m", df_15m, 0.6), ("1h", df_1h, 0.8), ("4h", df_4h, 1.0)]:
            latest = df.iloc[-1]
            
            # 주요 이동평균선
            for ma_name, strength_base in [
                ('EMA_20', 0.5), ('EMA_50', 0.6), ('EMA_100', 0.7), ('EMA_200', 0.8)
            ]:
                if ma_name in latest:
                    ma_value = float(latest[ma_name])
                    
                    # 저항 또는 지지 판단
                    if ma_value > current_price:
                        distance_pct = (ma_value - current_price) / current_price * 100
                        # 가까울수록 더 중요한 저항
                        strength = strength_base * weight * (1.0 - min(1.0, distance_pct / 5.0))
                        resistance_levels.append({
                            "price": ma_value,
                            "strength": min(1.0, strength),
                            "type": "MA",
                            "source": f"{ma_name}_{tf}"
                        })
                    else:
                        distance_pct = (current_price - ma_value) / ma_value * 100
                        # 가까울수록 더 중요한 지지
                        strength = strength_base * weight * (1.0 - min(1.0, distance_pct / 5.0))
                        support_levels.append({
                            "price": ma_value,
                            "strength": min(1.0, strength),
                            "type": "MA",
                            "source": f"{ma_name}_{tf}"
                        })
        
        # 2. 볼린저 밴드 기반 레벨
        for tf, df, weight in [("15m", df_15m, 0.7), ("1h", df_1h, 0.9)]:
            latest = df.iloc[-1]
            
            # 볼린저 상단
            if 'BB_upper' in latest:
                bb_upper = float(latest['BB_upper'])
                if bb_upper > current_price:
                    distance_pct = (bb_upper - current_price) / current_price * 100
                    strength = 0.7 * weight * (1.0 - min(1.0, distance_pct / 3.0))
                    resistance_levels.append({
                        "price": bb_upper,
                        "strength": min(1.0, strength),
                        "type": "BB",
                        "source": f"BB_upper_{tf}"
                    })
            
            # 볼린저 하단
            if 'BB_lower' in latest:
                bb_lower = float(latest['BB_lower'])
                if bb_lower < current_price:
                    distance_pct = (current_price - bb_lower) / bb_lower * 100
                    strength = 0.7 * weight * (1.0 - min(1.0, distance_pct / 3.0))
                    support_levels.append({
                        "price": bb_lower,
                        "strength": min(1.0, strength),
                        "type": "BB",
                        "source": f"BB_lower_{tf}"
                    })
        
        # 3. 피보나치 레벨
        fib_strength_map = {
            'Fibo_0.236': 0.5, 'Fibo_0.382': 0.65, 'Fibo_0.5': 0.7,
            'Fibo_0.618': 0.8, 'Fibo_0.786': 0.75, 'Fibo_1': 0.6
        }
        
        for tf, df, weight in [("1h", df_1h, 0.8), ("4h", df_4h, 1.0)]:
            latest = df.iloc[-1]
            
            for fib_name, strength_base in fib_strength_map.items():
                if fib_name in latest:
                    fib_value = float(latest[fib_name])
                    
                    if fib_value > current_price:
                        distance_pct = (fib_value - current_price) / current_price * 100
                        strength = strength_base * weight * (1.0 - min(1.0, distance_pct / 5.0))
                        resistance_levels.append({
                            "price": fib_value,
                            "strength": min(1.0, strength),
                            "type": "Fib",
                            "source": f"{fib_name}_{tf}"
                        })
                    else:
                        distance_pct = (current_price - fib_value) / fib_value * 100
                        strength = strength_base * weight * (1.0 - min(1.0, distance_pct / 5.0))
                        support_levels.append({
                            "price": fib_value,
                            "strength": min(1.0, strength),
                            "type": "Fib",
                            "source": f"{fib_name}_{tf}"
                        })
        
        # 4. 도치안 채널
        for tf, df, weight in [("15m", df_15m, 0.6), ("1h", df_1h, 0.9)]:
            latest = df.iloc[-1]
            
            if 'Donchian_High' in latest:
                don_high = float(latest['Donchian_High'])
                if don_high > current_price:
                    distance_pct = (don_high - current_price) / current_price * 100
                    strength = 0.8 * weight * (1.0 - min(1.0, distance_pct / 4.0))
                    resistance_levels.append({
                        "price": don_high,
                        "strength": min(1.0, strength),
                        "type": "Donchian",
                        "source": f"Donchian_High_{tf}"
                    })
            
            if 'Donchian_Low' in latest:
                don_low = float(latest['Donchian_Low'])
                if don_low < current_price:
                    distance_pct = (current_price - don_low) / don_low * 100
                    strength = 0.8 * weight * (1.0 - min(1.0, distance_pct / 4.0))
                    support_levels.append({
                        "price": don_low,
                        "strength": min(1.0, strength),
                        "type": "Donchian",
                        "source": f"Donchian_Low_{tf}"
                    })
        
        # 5. 피벗 포인트 및 기타 레벨
        for tf, df, weight in [("1h", df_1h, 0.8)]:
            latest = df.iloc[-1]
            
            # 피벗, 저항, 지지 레벨
            for level_type, field, base_strength in [
                ("Resistance", "Resistance1", 0.7),
                ("Resistance", "Resistance2", 0.6),
                ("Support", "Support1", 0.7),
                ("Support", "Support2", 0.6),
                ("Pivot", "Pivot", 0.65)
            ]:
                if field in latest:
                    level_value = float(latest[field])
                    
                    if level_type in ["Resistance", "Pivot"] and level_value > current_price:
                        distance_pct = (level_value - current_price) / current_price * 100
                        strength = base_strength * weight * (1.0 - min(1.0, distance_pct / 5.0))
                        resistance_levels.append({
                            "price": level_value,
                            "strength": min(1.0, strength),
                            "type": level_type,
                            "source": f"{field}_{tf}"
                        })
                    elif level_type in ["Support", "Pivot"] and level_value < current_price:
                        distance_pct = (current_price - level_value) / level_value * 100
                        strength = base_strength * weight * (1.0 - min(1.0, distance_pct / 5.0))
                        support_levels.append({
                            "price": level_value,
                            "strength": min(1.0, strength),
                            "type": level_type,
                            "source": f"{field}_{tf}"
                        })
        
        # 6. 오더북 데이터와 통합 (있는 경우)
        if orderbook_data and "resistance_levels" in orderbook_data:
            for level in orderbook_data["resistance_levels"]:
                resistance_levels.append({
                    "price": level["price"],
                    "strength": level["strength"] * 0.9,  # 약간 가중치 조정
                    "type": "Orderbook",
                    "source": "orderbook_cluster"
                })
        
        if orderbook_data and "support_levels" in orderbook_data:
            for level in orderbook_data["support_levels"]:
                support_levels.append({
                    "price": level["price"],
                    "strength": level["strength"] * 0.9,
                    "type": "Orderbook",
                    "source": "orderbook_cluster"
                })
        
        # 7. 심리적 레벨 추가 (천 단위, 오백 단위)
        current_price_thousands = int(current_price / 1000) * 1000
        
        # 상단 심리적 레벨
        for i in range(1, 4):  # 다음 3개 천 단위
            psych_price = current_price_thousands + (i * 1000)
            if psych_price > current_price:
                distance_pct = (psych_price - current_price) / current_price * 100
                # 거리에 따른 강도 조정, 가까울수록 강함
                base_strength = 0.85 if i == 1 else (0.75 if i == 2 else 0.65)
                strength = base_strength * (1.0 - min(1.0, distance_pct / 10.0))
                resistance_levels.append({
                    "price": psych_price,
                    "strength": min(1.0, strength),
                    "type": "Psychological",
                    "source": f"thousands_{psych_price}"
                })
        
        # 상단 500 단위
        for i in range(1, 6):  # 다음 5개 500 단위
            psych_price = current_price_thousands + (i * 500)
            if psych_price > current_price and psych_price % 1000 != 0:  # 천 단위와 중복 방지
                distance_pct = (psych_price - current_price) / current_price * 100
                # 500 단위는 천 단위보다 약한 심리적 영향
                base_strength = 0.65
                strength = base_strength * (1.0 - min(1.0, distance_pct / 8.0))
                resistance_levels.append({
                    "price": psych_price,
                    "strength": min(1.0, strength),
                    "type": "Psychological",
                    "source": f"five_hundreds_{psych_price}"
                })
        
        # 하단 심리적 레벨 (천 단위)
        for i in range(0, 3):  # 현재 및 이전 2개 천 단위
            psych_price = current_price_thousands - (i * 1000)
            if psych_price < current_price:
                distance_pct = (current_price - psych_price) / psych_price * 100
                base_strength = 0.85 if i == 0 else (0.75 if i == 1 else 0.65)
                strength = base_strength * (1.0 - min(1.0, distance_pct / 10.0))
                support_levels.append({
                    "price": psych_price,
                    "strength": min(1.0, strength),
                    "type": "Psychological",
                    "source": f"thousands_{psych_price}"
                })
        
        # 하단 500 단위
        for i in range(1, 6):
            psych_price = current_price_thousands - (i * 500)
            if psych_price < current_price and psych_price % 1000 != 0:
                distance_pct = (current_price - psych_price) / psych_price * 100
                base_strength = 0.65
                strength = base_strength * (1.0 - min(1.0, distance_pct / 8.0))
                support_levels.append({
                    "price": psych_price,
                    "strength": min(1.0, strength),
                    "type": "Psychological",
                    "source": f"five_hundreds_{psych_price}"
                })
        
        # 8. 가격 군집화 및 병합
        # 비슷한 가격대의 저항/지지 병합 (0.5% 이내)
        merged_resistance = []
        merged_support = []
        
        # 저항 병합
        resistance_sorted = sorted(resistance_levels, key=lambda x: x["price"])
        i = 0
        while i < len(resistance_sorted):
            current_level = resistance_sorted[i]
            current_price = current_level["price"]
            similar_levels = [current_level]
            
            j = i + 1
            while j < len(resistance_sorted) and abs(resistance_sorted[j]["price"] - current_price) / current_price < 0.005:
                similar_levels.append(resistance_sorted[j])
                j += 1
            
            # 여러 레벨이 비슷한 가격대에 있으면 강도 증가
            if len(similar_levels) > 1:
                avg_price = sum(level["price"] for level in similar_levels) / len(similar_levels)
                # 최대 강도는 더 높아질 수 있지만 1.0 제한
                base_strength = max(level["strength"] for level in similar_levels)
                combined_strength = min(1.0, base_strength + 0.1 * (len(similar_levels) - 1))
                
                source_types = ", ".join(set(level["type"] for level in similar_levels))
                merged_resistance.append({
                    "price": avg_price,
                    "strength": combined_strength,
                    "type": "Combined",
                    "source": source_types,
                    "count": len(similar_levels)
                })
            else:
                merged_resistance.append(current_level)
            
            i = j
        
        # 지지 병합 (저항과 동일한 로직)
        support_sorted = sorted(support_levels, key=lambda x: x["price"], reverse=True)
        i = 0
        while i < len(support_sorted):
            current_level = support_sorted[i]
            current_price = current_level["price"]
            similar_levels = [current_level]
            
            j = i + 1
            while j < len(support_sorted) and abs(support_sorted[j]["price"] - current_price) / current_price < 0.005:
                similar_levels.append(support_sorted[j])
                j += 1
            
            if len(similar_levels) > 1:
                avg_price = sum(level["price"] for level in similar_levels) / len(similar_levels)
                base_strength = max(level["strength"] for level in similar_levels)
                combined_strength = min(1.0, base_strength + 0.1 * (len(similar_levels) - 1))
                
                source_types = ", ".join(set(level["type"] for level in similar_levels))
                merged_support.append({
                    "price": avg_price,
                    "strength": combined_strength,
                    "type": "Combined",
                    "source": source_types,
                    "count": len(similar_levels)
                })
            else:
                merged_support.append(current_level)
            
            i = j
        
        # 9. 강도별 정렬 및 최종 결과 선택
        # 강도가 높은 순으로 정렬
        merged_resistance = sorted(merged_resistance, key=lambda x: x["strength"], reverse=True)
        merged_support = sorted(merged_support, key=lambda x: x["strength"], reverse=True)
        
        # 상위 레벨만 선택 (최대 10개)
        top_resistance = merged_resistance[:10]
        top_support = merged_support[:10]
        
        # 가격 순서로 재정렬
        top_resistance = sorted(top_resistance, key=lambda x: x["price"])
        top_support = sorted(top_support, key=lambda x: x["price"], reverse=True)
        
        # 근접한 주요 저항/지지 레벨 식별
        nearby_resistance = []
        nearby_support = []
        
        for level in top_resistance:
            distance_pct = (level["price"] - current_price) / current_price * 100
            if distance_pct < 5.0:  # 5% 이내 저항
                nearby_resistance.append(level)
        
        for level in top_support:
            distance_pct = (current_price - level["price"]) / level["price"] * 100
            if distance_pct < 5.0:  # 5% 이내 지지
                nearby_support.append(level)
        
        # 10. 포지션 타입에 따른 최적 레벨 선택
        optimal_levels = []
        if position_type == "Long":
            # 롱 포지션 - 위쪽 저항 레벨 중요
            for level in nearby_resistance[:3]:  # 상위 3개
                optimal_levels.append({
                    "price": level["price"],
                    "strength": level["strength"],
                    "distance_pct": (level["price"] - current_price) / current_price * 100,
                    "type": "resistance"
                })
        elif position_type == "Short":
            # 숏 포지션 - 아래쪽 지지 레벨 중요
            for level in nearby_support[:3]:  # 상위 3개
                optimal_levels.append({
                    "price": level["price"],
                    "strength": level["strength"],
                    "distance_pct": (current_price - level["price"]) / level["price"] * 100,
                    "type": "support"
                })
        
        # 결과 로깅
        print(f"[정보] 저항/지지 레벨 분석 완료 - 저항: {len(top_resistance)}개, 지지: {len(top_support)}개")
        print(f"[정보] 현재 가격 ${current_price:.2f} 근처의 주요 레벨:")
        
        for i, level in enumerate(nearby_resistance[:3], 1):
            print(f"  저항 {i}: ${level['price']:.2f} (강도: {level['strength']:.2f}, 출처: {level['source']})")
        
        for i, level in enumerate(nearby_support[:3], 1):
            print(f"  지지 {i}: ${level['price']:.2f} (강도: {level['strength']:.2f}, 출처: {level['source']})")
        
        # 최종 결과 반환
        result = {
            "current_price": current_price,
            "resistance_levels": top_resistance,
            "support_levels": top_support,
            "nearby_resistance": nearby_resistance,
            "nearby_support": nearby_support,
            "optimal_levels": optimal_levels if position_type else []
        }
        
        return result
        
    except Exception as e:
        print(f"[오류] 저항/지지 레벨 강도 분석 중 문제 발생: {e}")
        print(traceback.format_exc())
        
        # 오류 발생 시 기본 빈 결과 반환
        return {
            "current_price": 0.0,
            "resistance_levels": [],
            "support_levels": [],
            "nearby_resistance": [],
            "nearby_support": [],
            "optimal_levels": []
        }

def check_tp_orders(symbol=SYMBOL):
    """현재 설정된 TP(Take Profit) 주문 확인"""
    try:
        for position_idx in [1, 2]:  # 1: Long, 2: Short
            orders = bybit_client.get_open_orders(
                category="linear",
                symbol=symbol,
                positionIdx=position_idx,
                orderFilter="StopOrder"
            )
            
            if orders["retCode"] == 0:
                tp_orders = [order for order in orders["result"]["list"] 
                           if order["stopOrderType"] == "TakeProfit"]
                
                position_type = "Long" if position_idx == 1 else "Short"
                if tp_orders:
                    print(f"\n=== {position_type} 포지션 익절 주문 ({len(tp_orders)}개) ===")
                    for i, order in enumerate(tp_orders):
                        print(f"TP {i+1}: 가격 ${float(order['triggerPrice']):,.2f}, "
                              f"수량: {float(order.get('qty', 0)):,.3f} BTC")
                else:
                    print(f"\n{position_type} 포지션에 설정된 익절 주문이 없습니다.")
        return True
    except Exception as e:
        print(f"[오류] TP 주문 확인 중 문제 발생: {e}")
        return False

def check_sideways_strategy(df: pd.DataFrame) -> str:
    """
    개선된 횡보장 전략
    - RSI와 거래량도 함께 확인
    - ADX로 추세 강도 검증
    - 여러 지표의 일치성 확인
    """
    try:
        latest = df.iloc[-1]
        upper = float(latest['BB_upper'])
        lower = float(latest['BB_lower'])
        mid = float(latest['BB_mavg'])
        current = float(latest['Close'])
        range_width = upper - lower
        
        # 추가 지표 검증
        rsi = float(latest['RSI'])
        adx = float(latest['ADX'])
        volume = float(latest['Volume'])
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        
        # 횡보장 조건 검증 (ADX가 낮을 때)
        if adx < 25:  # 약한 추세
            if current <= lower + 0.1 * range_width:
                # 롱 진입 조건 강화
                if rsi < 40 and volume > avg_volume * 1.2:
                    return "Long"
            elif current >= upper - 0.1 * range_width:
                # 숏 진입 조건 강화
                if rsi > 60 and volume > avg_volume * 1.2:
                    return "Short"
        return "Hold"
    except Exception as e:
        print(f"[오류] check_sideways_strategy 실행 중 문제 발생: {e}")
        return "Hold"

def check_donchian_strategy(df: pd.DataFrame) -> str:
    """
    돈치안 채널 기반 횡보장 전략
    - 채널 돌파 신호와 RSI를 결합
    - 추세 강도 검증에 ADX 활용
    - 볼륨 확인으로 신호 검증
    """
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 돈치안 채널 값
        donchian_high = float(latest['Donchian_High'])
        donchian_low = float(latest['Donchian_Low'])
        donchian_mid = float(latest['Donchian_Mid'])
        current = float(latest['Close'])
        
        # 추가 지표 검증
        rsi = float(latest['RSI'])
        adx = float(latest['ADX'])
        volume = float(latest['Volume'])
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        
        # 변동성 계산
        channel_width = (donchian_high - donchian_low) / donchian_mid * 100
        
        # 횡보장 조건 검증 (ADX가 낮을 때)
        if adx < 25:  # 약한 추세
            # 채널 하단 근처에서 반등
            if current <= donchian_low * 1.01:
                # 롱 진입 조건 강화
                if rsi < 40 and volume > avg_volume * 1.2:
                    return "Long"
            # 채널 상단 근처에서 하락
            elif current >= donchian_high * 0.99:
                # 숏 진입 조건 강화
                if rsi > 60 and volume > avg_volume * 1.2:
                    return "Short"
            # 채널 중앙 부근에서 추가 조건
            elif donchian_mid * 0.98 <= current <= donchian_mid * 1.02:
                # 방향성 파악
                if current > prev['Close'] and rsi > 50 and rsi < 70:
                    return "Long"
                elif current < prev['Close'] and rsi < 50 and rsi > 30:
                    return "Short"
        # 강한 추세 (ADX가 높을 때) - 채널 돌파 트레이딩
        elif adx > 30:
            # 채널 상단 돌파
            if current > donchian_high and current > prev['Close']:
                if volume > avg_volume * 1.5:  # 볼륨 확인
                    return "Long"
            # 채널 하단 돌파
            elif current < donchian_low and current < prev['Close']:
                if volume > avg_volume * 1.5:  # 볼륨 확인
                    return "Short"
        
        return "Hold"
    except Exception as e:
        print(f"[오류] check_donchian_strategy 실행 중 문제 발생: {e}")
        return "Hold"

def get_final_trading_decision(market_data: Dict) -> TradingDecision:
    try:
        overall_trend = check_multiframe_trend(market_data['timeframes'])
        latest_data = market_data['timeframes']["15"].iloc[-1]
        volatility = latest_data['ATR'] / latest_data['Close'] * 100
        
        if overall_trend in ['up', 'down']:
            return get_ai_trading_decision(market_data)
        else:
            # 횡보장 전략 적용 (돈치안 채널 기반)
            sideways_signal = check_donchian_strategy(market_data['timeframes']["15"])
            
            # 변동성에 따른 포지션 크기 조정
            position_size = 20
            if volatility > 1.5:
                position_size = 15
            elif volatility > 2.0:
                position_size = 10
            
            # 추가: 횡보장일 때 추가 확인
            # 1. 볼륨 확인 - 평균보다 1.5배 이상일 때만 고려
            avg_volume = market_data['timeframes']["15"]['Volume'].rolling(20).mean().iloc[-1]
            current_volume = latest_data['Volume']
            volume_confirmed = current_volume > avg_volume * 1.5
            
            # 2. RSI 극단값 확인
            rsi_extreme = (latest_data['RSI'] < 30 and sideways_signal == "Long") or \
                           (latest_data['RSI'] > 70 and sideways_signal == "Short")
                
            # 횡보장에서는 두 가지 조건을 모두 만족할 때만 진입
            if sideways_signal in ["Long", "Short"] and (volume_confirmed or rsi_extreme):
                print(f"[정보] 횡보장 진입 조건 충족 - 볼륨: {volume_confirmed}, RSI 극단: {rsi_extreme}")
                direction = sideways_signal
                reason = "돈치안 채널 신호 + " + \
                        ("높은 볼륨" if volume_confirmed else "") + \
                        (" 및 " if volume_confirmed and rsi_extreme else "") + \
                        ("극단 RSI" if rsi_extreme else "")
                
                return TradingDecision(
                    decision=direction,
                    reason=reason,
                    position_size=position_size,
                    leverage=2,  # 횡보장에서는 낮은 레버리지
                    stop_loss_pct=0.8  # 좁은 스탑로스
                )
            else:
                if sideways_signal != "Hold":
                    print(f"[정보] 횡보장 진입 조건 불충족 - 추가 필터에 의해 홀드 결정됨")
                    print(f"볼륨 확인: {volume_confirmed}, RSI 극단값: {rsi_extreme}")
                
                return TradingDecision(
                    decision="Hold",
                    reason="횡보장 조건 불충족으로 홀드",
                    position_size=0,
                    leverage=None,
                    stop_loss_pct=0.1
                )
    except Exception as e:
        print(f"[오류] get_final_trading_decision 실행 중 문제 발생: {e}")
        return TradingDecision(
            decision="Hold",
            reason="오류 발생으로 인한 홀드",
            position_size=0,
            leverage=None,
            stop_loss_pct=0.1
        )

def set_leverage(symbol: str = SYMBOL, leverage: int = 1) -> bool:
    """레버리지 설정 (기존 값과 같으면 변경하지 않음)"""
    try:
        # 레버리지 범위 제한
        leverage = max(1, min(20, leverage))
        print(f"\n레버리지 설정 시도: {leverage}x")
        
        # 현재 레버리지 확인
        current_position = bybit_client.get_positions(category="linear", symbol=symbol)
        if isinstance(current_position.get("retCode"), str):
            current_ret_code = int(current_position["retCode"])
        else:
            current_ret_code = current_position["retCode"]
            
        if current_ret_code == 0:
            positions = current_position["result"]["list"]
            if positions:
                try:
                    current_leverage = float(positions[0]["leverage"])
                    if current_leverage == leverage:
                        print(f"[정보] 이미 레버리지가 {leverage}x로 설정되어 있습니다.")
                        return True
                except (ValueError, KeyError) as e:
                    print(f"[경고] 현재 레버리지 확인 중 오류: {e}")
        
        # 레버리지 설정
        response = bybit_client.set_leverage(
            category="linear",
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage)
        )
        
        # retCode 타입 검사 및 변환
        if isinstance(response.get("retCode"), str):
            ret_code = int(response["retCode"])
        else:
            ret_code = response["retCode"]
        
        # 응답 검증
        if ret_code == 0 or ret_code == 110043:
            print(f"[정보] 레버리지 {leverage}x 설정 완료")
            return True
        else:
            error_msg = response.get("retMsg", "알 수 없는 오류")
            print(f"[경고] 레버리지 설정 실패: {error_msg}")
            return False
            
    except Exception as e:
        print(f"[오류] 레버리지 설정 중 문제 발생: {e}")
        print(f"상세 오류: {str(e)}")
        return False


# 7. 데이터베이스 초기화 함수
def init_db():
    """데이터베이스 초기화 및 테이블 생성"""
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        c = conn.cursor()
        
        # 기존 trades 테이블 생성
        c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            current_price REAL,
            decision TEXT,
            reason TEXT,
            position_size INTEGER,
            leverage INTEGER,
            wallet_balance REAL,
            estimated_fee REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            ema_20 REAL,
            ema_50 REAL,
            bb_upper REAL,
            bb_lower REAL,
            fear_greed_value TEXT,
            success INTEGER,
            reflection TEXT,
            position_type TEXT,
            entry_price REAL,
            exit_price REAL,
            trade_status TEXT,
            profit_loss REAL,
            trade_id TEXT,
            ai_confidence_score REAL,
            position_duration INTEGER,
            entry_conditions TEXT,
            exit_conditions TEXT,
            social_sentiment TEXT,
            news_keywords TEXT,
            trading_analysis_ai TEXT,
            funding_fee REAL,
            total_fee REAL,
            roi_percentage REAL,
            stop_loss_price REAL,
            is_stop_loss INTEGER DEFAULT 0,
            stop_loss_pct REAL
        )
        ''')

        # 새로운 trading_summary 테이블 생성
        c.execute('''
        CREATE TABLE IF NOT EXISTS trading_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            
            -- 포트폴리오 성과
            initial_balance REAL,
            current_balance REAL,
            total_return_pct REAL,
            
            -- 거래 통계
            total_trades INTEGER,
            win_rate REAL,
            avg_profit_pct REAL,
            max_profit_pct REAL,
            max_loss_pct REAL,
            avg_leverage REAL,
            
            -- 거래 패턴
            max_consecutive_streak INTEGER,
            current_streak INTEGER,
            trades_per_hour REAL,
            
            -- 레버리지 분석
            high_leverage_trades INTEGER,
            medium_leverage_trades INTEGER,
            low_leverage_trades INTEGER,
            high_leverage_winrate REAL,
            medium_leverage_winrate REAL,
            low_leverage_winrate REAL,
            
            -- AI 분석
            ai_analysis TEXT,
            strategy_suggestions TEXT
        )
        ''')

        # 새로운 stop_loss_records 테이블 생성 (테스트 코드에서 가져옴)
        c.execute('''
        CREATE TABLE IF NOT EXISTS stop_loss_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_type TEXT,
            entry_price REAL,
            stop_price REAL,
            set_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            liquidated INTEGER DEFAULT 0,
            liquidation_price REAL DEFAULT NULL,
            liquidation_time DATETIME DEFAULT NULL,
            profit_loss REAL DEFAULT NULL
        );
        ''')
        
        # 새로운 take_profit_orders 테이블 생성
        c.execute('''
        CREATE TABLE IF NOT EXISTS take_profit_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT,
            position_type TEXT,
            tp_level INTEGER,
            tp_price REAL,
            tp_quantity REAL,
            status TEXT DEFAULT 'Active',
            created_at TEXT,
            executed_at TEXT,
            execution_price REAL,
            profit_pct REAL,
            order_id TEXT
        )
        ''')

        conn.commit()
        print("[정보] 데이터베이스 테이블이 성공적으로 생성되었습니다.")
        return conn
    except Exception as e:
        print(f"[오류] 데이터베이스 초기화 중 문제 발생: {e}")
        return None



def reset_take_profit_orders(conn, trade_id: str) -> bool:
    """
    지정된 trade_id에 대한 모든 익절 주문을 초기화(취소)하는 함수.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE take_profit_orders
            SET status = 'Reset'
            WHERE trade_id = ? AND status = 'Active'
        """, (trade_id,))
        conn.commit()
        print(f"[정보] 거래 {trade_id}의 익절 주문이 초기화되었습니다.")
        return True
    except Exception as e:
        print(f"[오류] 익절 주문 초기화 중 문제 발생: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def update_trade_entry_price(trade_id: str, actual_entry_price: float):
    """실제 진입가로 trades 테이블 내 진입가를 갱신합니다."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE trades
            SET entry_price = ?
            WHERE trade_id = ?
        ''', (actual_entry_price, trade_id))
        conn.commit()

        if cursor.rowcount == 0:
            logger.warning(f"[경고] trade_id {trade_id}를 찾지 못했습니다. 진입가 갱신되지 않음.")
        else:
            logger.info(f"[업데이트] 거래 ID {trade_id}의 진입가를 {actual_entry_price}로 갱신했습니다.")
    except Exception as e:
        logger.exception(f"[오류] 진입가 업데이트 실패 - 거래 ID: {trade_id}, 오류: {e}")
    finally:
        conn.close()



def clean_old_trades(conn, limit: int = 300):
    """
    오래된 거래 데이터를 정리하여 최근 거래 건수를 제한합니다.
    
    Args:
        conn: SQLite 데이터베이스 연결
        limit: 유지할 최근 거래 건수
    """
    try:
        c = conn.cursor()
        
        # 현재 거래 수 확인
        c.execute("SELECT COUNT(*) FROM trades")
        current_count = c.fetchone()[0]
        
        if current_count > limit:
            # 삭제할 거래 수 계산
            to_delete = current_count - limit
            
            # 가장 오래된 데이터부터 삭제
            c.execute("""
                DELETE FROM trades 
                WHERE id IN (
                    SELECT id FROM trades 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                )
            """, (to_delete,))
            
            conn.commit()
            print(f"[정보] 오래된 거래 데이터 {to_delete}건 정리 완료")
            
        return True
    except Exception as e:
        print(f"[오류] 거래 데이터 정리 중 문제 발생: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def update_trades_table_schema():
    """trades 테이블에 수수료 관련 컬럼 추가"""
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        
        # 컬럼 존재 여부 확인
        cursor.execute("PRAGMA table_info(trades)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # 필요한 컬럼들 추가
        columns_to_add = {
            'entry_fee': 'REAL DEFAULT 0',
            'exit_fee': 'REAL DEFAULT 0',
            'funding_fee': 'REAL DEFAULT 0'
        }
        
        for column, data_type in columns_to_add.items():
            if column not in columns:
                cursor.execute(f"ALTER TABLE trades ADD COLUMN {column} {data_type}")
                print(f"[정보] 컬럼 추가: {column}")
        
        conn.commit()
        conn.close()
        print("[정보] trades 테이블 스키마 업데이트 완료")
        return True
    except Exception as e:
        print(f"[오류] 테이블 스키마 업데이트 중 문제 발생: {e}")
        if 'conn' in locals():
            conn.close()
        return False

def update_db_schema():
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        
        # take_profit_orders 테이블이 있는지 확인하고 없으면 생성
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='take_profit_orders'")
        table_exists = cursor.fetchone()
        if not table_exists:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS take_profit_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                position_type TEXT,
                tp_level INTEGER,
                tp_price REAL,
                tp_quantity REAL,
                status TEXT DEFAULT 'Active',
                created_at TEXT,
                executed_at TEXT,
                execution_price REAL,
                profit_pct REAL,
                order_id TEXT
            )
            ''')
            print("[정보] take_profit_orders 테이블이 성공적으로 생성되었습니다.")
        
        # parameters_history 테이블이 있는지 확인하고 없으면 생성
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parameters_history'")
        table_exists = cursor.fetchone()
        if not table_exists:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameters_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                rsi_high REAL,
                rsi_low REAL,
                atr_multiplier REAL,
                tp_levels TEXT,
                volatility_threshold REAL,
                adx_threshold REAL,
                reason TEXT
            )
            ''')
            print("[정보] parameters_history 테이블이 성공적으로 생성되었습니다.")
        
        # market_analysis_records 테이블이 있는지 확인하고 없으면 생성
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_analysis_records'")
        table_exists = cursor.fetchone()
        if not table_exists:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TEXT,
                data TEXT
            )
            ''')
            print("[정보] market_analysis_records 테이블이 성공적으로 생성되었습니다.")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[오류] DB 스키마 업데이트 중 문제 발생: {e}")
        return False

def setup_market_analysis_table(cursor, conn):
    """
    market_analysis_records 테이블이 없으면 생성하고,
    필요한 컬럼이 누락된 경우 자동으로 추가합니다.
    """
    # 테이블 존재 여부 확인
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_analysis_records'")
    table_exists = cursor.fetchone()

    if not table_exists:
        # 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TEXT,
                data TEXT,
                timestamp TEXT,
                trend TEXT,
                direction TEXT,
                volume REAL
            )
        ''')
        conn.commit()
        print("[정보] market_analysis_records 테이블이 성공적으로 생성되었습니다.")
    else:
        # 기존 테이블 컬럼 정보 조회
        cursor.execute("PRAGMA table_info(market_analysis_records)")
        existing_columns = [col[1] for col in cursor.fetchall()]

        # 누락된 컬럼 추가
        required_columns = {
            "timestamp": "TEXT",
            "trend": "TEXT",
            "direction": "TEXT",
            "volume": "REAL"
        }

        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE market_analysis_records ADD COLUMN {column_name} {column_type}")
                print(f"[정보] '{column_name}' 컬럼이 테이블에 추가되었습니다.")
        
        conn.commit()


# 여기에 새 함수 추가
def update_tp_table_schema():
    """take_profit_orders 테이블에 추가 매수 표시 필드 추가"""
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        
        # 컬럼 존재 여부 확인
        cursor.execute("PRAGMA table_info(take_profit_orders)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # 필요한 컬럼 추가
        if 'is_additional' not in columns:
            cursor.execute("ALTER TABLE take_profit_orders ADD COLUMN is_additional INTEGER DEFAULT 0")
            print("[정보] take_profit_orders 테이블에 is_additional 컬럼 추가됨")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[오류] 테이블 스키마 업데이트 중 문제 발생: {e}")
        if 'conn' in locals():
            conn.close()
        return False

def sync_stop_loss_from_bybit(symbol: str = SYMBOL) -> bool:
    """
    바이비트 API에서 스탑로스 주문 실행 정보를 확인하고,
    stop_loss_records 테이블에 해당 정보를 업데이트하는 함수.
    
    바이비트의 'StopOrder' 주문 중 stopOrderType이 "StopLoss"인 주문을 조회합니다.
    주문 상태가 'Filled' 또는 'PartiallyFilled'인 경우, 
    해당 포지션(롱: positionIdx 1, 숏: positionIdx 2)에 대한 미청산 레코드를 찾아
    liquidation_price, liquidation_time, profit_loss를 업데이트합니다.
    
    Returns:
        bool: 업데이트 성공 여부.
    """
    try:
        response = bybit_client.get_open_orders(category="linear", symbol=symbol, orderFilter="StopOrder")
        if response.get("retCode") != 0:
            logger.warning(f"[경고] 스탑로스 주문 조회 실패: {response.get('retMsg', '알 수 없는 오류')}")
            return False
        
        orders = response["result"]["list"]
        if not orders:
            logger.info("[정보] 활성화된 스탑로스 주문이 없습니다.")
            return True
        
        updated = False
        for order in orders:
            stop_order_type = order.get("stopOrderType")
            order_status = order.get("orderStatus", "").lower()
            if stop_order_type != "StopLoss":
                continue
            if order_status not in ["filled", "partiallyfilled"]:
                continue
            
            position_idx = order.get("positionIdx")
            position_type = "Long" if position_idx == 1 else "Short"
            liquidation_price = float(order.get("triggerPrice", 0))
            if liquidation_price == 0:
                continue
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                select_sql = """
                    SELECT id, entry_price FROM stop_loss_records 
                    WHERE position_type = ? AND liquidated = 0
                    ORDER BY set_time DESC LIMIT 1
                """
                cursor.execute(select_sql, (position_type,))
                record = cursor.fetchone()
                if not record:
                    logger.warning(f"[경고] {position_type} 포지션에 업데이트할 스탑로스 레코드가 없습니다.")
                    continue
                
                record_id = record["id"]
                entry_price = float(record["entry_price"])
                if position_type == "Long":
                    profit_loss = ((liquidation_price - entry_price) / entry_price) * 100
                else:
                    profit_loss = ((entry_price - liquidation_price) / entry_price) * 100
                
                update_sql = """
                    UPDATE stop_loss_records
                    SET liquidated = 1,
                        liquidation_price = ?,
                        liquidation_time = ?,
                        profit_loss = ?
                    WHERE id = ?
                """
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(update_sql, (liquidation_price, current_time, profit_loss, record_id))
                conn.commit()
                logger.info(f"[정보] 스탑로스 청산 정보 업데이트 완료 (ID: {record_id}) - {position_type}, 손익: {profit_loss:.2f}%")
                updated = True
        return updated
    except Exception as e:
        logger.error(f"[오류] 스탑로스 데이터 동기화 중 문제 발생: {e}")
        return False


def sync_take_profit_from_bybit(symbol: str = SYMBOL) -> bool:
    """
    바이비트 API에서 익절(TakeProfit) 주문 실행 정보를 확인하고,
    take_profit_orders 테이블에 해당 정보를 업데이트하는 함수.
    
    바이비트의 'StopOrder' 주문 중 stopOrderType이 "TakeProfit"인 주문을 조회하고,
    주문 상태가 'Filled' 또는 'PartiallyFilled'인 경우, 해당 익절 주문의 실행 정보를 DB에 업데이트합니다.
    
    Returns:
        bool: 업데이트 성공 여부.
    """
    try:
        response = bybit_client.get_open_orders(category="linear", symbol=symbol, orderFilter="StopOrder")
        if response.get("retCode") != 0:
            logger.warning(f"[경고] 익절 주문 조회 실패: {response.get('retMsg', '알 수 없는 오류')}")
            return False
        
        orders = response["result"]["list"]
        if not orders:
            logger.info("[정보] 활성화된 익절 주문이 없습니다.")
            return True
        
        updated = False
        for order in orders:
            stop_order_type = order.get("stopOrderType")
            order_status = order.get("orderStatus", "").lower()
            if stop_order_type != "TakeProfit":
                continue
            if order_status not in ["filled", "partiallyfilled"]:
                continue
            
            position_idx = order.get("positionIdx")
            position_type = "Long" if position_idx == 1 else "Short"
            execution_price = float(order.get("avgPrice", order.get("triggerPrice", 0)))
            if execution_price == 0:
                continue
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                select_sql = """
                    SELECT id, tp_price, tp_quantity FROM take_profit_orders 
                    WHERE position_type = ? AND status IN ('Active', 'Pending')
                    ORDER BY tp_level ASC LIMIT 1
                """
                cursor.execute(select_sql, (position_type,))
                record = cursor.fetchone()
                if not record:
                    logger.warning(f"[경고] {position_type} 포지션에 업데이트할 익절 주문 레코드가 없습니다.")
                    continue
                
                record_id = record["id"]
                db_tp_price = float(record["tp_price"])
                if position_type == "Long":
                    profit_pct = ((execution_price - db_tp_price) / db_tp_price) * 100
                else:
                    profit_pct = ((db_tp_price - execution_price) / db_tp_price) * 100
                
                update_sql = """
                    UPDATE take_profit_orders
                    SET status = 'Executed',
                        executed_at = ?,
                        execution_price = ?,
                        profit_pct = ?
                    WHERE id = ?
                """
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(update_sql, (current_time, execution_price, profit_pct, record_id))
                conn.commit()
                logger.info(f"[정보] 익절 주문 업데이트 완료 (ID: {record_id}) - {position_type}, 수익률: {profit_pct:.2f}%")
                updated = True
        return updated
    except Exception as e:
        logger.error(f"[오류] 익절 주문 데이터 동기화 중 문제 발생: {e}")
        return False


def sync_open_positions_from_bybit(symbol: str = SYMBOL) -> bool:
    """
    바이비트 API에서 현재 오픈 포지션 정보를 조회하고,
    trades 테이블에 기록된 거래 상태를 업데이트하여
    내부 데이터와 바이비트 데이터 간의 일치를 도모하는 함수.
    
    - 바이비트 API의 get_positions() 메서드를 사용하여 현재 포지션 정보를 가져옵니다.
    - 각 포지션의 trade_id가 존재할 경우, 해당 거래의 상태를 'Open'으로 업데이트합니다.
    
    Args:
        symbol (str): 조회할 거래 심볼 (기본값: SYMBOL).
        
    Returns:
        bool: 동기화 성공 여부.
    """
    try:
        response = bybit_client.get_positions(category="linear", symbol=symbol)
        if response.get("retCode") != 0:
            logger.warning(f"[경고] 오픈 포지션 조회 실패 (심볼: {symbol}): {response.get('retMsg', '알 수 없는 오류')}")
            return False
        
        positions = response["result"]["list"]
        logger.info(f"[정보] 심볼 {symbol}의 오픈 포지션 {len(positions)}건 조회 완료.")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for pos in positions:
                trade_id = pos.get("tradeId")
                current_size = float(pos.get("size", 0))
                if not trade_id:
                    logger.warning(f"[경고] 포지션 식별자(tradeId)가 없는 포지션 발견. 포지션 크기: {current_size}")
                    continue
                update_sql = """
                    UPDATE trades
                    SET trade_status = 'Open'
                    WHERE trade_id = ?
                """
                cursor.execute(update_sql, (trade_id,))
            conn.commit()
            logger.info("[정보] 오픈 포지션 데이터와 trades 테이블 동기화 완료.")
        return True
    except Exception as e:
        logger.error(f"[오류] 오픈 포지션 동기화 중 문제 발생: {e}")
        return False

def store_stop_loss_db(position_type: str, entry_price: float, stop_price: float):
    """
    DB에 스탑로스 정보 저장
    """
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()

        # 현재 시간
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        insert_query = """
        INSERT INTO stop_loss_records (position_type, entry_price, stop_price, set_time)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(insert_query, (position_type, entry_price, stop_price, current_time))
        conn.commit()
        conn.close()

        print(f"[정보] 스탑로스 DB 저장 완료. (포지션: {position_type}, 진입가: {entry_price}, 스탑가: {stop_price})")
    except Exception as e:
        print(f"[오류] 스탑로스 DB 저장 중 문제 발생: {e}")

def update_database_schema(conn) -> bool:
    """
    데이터베이스 스키마를 최신 버전으로 업데이트하는 함수
    - 새로운 테이블 생성
    - 기존 테이블에 필요한 컬럼 추가
    
    Args:
        conn: 데이터베이스 연결 객체
        
    Returns:
        bool: 성공 여부
    """
    try:
        cursor = conn.cursor()
        
        # 1. trades 테이블 업데이트
        # trades 테이블에 새 컬럼 추가
        columns_to_add = [
            ("close_reason", "TEXT"),
            ("trailing_stop_price", "REAL"),
            ("trailing_stop_level", "INTEGER"),
            ("trailing_stop_updated_at", "TEXT"),
            ("tp_strategy", "TEXT"),
            ("risk_amount", "REAL"),
            ("market_condition", "TEXT")
        ]
        
        # trades 테이블의 현재 컬럼 정보 가져오기
        cursor.execute("PRAGMA table_info(trades)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        
        # 존재하지 않는 컬럼만 추가
        for column_name, column_type in columns_to_add:
            if column_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE trades ADD COLUMN {column_name} {column_type}")
                    print(f"[정보] trades 테이블에 {column_name} 컬럼 추가됨")
                except sqlite3.OperationalError as e:
                    # 이미 존재하는 컬럼에 대한 오류는 무시
                    if "duplicate column name" not in str(e):
                        print(f"[경고] 컬럼 추가 실패 ({column_name}): {e}")
        
        # 2. take_profit_orders 테이블 업데이트
        # take_profit_orders 테이블에 새 컬럼 추가
        tp_columns_to_add = [
            ("metadata", "TEXT"),
            ("is_time_based", "INTEGER DEFAULT 0")
        ]
        
        cursor.execute("PRAGMA table_info(take_profit_orders)")
        existing_tp_columns = [column[1] for column in cursor.fetchall()]
        
        for column_name, column_type in tp_columns_to_add:
            if column_name not in existing_tp_columns:
                try:
                    cursor.execute(f"ALTER TABLE take_profit_orders ADD COLUMN {column_name} {column_type}")
                    print(f"[정보] take_profit_orders 테이블에 {column_name} 컬럼 추가됨")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        print(f"[경고] 컬럼 추가 실패 ({column_name}): {e}")
        
        # 3. position_close_reasons 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_close_reasons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                close_reason TEXT,
                additional_data TEXT,
                close_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 4. time_based_tp_records 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS time_based_tp_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                position_type TEXT,
                holding_hours REAL,
                pnl_pct REAL,
                tp_target REAL,
                tp_ratio REAL,
                market_condition TEXT,
                execution_time TEXT
            )
        ''')
        
        # 5. market_analysis_records 테이블 생성 (시장 분석 결과 저장)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market_type TEXT,
                direction TEXT,
                strength REAL,
                confidence REAL,
                key_levels TEXT,
                volatility REAL,
                adx REAL,
                rsi TEXT,
                analysis_summary TEXT
            )
        ''')
        
        # 추가: market_analysis_records 테이블에 timestamp 컬럼이 누락되었으면 추가
        add_timestamp_column_to_market_analysis_records(conn)
        
        conn.commit()
        print("[정보] 데이터베이스 스키마 업데이트 완료")
        return True
        
    except Exception as e:
        print(f"[오류] 데이터베이스 스키마 업데이트 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'conn' in locals():
            conn.rollback()
        return False

def add_timestamp_column_to_market_analysis_records(conn) -> bool:
    """market_analysis_records 테이블에 timestamp 컬럼이 없으면 추가하고, 성공 여부를 반환합니다."""
    try:
        if not conn:
            print("[오류] 데이터베이스 연결이 없습니다.")
            return False
        c = conn.cursor()
        c.execute("PRAGMA table_info(market_analysis_records)")
        columns = [row["name"] for row in c.fetchall()]
        if "timestamp" in columns:
            print("[정보] market_analysis_records 테이블에 이미 timestamp 컬럼이 존재합니다.")
            return True
        c.execute("ALTER TABLE market_analysis_records ADD COLUMN timestamp TEXT")
        conn.commit()
        print("[정보] market_analysis_records 테이블에 timestamp 컬럼이 추가되었습니다.")
        return True
    except Exception as e:
        print(f"[오류] timestamp 컬럼 추가 중 문제 발생: {e}")
        return False

def track_position_close_reason(conn, trade_id: str, close_reason: str, additional_data: dict = None) -> bool:
    """
    포지션 종료 원인을 추적하고 기록하는 함수
    
    Args:
        conn: 데이터베이스 연결 객체
        trade_id: 거래 ID
        close_reason: 종료 원인 (예: "stop_loss", "take_profit", "manual", "forced_liquidation" 등)
        additional_data: 추가 정보 (선택적)
        
    Returns:
        bool: 성공 여부
    """
    try:
        cursor = conn.cursor()
        
        # position_close_reasons 테이블이 없으면 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_close_reasons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                close_reason TEXT,
                additional_data TEXT,
                close_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # DB에 종료 원인 기록
        cursor.execute('''
            INSERT INTO position_close_reasons (trade_id, close_reason, additional_data)
            VALUES (?, ?, ?)
        ''', (
            trade_id,
            close_reason,
            json.dumps(additional_data) if additional_data else None
        ))
        
        # 거래 테이블에도 종료 원인 업데이트
        cursor.execute('''
            UPDATE trades
            SET close_reason = ?
            WHERE trade_id = ?
        ''', (close_reason, trade_id))
        
        conn.commit()
        print(f"[정보] 포지션 종료 원인 기록 완료: {trade_id}, 원인: {close_reason}")
        return True
        
    except Exception as e:
        print(f"[오류] 포지션 종료 원인 기록 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'conn' in locals():
            conn.rollback()
        return False

def update_liquidation_info(position_type: str, liquidation_price: float, profit_loss: float):
    """
    스탑로스 청산 정보를 기존 레코드에 업데이트
    """
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        
        liquidation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 먼저 미청산 레코드 ID 찾기
        find_query = """
        SELECT id FROM stop_loss_records
        WHERE position_type = ? AND liquidated = 0
        ORDER BY set_time DESC LIMIT 1
        """
        cursor.execute(find_query, (position_type,))
        record = cursor.fetchone()
        
        if not record:
            print(f"[경고] 업데이트할 {position_type} 포지션 레코드를 찾을 수 없습니다.")
            conn.close()
            return
            
        record_id = record[0]
        
        # 해당 ID의 레코드 업데이트
        update_query = """
        UPDATE stop_loss_records
        SET liquidated = 1,
            liquidation_price = ?,
            liquidation_time = ?,
            profit_loss = ?
        WHERE id = ?
        """
        
        cursor.execute(update_query, (
            liquidation_price,
            liquidation_time,
            profit_loss,
            record_id
        ))
        
        print(f"[정보] 스탑로스 청산 정보 업데이트 완료 (ID: {record_id}): {position_type}, 손익: {profit_loss:.2f}%")
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"[오류] 청산 정보 업데이트 중 문제 발생: {e}")

# 8. 기술적 지표 관련 함수들 (동일)
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 추가"""
    try:
        # MACD
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd().fillna(0)
        df['MACD_signal'] = macd.macd_signal().fillna(0)
        df['MACD_diff'] = macd.macd_diff().fillna(0)

        # RSI
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi().fillna(50)

        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband().fillna(df['Close'])
        df['BB_lower'] = bb.bollinger_lband().fillna(df['Close'])
        df['BB_mavg'] = bb.bollinger_mavg().fillna(df['Close'])

        # Moving Averages
        df['EMA_10'] = EMAIndicator(close=df['Close'], window=10).ema_indicator().fillna(df['Close'])
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator().fillna(df['Close'])
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator().fillna(df['Close'])
        df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator().fillna(df['Close'])
        df['EMA_100'] = EMAIndicator(close=df['Close'], window=100).ema_indicator().fillna(df['Close'])
        df['EMA_200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator().fillna(df['Close'])

        # ATR
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ATR'] = atr.average_true_range().fillna(0)

        # ADX
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx.adx().fillna(0)
        df['+DI'] = adx.adx_pos().fillna(0)
        df['-DI'] = adx.adx_neg().fillna(0)

        # Williams Alligator
        df['Alligator_Jaws'] = df['Close'].rolling(window=13).mean().shift(8).fillna(df['Close'])
        df['Alligator_Teeth'] = df['Close'].rolling(window=8).mean().shift(5).fillna(df['Close'])
        df['Alligator_Lips'] = df['Close'].rolling(window=5).mean().shift(3).fillna(df['Close'])

        # Fractal Indicator
        df['Fractal_Up'] = df['High'].rolling(window=5, center=True).apply(lambda x: 1 if x[2] == max(x) else 0, raw=True).fillna(0)
        df['Fractal_Down'] = df['Low'].rolling(window=5, center=True).apply(lambda x: 1 if x[2] == min(x) else 0, raw=True).fillna(0)

        # Fibonacci Levels
        high = df['High'].max()
        low = df['Low'].min()
        df['Fibo_0.236'] = high - (high - low) * 0.236
        df['Fibo_0.382'] = high - (high - low) * 0.382
        df['Fibo_0.5'] = (high + low) / 2
        df['Fibo_0.618'] = high - (high - low) * 0.618
        df['Fibo_0.786'] = high - (high - low) * 0.786
        df['Fibo_1'] = low
        df['Fibo_-0.236'] = low - (high - low) * 0.236
        df['Fibo_-0.382'] = low - (high - low) * 0.382
        df['Fibo_-0.5'] = low - (high - low) * 0.5
        df['Fibo_-0.618'] = low - (high - low) * 0.618
        df['Fibo_-0.786'] = low - (high - low) * 0.786
        df['Fibo_-1'] = low - (high - low)

        # [수정] 피벗 포인트 대신 돈치안 채널 지표 추가
        period = 20
        df['Donchian_High'] = df['High'].rolling(window=period).max().fillna(df['High'])
        df['Donchian_Low'] = df['Low'].rolling(window=period).min().fillna(df['Low'])
        df['Donchian_Mid'] = (df['Donchian_High'] + df['Donchian_Low']) / 2

        # 추가 지지/저항 레벨 계산
        df['Support1'] = df['Donchian_Low']
        df['Resistance1'] = df['Donchian_High']
        df['Support2'] = df['Donchian_Low'] - (df['ATR'] * 1.5)
        df['Resistance2'] = df['Donchian_High'] + (df['ATR'] * 1.5)
        df['Pivot'] = df['Donchian_Mid']

        return df
    except Exception as e:
        print(f"기술적 지표 추가 중 오류 발생: {e}")
        return df


def detect_trendlines(df: pd.DataFrame, lookback_period: int = 20, position_type: str = "Long") -> Optional[float]:
    """
    추세선 분석을 통한 주요 지지/저항 수준 탐지
    
    Args:
        df: 캔들스틱 데이터
        lookback_period: 분석할 과거 기간
        position_type: "Long" 또는 "Short"
        
    Returns:
        추세 기반 지지/저항 가격
    """
    try:
        # 분석 대상 데이터 추출
        recent_data = df.tail(lookback_period)
        
        if position_type == "Long":
            # 롱 포지션의 경우 지지 추세선 탐지 (저점 연결)
            local_lows = []
            
            # 이전 2개 봉보다 낮은 저점 식별
            for i in range(2, len(recent_data)-2):
                if (recent_data.iloc[i]['Low'] < recent_data.iloc[i-1]['Low'] and 
                    recent_data.iloc[i]['Low'] < recent_data.iloc[i-2]['Low'] and
                    recent_data.iloc[i]['Low'] < recent_data.iloc[i+1]['Low'] and
                    recent_data.iloc[i]['Low'] < recent_data.iloc[i+2]['Low']):
                    local_lows.append((i, float(recent_data.iloc[i]['Low'])))
            
            # 최소 2개의 저점이 필요
            if len(local_lows) >= 2:
                # 선형 회귀로 추세 계산
                x = np.array([point[0] for point in local_lows])
                y = np.array([point[1] for point in local_lows])
                
                # 상승 추세선만 고려 (기울기 양수)
                slope, intercept = np.polyfit(x, y, 1)
                if slope > 0:
                    # 현재 봉에 대한 추세선 가격 계산
                    current_index = len(recent_data) - 1
                    trendline_price = slope * current_index + intercept
                    return trendline_price
        else:  # Short
            # 숏 포지션의 경우 저항 추세선 탐지 (고점 연결)
            local_highs = []
            
            # 이전 2개 봉보다 높은 고점 식별
            for i in range(2, len(recent_data)-2):
                if (recent_data.iloc[i]['High'] > recent_data.iloc[i-1]['High'] and 
                    recent_data.iloc[i]['High'] > recent_data.iloc[i-2]['High'] and
                    recent_data.iloc[i]['High'] > recent_data.iloc[i+1]['High'] and
                    recent_data.iloc[i]['High'] > recent_data.iloc[i+2]['High']):
                    local_highs.append((i, float(recent_data.iloc[i]['High'])))
            
            # 최소 2개의 고점이 필요
            if len(local_highs) >= 2:
                # 선형 회귀로 추세 계산
                x = np.array([point[0] for point in local_highs])
                y = np.array([point[1] for point in local_highs])
                
                # 하락 추세선만 고려 (기울기 음수)
                slope, intercept = np.polyfit(x, y, 1)
                if slope < 0:
                    # 현재 봉에 대한 추세선 가격 계산
                    current_index = len(recent_data) - 1
                    trendline_price = slope * current_index + intercept
                    return trendline_price
        
        return None
    except Exception as e:
        print(f"[오류] 추세선 분석 중 문제 발생: {e}")
        return None

def fetch_candle_data(interval: str = "15", limit: int = 70) -> pd.DataFrame:
    """캔들스틱 데이터 조회"""
    try:
        response = bybit_client.get_kline(
            category="linear",
            symbol=SYMBOL,
            interval=interval,
            limit=limit
        )
        
        if response["retCode"] != 0:
            print(f"캔들스틱 데이터 조회 실패: {response['retMsg']}") 
            return pd.DataFrame()
            
        df = pd.DataFrame(
            response['result']['list'],
            columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Value']
        )
        
        print("Raw Time value:", df['Time'].iloc[0])
        # 수정: np.int64를 사용하여 변환
        # df['Time'] = pd.to_datetime(df['Time'].astype(np.int64), unit='ms')
        df['Time'] = pd.to_datetime(df['Time'].astype(np.int64), unit='ms', utc=True).dt.tz_localize(None)
        df = df.astype({
            'Open': float,
            'High': float,
            'Low': float,
            'Close': float,
            'Volume': float
        })
        
        df.set_index('Time', inplace=True)
        return df
        
    except Exception as e:
        print(f"캔들스틱 데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()

def check_multiframe_trend(timeframe_data: Dict[str, pd.DataFrame]) -> str:
    try:
        m15_df = timeframe_data["15"]
        h1_df = timeframe_data["60"]
        h4_df = timeframe_data["240"]

        latest_h4 = h4_df.iloc[-1]
        latest_h1 = h1_df.iloc[-1]
        latest_m15 = m15_df.iloc[-1]

        # 1. 4시간봉 추세 분석 (장기 추세 참고)
        print("\n[4시간봉 추세 분석]")
        trend_strength = "weak"
        trend_direction_h4 = "neutral"
        if latest_h4['ADX'] >= 25:
            trend_strength = "strong"
            print(f"4시간봉: 강한 추세 감지 (ADX: {latest_h4['ADX']:.2f})")
        elif latest_h4['ADX'] >= 20:
            trend_strength = "moderate"
            print(f"4시간봉: 중간 추세 (ADX: {latest_h4['ADX']:.2f})")
        else:
            print(f"4시간봉: 약한 추세 (ADX: {latest_h4['ADX']:.2f})")

        if (latest_h4['MACD'] > latest_h4['MACD_signal'] and 
            latest_h4['EMA_20'] > latest_h4['EMA_50']):
            trend_direction_h4 = "bullish"
            print("4시간봉: 상승 추세")
        elif (latest_h4['MACD'] < latest_h4['MACD_signal'] and 
              latest_h4['EMA_20'] < latest_h4['EMA_50']):
            trend_direction_h4 = "bearish"
            print("4시간봉: 하락 추세")

        # 2. 1시간봉 신호 분석 (중기 트렌드)
        print("\n[1시간봉 중기 트렌드 분석]")
        signals = {"bullish": 0, "bearish": 0, "neutral": 0}
        if latest_h1['RSI'] > 60:
            signals["bullish"] += 1
            print("1시간봉: RSI 상승세")
        elif latest_h1['RSI'] < 40:
            signals["bearish"] += 1
            print("1시간봉: RSI 하락세")
        else:
            signals["neutral"] += 1
            print("1시간봉: RSI 중립")
            
        if latest_h1['MACD'] > latest_h1['MACD_signal']:
            signals["bullish"] += 1
            print("1시간봉: MACD 상승세")
        else:
            signals["bearish"] += 1
            print("1시간봉: MACD 하락세")
            
        bb_position = (latest_h1['Close'] - latest_h1['BB_lower']) / (latest_h1['BB_upper'] - latest_h1['BB_lower'])
        if bb_position > 0.8:
            signals["bearish"] += 1
            print("1시간봉: 볼린저 밴드 상단 접근 (하락 신호)")
        elif bb_position < 0.2:
            signals["bullish"] += 1
            print("1시간봉: 볼린저 밴드 하단 접근 (상승 신호)")
        else:
            signals["neutral"] += 1
            print("1시간봉: 볼린저 밴드 중립")

        # 3. 15분봉 모멘텀 분석 (단기 신호)
        print("\n[15분봉 모멘텀 분석]")
        momentum = "neutral"
        vol_increase = latest_m15['Volume'] > m15_df['Volume'].rolling(10).mean().iloc[-1]
        if (latest_m15['MACD'] > latest_m15['MACD_signal'] and 
            latest_m15['Close'] > latest_m15['EMA_20'] and vol_increase):
            momentum = "bullish"
            print("15분봉: 상승 모멘텀")
        elif (latest_m15['MACD'] < latest_m15['MACD_signal'] and 
              latest_m15['Close'] < latest_m15['EMA_20'] and vol_increase):
            momentum = "bearish"
            print("15분봉: 하락 모멘텀")
        else:
            print("15분봉: 모멘텀 중립")

        # 4. 최종 신호 결정 - 1시간봉을 중기 트렌드로, 4시간봉을 장기 트렌드로 사용
        print("\n=== 최종 신호 결정 ===")
        if trend_strength == "strong":
            # 강한 트렌드에서는 4시간봉 신호를 우선시
            print("강한 추세 (강조): 4시간봉 신호 우선")
            if trend_direction_h4 == "bullish":
                # 단, 1시간봉과 15분봉이 모두 반대 방향인 경우 신중하게 접근
                if signals["bearish"] >= 2 and momentum == "bearish":
                    print("주의: 단기/중기 시그널이 장기 추세와 반대 - 중립 유지")
                    return "sideways"
                return "up"
            elif trend_direction_h4 == "bearish":
                # 단, 1시간봉과 15분봉이 모두 반대 방향인 경우 신중하게 접근
                if signals["bullish"] >= 2 and momentum == "bullish":
                    print("주의: 단기/중기 시그널이 장기 추세와 반대 - 중립 유지")
                    return "sideways"
                return "down"
        elif trend_strength == "moderate":
            print("중간 강도: 1시간봉과 15분봉 신호 조합, 4시간봉으로 검증")
            # 1시간봉 신호에 가중치를 주고, 15분봉으로 타이밍, 4시간봉으로 확인
            score = signals["bullish"] - signals["bearish"]
            if momentum == "bullish":
                score += 0.5
            elif momentum == "bearish":
                score -= 0.5
            
            # 4시간봉 추세 확인 - 중기와 장기가 일치할 때 신호 강화
            if trend_direction_h4 == "bullish" and score > 0:
                return "up"
            elif trend_direction_h4 == "bearish" and score < 0:
                return "down"
            # 중기와 장기가 불일치하면 추가 분석 필요
            elif score > 1:  # 중기 신호가 매우 강한 경우
                return "up"
            elif score < -1:  # 중기 신호가 매우 강한 경우
                return "down"
        else:
            print("약한 추세: 15분 및 1시간봉 중시, 배수 설정 감소")
            # 약한 추세에서는 단기 신호에 더 집중
            if signals["bullish"] > signals["bearish"] and momentum == "bullish":
                # 단, 4시간봉이 매우 반대 방향이면 조심
                if latest_h4['ADX'] > 25 and trend_direction_h4 == "bearish":
                    print("주의: 장기 추세가 강하게 반대 - 배수 감소 필요")
                return "up"
            elif signals["bearish"] > signals["bullish"] and momentum == "bearish":
                # 단, 4시간봉이 매우 반대 방향이면 조심
                if latest_h4['ADX'] > 25 and trend_direction_h4 == "bullish":
                    print("주의: 장기 추세가 강하게 반대 - 배수 감소 필요")
                return "down"

        print("명확한 방향성 부재: Sideways 판정")
        return "sideways"
    except Exception as e:
        print(f"[오류] check_multiframe_trend 실행 중 문제 발생: {e}")
        return 'error'


def check_tp_orders_in_db(conn):
    """데이터베이스에 저장된 TP(Take Profit) 주문 확인"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT trade_id, position_type, tp_level, tp_price, tp_quantity, status
            FROM take_profit_orders
            WHERE status = 'Active'
            ORDER BY position_type, tp_level
        """)
        
        orders = cursor.fetchall()
        if orders:
            print(f"\n=== 데이터베이스에 저장된 익절 주문 ({len(orders)}개) ===")
            current_trade_id = None
            for order in orders:
                trade_id, pos_type, level, price, qty, status = order
                if trade_id != current_trade_id:
                    print(f"\n거래 ID: {trade_id}, 포지션: {pos_type}")
                    current_trade_id = trade_id
                print(f"TP {level}: 가격 ${float(price):,.2f}, 수량: {float(qty):,.3f} BTC")
        else:
            print("\n데이터베이스에 활성화된 익절 주문이 없습니다.")
        return True
    except Exception as e:
        print(f"[오류] DB 익절 주문 확인 중 문제 발생: {e}")
        return False


# 5. 익절 주문 실행 확인 함수 수정
def check_take_profit_executions(conn):
    """
    익절 주문 실행 여부를 확인하고 후속 조치 실행 (개선된 버전)
    """
    try:
        # 기존 conn 대신 새로운 전용 연결 사용
        with get_db_connection() as db_conn:
            cursor = db_conn.cursor()
            
            # 활성 상태의 익절 주문 조회
            cursor.execute("""
                SELECT tpo.id, tpo.trade_id, tpo.position_type, tpo.tp_level, tpo.tp_price, tpo.tp_quantity, tpo.order_id,
                    t.entry_price, t.leverage, tpo.is_additional
                FROM take_profit_orders tpo
                JOIN trades t ON tpo.trade_id = t.trade_id
                WHERE tpo.status = 'Active'
                ORDER BY tpo.position_type, tpo.tp_level
            """)
            
            active_tp_orders = cursor.fetchall()
            if not active_tp_orders:
                return
                
            print(f"[정보] {len(active_tp_orders)}개의 활성 익절 주문 확인 중...")
            
            # 현재 가격 정보 가져오기
            current_price = float(fetch_candle_data(interval="1", limit=1)['Close'].iloc[-1])
            
            # 각 익절 주문 개별 처리
            for tp_order in active_tp_orders:
                process_single_tp_order(db_conn, tp_order, current_price)
        
    except Exception as e:
        print(f"[오류] 익절 주문 확인 중 문제 발생: {e}")
        print(traceback.format_exc())
        
    except Exception as e:
        print(f"[오류] 익절 주문 확인 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'cursor' in locals() and 'conn' in locals():
            conn.rollback()

def check_take_profit_executions_enhanced(conn):
    """
    향상된 익절 주문 실행 여부 확인 및 후속 조치 실행 함수
    - 시장 상황에 따른 동적 대응 추가
    - 다단계 익절 전략 지원
    - 익절 후 트레일링 스탑 자동 적용
    """
    try:
        cursor = conn.cursor()
        
        # 1. 활성 상태의 익절 주문 조회
        cursor.execute("""
            SELECT tpo.id, tpo.trade_id, tpo.position_type, tpo.tp_level, tpo.tp_price, tpo.tp_quantity, tpo.order_id,
                   t.entry_price, t.leverage, tpo.is_additional
            FROM take_profit_orders tpo
            JOIN trades t ON tpo.trade_id = t.trade_id
            WHERE tpo.status = 'Active'
            ORDER BY tpo.position_type, tpo.tp_level
        """)
        
        active_tp_orders = cursor.fetchall()
        if not active_tp_orders:
            return
            
        print(f"[정보] {len(active_tp_orders)}개의 활성 익절 주문 확인 중...")
        
        # 2. 최신 시장 데이터 가져오기
        market_data = get_market_data()
        if not market_data or market_data['candlestick'].empty:
            print("[경고] 시장 데이터를 가져올 수 없습니다. 익절 확인 중단.")
            return
            
        current_price = float(market_data['candlestick'].iloc[-1]['Close'])
        
        # 3. 각 TP 주문 처리
        for tp_order in active_tp_orders:
            tp_id, trade_id, pos_type, tp_level, tp_price, tp_qty, order_id, entry_price, leverage, is_additional = tp_order
            
            # 각 TP 주문 개별 처리
            process_single_tp_order_enhanced(
                conn, tp_id, trade_id, pos_type, tp_level, tp_price, tp_qty, order_id, 
                entry_price, leverage, is_additional, current_price, market_data
            )
        
        print("[정보] 익절 주문 확인 완료")
        
    except Exception as e:
        print(f"[오류] 익절 주문 확인 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'cursor' in locals() and 'conn' in locals():
            conn.rollback()

def process_single_tp_order_enhanced(conn, tp_id, trade_id, position_type, tp_level, tp_price, tp_qty, order_id, 
                                   entry_price, leverage, is_additional, current_price, market_data):
    """
    단일 TP 주문 처리 보조 함수
    """
    try:
        cursor = conn.cursor()
        
        # 1. 익절 도달 여부 확인
        price_reached = False
        if position_type == "Long" and current_price >= tp_price:
            price_reached = True
            print(f"[감지] 롱 포지션 익절 가격 도달: ${current_price:.2f} >= ${tp_price:.2f}")
        elif position_type == "Short" and current_price <= tp_price:
            price_reached = True
            print(f"[감지] 숏 포지션 익절 가격 도달: ${current_price:.2f} <= ${tp_price:.2f}")
                
        # 2. 주문 상태 확인
        order_executed = False
        execution_price = tp_price  # 기본값
        
        if order_id and price_reached:
            try:
                order_info = bybit_client.get_order_history(
                    category="linear",
                    symbol=SYMBOL,
                    orderId=order_id
                )
                if order_info["retCode"] == 0 and order_info["result"]["list"]:
                    order_status = order_info["result"]["list"][0]["orderStatus"]
                    if order_status in ["Filled", "PartiallyFilled"]:
                        order_executed = True
                        execution_price = float(order_info["result"]["list"][0]["avgPrice"])
                        print(f"[확인] 거래소에서 주문 실행 확인됨: {order_status}, 실행가: ${execution_price:.2f}")
            except Exception as e:
                print(f"[경고] 주문 {order_id} 상태 확인 중 오류: {e}")
        
        # 3. 가격이 도달했거나 주문이 체결된 경우
        if price_reached or order_executed:
            # 3.1 수익률 계산
            if position_type == "Long":
                profit_pct = ((tp_price - entry_price) / entry_price) * 100 * leverage
            else:
                profit_pct = ((entry_price - tp_price) / entry_price) * 100 * leverage
                
            # 3.2 DB 업데이트
            cursor.execute("""
                UPDATE take_profit_orders
                SET status = 'Executed',
                    executed_at = ?,
                    execution_price = ?,
                    profit_pct = ?
                WHERE id = ?
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                execution_price,
                profit_pct,
                tp_id
            ))
            
            print(f"[성공] {position_type} 포지션 {tp_level}단계 익절 실행! 수익률: {profit_pct:.2f}%")
            
            # 3.3 AI 분석 요청 (있는 경우)
            try:
                # AI 분석 요청
                ai_analysis_data = {
                    "trade_id": trade_id,
                    "position_type": position_type,
                    "entry_price": entry_price,
                    "exit_price": tp_price,
                    "leverage": leverage,
                    "profit_percentage": profit_pct,
                    "tp_level": tp_level
                }
                
                request_ai_trade_analysis(ai_analysis_data)
            except Exception as ai_error:
                print(f"[경고] AI 분석 요청 중 오류: {ai_error}")

            # 3.4 추가 매수에 의한 익절 처리
            if is_additional == 1:
                print(f"[정보] 추가 매수에 대한 익절 처리 완료")
                # 추가 매수 익절은 다음 단계 설정 없이 단독으로 처리
                conn.commit()
                return
                
            
            # 3.5 다음 TP 단계 설정 (마지막 단계가 아닌 경우)
            if tp_level < 3:  # 3단계가 마지막
                # 향상된 버전의 함수를 직접 호출
                next_tp_result = activate_next_tp_order_enhanced(
                    conn, trade_id, position_type, tp_level, market_data
                )
                    
                if next_tp_result:
                    print(f"[성공] {tp_level+1}단계 TP가 활성화되었습니다.")        
                    
            
            # 3.6 익절 단계별 트레일링 스탑 적용
            if tp_level == 1:
                # 첫 번째 익절 후 손절가를 진입가 근처로 조정
                adjust_stop_loss_after_first_tp(conn, trade_id, position_type, entry_price)
                
                # 트레일링 스탑 활성화 (수익의 50%를 보호)
                apply_trailing_stop_after_tp(conn, trade_id, position_type, entry_price, leverage, 
                                        profit_pct, tp_level, market_data)
                
            elif tp_level == 2:
                # 두 번째 익절 후 손절가를 손익분기점으로 조정
                set_breakeven_stop_loss(conn, trade_id, position_type, entry_price)
                
                # 트레일링 스탑 활성화 (수익의 70%를 보호)
                apply_trailing_stop_after_tp(conn, trade_id, position_type, entry_price, leverage, 
                                        profit_pct, tp_level, market_data)
                
            elif tp_level == 3:
                # 마지막 익절 후 남은 포지션 정리
                close_remaining_position(conn, trade_id, position_type)

            conn.commit()
        
        
    except Exception as e:
        print(f"[오류] TP 주문 처리 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'cursor' in locals() and 'conn' in locals():
            conn.rollback()

# 6. 단일 익절 주문 처리 함수 추가
def process_single_tp_order(conn, tp_order, current_price):
    """
    단일 익절 주문 처리 (개별 처리로 동시성 문제 완화)
    - 시간 기반 익절 조건 통합
    """
    tp_id, trade_id, pos_type, tp_level, tp_price, tp_qty, order_id, entry_price, leverage, is_additional = tp_order
    
    try:
        print(f"[검사] {pos_type} 포지션 {tp_level}단계 익절 (가격: ${tp_price:.2f}, 수량: {tp_qty})")
        
        # 시간 기반 익절 조건 확인 (추가)
        market_data = get_market_data()
        time_based_tp_result, time_based_tp_price, time_based_tp_ratio = check_time_based_take_profit(
            conn, trade_id, pos_type, market_data
        )
        
        # 시간 기반 익절 실행 (추가)
        if time_based_tp_result and time_based_tp_price and time_based_tp_ratio:
            # 현재 포지션 정보 가져오기
            position_info = get_position_info(bybit_client)
            if position_info and position_info[pos_type]["size"] > 0:
                current_size = position_info[pos_type]["size"]
                
                # 청산할 수량 계산
                execute_qty = round(current_size * time_based_tp_ratio, 3)
                
                # 최소 거래량 체크
                if execute_qty >= MIN_QTY:
                    print(f"[정보] 시간 기반 익절 실행: {pos_type} 포지션 {execute_qty} BTC ({time_based_tp_ratio*100:.0f}%)")
                    
                    # 익절 주문 실행
                    position_idx = 1 if pos_type == "Long" else 2
                    order_type = "sell" if pos_type == "Long" else "buy"
                    
                    order_success = place_order(
                        order_type=order_type, 
                        symbol=SYMBOL, 
                        qty=execute_qty, 
                        position_idx=position_idx
                    )
                    
                    if order_success:
                        print(f"[성공] 시간 기반 익절 주문 실행 완료")
                        
                        # 수익률 계산
                        if pos_type == "Long":
                            profit_pct = ((time_based_tp_price - entry_price) / entry_price) * 100 * leverage
                        else:
                            profit_pct = ((entry_price - time_based_tp_price) / entry_price) * 100 * leverage
                            
                        # DB에 시간 기반 익절 정보 기록
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO take_profit_orders 
                            (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, 
                             order_id, is_additional, is_time_based)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            trade_id,
                            pos_type,
                            tp_level,  # 기존 TP 레벨 유지
                            time_based_tp_price,
                            execute_qty,
                            'Executed',
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            None,  # 주문 ID는 추적하지 않음 (시장가 주문)
                            0,     # 추가 매수가 아님
                            1      # 시간 기반 익절 표시
                        ))
                        conn.commit()
                        
                        # 남은 포지션이 있는지 확인하고 다음 단계 TP 설정
                        remaining_size = current_size - execute_qty
                        if remaining_size < MIN_QTY:
                            # 포지션이 거의 없음 - 완전 청산으로 간주
                            print(f"[정보] 포지션 대부분 청산됨 - 다음 TP 설정 없음")
                            return
                            
                        # 익절 후 트레일링 스탑 설정 (시간 기반 익절 후에도 적용)
                        apply_trailing_stop_after_tp(
                            conn, trade_id, pos_type, entry_price, 
                            leverage, profit_pct, tp_level, market_data
                        )
        
        # 기존 가격 기반 익절 로직 (이하 기존 코드와 동일)
        price_reached = False
        if pos_type == "Long" and current_price >= tp_price:
            price_reached = True
            print(f"[감지] 롱 포지션 익절 가격 도달: ${current_price:.2f} >= ${tp_price:.2f}")
        elif pos_type == "Short" and current_price <= tp_price:
            price_reached = True
            print(f"[감지] 숏 포지션 익절 가격 도달: ${current_price:.2f} <= ${tp_price:.2f}")
                
        # 주문 상태 확인
        order_executed = False
        execution_price = tp_price  # 기본값
        
        if order_id and price_reached:
            try:
                order_info = bybit_client.get_order_history(
                    category="linear",
                    symbol=SYMBOL,
                    orderId=order_id
                )
                if order_info["retCode"] == 0 and order_info["result"]["list"]:
                    order_status = order_info["result"]["list"][0]["orderStatus"]
                    if order_status in ["Filled", "PartiallyFilled"]:
                        order_executed = True
                        execution_price = float(order_info["result"]["list"][0]["avgPrice"])
                        print(f"[확인] 거래소에서 주문 실행 확인됨: {order_status}, 실행가: ${execution_price:.2f}")
            except Exception as e:
                print(f"[경고] 주문 {order_id} 상태 확인 중 오류: {e}")
        
        # 가격이 도달했거나 주문이 체결된 경우
        if price_reached or order_executed:
            # 수익률 계산
            if pos_type == "Long":
                profit_pct = ((tp_price - entry_price) / entry_price) * 100 * leverage
            else:
                profit_pct = ((entry_price - tp_price) / entry_price) * 100 * leverage
                
            # DB 업데이트 (별도 작업으로 분리)
            def update_tp_status(db_conn):
                cursor = db_conn.cursor()
                cursor.execute("""
                    UPDATE take_profit_orders
                    SET status = 'Executed',
                        executed_at = ?,
                        execution_price = ?,
                        profit_pct = ?
                    WHERE id = ?
                """, (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    execution_price,
                    profit_pct,
                    tp_id
                ))
            
            # 재시도 로직 적용하여 DB 업데이트
            execute_with_retry(update_tp_status)
            
            print(f"[성공] {pos_type} 포지션 {tp_level}단계 익절 실행! 수익률: {profit_pct:.2f}%")
            
            # AI 분석을 위한 데이터 준비
            ai_analysis_data = {
                "trade_id": trade_id,
                "position_type": pos_type,
                "entry_price": entry_price,
                "exit_price": tp_price,
                "leverage": leverage,
                "profit_percentage": profit_pct,
                "tp_level": tp_level
            }
            
            # AI에게 분석 요청 (비동기)
            request_ai_trade_analysis_async(ai_analysis_data)
            
            # 추가 매수 여부 확인 
            is_additional = is_additional if is_additional else 0            
            # 추가 매수가 아닌 경우에만 다음 단계 TP 설정 및 스탑로스 조정
            if is_additional != 1:
                # 다음 단계 TP 활성화 전에 포지션이 여전히 존재하는지 확인
                position_info = get_position_info(bybit_client)
                if position_info and position_info[pos_type]["size"] > 0:
                    # 다음 단계 TP 활성화
                    if tp_level < 3:  # 3단계가 마지막
                        # market_data 매개변수가 필요하지만 사용할 수 없는 경우 None 전달
                        activate_next_tp_order_enhanced(conn, trade_id, pos_type, tp_level, market_data)
            
                    
                    # 익절 단계별 후속 조치
                    if tp_level == 1:
                        # 첫 번째 익절 후 손절가를 진입가 근처로 조정
                        adjust_stop_loss_after_first_tp(conn, trade_id, pos_type, entry_price)
                    elif tp_level == 2:
                        # 두 번째 단계 익절 후 스탑로스를 손익분기점으로 조정
                        set_breakeven_stop_loss(conn, trade_id, pos_type, entry_price)
                    elif tp_level == 3:
                        # 마지막 단계 익절 후 남은 포지션 정리
                        close_remaining_position(conn, trade_id, pos_type)
                else:
                    print(f"[경고] {pos_type} 포지션 정보를 찾을 수 없습니다.")
            else:
                print(f"[정보] 추가 매수에 대한 익절 처리 완료 - 다음 TP 설정 없음")

    except Exception as e:
        print(f"[오류] TP 주문 {tp_id} 처리 중 문제 발생: {e}")
        print(traceback.format_exc())

# 7. 스레드 풀 구현 (병렬 작업 처리용)
thread_pool = ThreadPoolExecutor(max_workers=5)

def run_in_thread(func, *args, **kwargs):
    """
    함수를 스레드 풀에서 실행
    """
    return thread_pool.submit(func, *args, **kwargs)


# 8. init_db() 함수 수정
def init_db():
    """데이터베이스 초기화 및 테이블 생성 (개선된 버전)"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # 기존 trades 테이블 생성
            c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                current_price REAL,
                decision TEXT,
                reason TEXT,
                position_size INTEGER,
                leverage INTEGER,
                wallet_balance REAL,
                estimated_fee REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                ema_20 REAL,
                ema_50 REAL,
                bb_upper REAL,
                bb_lower REAL,
                fear_greed_value TEXT,
                success INTEGER,
                reflection TEXT,
                position_type TEXT,
                entry_price REAL,
                exit_price REAL,
                trade_status TEXT,
                profit_loss REAL,
                trade_id TEXT,
                ai_confidence_score REAL,
                position_duration INTEGER,
                entry_conditions TEXT,
                exit_conditions TEXT,
                social_sentiment TEXT,
                news_keywords TEXT,
                trading_analysis_ai TEXT,
                funding_fee REAL,
                total_fee REAL,
                roi_percentage REAL,
                stop_loss_price REAL,
                is_stop_loss INTEGER DEFAULT 0,
                stop_loss_pct REAL
            )
            ''')

            # market_analysis_records 테이블 생성
            c.execute('''
            CREATE TABLE IF NOT EXISTS market_analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level_type TEXT,
                price REAL,
                strength REAL,
                source TEXT,
                timeframe TEXT
            )
            ''')

            # timestamp 컬럼 추가 시도
            add_timestamp_column_to_market_analysis_records(conn)

            conn.commit()
            print("[정보] 데이터베이스 테이블이 성공적으로 생성되었습니다.")
            return conn
    except Exception as e:
        print(f"[오류] 데이터베이스 초기화 중 문제 발생: {e}")
        return None

# 9. 기존 DB 종속 함수들 수정 예시
def store_take_profit_orders(trade_id, position_type, tp_levels, tp_prices, tp_quantities, order_ids=None):
    """
    익절 주문 정보를 DB에 저장 (개선된 버전)
    """
    def _store_operation(conn):
        cursor = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 기존 주문 초기화
        cursor.execute("""
            UPDATE take_profit_orders
            SET status = 'Cancelled'
            WHERE trade_id = ? AND status = 'Active'
        """, (trade_id,))
        
        # 새로운 주문 저장
        for i, (level, price, qty) in enumerate(zip(tp_levels, tp_prices, tp_quantities)):
            order_id = order_ids[i] if order_ids and i < len(order_ids) else None
            
            cursor.execute("""
                INSERT INTO take_profit_orders 
                (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                position_type,
                level,
                price,
                qty,
                'Active' if i == 0 else 'Pending',  # 첫 번째만 Active
                created_at,
                order_id
            ))
        
        return True
    
    try:
        result = execute_with_retry(_store_operation)
        print(f"[정보] {len(tp_levels)}개의 익절 계획이 DB에 저장되었습니다.")
        return result
    except Exception as e:
        print(f"[오류] 익절 주문 정보 저장 중 문제 발생: {e}")
        return False 


def store_take_profit_orders_enhanced(conn, trade_id: str, position_type: str, tp_prices: List[float], 
                                    tp_quantities: List[float], strategy_type: str = "STANDARD", 
                                    first_active_order_id: str = None) -> bool:
    """
    익절 주문 계획을 DB에 저장하는 향상된 함수
    
    Args:
        conn: 데이터베이스 연결 객체
        trade_id: 거래 ID
        position_type: 포지션 유형 ("Long" 또는 "Short")
        tp_prices: 익절 가격 리스트
        tp_quantities: 익절 수량 리스트
        strategy_type: 전략 유형 ("TREND", "RANGE", "VOLATILE", "BREAKOUT", "STANDARD" 등)
        first_active_order_id: 첫 번째 활성화된 주문 ID (선택적)
        
    Returns:
        bool: 성공 여부
    """
    try:
        cursor = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. 기존 주문 초기화
        cursor.execute("""
            UPDATE take_profit_orders
            SET status = 'Cancelled'
            WHERE trade_id = ? AND status IN ('Active', 'Pending')
        """, (trade_id,))
        
        # 2. 새로운 주문 계획 모두 저장
        for i, (price, qty) in enumerate(zip(tp_prices, tp_quantities)):
            level = i + 1
            status = 'Active' if i == 0 else 'Pending'
            current_order_id = first_active_order_id if i == 0 else None
            
            # 2.1 메타데이터 JSON 생성 (전략 유형, 생성 방법 등)
            metadata = json.dumps({
                'strategy_type': strategy_type,
                'creation_method': 'AUTO',
                'price_basis': 'RISK_REWARD' if strategy_type != "STANDARD" else 'PERCENTAGE',
                'market_condition': strategy_type.lower(),
                'creation_timestamp': created_at
            })
            
            # 2.2 TP 주문 정보 DB에 저장
            try:
                cursor.execute("""
                    INSERT INTO take_profit_orders 
                    (trade_id, position_type, tp_level, tp_price, tp_quantity, status, 
                     created_at, order_id, metadata, is_additional)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    position_type,
                    level,
                    price,
                    qty,
                    status,
                    created_at,
                    current_order_id,
                    metadata,
                    0  # 추가 매수 아님
                ))
            except sqlite3.OperationalError:
                # metadata 컬럼이 없는 경우, 기존 방식으로 저장
                cursor.execute("""
                    INSERT INTO take_profit_orders 
                    (trade_id, position_type, tp_level, tp_price, tp_quantity, status, 
                     created_at, order_id, is_additional)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    position_type,
                    level,
                    price,
                    qty,
                    status,
                    created_at,
                    current_order_id,
                    0  # 추가 매수 아님
                ))
        
        # 3. 거래 테이블 TP 정보 업데이트
        # 향후 조회를 쉽게 하기 위해 거래 정보에도 TP 요약 저장
        try:
            tp_summary = json.dumps({
                'strategy': strategy_type,
                'levels': len(tp_prices),
                'prices': tp_prices,
                'quantities': [float(q) for q in tp_quantities]
            })
            
            cursor.execute("""
                UPDATE trades
                SET tp_strategy = ?
                WHERE trade_id = ?
            """, (tp_summary, trade_id))
        except sqlite3.OperationalError:
            # tp_strategy 컬럼이 없는 경우 무시
            pass
            
        conn.commit()
        print(f"[성공] {len(tp_prices)}단계 익절 계획이 DB에 저장되었습니다.")
        return True
        
    except Exception as e:
        print(f"[오류] 익절 주문 정보 저장 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'cursor' in locals():
            conn.rollback()
        return False

def store_additional_tp_order(conn, trade_id, position_type, tp_price, tp_qty):
    """추가 매수에 대한 익절 주문 정보만 DB에 저장"""
    try:
        cursor = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 새로운 주문 저장 (추가 매수 표시)
        cursor.execute("""
            INSERT INTO take_profit_orders 
            (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, is_additional)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id,
            position_type,
            1,  # 항상 첫 번째 레벨에 추가
            tp_price,
            tp_qty,
            'Active',
            created_at,
            1  # 추가 매수 표시
        ))
        
        conn.commit()
        print(f"[정보] 추가 매수에 대한 익절 계획이 DB에 저장되었습니다.")
        return True
        
    except Exception as e:
        print(f"[오류] 추가 익절 주문 정보 저장 중 문제 발생: {e}")
        if 'cursor' in locals():
            conn.rollback()
        return False
    
def check_and_update_tp_orders(conn, trade_id: str, position_type: str) -> bool:
    """
    TP 주문 실행 확인 및 다음 단계 설정
    """
    try:
        cursor = conn.cursor()
        
        # 현재 활성화된 TP 주문 확인
        cursor.execute("""
            SELECT tp_level, tp_price, tp_quantity, order_id, status
            FROM take_profit_orders
            WHERE trade_id = ? AND position_type = ?
            ORDER BY tp_level
        """, (trade_id, position_type))
        
        tp_orders = cursor.fetchall()
        if not tp_orders:
            print(f"[경고] 거래 ID {trade_id}의 TP 주문 정보를 찾을 수 없음")
            return False
            
        # 실행된 TP 찾기
        executed_level = None
        for order in tp_orders:
            level, price, quantity, order_id, status = order
            if status == 'Executed':
                executed_level = level
                break
                
        if executed_level and executed_level < 3:  # 3단계가 마지막
            # 다음 단계 TP 설정
            next_level = executed_level + 1
            next_order = tp_orders[next_level - 1]  # 인덱스는 0부터 시작
            
            position_idx = 1 if position_type == "Long" else 2
            response = bybit_client.set_trading_stop(
                category="linear",
                symbol=SYMBOL,
                positionIdx=position_idx,
                takeProfit=str(next_order[1]),  # tp_price
                tpSize=str(next_order[2]),      # tp_quantity
                tpTriggerBy="LastPrice",
                tpslMode="Partial"  # 파라미터 추가
            )
            
            if response["retCode"] == 0:
                print(f"[성공] {next_level}단계 TP 설정 완료: ${next_order[1]}, 수량: {next_order[2]}")
                
                # DB 업데이트
                cursor.execute("""
                    UPDATE take_profit_orders
                    SET order_id = ?, status = 'Active'
                    WHERE trade_id = ? AND tp_level = ?
                """, (response["result"].get("orderId"), trade_id, next_level))
                conn.commit()
                return True
            else:
                print(f"[경고] {next_level}단계 TP 설정 실패: {response['retMsg']}")
                return False
                
        return True
        
    except Exception as e:
        print(f"[오류] TP 주문 업데이트 중 문제 발생: {e}")
        if 'cursor' in locals():
            conn.rollback()
        return False

def adjust_stop_loss_after_first_tp(conn, trade_id, position_type, entry_price):
    """
    첫 번째 익절 이후 스탑로스 조정
    """
    try:
        # 현재 포지션 정보 가져오기
        position_info = get_position_info(bybit_client)
        if not position_info or position_info[position_type]["size"] <= 0:
            print(f"[경고] {position_type} 포지션 정보를 찾을 수 없습니다.")
            return False
            
        # 현재 포지션 수량 가져오기
        current_size = position_info[position_type]["size"]
        
        # 현재 가격 확인
        current_price = float(fetch_candle_data(interval="1", limit=1)['Close'].iloc[-1])
            
        # 손절가를 진입가 근처로 조정
        if position_type == "Long":
            # 롱 포지션은 진입가보다 낮게 설정
            adjustment = 0.003  # 0.3% 조정
            new_stop_price = entry_price * (1 - adjustment)
        else:
            # 숏 포지션은 진입가보다 높게 설정 (최소 1% 이상)
            adjustment = 0.01  # 1% 조정
            new_stop_price = entry_price * (1 + adjustment)
            
            # 추가: 현재 가격이 진입가보다 낮다면, 현재 가격보다 1% 높게 설정
            if current_price < entry_price:
                new_stop_price = current_price * 1.01
                
        position_idx = 1 if position_type == "Long" else 2
        
        # 기존 스탑로스 주문 취소
        cancel_stop_orders(SYMBOL, position_idx)
        time.sleep(1)  # API 제한 방지
        
        # 새 스탑로스 설정
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=SYMBOL,
            positionIdx=position_idx,
            stopLoss=str(new_stop_price),
            slSize=str(current_size),  # 전체 포지션 수량 사용
            slTriggerBy="LastPrice",
            tpslMode="Partial"
        )
        
        if response["retCode"] == 0:
            print(f"[성공] 첫 익절 이후 스탑로스 조정 완료: ${new_stop_price:.2f}")
            # DB에 스탑로스 조정 내역 기록
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET stop_loss_price = ?
                WHERE trade_id = ?
            """, (new_stop_price, trade_id))
            conn.commit()
            return True
        else:
            print(f"[경고] 스탑로스 조정 실패: {response['retMsg']}")
            return False
    except Exception as e:
        print(f"[오류] 스탑로스 조정 중 문제 발생: {e}")
        return False

def set_breakeven_stop_loss(conn, trade_id, position_type, entry_price):
    """
    두 번째 익절 이후 스탑로스를 손익분기점으로 조정
    """
    try:
        position_info = get_position_info(bybit_client)
        if not position_info or position_info[position_type]["size"] <= 0:
            print(f"[경고] {position_type} 포지션 정보를 찾을 수 없습니다.")
            return False
            
        # 현재 포지션 수량 가져오기
        current_size = position_info[position_type]["size"]
        
        # 현재 가격 확인
        current_price = float(fetch_candle_data(interval="1", limit=1)['Close'].iloc[-1])
        
        # 약간의 마진을 추가하여 수수료를 커버
        fee_adjustment = 0.001  # 0.1% 수수료 고려
        
        if position_type == "Long":
            # 롱 포지션의 경우 손익분기점은 진입가 + 수수료
            breakeven_price = entry_price * (1 + fee_adjustment)
        else:
            # 숏 포지션의 경우 손익분기점 로직 수정
            # 진입가보다 높게 설정해야 함 (최소 1% 이상)
            breakeven_price = entry_price * 1.01
            
            # 추가: 현재 가격이 진입가보다 낮다면, 현재 가격보다 1% 높게 설정
            if current_price < entry_price:
                breakeven_price = current_price * 1.01
            
        position_idx = 1 if position_type == "Long" else 2
        
        # 기존 스탑로스 주문 취소
        cancel_stop_orders(SYMBOL, position_idx)
        time.sleep(1)  # API 제한 방지
        
        # 새 스탑로스 설정
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=SYMBOL,
            positionIdx=position_idx,
            stopLoss=str(breakeven_price),
            slSize=str(current_size),  # 전체 포지션 수량 사용
            slTriggerBy="LastPrice",
            tpslMode="Partial"
        )
        
        if response["retCode"] == 0:
            print(f"[성공] 손익분기점 스탑로스 설정 완료: ${breakeven_price:.2f}")
            # DB에 스탑로스 조정 내역 기록
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET stop_loss_price = ?
                WHERE trade_id = ?
            """, (breakeven_price, trade_id))
            conn.commit()
            return True
        else:
            print(f"[경고] 손익분기점 스탑로스 설정 실패: {response['retMsg']}")
            return False
    except Exception as e:
        print(f"[오류] 손익분기점 스탑로스 설정 중 문제 발생: {e}")
        return False

def apply_trailing_stop_after_tp(conn, trade_id: str, position_type: str, entry_price: float, 
                                leverage: int, profit_pct: float, tp_level: int, 
                                market_data: Dict = None) -> bool:
    """
    익절 후 트레일링 스탑을 적용하는 향상된 함수
    - 시장 상황에 따른 동적 트레일링 스탑 조정
    - 익절 단계별 최적화된 전략
    - 급격한 가격 변동 대응 메커니즘
    
    Args:
        conn: 데이터베이스 연결 객체
        trade_id: 거래 ID
        position_type: 포지션 유형 ("Long" 또는 "Short")
        entry_price: 진입 가격
        leverage: 레버리지
        profit_pct: 현재 익절의 수익률(%)
        tp_level: 현재 익절 레벨
        market_data: 시장 데이터 (선택적)
        
    Returns:
        bool: 성공 여부
    """
    try:
        position_idx = 1 if position_type == "Long" else 2
        
        # 현재 가격 확인
        current_price = None
        if market_data:
            current_price = float(market_data['timeframes']["15"].iloc[-1]['Close'])
        
        if not current_price:
            # 시장 데이터가 없으면 캔들 데이터 직접 조회
            candle_data = fetch_candle_data(interval="1", limit=1)
            if candle_data.empty:
                print("[경고] 현재 가격을 확인할 수 없습니다. 트레일링 스탑 적용 취소.")
                return False
            current_price = float(candle_data['Close'].iloc[0])
        
        # 변동성 계산 - 동적 조정에 사용
        volatility = 0.01  # 기본값
        if market_data:
            atr = float(market_data['timeframes']["15"].iloc[-1]['ATR'])
            volatility = atr / current_price  # 현재 가격 대비 ATR 비율
        
        # 변동성 기반 안전 계수 계산
        safety_factor = 1.0
        if volatility > 0.02:  # 2% 이상 높은 변동성
            safety_factor = 1.3  # 더 보수적 접근
        elif volatility > 0.01:  # 1% 이상 중간 변동성
            safety_factor = 1.1
        elif volatility < 0.005:  # 0.5% 미만 낮은 변동성
            safety_factor = 0.9  # 더 공격적 접근
        
        # 익절 레벨에 따른 보호 비율 설정 (기본 로직)
        if tp_level == 1:
            base_protection_ratio = 0.5  # 첫 번째 TP 후 수익의 50% 보호
        elif tp_level == 2:
            base_protection_ratio = 0.7  # 두 번째 TP 후 수익의 70% 보호
        else:
            base_protection_ratio = 0.9  # 세 번째 TP 후 수익의 90% 보호
        
        # 변동성과 이익에 따른 동적 보호 비율 조정
        dynamic_ratio = base_protection_ratio
        
        # 높은 이익일수록 더 많이 보호
        if profit_pct > 10:
            dynamic_ratio = min(0.95, base_protection_ratio + 0.15)  # 최대 95%까지 보호
        elif profit_pct > 5:
            dynamic_ratio = min(0.9, base_protection_ratio + 0.1)
            
        # 변동성 기반 추가 조정
        dynamic_ratio *= safety_factor
        
        # 최종 보호 비율은 40%~95% 범위로 제한
        final_protection_ratio = max(0.4, min(0.95, dynamic_ratio))
        
        # 시장 상태 평가 - ADX로 추세 강도 확인
        trend_strength = 0
        if market_data:
            adx = float(market_data['timeframes'].get("60", market_data['timeframes']["15"]).iloc[-1]['ADX'])
            trend_strength = adx
            
            # 추세가 강하면 더 느슨하게, 약하면 더 타이트하게 조정
            if adx > 30:  # 강한 추세
                if position_type == "Long" and market_data['timeframes']["15"].iloc[-1]['+DI'] > market_data['timeframes']["15"].iloc[-1]['-DI']:
                    final_protection_ratio = max(0.4, final_protection_ratio - 0.1)  # 상승 추세에서 롱 포지션은 더 느슨하게
                elif position_type == "Short" and market_data['timeframes']["15"].iloc[-1]['+DI'] < market_data['timeframes']["15"].iloc[-1]['-DI']:
                    final_protection_ratio = max(0.4, final_protection_ratio - 0.1)  # 하락 추세에서 숏 포지션은 더 느슨하게
            elif adx < 20:  # 약한 추세
                final_protection_ratio = min(0.95, final_protection_ratio + 0.1)  # 더 타이트하게
        
        # 이익 금액 계산 (레버리지 고려)
        if position_type == "Long":
            profit_amount = current_price - entry_price
        else:  # Short
            profit_amount = entry_price - current_price
        
        # 보호할 이익 계산
        protected_profit = profit_amount * final_protection_ratio
        
        # 트레일링 스탑 가격 계산
        if position_type == "Long":
            trailing_stop_price = max(entry_price, round(current_price - protected_profit, 1))
        else:  # Short
            trailing_stop_price = min(entry_price, round(current_price + protected_profit, 1))
        
        # 진입가보다는 유리한 위치에 스탑 설정 보장 (브레이크이븐 보장)
        if tp_level >= 2:  # 두 번째 이상 익절에서는 최소 손익분기점 이상 보장
            fee_buffer = entry_price * 0.001  # 수수료 버퍼 (0.1%)
            if position_type == "Long":
                trailing_stop_price = max(trailing_stop_price, entry_price + fee_buffer)
            else:
                trailing_stop_price = min(trailing_stop_price, entry_price - fee_buffer)
        
        print(f"[정보] {tp_level}단계 익절 후 트레일링 스탑 설정:")
        print(f"  진입가: ${entry_price:.2f}")
        print(f"  현재가: ${current_price:.2f}")
        print(f"  변동성: {volatility*100:.2f}%")
        print(f"  추세강도(ADX): {trend_strength:.1f}")
        print(f"  보호비율: {final_protection_ratio*100:.1f}%")
        print(f"  트레일링 스탑 가격: ${trailing_stop_price:.2f}")
        
        # 현재 포지션 확인
        position_info = get_position_info(bybit_client)
        if not position_info or position_info[position_type]["size"] <= 0:
            print(f"[경고] {position_type} 포지션이 없습니다. 트레일링 스탑 적용 취소.")
            return False
        
        # 기존 스탑로스 주문 취소
        cancel_stop_orders(SYMBOL, position_idx)
        time.sleep(0.5)  # API 제한 방지
        
        # 현재 포지션 수량 가져오기
        current_size = position_info[position_type]["size"]
        
        # 트레일링 스탑 설정
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=SYMBOL,
            positionIdx=position_idx,
            stopLoss=str(trailing_stop_price),
            slSize=str(current_size),  # 전체 남은 포지션에 적용
            slTriggerBy="LastPrice",
            tpslMode="Partial"  # 부분 청산 모드
        )
        
        if response["retCode"] == 0:
            print(f"[성공] 트레일링 스탑 설정 완료: ${trailing_stop_price:.2f}")
            
            # DB에 트레일링 스탑 정보 업데이트
            try:
                cursor = conn.cursor()
                
                # 트레일링 스탑 정보를 저장할 컬럼이 있는지 확인
                cursor.execute("PRAGMA table_info(trades)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'trailing_stop_price' in columns and 'trailing_stop_level' in columns and 'trailing_stop_updated_at' in columns:
                    cursor.execute("""
                        UPDATE trades
                        SET trailing_stop_price = ?,
                            trailing_stop_level = ?,
                            trailing_stop_updated_at = ?
                        WHERE trade_id = ?
                    """, (
                        trailing_stop_price,
                        tp_level,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        trade_id
                    ))
                else:
                    # 컬럼이 없다면 일반 스탑 로스 정보 업데이트
                    cursor.execute("""
                        UPDATE trades
                        SET stop_loss_price = ?
                        WHERE trade_id = ?
                    """, (trailing_stop_price, trade_id))
                
                conn.commit()
                
                # 스탑로스 DB 업데이트
                store_stop_loss_db(position_type, entry_price, trailing_stop_price)
            except Exception as db_error:
                print(f"[경고] DB 업데이트 중 오류: {db_error}")
                if 'cursor' in locals() and conn:
                    conn.rollback()
            
            return True
        else:
            print(f"[경고] 트레일링 스탑 설정 실패: {response['retMsg']}")
            return False
    
    except Exception as e:
        print(f"[오류] 트레일링 스탑 적용 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'conn' in locals():
            conn.rollback()
        return False

# 여기 아래에 check_auxiliary_signals 함수를 추가합니다.
def check_auxiliary_signals(df: pd.DataFrame, position: str) -> bool:
    """
    개선된 진입 조건 체크 - 15분, 60분, 240분 타임프레임 분석 반영
    - RSI: 범위 확대
    - MACD: 신호선 교차 강도 완화
    - 볼린저 밴드: 스퀴즈 조건 완화
    """
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 변동성 체크 (0.5%로 완화)
        volatility = latest['ATR'] / latest['Close'] * 100
        high_volatility = volatility > 0.5
        
        # 볼린저 밴드 조건
        bb_width = (latest['BB_upper'] - latest['BB_lower']) / latest['BB_mavg']
        squeeze = bb_width < 0.05  # 5%로 완화
        
        if position == 'Long':
            # Long 진입 조건
            return (
                # RSI 상단 제한 완화 (70)
                latest['RSI'] < 70 and
                
                # MACD 조건 완화 (Signal의 70%)
                latest['MACD'] > latest['MACD_signal'] * 0.7 and
                
                # EMA 조건 (3% 여유)
                latest['Close'] > latest['EMA_20'] * 0.97 and
                
                # 볼륨 체크 (20일 평균의 60%)
                latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] * 0.6
            )
                
        elif position == 'Short':
            # Short 진입 조건
            return (
                # RSI 하단 제한 완화 (30)
                latest['RSI'] > 30 and
                
                # MACD 조건 완화 (Signal의 130%)
                latest['MACD'] < latest['MACD_signal'] * 1.3 and
                
                # EMA 조건 (3% 여유)
                latest['Close'] < latest['EMA_20'] * 1.03 and
                
                # 볼륨 체크 (20일 평균의 60%)
                latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] * 0.6
            )
        
        return False
        
    except Exception as e:
        print(f"[오류] check_auxiliary_signals 실행 중 문제 발생: {e}")
        return False

def set_take_profit(symbol: str, qty: float, entry_price: float, position_type: str, df: pd.DataFrame, leverage: int, conn=None, trade_id=None) -> bool:    
    try:
        # 시장 상태 분석
        latest = df.iloc[-1]
        adx = float(latest['ADX'])
        atr = float(latest['ATR'])
        bb_width = (float(latest['BB_upper']) - float(latest['BB_lower'])) / float(latest['BB_mavg'])
        volatility = atr / entry_price
        rsi = float(latest['RSI'])
        
        # 기본 수수료 고려 (진입 + 청산)
        total_fee = FEE_RATE * 2  # 0.12%
        
        # 최소 수익 목표를 수수료의 3배로 설정
        min_target = total_fee * 3  # 0.36%
        
        # 시장 상태에 따른 동적 목표 설정
        base_target = 0.15  # 기본 15%
        
        # 추세 강도에 따른 조정
        trend_multiplier = min(1.5, max(0.5, adx/25))
        
        # 변동성에 따른 조정
        vol_multiplier = min(1.3, max(0.7, volatility * 100))
        
        # RSI 기반 조정
        if position_type == "Long":
            rsi_multiplier = min(1.2, max(0.8, (70 - rsi) / 30))
        else:
            rsi_multiplier = min(1.2, max(0.8, (rsi - 30) / 30))
            
        # 최종 목표 수익률 계산
        if adx > 25 and bb_width > 0.03:  # 추세장
            target_profits = [
                max(min_target, base_target * trend_multiplier * vol_multiplier * rsi_multiplier),
                max(min_target * 2, base_target * 1.5 * trend_multiplier * vol_multiplier * rsi_multiplier),
                max(min_target * 3, base_target * 2.0 * trend_multiplier * vol_multiplier * rsi_multiplier)
            ]
            qty_distribution = [0.3, 0.3, 0.4]
        else:  # 횡보장
            target_profits = [
                max(min_target, base_target * 0.7 * vol_multiplier * rsi_multiplier),
                max(min_target * 1.5, base_target * 1.0 * vol_multiplier * rsi_multiplier),
                max(min_target * 2, base_target * 1.3 * vol_multiplier * rsi_multiplier)
            ]
            qty_distribution = [0.4, 0.4, 0.2]

        print(f"\n=== 이익실현 설정 분석 ===")
        print(f"ADX: {adx:.2f}")
        print(f"변동성: {volatility:.4f}")
        print(f"RSI: {rsi:.2f}")
        print(f"추세 승수: {trend_multiplier:.2f}")
        print(f"변동성 승수: {vol_multiplier:.2f}")
        print(f"RSI 승수: {rsi_multiplier:.2f}")
        print(f"총 수수료: {total_fee*100:.2f}%")
        print(f"최소 수익 목표: {min_target*100:.2f}%")
        print(f"시장상태: {'추세장' if (adx > 25 and bb_width > 0.03) else '횡보장'}")

        # 레버리지를 고려한 실제 가격 변동 필요량 계산
        price_moves = [profit/leverage for profit in target_profits]

        # 이익실현 가격 계산
        if position_type == "Long":
            tp_prices = [
                round(entry_price * (1 + move), 1) 
                for move in price_moves
            ]
        else:  # Short
            tp_prices = [
                round(entry_price * (1 - move), 1)
                for move in price_moves
            ]

        # 각 단계별 수량 계산
        qty_per_level = [round(qty * dist, 3) for dist in qty_distribution]
        
        print("\n[이익실현 설정 상세]")
        for i, (price, amount, target) in enumerate(zip(tp_prices, qty_per_level, target_profits)):
            print(f"단계 {i+1}:")
            print(f"- 목표 수익률: {target*100:.1f}%")
            print(f"- 필요 가격 변동: {price_moves[i]*100:.2f}%")
            print(f"- 청산 가격: ${price:,.2f}")
            print(f"- 수량: {amount} BTC")

        # 이익실현 주문 설정
        tp_order_ids = []
        for i, (tp_price, tp_qty) in enumerate(zip(tp_prices, qty_per_level)):
            response = bybit_client.set_trading_stop(
                category="linear",
                symbol=symbol,
                positionIdx=1 if position_type == "Long" else 2,
                takeProfit=str(tp_price),
                tpSize=str(tp_qty),
                tpTriggerBy="LastPrice",
                tpslMode="Partial"  # 파라미터 추가
            )
            if response["retCode"] != 0:
                print(f"[경고] {i+1}번째 이익실현 주문 설정 실패: {response['retMsg']}")
                return False
                
            # 주문 ID 저장 (만약 API 응답에 주문 ID가 있다면)
            if "result" in response and "orderId" in response["result"]:
                tp_order_ids.append(response["result"]["orderId"])
            else:
                tp_order_ids.append(None)
        
        # DB에 익절 주문 정보 저장
        if conn and trade_id:
            tp_levels = list(range(1, len(tp_prices) + 1))  # 1, 2, 3, ...
            store_take_profit_orders(conn, trade_id, position_type, tp_levels, tp_prices, qty_per_level, tp_order_ids)
            
        print(f"[성공] 단계별 이익실현 설정 완료")
        return True
        
    except Exception as e:
        print(f"[오류] 이익실현 주문 설정 중 문제 발생: {e}")
        return False


def determine_tp_strategy(market_structure: Dict, position_type: str, key_levels: List[Dict], 
                        entry_price: float, stop_loss_price: float = None) -> Dict:
    """
    시장 구조와 주요 레벨을 기반으로 최적의 이익실현 전략 결정 (개선된 버전)
    - 시장 구조별 맞춤형 전략 적용
    - 저항/지지선 강도 기반 정밀 익절 가격 설정
    - 동적 손익비 및 시장 심리 반영
    - 스마트 수량 분배로 최적 수익 실현
    
    Args:
        market_structure: 시장 구조 정보 {'type': 'TREND'|'RANGE'|'VOLATILE'|'CONTRACTION', 
                                     'direction': 'UP'|'DOWN'|'NEUTRAL',
                                     'strength': float,
                                     'confidence': float}
        position_type: 포지션 유형 ("Long" 또는 "Short")
        key_levels: 주요 지지/저항 레벨 리스트 [{'price': float, 'strength': float, 'type': str}, ...]
        entry_price: 진입 가격
        stop_loss_price: 손절 가격 (선택적)
    
    Returns:
        Dict: {
            'tp_prices': [가격1, 가격2, 가격3], 
            'tp_quantities': [수량1, 수량2, 수량3],
            'strategy_type': 'TREND'|'RANGE'|'VOLATILE'|'BREAKOUT'|'SCALP'
        }
    """
    try:
        strategy = {}
        
        # 기본값 설정
        tp_prices = []
        tp_quantities = []
        strategy_type = "STANDARD"  # 기본 전략 유형
        
        # 1. 시장 구조 분석 및 기본 전략 결정
        market_type = market_structure.get('type', 'RANGE')
        market_direction = market_structure.get('direction', 'NEUTRAL')
        market_strength = market_structure.get('strength', 0.5)
        market_confidence = market_structure.get('confidence', 0.5)
        
        # 포지션과 시장 방향 일치 여부
        direction_aligned = (position_type == "Long" and market_direction == "UP") or \
                           (position_type == "Short" and market_direction == "DOWN")
        
        # 2. 손익비(Risk-Reward Ratio) 계산
        risk_per_unit = 0.0
        if stop_loss_price:
            if position_type == "Long":
                risk_per_unit = abs(entry_price - stop_loss_price)
            else:
                risk_per_unit = abs(stop_loss_price - entry_price)
        else:
            # 손절가 없을 경우 진입가의 2%를 기본 리스크로 설정
            risk_per_unit = entry_price * 0.02
        
        # 3. 주요 저항/지지 레벨 분석 및 강도 평가
        if position_type == "Long":
            # 롱 포지션의 경우 entry_price보다 높은 저항 레벨을 대상으로 함
            valid_levels = [level for level in key_levels 
                            if level['price'] > entry_price and level['type'] == 'resistance']
            # 가격 오름차순 정렬
            valid_levels = sorted(valid_levels, key=lambda x: x['price'])
        else:
            # 숏 포지션의 경우 entry_price보다 낮은 지지 레벨을 대상으로 함
            valid_levels = [level for level in key_levels 
                           if level['price'] < entry_price and level['type'] == 'support']
            # 가격 내림차순 정렬
            valid_levels = sorted(valid_levels, key=lambda x: x['price'], reverse=True)
        
        # 4. 시장 구조별 전략 결정 및 익절 계산
        # 4.1 추세장 전략 - 추세 강도와 방향성 일치도에 따른 최적화
        if market_type == "TREND":
            strategy_type = "TREND"
            print(f"[전략] 추세장 전략 적용 (강도: {market_strength:.2f}, 일치도: {market_confidence:.2f})")
            
            # 추세 강도가 강할수록 더 높은 목표 설정
            strength_factor = 1.0 + (market_strength * 0.5)
            
            # 방향 일치 여부에 따른 조정
            if direction_aligned:
                # 방향 일치시 더 공격적인 목표
                confidence_factor = 1.0 + (market_confidence * 0.3)
                print(f"  - 추세 방향 일치: 목표 상향 조정 (x{confidence_factor:.2f})")
            else:
                # 방향 불일치시 더 보수적인 목표
                confidence_factor = 0.7 + (market_confidence * 0.2)
                print(f"  - 추세 방향 불일치: 목표 하향 조정 (x{confidence_factor:.2f})")
            
            # 레벨 기반 목표 설정 (레벨이 충분한 경우)
            if len(valid_levels) >= 3:
                # 유효 레벨 3개 선택
                selected_levels = valid_levels[:3]
                
                # 레벨 강도에 따른 진입 지점 최적화
                tp_prices = []
                for level in selected_levels:
                    # 강한 저항/지지선은 약간 앞에서 익절
                    strength_adjustment = 1.0 - (level['strength'] * 0.1)  # 강도가 높을수록 더 일찍 익절
                    
                    if position_type == "Long":
                        # 롱: 저항선 직전에 익절 (강도가 높을수록 더 일찍)
                        price_adjustment = level['price'] * strength_adjustment if strength_adjustment < 1.0 else level['price'] * 0.995
                        tp_prices.append(round(price_adjustment, 1))
                    else:
                        # 숏: 지지선 직전에 익절 (강도가 높을수록 더 일찍)
                        price_adjustment = level['price'] * (2 - strength_adjustment) if strength_adjustment < 1.0 else level['price'] * 1.005
                        tp_prices.append(round(price_adjustment, 1))
                    
                print(f"  - 주요 레벨 기반 익절 가격: {tp_prices}")
            else:
                # 주요 레벨이 부족한 경우 손익비 기반 설정
                # 추세장에서는 기본적으로 더 큰 손익비 적용
                base_rr_ratios = [1.5, 2.5, 4.0]
                rr_ratios = [r * strength_factor * confidence_factor for r in base_rr_ratios]
                
                if position_type == "Long":
                    tp_prices = [
                        round(entry_price + (risk_per_unit * rr_ratios[0]), 1),
                        round(entry_price + (risk_per_unit * rr_ratios[1]), 1),
                        round(entry_price + (risk_per_unit * rr_ratios[2]), 1)
                    ]
                else:
                    tp_prices = [
                        round(entry_price - (risk_per_unit * rr_ratios[0]), 1),
                        round(entry_price - (risk_per_unit * rr_ratios[1]), 1),
                        round(entry_price - (risk_per_unit * rr_ratios[2]), 1)
                    ]
                
                print(f"  - 손익비 기반 익절 가격: {tp_prices} (손익비: {[f'{r:.1f}' for r in rr_ratios]})")
                
                # 가장 가까운 저항/지지선 활용하여 조정
                if valid_levels:
                    for i in range(len(tp_prices)):
                        current_tp = tp_prices[i]
                        
                        # 현재 TP 가격과 가장 가까운 레벨 찾기
                        closest_idx = None
                        min_dist_pct = float('inf')
                        
                        for j, level in enumerate(valid_levels):
                            dist_pct = abs(level['price'] - current_tp) / current_tp
                            if dist_pct < min_dist_pct:
                                min_dist_pct = dist_pct
                                closest_idx = j
                        
                        # 5% 이내 거리에 레벨이 있다면 조정
                        if closest_idx is not None and min_dist_pct < 0.05:
                            level = valid_levels[closest_idx]
                            strength_adjustment = 1.0 - (level['strength'] * 0.1)
                            
                            if position_type == "Long":
                                if level['price'] > current_tp:  # 위쪽 저항선
                                    new_tp = round(level['price'] * strength_adjustment, 1)
                                    tp_prices[i] = min(new_tp, level['price'] * 0.995)  # 저항선 직전에서 익절
                            else:
                                if level['price'] < current_tp:  # 아래쪽 지지선
                                    new_tp = round(level['price'] * (2 - strength_adjustment), 1)
                                    tp_prices[i] = max(new_tp, level['price'] * 1.005)  # 지지선 직전에서 익절
                            
                            print(f"  - 레벨 조정된 TP{i+1}: ${tp_prices[i]:.1f} (레벨 ${level['price']:.1f}, 강도 {level['strength']:.2f})")
            
            # 추세장에서는 더 큰 목표에 더 많은 물량 배치
            # 추세 강도가 강할수록 마지막 TP에 더 많은 물량 배치
            last_tp_weight = 0.35 + (market_strength * 0.15)  # 0.35 ~ 0.5
            mid_tp_weight = 0.35 - (market_strength * 0.05)   # 0.35 ~ 0.3
            first_tp_weight = 1.0 - last_tp_weight - mid_tp_weight
            
            tp_quantities = [first_tp_weight, mid_tp_weight, last_tp_weight]
            
            # 방향 불일치 시 수량 분배 조정 (안전한 방향으로)
            if not direction_aligned:
                # 첫 번째 목표에 더 많은 물량 배치
                tp_quantities = [0.5, 0.3, 0.2]
                print("  - 추세 방향 불일치: 안전 분배 적용 (50/30/20)")
        
        # 4.2 횡보장 전략 - 좁은 가격 범위와 빠른 이익실현
        elif market_type == "RANGE":
            strategy_type = "RANGE"
            print(f"[전략] 횡보장 전략 적용 (강도: {market_strength:.2f})")
            
            # 횡보장에서는 좁은 범위 내에서 더 일찍 이익 실현
            # 횡보 강도가 강할수록 더 좁은 목표 설정
            range_factor = 1.0 - (market_strength * 0.3)  # 0.7 ~ 1.0
            
            # 주요 레벨 활용 (우선순위)
            if len(valid_levels) >= 2:
                # 첫 두 레벨 사용
                tp_prices = []
                
                # 첫 두 레벨을 사용하되, 횡보 강도에 따라 조정
                for i, level in enumerate(valid_levels[:2]):
                    # 강한 레벨일수록 더 안전하게 익절
                    strength_adjustment = 1.0 - (level['strength'] * 0.15)  # 최대 15% 조정
                    
                    if position_type == "Long":
                        # 롱: 저항선 직전에 익절
                        tp_prices.append(round(level['price'] * strength_adjustment, 1))
                    else:
                        # 숏: 지지선 직전에 익절
                        tp_prices.append(round(level['price'] * (2 - strength_adjustment), 1))
                
                # 세 번째 TP는 첫 두 레벨과 거리가 충분히 나는 다음 레벨 또는
                # 손익비 기반으로 계산
                if len(valid_levels) >= 3:
                    # 이미 선택한 레벨보다 충분히 멀리 있는 레벨 찾기
                    last_tp = None
                    for level in valid_levels[2:]:
                        if position_type == "Long":
                            if level['price'] > tp_prices[-1] * 1.015:  # 최소 1.5% 이상 거리
                                last_tp = round(level['price'] * (1.0 - (level['strength'] * 0.1)), 1)
                                break
                        else:
                            if level['price'] < tp_prices[-1] * 0.985:  # 최소 1.5% 이상 거리
                                last_tp = round(level['price'] * (1.0 + (level['strength'] * 0.1)), 1)
                                break
                    
                    # 적절한 레벨을 찾지 못했다면 손익비 기반으로 계산
                    if not last_tp:
                        rr_ratio = 2.0 * range_factor
                        if position_type == "Long":
                            last_tp = round(entry_price + (risk_per_unit * rr_ratio), 1)
                        else:
                            last_tp = round(entry_price - (risk_per_unit * rr_ratio), 1)
                else:
                    # 손익비 기반으로 계산
                    rr_ratio = 2.0 * range_factor
                    if position_type == "Long":
                        last_tp = round(entry_price + (risk_per_unit * rr_ratio), 1)
                    else:
                        last_tp = round(entry_price - (risk_per_unit * rr_ratio), 1)
                
                tp_prices.append(last_tp)
                print(f"  - 주요 레벨 기반 익절 가격: {tp_prices}")
            else:
                # 주요 레벨이 부족한 경우 손익비 기반 설정
                # 횡보장에서는 좁은 범위의 손익비 적용
                base_rr_ratios = [1.0, 1.5, 2.0]
                rr_ratios = [r * range_factor for r in base_rr_ratios]
                
                if position_type == "Long":
                    tp_prices = [
                        round(entry_price + (risk_per_unit * rr_ratios[0]), 1),
                        round(entry_price + (risk_per_unit * rr_ratios[1]), 1),
                        round(entry_price + (risk_per_unit * rr_ratios[2]), 1)
                    ]
                else:
                    tp_prices = [
                        round(entry_price - (risk_per_unit * rr_ratios[0]), 1),
                        round(entry_price - (risk_per_unit * rr_ratios[1]), 1),
                        round(entry_price - (risk_per_unit * rr_ratios[2]), 1)
                    ]
                
                print(f"  - 손익비 기반 익절 가격: {tp_prices} (손익비: {[f'{r:.1f}' for r in rr_ratios]})")
            
            # 횡보장에서는 초반에 더 많은 물량 배치
            tp_quantities = [0.5, 0.3, 0.2]
            
        # 4.3 변동성 장 전략 - 빠른 익절과 안전 우선
        elif market_type == "VOLATILE":
            strategy_type = "SCALP"
            print(f"[전략] 변동성 장 전략 적용 (강도: {market_strength:.2f})")
            
            # 변동성이 큰 장에서는 빠른 이익실현
            # 변동성 강도가 강할수록 더 빠른 이익실현
            volatility_factor = 0.7 + (market_strength * 0.15)  # 0.7 ~ 0.85
            
            # 추세 방향이 일치할 경우 조정
            if direction_aligned:
                volatility_factor *= 1.2  # 20% 증가
                print(f"  - 방향 일치: 목표 상향 조정 (x1.2)")
            
            # 주요 레벨 활용하되 빠른 익절 고려
            if valid_levels:
                # 첫 번째 TP는 진입가와 가장 가까운 저항/지지선 50-80% 지점
                if position_type == "Long":
                    # 진입가보다 높은 첫 번째 저항선
                    first_level = valid_levels[0]
                    # 저항선의 50-80% 지점에 첫 번째 TP 설정
                    tp1_ratio = 0.5 + (volatility_factor * 0.3)  # 0.65 ~ 0.755
                    tp1_price = round(entry_price + ((first_level['price'] - entry_price) * tp1_ratio), 1)
                else:
                    # 진입가보다 낮은 첫 번째 지지선
                    first_level = valid_levels[0]
                    # 지지선의 50-80% 지점에 첫 번째 TP 설정
                    tp1_ratio = 0.5 + (volatility_factor * 0.3)  # 0.65 ~ 0.755
                    tp1_price = round(entry_price - ((entry_price - first_level['price']) * tp1_ratio), 1)
                
                # 두 번째 TP는 첫 번째 저항/지지선 바로 앞
                if position_type == "Long":
                    tp2_price = round(first_level['price'] * 0.995, 1)
                else:
                    tp2_price = round(first_level['price'] * 1.005, 1)
                
                # 세 번째 TP는 두 번째 저항/지지선(있는 경우) 또는 손익비 기반
                if len(valid_levels) >= 2:
                    second_level = valid_levels[1]
                    # 두 번째 저항/지지선 직전에 설정
                    if position_type == "Long":
                        tp3_price = round(second_level['price'] * 0.995, 1)
                    else:
                        tp3_price = round(second_level['price'] * 1.005, 1)
                else:
                    # 손익비 기반으로 설정
                    rr_ratio = 2.0 * volatility_factor
                    if position_type == "Long":
                        tp3_price = round(entry_price + (risk_per_unit * rr_ratio), 1)
                    else:
                        tp3_price = round(entry_price - (risk_per_unit * rr_ratio), 1)
                
                tp_prices = [tp1_price, tp2_price, tp3_price]
                print(f"  - 레벨 기반 익절 가격: {tp_prices}")
            else:
                # 저항/지지선 정보가 없는 경우, 손익비 기반
                base_rr_ratios = [0.8, 1.3, 2.0]
                rr_ratios = [r * volatility_factor for r in base_rr_ratios]
                
                if position_type == "Long":
                    tp_prices = [
                        round(entry_price + (risk_per_unit * rr_ratios[0]), 1),
                        round(entry_price + (risk_per_unit * rr_ratios[1]), 1),
                        round(entry_price + (risk_per_unit * rr_ratios[2]), 1)
                    ]
                else:
                    tp_prices = [
                        round(entry_price - (risk_per_unit * rr_ratios[0]), 1),
                        round(entry_price - (risk_per_unit * rr_ratios[1]), 1),
                        round(entry_price - (risk_per_unit * rr_ratios[2]), 1)
                    ]
                
                print(f"  - 손익비 기반 익절 가격: {tp_prices} (손익비: {[f'{r:.1f}' for r in rr_ratios]})")
            
            # 변동성 장에서는 초반에 매우 많은 물량 배치 (안전 우선)
            # 변동성 강도가 강할수록 초반에 더 많은 비중
            first_tp_weight = 0.6 + (market_strength * 0.1)  # 0.6 ~ 0.7
            tp_quantities = [first_tp_weight, (1.0 - first_tp_weight) * 0.7, (1.0 - first_tp_weight) * 0.3]
        
        # 4.4 변동성 수축 구간 (돌파 예상)
        elif market_type == "CONTRACTION":
            strategy_type = "BREAKOUT"
            print(f"[전략] 변동성 수축 → 돌파 전략 적용 (강도: {market_strength:.2f}, 확신도: {market_confidence:.2f})")
            
            # 변동성 수축 후 돌파가 예상되는 경우
            # 확신도가 높을수록 더 큰 목표 설정
            breakout_factor = 0.8 + (market_confidence * 0.4)  # 0.8 ~ 1.2
            
            # 방향 일치여부에 따른 조정
            if direction_aligned:
                # 방향 일치시 더 공격적인 목표
                breakout_factor *= 1.2
                print(f"  - 돌파 방향 일치: 목표 상향 조정 (x1.2)")
            
            # 주요 레벨 중 가장 강한 저항/지지선 찾기
            strongest_level = None
            if valid_levels:
                strongest_level = max(valid_levels, key=lambda x: x['strength'])
            
            if strongest_level and strongest_level['strength'] > 0.7:
                print(f"  - 강한 주요 레벨 활용 (강도: {strongest_level['strength']:.2f})")
                # 첫 번째 TP는 가까운 저항/지지선 80% 지점
                level_distance = abs(strongest_level['price'] - entry_price)
                
                if position_type == "Long":
                    tp1_price = round(entry_price + (level_distance * 0.7 * breakout_factor), 1)
                    # 두 번째 TP는 저항선 돌파 직후 (1-2% 상승)
                    tp2_price = round(strongest_level['price'] * 1.01, 1)
                    # 세 번째 TP는 돌파 후 3-5% 상승
                    tp3_price = round(strongest_level['price'] * (1.03 + (breakout_factor * 0.02)), 1)
                else:
                    tp1_price = round(entry_price - (level_distance * 0.7 * breakout_factor), 1)
                    # 두 번째 TP는 지지선 돌파 직후 (1-2% 하락)
                    tp2_price = round(strongest_level['price'] * 0.99, 1)
                    # 세 번째 TP는 돌파 후 3-5% 하락
                    tp3_price = round(strongest_level['price'] * (0.97 - (breakout_factor * 0.02)), 1)
                
                tp_prices = [tp1_price, tp2_price, tp3_price]
                print(f"  - 돌파 시나리오 기반 익절 가격: {tp_prices}")
            else:
                # 주요 레벨이 없거나 강도가 약한 경우 손익비 기반 설정
                # 변동성 수축은 큰 움직임 예상, 넓은 범위의 손익비 적용
                base_rr_ratios = [1.2, 2.0, 3.5]
                rr_ratios = [r * breakout_factor for r in base_rr_ratios]
                
                if position_type == "Long":
                    tp_prices = [
                        round(entry_price + (risk_per_unit * rr_ratios[0]), 1),
                        round(entry_price + (risk_per_unit * rr_ratios[1]), 1),
                        round(entry_price + (risk_per_unit * rr_ratios[2]), 1)
                    ]
                else:
                    tp_prices = [
                        round(entry_price - (risk_per_unit * rr_ratios[0]), 1),
                        round(entry_price - (risk_per_unit * rr_ratios[1]), 1),
                        round(entry_price - (risk_per_unit * rr_ratios[2]), 1)
                    ]
                
                print(f"  - 손익비 기반 익절 가격: {tp_prices} (손익비: {[f'{r:.1f}' for r in rr_ratios]})")
            
            # 변동성 수축 구간에서는 중간 TP에 가장 큰 비중
            if direction_aligned and market_confidence > 0.7:
                # 방향성 확신이 높은 경우 후반 비중 증가
                tp_quantities = [0.3, 0.4, 0.3]
                print("  - 높은 돌파 확신: 균형 분배 적용 (30/40/30)")
            else:
                # 기본적으로는 중간 단계 비중 높임
                tp_quantities = [0.35, 0.45, 0.2]
                print("  - 표준 돌파 전략: 중간 비중 강화 (35/45/20)")
        
        # 4.5 기본 전략 (위 카테고리에 속하지 않는 경우)
        else:
            strategy_type = "STANDARD"
            print("[전략] 기본 전략 적용 (균형 접근)")
            
            # 기본 손익비 비율
            base_rr_ratios = [1.0, 1.7, 2.5]
            
            if position_type == "Long":
                tp_prices = [
                    round(entry_price + (risk_per_unit * base_rr_ratios[0]), 1),
                    round(entry_price + (risk_per_unit * base_rr_ratios[1]), 1),
                    round(entry_price + (risk_per_unit * base_rr_ratios[2]), 1)
                ]
            else:
                tp_prices = [
                    round(entry_price - (risk_per_unit * base_rr_ratios[0]), 1),
                    round(entry_price - (risk_per_unit * base_rr_ratios[1]), 1),
                    round(entry_price - (risk_per_unit * base_rr_ratios[2]), 1)
                ]
            
            # 주요 레벨 고려하여 조정
            # 주요 레벨 고려하여 조정 
            if valid_levels:
                for i in range(len(tp_prices)):
                    closest_level = None
                    min_dist = float('inf')
                    
                    # 현재 TP 가격과 가장 가까운 레벨 찾기
                    for level in valid_levels:
                        dist = abs(level['price'] - tp_prices[i]) / tp_prices[i]
                        if dist < min_dist and dist < 0.05:  # 5% 이내
                            min_dist = dist
                            closest_level = level
                    
                    # 가까운 레벨이 있으면 그에 맞게 조정
                    if closest_level:
                        if position_type == "Long":
                            # 저항선보다 약간 낮게 조정
                            tp_prices[i] = round(closest_level['price'] * 0.995, 1)
                            print(f"  - TP{i+1} 레벨 조정: ${tp_prices[i]:.1f} (저항선 ${closest_level['price']:.1f})")
                        else:
                            # 지지선보다 약간 높게 조정
                            tp_prices[i] = round(closest_level['price'] * 1.005, 1)
                            print(f"  - TP{i+1} 레벨 조정: ${tp_prices[i]:.1f} (지지선 ${closest_level['price']:.1f})")
            
            # 기본 전략은 균형 잡힌 배분
            tp_quantities = [0.35, 0.35, 0.3]
        
        # 5. 최종 가격 및 수량 유효성 검증
        # 5.1 TP 가격 순서 확인
        for i in range(1, len(tp_prices)):
            if position_type == "Long":
                if tp_prices[i] <= tp_prices[i-1]:
                    # 더 높은 가격으로 조정 (1% 이상)
                    tp_prices[i] = max(tp_prices[i], round(tp_prices[i-1] * 1.01, 1))
            else:  # Short
                if tp_prices[i] >= tp_prices[i-1]:
                    # 더 낮은 가격으로 조정 (1% 이상)
                    tp_prices[i] = min(tp_prices[i], round(tp_prices[i-1] * 0.99, 1))
        
        # 5.2 가격 유효성 검증
        for i in range(len(tp_prices)):
            if position_type == "Long" and tp_prices[i] <= entry_price:
                original_price = tp_prices[i]
                tp_prices[i] = round(entry_price * (1.01 + (i * 0.015)), 1)  # 최소 1%, 1.5% 증가 간격
                print(f"[경고] TP{i+1} 가격(${original_price:.1f})이 진입가보다 낮음. 조정: ${tp_prices[i]:.1f}")
            elif position_type == "Short" and tp_prices[i] >= entry_price:
                original_price = tp_prices[i]
                tp_prices[i] = round(entry_price * (0.99 - (i * 0.015)), 1)  # 최소 1%, 1.5% 감소 간격
                print(f"[경고] TP{i+1} 가격(${original_price:.1f})이 진입가보다 높음. 조정: ${tp_prices[i]:.1f}")
        
        # 5.3 수량 유효성 검증
        if sum(tp_quantities) > 1.01:  # 1%의 오차 허용
            # 정규화
            total = sum(tp_quantities)
            tp_quantities = [qty / total for qty in tp_quantities]
            print(f"[정보] 수량 비율 정규화: {[f'{q:.2f}' for q in tp_quantities]}")

        # 5.4 최종 비율 반올림 (소수점 2자리까지)
        tp_quantities = [round(qty, 2) for qty in tp_quantities]
        
        # 결과 반환
        strategy['tp_prices'] = tp_prices
        strategy['tp_quantities'] = tp_quantities
        strategy['strategy_type'] = strategy_type
        
        print(f"\n[익절 전략 결정 완료]")
        print(f"- 전략 유형: {strategy_type}")
        print(f"- 가격: {tp_prices}")
        print(f"- 수량 분배: {tp_quantities}")
        
        return strategy
        
    except Exception as e:
        print(f"[오류] 익절 전략 결정 중 문제 발생: {e}")
        import traceback
        print(traceback.format_exc())
        
        # 기본 전략 반환
        if position_type == "Long":
            tp_prices = [
                round(entry_price * 1.015, 1),
                round(entry_price * 1.025, 1),
                round(entry_price * 1.04, 1)
            ]
        else:
            tp_prices = [
                round(entry_price * 0.985, 1),
                round(entry_price * 0.975, 1),
                round(entry_price * 0.96, 1)
            ]
            
        return {
            'tp_prices': tp_prices,
            'tp_quantities': [0.4, 0.3, 0.3],
            'strategy_type': "STANDARD"
        }

def check_time_based_take_profit(conn, trade_id: str, position_type: str, market_data: Dict = None) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    포지션 보유 시간에 기반한 익절 조건을 확인하는 함수
    - 일정 시간 경과 후 부분 익절 또는 전체 익절 결정
    - 시장 상황과 보유 시간을 조합하여 동적 익절 조건 설정
    - 손익 목표를 시간에 따라 점진적으로 조정
    
    Args:
        conn: 데이터베이스 연결 객체
        trade_id: 거래 ID
        position_type: 포지션 유형 ("Long" 또는 "Short")
        market_data: 시장 데이터 (선택적)
        
    Returns:
        Tuple[bool, Optional[float], Optional[float]]: 
            - 익절 실행 여부
            - 익절 가격 (없으면 None)
            - 익절 수량 비율 (없으면 None)
    """
    try:
        if not conn or not trade_id:
            return False, None, None
            
        cursor = conn.cursor()
        
        # 거래 정보 조회
        cursor.execute("""
            SELECT timestamp, entry_price, leverage, position_size
            FROM trades
            WHERE trade_id = ? AND trade_status = 'Open'
        """, (trade_id,))
        
        trade_info = cursor.fetchone()
        if not trade_info:
            print(f"[경고] 거래 ID {trade_id}에 대한 정보를 찾을 수 없습니다.")
            return False, None, None
            
        entry_time_str, entry_price, leverage, position_size = trade_info
        entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        
        # 포지션 보유 시간 계산 (시간)
        holding_hours = (current_time - entry_time).total_seconds() / 3600
        
        # 현재 가격 정보
        current_price = 0.0
        if market_data and 'candlestick' in market_data and not market_data['candlestick'].empty:
            current_price = float(market_data['candlestick'].iloc[-1]['Close'])
        else:
            # 시장 데이터가 없으면 직접 조회
            candle_data = fetch_candle_data(interval="1", limit=1)
            if not candle_data.empty:
                current_price = float(candle_data['Close'].iloc[0])
        
        if current_price <= 0:
            print("[경고] 현재 가격을 확인할 수 없습니다.")
            return False, None, None
        
        # 현재 미실현 손익 계산
        pnl_pct = 0.0
        if position_type == "Long":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
        else:  # Short
            pnl_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
        
        # 시장 상태 평가 (선택적)
        market_condition = "normal"
        adx = 0
        
        if market_data and 'timeframes' in market_data and "60" in market_data['timeframes']:
            # ADX로 추세 강도 확인
            adx = float(market_data['timeframes']["60"].iloc[-1]['ADX'])
            
            if adx > 30:
                # 강한 추세
                if (position_type == "Long" and float(market_data['timeframes']["60"].iloc[-1]['+DI']) > 
                    float(market_data['timeframes']["60"].iloc[-1]['-DI'])) or \
                   (position_type == "Short" and float(market_data['timeframes']["60"].iloc[-1]['+DI']) < 
                    float(market_data['timeframes']["60"].iloc[-1]['-DI'])):
                    market_condition = "strong_trend_aligned"
                else:
                    market_condition = "strong_trend_against"
            elif adx < 20:
                market_condition = "range_bound"
        
        # 시간 기반 익절 조건
        # 1. 기본 보유 시간 기준점 설정
        time_thresholds = {
            "short_term": 4,     # 4시간
            "medium_term": 12,   # 12시간
            "long_term": 24,     # 24시간
            "extended": 48       # 48시간
        }
        
        # 2. 시간대별 익절 목표 설정 (포지션 방향 및 시장 상황에 따라 조정)
        take_profit_targets = {
            # 기본 목표 (이익 %)
            "normal": {
                "short_term": 1.5 * leverage,    # 4시간 후 목표
                "medium_term": 2.0 * leverage,   # 12시간 후 목표
                "long_term": 2.5 * leverage,     # 24시간 후 목표
                "extended": 1.0 * leverage       # 48시간 후 목표 (하향 조정)
            },
            # 강한 추세 (포지션과 일치)
            "strong_trend_aligned": {
                "short_term": 2.0 * leverage,
                "medium_term": 3.0 * leverage,
                "long_term": 4.0 * leverage,
                "extended": 2.0 * leverage
            },
            # 강한 추세 (포지션과 반대)
            "strong_trend_against": {
                "short_term": 1.0 * leverage,
                "medium_term": 1.2 * leverage,
                "long_term": 0.8 * leverage,
                "extended": 0.5 * leverage
            },
            # 횡보장
            "range_bound": {
                "short_term": 1.2 * leverage,
                "medium_term": 1.5 * leverage,
                "long_term": 1.8 * leverage,
                "extended": 1.0 * leverage
            }
        }
        
        # 3. 시간대별 익절 수량 비율 설정
        quantity_ratios = {
            "short_term": 0.3,   # 30% 물량
            "medium_term": 0.5,  # 50% 물량
            "long_term": 0.7,    # 70% 물량
            "extended": 1.0      # 100% 물량 (전량 청산)
        }
        
        # 4. 익절 타이밍 및 조건 평가
        execute_tp = False
        tp_target = 0.0
        tp_ratio = 0.0
        
        # 시간대 확인
        time_bracket = None
        
        if holding_hours >= time_thresholds["extended"]:
            time_bracket = "extended"
        elif holding_hours >= time_thresholds["long_term"]:
            time_bracket = "long_term"
        elif holding_hours >= time_thresholds["medium_term"]:
            time_bracket = "medium_term"
        elif holding_hours >= time_thresholds["short_term"]:
            time_bracket = "short_term"
        
        if time_bracket:
            # 목표 손익 설정
            tp_target = take_profit_targets[market_condition][time_bracket]
            tp_ratio = quantity_ratios[time_bracket]
            
            # 손익이 목표를 초과하는지 확인
            if pnl_pct >= tp_target:
                execute_tp = True
            
            # 48시간 이상 보유 시 목표 손익을 하향 조정 (시간이 길어질수록 목표 낮춤)
            if time_bracket == "extended":
                hours_beyond_extended = holding_hours - time_thresholds["extended"]
                # 추가 24시간마다 목표를 10%씩 낮춤
                additional_reduction = min(0.5, hours_beyond_extended / 24 * 0.1)
                adjusted_target = tp_target * (1.0 - additional_reduction)
                
                if pnl_pct >= adjusted_target:
                    execute_tp = True
                    tp_target = adjusted_target
            
            # 특별 조건: 매우 긴 시간 보유 시 의무 청산 (72시간 이상)
            if holding_hours >= 72:
                # 이익이 있으면 청산
                if pnl_pct > 0:
                    execute_tp = True
                    tp_ratio = 1.0  # 전량 청산
                    print(f"[정보] 72시간 이상 포지션 보유로 의무 청산 실행")
        
        # 실행 결정 시 TP 가격 계산
        tp_price = None
        if execute_tp:
            if position_type == "Long":
                tp_price = round(current_price, 1)  # 현재가로 지정 (시장가 주문)
            else:  # Short
                tp_price = round(current_price, 1)  # 현재가로 지정 (시장가 주문)
            
            print(f"[정보] 시간 기반 익절 조건 충족:")
            print(f"  포지션 타입: {position_type}")
            print(f"  보유 시간: {holding_hours:.1f}시간 (시간대: {time_bracket})")
            print(f"  현재 손익: {pnl_pct:.2f}% (목표: {tp_target:.2f}%)")
            print(f"  시장 상태: {market_condition} (ADX: {adx:.1f})")
            print(f"  익절 수량 비율: {tp_ratio:.2f} (즉시 실행)")
            
            # DB에 시간 기반 TP 설정 기록
            try:
                cursor.execute("""
                    INSERT INTO time_based_tp_records
                    (trade_id, position_type, holding_hours, pnl_pct, tp_target, tp_ratio, market_condition, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, 
                    position_type, 
                    holding_hours, 
                    pnl_pct, 
                    tp_target, 
                    tp_ratio, 
                    market_condition,
                    current_time.strftime("%Y-%m-%d %H:%M:%S")
                ))
                conn.commit()
            except sqlite3.OperationalError:
                # 테이블이 없는 경우 생성
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS time_based_tp_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT,
                        position_type TEXT,
                        holding_hours REAL,
                        pnl_pct REAL,
                        tp_target REAL,
                        tp_ratio REAL,
                        market_condition TEXT,
                        execution_time TEXT
                    )
                """)
                conn.commit()
                
                # 다시 시도
                cursor.execute("""
                    INSERT INTO time_based_tp_records
                    (trade_id, position_type, holding_hours, pnl_pct, tp_target, tp_ratio, market_condition, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, 
                    position_type, 
                    holding_hours, 
                    pnl_pct, 
                    tp_target, 
                    tp_ratio, 
                    market_condition,
                    current_time.strftime("%Y-%m-%d %H:%M:%S")
                ))
                conn.commit()
            
        return execute_tp, tp_price, tp_ratio
        
    except Exception as e:
        print(f"[오류] 시간 기반 익절 조건 확인 중 문제 발생: {e}")
        print(traceback.format_exc())
        return False, None, None


def set_take_profit_enhanced(symbol: str, qty: float, entry_price: float, position_type: str, 
                       market_data: Dict, leverage: int, conn=None, trade_id=None, 
                       decision=None) -> bool:

    """
    향상된 동적 익절 설정 함수
    - 멀티 타임프레임 분석 반영
    - 주요 지지/저항선 고려
    - 시장 심리 반영
    - 과거 거래 성과 기반 최적화
    """
    try:
        # 로깅 추가 (함수 시작 부분)
        print("\n=== 이익실현 주문 설정 시작 ===")
        print(f"포지션 타입: {position_type}")
        print(f"진입가: ${entry_price:.2f}")
        print(f"수량: {qty} BTC")
        print(f"레버리지: {leverage}x")
        print(f"거래 ID: {trade_id}")
        
        # 시장 상태 분석 (15분봉 기준)
        df_15m = market_data['timeframes']["15"]
        df_1h = market_data['timeframes'].get("60", df_15m)  # 1시간봉 (없으면 15분봉 사용)
        df_4h = market_data['timeframes'].get("240", df_15m)  # 4시간봉 (없으면 15분봉 사용)
        
        latest_15m = df_15m.iloc[-1]
        latest_1h = df_1h.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        
        # === 1. 기본 지표 분석 ===
        adx_15m = float(latest_15m['ADX'])
        adx_1h = float(latest_1h['ADX'])
        adx_4h = float(latest_4h['ADX'])
        
        atr_15m = float(latest_15m['ATR'])
        atr_1h = float(latest_1h['ATR'])
        
        bb_width_15m = (float(latest_15m['BB_upper']) - float(latest_15m['BB_lower'])) / float(latest_15m['BB_mavg'])
        volatility = atr_15m / entry_price  # 현재 가격 대비 ATR
        
        rsi_15m = float(latest_15m['RSI'])
        rsi_1h = float(latest_1h['RSI'])
        
        # === 2. 시장 트렌드 평가 ===
        # 타임프레임별 가중치 설정
        tf_weights = {
            "15m": 0.5,  # 단기 (50%)
            "1h": 0.3,   # 중기 (30%)
            "4h": 0.2    # 장기 (20%)
        }
        
        # 타임프레임별 추세 강도 평가
        trend_strength = {
            "15m": min(1.0, adx_15m / 30),
            "1h": min(1.0, adx_1h / 30),
            "4h": min(1.0, adx_4h / 30)
        }
        
        # 종합 추세 강도 (가중 평균)
        overall_trend_strength = (
            trend_strength["15m"] * tf_weights["15m"] +
            trend_strength["1h"] * tf_weights["1h"] +
            trend_strength["4h"] * tf_weights["4h"]
        )
        
        # 방향성 평가 (각 타임프레임별)
        is_uptrend = {
            "15m": float(latest_15m['Close']) > float(latest_15m['EMA_20']) and float(latest_15m['MACD']) > float(latest_15m['MACD_signal']),
            "1h": float(latest_1h['Close']) > float(latest_1h['EMA_20']) and float(latest_1h['MACD']) > float(latest_1h['MACD_signal']),
            "4h": float(latest_4h['Close']) > float(latest_4h['EMA_20']) and float(latest_4h['MACD']) > float(latest_4h['MACD_signal'])
        }
        
        is_downtrend = {
            "15m": float(latest_15m['Close']) < float(latest_15m['EMA_20']) and float(latest_15m['MACD']) < float(latest_15m['MACD_signal']),
            "1h": float(latest_1h['Close']) < float(latest_1h['EMA_20']) and float(latest_1h['MACD']) < float(latest_1h['MACD_signal']),
            "4h": float(latest_4h['Close']) < float(latest_4h['EMA_20']) and float(latest_4h['MACD']) < float(latest_4h['MACD_signal'])
        }
        
        # 트렌드 일치도 평가
        trend_agreement = 0
        if position_type == "Long":
            # 롱 포지션의 경우 상승 트렌드 일치도 확인
            trend_agreement = (
                (1 if is_uptrend["15m"] else 0) * tf_weights["15m"] +
                (1 if is_uptrend["1h"] else 0) * tf_weights["1h"] +
                (1 if is_uptrend["4h"] else 0) * tf_weights["4h"]
            )
        else:
            # 숏 포지션의 경우 하락 트렌드 일치도 확인
            trend_agreement = (
                (1 if is_downtrend["15m"] else 0) * tf_weights["15m"] +
                (1 if is_downtrend["1h"] else 0) * tf_weights["1h"] +
                (1 if is_downtrend["4h"] else 0) * tf_weights["4h"]
            )
        
        # === 3. 주요 지지/저항선 분석 ===
        key_levels = []
        
        # 돈치안 채널 레벨
        if 'Donchian_High' in latest_1h:
            if position_type == "Long":
                key_levels.append(float(latest_1h['Donchian_High']))  # 롱의 경우 상단 저항선
                key_levels.append(float(latest_1h['Resistance1']))
                key_levels.append(float(latest_1h['Resistance2']))
            else:
                key_levels.append(float(latest_1h['Donchian_Low']))   # 숏의 경우 하단 지지선
                key_levels.append(float(latest_1h['Support1']))
                key_levels.append(float(latest_1h['Support2']))
        
        # 피보나치 레벨 추가
        if position_type == "Long":
            key_levels.append(float(latest_1h['Fibo_0.236']))
            key_levels.append(float(latest_1h['Fibo_0.382']))
            key_levels.append(float(latest_1h['Fibo_0.5']))
            key_levels.append(float(latest_1h['Fibo_0.618']))
        else:
            # 숏의 경우 반전된 피보나치 레벨 사용 (음수값)
            if 'Fibo_-0.236' in latest_1h:
                key_levels.append(float(latest_1h['Fibo_-0.236']))
                key_levels.append(float(latest_1h['Fibo_-0.382']))
                key_levels.append(float(latest_1h['Fibo_-0.5']))
                key_levels.append(float(latest_1h['Fibo_-0.618']))
        
        # EMA 레벨 추가
        if position_type == "Long":
            key_levels.append(float(latest_1h['EMA_50']))
            key_levels.append(float(latest_1h['EMA_100']))
            key_levels.append(float(latest_1h['EMA_200']))
        else:
            key_levels.append(float(latest_1h['EMA_50']))
            key_levels.append(float(latest_1h['EMA_100']))
            key_levels.append(float(latest_1h['EMA_200']))
        
        # 볼린저 밴드 추가
        if position_type == "Long":
            key_levels.append(float(latest_1h['BB_upper']))
        else:
            key_levels.append(float(latest_1h['BB_lower']))
            
        # === 4. 시장 심리 지표 분석 ===
        # Fear & Greed Index 평가
        fear_greed = market_data.get('fear_greed', {}).get('value', 50)
        if isinstance(fear_greed, str):
            try:
                fear_greed = int(fear_greed)
            except:
                fear_greed = 50  # 기본값
                
        # 심리 지표 정규화 (0-1 범위)
        market_sentiment = fear_greed / 100
        
        # === 5. 과거 거래 성과 반영 ===
        # 최근 거래 성과 분석 (DB에서 조회)
        success_rate = 0.5  # 기본값
        avg_profit = 0.0    # 기본값
        
        if conn:
            try:
                cursor = conn.cursor()
                # 최근 10개 완료된 거래의 승률 계산
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                           AVG(profit_loss) as avg_profit
                    FROM trades
                    WHERE trade_status = 'Closed'
                    ORDER BY timestamp DESC
                    LIMIT 10
                """)
                result = cursor.fetchone()
                if result and result[0] > 0:
                    success_rate = result[1] / result[0]
                    avg_profit = result[2] if result[2] else 0.0
            except Exception as e:
                print(f"[경고] 과거 거래 성과 분석 중 오류: {e}")
                # 기본값 사용
        
        # === 6. 최종 익절 전략 결정 ===
        # 기본 수수료 고려 (진입 + 청산)
        total_fee = 0.0012  # 0.12%

        # 최소 수익 목표를 수수료의 2배로 설정 (최소한 수수료보다는 높아야 함)
        min_target = total_fee * 2  # 0.24%

        # AI가 제공한 익절 목표 사용 (없으면 기본값 사용)
        if hasattr(decision, 'take_profit_targets') and decision.take_profit_targets and len(decision.take_profit_targets) >= 3:
            # AI가 제공한 타겟을 사용 (레버리지로 나눠서 실제 가격 목표로 변환)
            target_profits = [target / leverage for target in decision.take_profit_targets]
            print(f"[정보] AI 제공 익절 목표 사용: {decision.take_profit_targets}")
        else:
            # 기본 목표 설정 (레버리지 고려)
            base_target_pct = 0.15 / leverage  # 기본 15%를 레버리지로 나눔
            
            # 트렌드 강도와 일치도에 따른 조정
            trend_multiplier = 1.0 + (overall_trend_strength * 0.5) + (trend_agreement * 0.5)
            
            # 변동성에 따른 조정 (높은 변동성 = 더 큰 목표)
            vol_multiplier = 1.0
            if volatility > 0.015:  # 1.5% 이상 변동성
                vol_multiplier = 1.3
            elif volatility > 0.01:  # 1.0% 이상 변동성
                vol_multiplier = 1.2
            elif volatility < 0.005:  # 0.5% 미만 변동성
                vol_multiplier = 0.8
                
            # RSI 기반 조정 (극단값에 가까울수록 익절 타겟 축소)
            rsi_multiplier = 1.0
            if position_type == "Long":
                if rsi_15m > 70:  # 과매수 영역
                    rsi_multiplier = 0.8
                elif rsi_15m > 60:
                    rsi_multiplier = 0.9
                elif rsi_15m < 40:  # 과매도 영역에서 롱 진입 시 상승 여지 많음
                    rsi_multiplier = 1.2
            else:  # Short
                if rsi_15m < 30:  # 과매도 영역
                    rsi_multiplier = 0.8
                elif rsi_15m < 40:
                    rsi_multiplier = 0.9
                elif rsi_15m > 60:  # 과매수 영역에서 숏 진입 시 하락 여지 많음
                    rsi_multiplier = 1.2
                    
            # 시장 심리에 따른 조정
            sentiment_multiplier = 1.0
            if position_type == "Long":
                if market_sentiment < 0.3:  # 극단적 공포 (상승 여지 높음)
                    sentiment_multiplier = 1.2
                elif market_sentiment > 0.7:  # 극단적 탐욕 (상승 제한적)
                    sentiment_multiplier = 0.8
            else:  # Short
                if market_sentiment > 0.7:  # 극단적 탐욕 (하락 여지 높음)
                    sentiment_multiplier = 1.2
                elif market_sentiment < 0.3:  # 극단적 공포 (하락 제한적)
                    sentiment_multiplier = 0.8
                    
            # 과거 거래 성과에 따른 조정
            performance_multiplier = 1.0
            if success_rate > 0.7:  # 높은 승률 (현재 전략 유지)
                performance_multiplier = 1.0
            elif success_rate < 0.3:  # 낮은 승률 (더 보수적인 익절)
                performance_multiplier = 0.85
                
            # === 7. 최종 타겟 계산 ===
            # 종합 승수 계산
            combined_multiplier = (
                trend_multiplier * 
                vol_multiplier * 
                rsi_multiplier * 
                sentiment_multiplier * 
                performance_multiplier
            )
            
            # 시장 상태에 따른 단계별 익절 설정
            is_trend_market = adx_1h > 25 and bb_width_15m > 0.03
            
            if is_trend_market:  # 추세장
                # 트렌드를 따라가는 보다 공격적인 익절 목표
                target_profits = [
                    max(min_target, base_target_pct * 1.0 * combined_multiplier),
                    max(min_target * 2, base_target_pct * 1.6 * combined_multiplier),
                    max(min_target * 3, base_target_pct * 2.5 * combined_multiplier)
                ]
            else:  # 횡보장
                # 작은 이익을 더 빨리 실현하는 보수적 목표
                target_profits = [
                    max(min_target, base_target_pct * 0.7 * combined_multiplier),
                    max(min_target * 1.5, base_target_pct * 1.2 * combined_multiplier),
                    max(min_target * 2, base_target_pct * 1.8 * combined_multiplier)
                ]

        # AI가 제공한 분배 비율 사용 (없으면 기본값 사용)
        if hasattr(decision, 'take_profit_distribution') and decision.take_profit_distribution and len(decision.take_profit_distribution) >= 3:
            qty_distribution = decision.take_profit_distribution
            print(f"[정보] AI 제공 익절 분배 비율 사용: {decision.take_profit_distribution}")
        else:
            # 시장 상태에 따른 수량 분배
            if is_trend_market:  # 추세장
                qty_distribution = [0.25, 0.35, 0.4]  # 트렌드를 따라가도록 수량 배분 (마지막에 더 많이)
            else:  # 횡보장
                qty_distribution = [0.4, 0.35, 0.25]  # 초반에 더 많이 실현하도록 수량 배분
                    
                # === 8. 주요 지지/저항선 기반 익절 가격 최적화 ===
                if key_levels:
                    # 정렬 및 중복 제거
                    if position_type == "Long":
                        key_levels = sorted([level for level in key_levels if level > entry_price])
                    else:
                        key_levels = sorted([level for level in key_levels if level < entry_price], reverse=True)
                    
                    # 가격 이동 필요 비율 계산
                    price_move_pcts = []
                    for level in key_levels:
                        if position_type == "Long":
                            pct = (level - entry_price) / entry_price
                        else:
                            pct = (entry_price - level) / entry_price
                        price_move_pcts.append(pct)
                    
                    # 익절 목표치에 근접한 주요 레벨 찾기
                    optimized_targets = []
                    for target in target_profits:
                        target_move = target / leverage  # 레버리지 고려한 실제 가격 변동률
                        
                        # 가장 가까운 주요 레벨 찾기
                        closest_level_idx = None
                        min_diff = float('inf')
                        
                        for i, pct in enumerate(price_move_pcts):
                            if pct > 0:  # 유효한 이동 방향인 경우만
                                diff = abs(pct - target_move)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_level_idx = i
                        
                        # 주요 레벨이 있으면 조정, 없으면 원래 목표 사용
                        if closest_level_idx is not None and min_diff < 0.05:  # 5% 이내 차이면 조정
                            optimized_targets.append(price_move_pcts[closest_level_idx] * leverage)
                        else:
                            optimized_targets.append(target)
                    
                    # 만약 유효한 최적화 타겟이 있으면 교체
                    if optimized_targets and len(optimized_targets) == len(target_profits):
                        target_profits = optimized_targets
        
        # === 9. 디버깅 정보 출력 ===
        print(f"\n=== 향상된 익절 설정 분석 ===")
        print(f"포지션 타입: {position_type}")
        print(f"진입가: ${entry_price:.2f}")
        print(f"레버리지: {leverage}x")
        
        print(f"\n[시장 상태]")
        print(f"15분봉 ADX: {adx_15m:.2f}")
        print(f"1시간봉 ADX: {adx_1h:.2f}")
        print(f"4시간봉 ADX: {adx_4h:.2f}")
        print(f"볼린저 밴드 폭: {bb_width_15m:.4f}")
        print(f"ATR(15분): {atr_15m:.2f}")
        print(f"변동성: {volatility*100:.2f}%")
        print(f"시장 유형: {'추세장' if is_trend_market else '횡보장'}")
        
        print(f"\n[트렌드 분석]")
        print(f"종합 추세 강도: {overall_trend_strength:.2f}")
        print(f"트렌드 일치도: {trend_agreement:.2f}")
        
        print(f"\n[기술적 지표]")
        print(f"RSI(15분): {rsi_15m:.2f}")
        print(f"RSI(1시간): {rsi_1h:.2f}")
        print(f"시장 심리(Fear & Greed): {fear_greed}")
        
        if conn:
            print(f"\n[과거 거래 성과]")
            print(f"최근 승률: {success_rate:.2f}")
            print(f"평균 수익률: {avg_profit:.2f}%")
        
        print(f"\n[승수 분석]")
        print(f"트렌드 승수: {trend_multiplier:.2f}")
        print(f"변동성 승수: {vol_multiplier:.2f}")
        print(f"RSI 승수: {rsi_multiplier:.2f}")
        print(f"심리 승수: {sentiment_multiplier:.2f}")
        print(f"성과 승수: {performance_multiplier:.2f}")
        print(f"종합 승수: {combined_multiplier:.2f}")
        
        # === 10. 레버리지 고려한 가격 수준 계산 ===
        price_moves = [profit/leverage for profit in target_profits]
        
        # 익절 가격 계산
        if position_type == "Long":
            tp_prices = [
                round(entry_price * (1 + move), 1) 
                for move in price_moves
            ]
        else:  # Short
            tp_prices = [
                round(entry_price * (1 - move), 1)
                for move in price_moves
            ]
        
        # === 11. 실제 이익실현 주문 설정 ===
        # === 여기서부터 수정 ===
        # 각 단계별 수량 계산
        qty_per_level = [round(qty * dist, 3) for dist in qty_distribution]
        
        print("\n[최종 익절 설정]")
        for i, (price, amount, target) in enumerate(zip(tp_prices, qty_per_level, target_profits)):
            print(f"단계 {i+1}:")
            print(f"- 목표 수익률: {target*100:.1f}%")
            print(f"- 필요 가격 변동: {price_moves[i]*100:.2f}%")
            print(f"- 청산 가격: ${price:,.2f}")
            print(f"- 수량: {amount} BTC")
        
        position_idx = 1 if position_type == "Long" else 2
        
        # 기존 TP 주문이 있다면 취소
        cancel_all_tp_orders(symbol, position_idx)
        time.sleep(1)  # API 레이트 리밋 방지
            
        # 첫 번째 TP 주문만 설정
        first_tp_price = tp_prices[0]
        first_tp_qty = qty_per_level[0]
        
        # 최소 수량 체크
        if first_tp_qty < MIN_QTY:
            print(f"[경고] TP 수량이 최소 거래량보다 작음: {first_tp_qty} < {MIN_QTY}")
            return False
        
        print(f"\n[TP 주문 설정]")
        print(f"1단계 TP: ${first_tp_price} (수량: {first_tp_qty} BTC)")
            
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            takeProfit=str(first_tp_price),
            tpSize=str(first_tp_qty),
            tpTriggerBy="LastPrice",
            tpslMode="Partial"  # 이 파라미터가 핵심 - 부분 포지션 모드 활성화
        )
        
        if response["retCode"] == 0:
            print(f"[성공] 1단계 TP 설정 완료")
            order_id = response["result"].get("orderId")
            
            # DB에 모든 TP 계획 저장 (첫 번째는 Active, 나머지는 Pending)
            if conn and trade_id:
                # 기존 주문 초기화
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE take_profit_orders
                    SET status = 'Cancelled'
                    WHERE trade_id = ? AND status IN ('Active', 'Pending')
                """, (trade_id,))
                
                # 새로운 주문 계획 모두 저장
                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                for i, (price, amount) in enumerate(zip(tp_prices, qty_per_level)):
                    level = i + 1
                    status = 'Active' if i == 0 else 'Pending'
                    current_order_id = order_id if i == 0 else None
                    
                    cursor.execute("""
                        INSERT INTO take_profit_orders 
                        (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, order_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_id,
                        position_type,
                        level,
                        price,
                        amount,
                        status,
                        created_at,
                        current_order_id
                    ))
                
                conn.commit()
                print(f"[성공] {len(tp_prices)}단계 익절 계획이 DB에 저장되었습니다.")
                
            print(f"[성공] 향상된 단계별 이익실현 설정 완료")
            return True
        else:
            print(f"[경고] TP 설정 실패: {response['retMsg']}")
            return False
            
    except Exception as e:
        print(f"[오류] 이익실현 주문 설정 중 문제 발생: {e}")
        print(traceback.format_exc())
        return False


def set_take_profit_for_additional_entry(symbol: str, qty: float, entry_price: float, position_type: str,
                                        market_data: Dict, leverage: int, conn=None, trade_id=None,
                                        market_structure=None, key_levels=None) -> bool:
    """
    추가 매수 포지션에 대한 특화된 익절 설정 함수
    """
    try:
        print(f"\n[추가 매수 익절 전략 설정]")
        position_idx = 1 if position_type == "Long" else 2
        
        # 1. 기존 익절 주문 정보 가져오기 (취소하지 않음)
        existing_tp_orders = []
        orders = bybit_client.get_open_orders(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            orderFilter="StopOrder"
        )
        
        if orders["retCode"] == 0:
            for order in orders["result"]["list"]:
                if order["stopOrderType"] == "TakeProfit":
                    existing_tp_orders.append({
                        "order_id": order["orderId"],
                        "price": float(order["triggerPrice"]),
                        "qty": float(order["qty"])
                    })
                    
            print(f"[정보] 기존 익절 주문 {len(existing_tp_orders)}개 확인됨")
            
        # 2. 마켓 스트럭처 기반 추가 매수 익절 전략 
        if market_structure and market_structure["type"] == "TREND":
            # 추세장에서 추가 매수는 더 멀리 목표 설정
            tp_pct = 0.03 if position_type == "Long" else 0.03  # 3% 목표
        elif market_structure and market_structure["type"] == "RANGE":
            # 횡보장에서는 짧게 익절
            tp_pct = 0.015 if position_type == "Long" else 0.015  # 1.5% 목표
        elif market_structure and market_structure["type"] == "VOLATILE":
            # 변동성 장에서는 더 짧게 익절
            tp_pct = 0.01 if position_type == "Long" else 0.01  # 1% 목표
        else:
            # 기본 목표
            tp_pct = 0.02 if position_type == "Long" else 0.02  # 2% 기본 목표
            
        # 3. 시장 방향과 포지션 일치 여부에 따라 목표 조정
        if market_structure:
            direction_aligned = (position_type == "Long" and market_structure["direction"] == "UP") or \
                              (position_type == "Short" and market_structure["direction"] == "DOWN")
            if direction_aligned:
                tp_pct *= 1.2  # 방향 일치 시 목표 20% 증가
            else:
                tp_pct *= 0.8  # 방향 불일치 시 목표 20% 감소
                
        # 4. 추가 매수에 대한 익절 가격 계산
        if position_type == "Long":
            tp_price = round(entry_price * (1 + tp_pct), 1)
        else:
            tp_price = round(entry_price * (1 - tp_pct), 1)
            
        # 5. 주요 저항/지지 레벨 고려하여 TP 가격 조정
        if key_levels:
            # 롱 포지션은 위쪽 레벨(저항선), 숏 포지션은 아래쪽 레벨(지지선) 찾기
            relevant_levels = []
            if position_type == "Long":
                # 진입가보다 높고 계산된 TP보다 낮은 저항선
                relevant_levels = [level for level in key_levels 
                                  if entry_price < level['price'] < tp_price and level['type'] == 'resistance']
            else:
                # 진입가보다 낮고 계산된 TP보다 높은 지지선
                relevant_levels = [level for level in key_levels 
                                  if entry_price > level['price'] > tp_price and level['type'] == 'support']
                                  
            # 적절한 레벨이 있으면 조정
            if relevant_levels:
                # 가장 가까운 레벨 사용
                if position_type == "Long":
                    nearest_level = min(relevant_levels, key=lambda x: x['price'])
                    # 강한 저항선 직전에 TP 설정
                    tp_price = nearest_level['price'] * 0.995
                else:
                    nearest_level = max(relevant_levels, key=lambda x: x['price'])
                    # 강한 지지선 직전에 TP 설정
                    tp_price = nearest_level['price'] * 1.005
                    
                print(f"[정보] 주요 레벨 기반 TP 가격 조정: ${tp_price:.2f}")
        
        # 6. 기존 TP와 유사한 위치에 설정
        if existing_tp_orders:
            existing_prices = [order["price"] for order in existing_tp_orders]
            avg_existing_price = sum(existing_prices) / len(existing_prices)
            
            # 기존 TP가 이미 이익 구간에 있다면 그 근처에 설정
            if (position_type == "Long" and avg_existing_price > entry_price) or \
               (position_type == "Short" and avg_existing_price < entry_price):
                # 계산된 TP와 기존 TP의 절충
                tp_price = round((tp_price + avg_existing_price) / 2, 1)
                print(f"[정보] 기존 TP와 절충한 가격: ${tp_price:.2f}")
        
        # 7. 최종 설정
        print(f"[정보] 추가 매수 익절 가격: ${tp_price:.2f}, 수량: {qty} BTC")
        
        # 익절 주문 설정
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            takeProfit=str(tp_price),
            tpSize=str(qty),
            tpTriggerBy="LastPrice",
            tpslMode="Partial"  # 부분 청산 모드
        )
        
        if response["retCode"] == 0:
            print(f"[성공] 추가 매수 익절 설정 완료: 가격 ${tp_price:.2f}, 수량 {qty} BTC")
            
            # DB에 추가 익절 주문 저장
            if conn and trade_id:
                try:
                    cursor = conn.cursor()
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    cursor.execute("""
                        INSERT INTO take_profit_orders 
                        (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, order_id, is_additional)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_id,
                        position_type,
                        1,  # 항상 첫 번째 레벨로 설정
                        tp_price,
                        qty,
                        'Active',
                        created_at,
                        response["result"].get("orderId"),
                        1  # 추가 매수 표시
                    ))
                    
                    conn.commit()
                    print(f"[정보] 추가 매수에 대한 익절 계획이 DB에 저장되었습니다.")
                except Exception as db_error:
                    print(f"[오류] DB 저장 중 문제 발생: {db_error}")
                    if 'cursor' in locals() and conn:
                        conn.rollback()
            
            return True
        else:
            print(f"[경고] 추가 매수 익절 설정 실패: {response['retMsg']}")
            return False
            
    except Exception as e:
        print(f"[오류] 추가 매수 익절 설정 중 문제 발생: {e}")
        print(traceback.format_exc())
        return False


def set_take_profit_enhanced_improved(symbol: str, qty: float, entry_price: float, position_type: str, 
                             market_data: Dict, leverage: int, conn=None, trade_id=None, 
                             decision=None, is_additional_entry=False, 
                             additional_qty=None) -> bool:
   """
   손익비(Risk-Reward Ratio) 기반 이익실현 설정 함수 - 추가 매수 지원 기능 포함
   - 저항선/지지선 분석을 통한 더욱 정밀한 익절 가격 설정
   
   Args:
       symbol: 거래 심볼 (예: "BTCUSDT")
       qty: 전체 포지션 수량
       entry_price: 진입 가격
       position_type: 포지션 타입 ("Long" 또는 "Short")
       market_data: 시장 데이터 딕셔너리
       leverage: 레버리지 배수
       conn: 데이터베이스 연결 객체 (옵션)
       trade_id: 거래 ID (옵션)
       decision: 거래 결정 객체 (옵션)
       is_additional_entry: 추가 매수 여부
       additional_qty: 추가된 수량 (전체 수량이 아님)
   """
   try:
       # 로깅 추가 
       print("\n=== 저항선 기반 이익실현 주문 설정 시작 ===")
       print(f"포지션 타입: {position_type}")
       print(f"진입가: ${entry_price:.2f}")
       print(f"수량: {qty} BTC")
       print(f"레버리지: {leverage}x")
       print(f"거래 ID: {trade_id}")
       print(f"추가 매수 여부: {is_additional_entry}")
       if is_additional_entry:
           print(f"추가 매수 수량: {additional_qty} BTC")
       
       position_idx = 1 if position_type == "Long" else 2
       
       # 기존 익절 주문 정보 조회
       existing_tp_orders = []
       if is_additional_entry:
           orders = bybit_client.get_open_orders(
               category="linear",
               symbol=symbol,
               positionIdx=position_idx,
               orderFilter="StopOrder"
           )
           
           if orders["retCode"] == 0:
               for order in orders["result"]["list"]:
                   if order["stopOrderType"] == "TakeProfit":
                       existing_tp_orders.append({
                           "order_id": order["orderId"],
                           "price": float(order["triggerPrice"]),
                           "qty": float(order["qty"])
                       })
                       
               print(f"[정보] 기존 익절 주문 {len(existing_tp_orders)}개 발견")
               for i, order in enumerate(existing_tp_orders):
                   print(f"  {i+1}. 가격: ${order['price']:.2f}, 수량: {order['qty']} BTC")
       
       # 시장 상태 분석 (15분봉 기준)
       df_15m = market_data['timeframes']["15"]
       df_1h = market_data['timeframes'].get("60", df_15m)  # 1시간봉 (없으면 15분봉 사용)
       df_4h = market_data['timeframes'].get("240", df_15m)  # 4시간봉 (없으면 15분봉 사용)
       
       latest_15m = df_15m.iloc[-1]
       latest_1h = df_1h.iloc[-1]
       latest_4h = df_4h.iloc[-1]
       
       # === 1. 기본 지표 분석 ===
       adx_15m = float(latest_15m['ADX'])
       adx_1h = float(latest_1h['ADX'])
       adx_4h = float(latest_4h['ADX'])
       
       atr_15m = float(latest_15m['ATR'])
       atr_1h = float(latest_1h['ATR'])
       
       bb_width_15m = (float(latest_15m['BB_upper']) - float(latest_15m['BB_lower'])) / float(latest_15m['BB_mavg'])
       volatility = atr_15m / entry_price  # 현재 가격 대비 ATR
       
       rsi_15m = float(latest_15m['RSI'])
       rsi_1h = float(latest_1h['RSI'])
       
       # === 2. 시장 트렌드 평가 ===
       # 타임프레임별 가중치 설정
       tf_weights = {
           "15m": 0.5,  # 단기 (50%)
           "1h": 0.3,   # 중기 (30%)
           "4h": 0.2    # 장기 (20%)
       }
       
       # 타임프레임별 추세 강도 평가
       trend_strength = {
           "15m": min(1.0, adx_15m / 30),
           "1h": min(1.0, adx_1h / 30),
           "4h": min(1.0, adx_4h / 30)
       }
       
       # 종합 추세 강도 (가중 평균)
       overall_trend_strength = (
           trend_strength["15m"] * tf_weights["15m"] +
           trend_strength["1h"] * tf_weights["1h"] +
           trend_strength["4h"] * tf_weights["4h"]
       )
       
       # 방향성 평가 (각 타임프레임별)
       is_uptrend = {
           "15m": float(latest_15m['Close']) > float(latest_15m['EMA_20']) and float(latest_15m['MACD']) > float(latest_15m['MACD_signal']),
           "1h": float(latest_1h['Close']) > float(latest_1h['EMA_20']) and float(latest_1h['MACD']) > float(latest_1h['MACD_signal']),
           "4h": float(latest_4h['Close']) > float(latest_4h['EMA_20']) and float(latest_4h['MACD']) > float(latest_4h['MACD_signal'])
       }
       
       is_downtrend = {
           "15m": float(latest_15m['Close']) < float(latest_15m['EMA_20']) and float(latest_15m['MACD']) < float(latest_15m['MACD_signal']),
           "1h": float(latest_1h['Close']) < float(latest_1h['EMA_20']) and float(latest_1h['MACD']) < float(latest_1h['MACD_signal']),
           "4h": float(latest_4h['Close']) < float(latest_4h['EMA_20']) and float(latest_4h['MACD']) < float(latest_4h['MACD_signal'])
       }
       
       # 트렌드 일치도 평가 
       tf_agreement = 0
       if position_type == "Long":
           # 롱 포지션의 경우 상승 트렌드 일치도 확인
           tf_agreement = (
               (1 if is_uptrend["15m"] else 0) * tf_weights["15m"] +
               (1 if is_uptrend["1h"] else 0) * tf_weights["1h"] +
               (1 if is_uptrend["4h"] else 0) * tf_weights["4h"]
           )
       else:
           # 숏 포지션의 경우 하락 트렌드 일치도 확인
           tf_agreement = (
               (1 if is_downtrend["15m"] else 0) * tf_weights["15m"] +
               (1 if is_downtrend["1h"] else 0) * tf_weights["1h"] +
               (1 if is_downtrend["4h"] else 0) * tf_weights["4h"]
           )
       
       # 시장 상태 확인 (추세장 vs 횡보장)
       is_trend_market = adx_1h > 25 and bb_width_15m > 0.03
       
       # === 3. 스탑로스 가격 확인/계산 (손익비 계산에 필요) ===
       stop_loss_price = None
       
       # 3.1 거래소 API에서 현재 설정된 스탑로스 가격 확인
       try:
           orders = bybit_client.get_open_orders(
               category="linear",
               symbol=symbol,
               positionIdx=position_idx,
               orderFilter="StopOrder"
           )
           
           if orders["retCode"] == 0:
               for order in orders["result"]["list"]:
                   if order["stopOrderType"] == "StopLoss":
                       stop_loss_price = float(order["triggerPrice"])
                       print(f"[정보] 기존 스탑로스 가격 확인: ${stop_loss_price:.2f}")
                       break
           
           # 3.2 스탑로스 가격이 없으면 DB에서 조회
           if stop_loss_price is None and conn and trade_id:
               cursor = conn.cursor()
               cursor.execute("""
                   SELECT stop_loss_price FROM trades
                   WHERE trade_id = ?
               """, (trade_id,))
               
               result = cursor.fetchone()
               if result and result[0]:
                   stop_loss_price = float(result[0])
                   print(f"[정보] DB에서 스탑로스 가격 확인: ${stop_loss_price:.2f}")
       except Exception as e:
           print(f"[경고] 스탑로스 가격 조회 중 오류: {e}")
       
       # 3.3 여전히 스탑로스 가격이 없으면 기본값 계산
       if stop_loss_price is None:
           # 기본 스탑로스 비율 (ATR 기반 또는 고정 비율)
           default_sl_pct = 2.0  # 기본 2%
           
           if market_data:
               # ATR 기반 동적 손절 계산
               atr = float(market_data['timeframes']["15"].iloc[-1]['ATR'])
               volatility = atr / entry_price * 100
               default_sl_pct = max(1.0, min(3.0, volatility * 2))  # 1%~3% 범위 내에서 조정
           
           # 레버리지 조정
           default_sl_pct = min(default_sl_pct, 5.0 / leverage)  # 최대 리스크 5% 제한
           
           if position_type == "Long":
               stop_loss_price = round(entry_price * (1 - default_sl_pct/100), 1)
           else:  # Short
               stop_loss_price = round(entry_price * (1 + default_sl_pct/100), 1)
               
           print(f"[정보] 계산된 기본 스탑로스 가격: ${stop_loss_price:.2f} ({default_sl_pct:.2f}%)")
       
       # === 4. 추가 매수 시 처리 로직 ===
       if is_additional_entry and existing_tp_orders and additional_qty:
           print(f"[정보] 추가 매수를 위한 익절 전략 적용")
           
           # 시장 구조 분석
           market_structure = identify_market_structure(market_data)
           print(f"\n[시장 구조 분석]")
           print(f"시장 유형: {market_structure['type']}")
           print(f"방향성: {market_structure['direction']}")
           print(f"강도: {market_structure['strength']:.2f}")
           
           # 주요 저항/지지선 분석
           key_levels = identify_key_levels(market_data, position_type)
           
           # 가장 가까운 저항/지지선을 기준으로 TP 설정
           if key_levels:
               # 롱은 위쪽 저항선, 숏은 아래쪽 지지선 관련 레벨 필터링
               relevant_levels = []
               current_price = float(market_data['timeframes']["15"].iloc[-1]['Close'])
               
               if position_type == "Long":
                   # 진입가와 현재가 사이의 저항선 찾기
                   relevant_levels = [level for level in key_levels 
                                     if level['price'] > current_price and
                                     level['type'] == 'resistance']
               else:
                   # 진입가와 현재가 사이의 지지선 찾기
                   relevant_levels = [level for level in key_levels 
                                     if level['price'] < current_price and
                                     level['type'] == 'support']
               
               # 저항/지지선 기반 익절 가격 설정
               if relevant_levels:
                   if position_type == "Long":
                       # 가장 가까운 저항선에 약간 못 미치는 가격으로 설정
                       nearest_level = min(relevant_levels, key=lambda x: x['price'])
                       new_tp_price = round(nearest_level['price'] * 0.995, 1)
                   else:
                       # 가장 가까운 지지선보다 약간 높은 가격으로 설정
                       nearest_level = max(relevant_levels, key=lambda x: x['price'])
                       new_tp_price = round(nearest_level['price'] * 1.005, 1)
                   
                   print(f"[정보] 주요 레벨 기반 추가 매수 익절 설정: ${new_tp_price:.2f}")
               else:
                   # 기존 TP와 유사한 비율로 설정
                   if existing_tp_orders:
                       existing_tp = existing_tp_orders[0]
                       # 시장 구조에 기반한 가격 조정
                       adjustment_factor = 1.0
                       if tf_agreement > 0.7:  # 강한 일치성
                           adjustment_factor = 1.05 if position_type == "Long" else 0.95
                       elif tf_agreement < 0.3:  # 약한 일치성
                           adjustment_factor = 0.95 if position_type == "Long" else 1.05
                       
                       new_tp_price = round(existing_tp["price"] * adjustment_factor, 1)
                       print(f"[정보] 기존 TP 가격 ${existing_tp['price']:.2f} 기반 익절 설정: ${new_tp_price:.2f}")
                   else:
                       # 기본 수익률로 설정
                       tp_pct = 0.02 if position_type == "Long" else 0.02  # 2% 기본 목표
                       new_tp_price = round(entry_price * (1 + tp_pct) if position_type == "Long" else entry_price * (1 - tp_pct), 1)
                       print(f"[정보] 기본 비율(2%) 기반 익절 설정: ${new_tp_price:.2f}")
           else:
               # key_levels가 없는 경우 기존 로직 사용
               if existing_tp_orders:
                   existing_tp = existing_tp_orders[0]
                   existing_tp_price = existing_tp["price"]
                   
                   # 시장 구조에 기반한 가격 조정
                   adjustment_factor = 1.0
                   if tf_agreement > 0.7:  # 강한 일치성
                       adjustment_factor = 1.05 if position_type == "Long" else 0.95
                   elif tf_agreement < 0.3:  # 약한 일치성
                       adjustment_factor = 0.95 if position_type == "Long" else 1.05
                   
                   new_tp_price = round(existing_tp_price * adjustment_factor, 1)
               else:
                   # 기본 익절 비율 사용
                   tp_pct = 0.02
                   if position_type == "Long":
                       new_tp_price = round(entry_price * (1 + tp_pct), 1)
                   else:
                       new_tp_price = round(entry_price * (1 - tp_pct), 1)
           
           # 추가 매수 수량에 대한 익절 물량 계산
           # 시장 상태에 따른 수량 분배 조정
           if is_trend_market:  # 추세장
               additional_tp_qty = round(additional_qty * 0.7, 3)  # 추세장에서는 더 많이 유지
           else:  # 횡보장
               additional_tp_qty = round(additional_qty * 0.8, 3)  # 횡보장에서는 더 빨리 청산
           
           # 새로 추가: TP 가격 유효성 검증
           if position_type == "Long" and new_tp_price <= entry_price:
               print(f"[경고] 추가 매수의 TP 가격(${new_tp_price:.2f})이 진입가(${entry_price:.2f})보다 낮습니다. 조정 필요.")
               new_tp_price = round(entry_price * 1.005, 1)
           elif position_type == "Short" and new_tp_price >= entry_price:
               print(f"[경고] 추가 매수의 TP 가격(${new_tp_price:.2f})이 진입가(${entry_price:.2f})보다 높습니다. 조정 필요.")
               new_tp_price = round(entry_price * 0.995, 1)
               
           print(f"[정보] 추가 매수 익절 설정: 최종 가격 ${new_tp_price:.2f}, 수량 {additional_tp_qty} BTC")
           
           # 익절 주문 설정
           response = bybit_client.set_trading_stop(
               category="linear",
               symbol=symbol,
               positionIdx=position_idx,
               takeProfit=str(new_tp_price),
               tpSize=str(additional_tp_qty),
               tpTriggerBy="LastPrice",
               tpslMode="Partial"  # 부분 포지션 모드 활성화
           )
           
           if response["retCode"] == 0:
               print(f"[성공] 추가 매수 익절 설정 완료: 가격 ${new_tp_price:.2f}, 수량 {additional_tp_qty} BTC")
               
               # DB에 추가 익절 주문 저장
               if conn and trade_id:
                   try:
                       cursor = conn.cursor()
                       created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                       
                       # 새로운 주문 저장 (추가 매수 표시)
                       cursor.execute("""
                           INSERT INTO take_profit_orders 
                           (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, order_id, is_additional)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       """, (
                           trade_id,
                           position_type,
                           1,  # 항상 첫 번째 레벨에 추가
                           new_tp_price,
                           additional_tp_qty,
                           'Active',
                           created_at,
                           response["result"].get("orderId"),
                           1  # 추가 매수 표시
                       ))
                       
                       conn.commit()
                       print(f"[정보] 추가 매수에 대한 익절 계획이 DB에 저장되었습니다.")
                   except Exception as db_error:
                       print(f"[오류] DB 저장 중 문제 발생: {db_error}")
                       if 'cursor' in locals() and conn:
                           conn.rollback()
               
               return True
           else:
               print(f"[경고] 추가 매수 익절 설정 실패: {response['retMsg']}")
               # 실패 시 기존 전략 사용
       
       # === 5. 저항선 기반 익절 전략 결정 ===
       # 시장 구조 분석
       market_structure = identify_market_structure(market_data)
       print(f"\n[시장 구조 분석]")
       print(f"시장 유형: {market_structure['type']}")
       print(f"방향성: {market_structure['direction']}")
       print(f"강도: {market_structure['strength']:.2f}")
       print(f"타임프레임 일치도: {tf_agreement:.2f}")
       
       # 주요 저항/지지선 분석
       key_levels = identify_key_levels(market_data, position_type)
       
       # 주요 레벨 기반 익절 전략 결정
       tp_strategy = determine_tp_strategy(
           market_structure, 
           position_type, 
           key_levels, 
           entry_price, 
           stop_loss_price
       )
       
       # 익절 전략 적용
       tp_prices = tp_strategy['tp_prices']
       qty_distribution = tp_strategy['tp_quantities']
       strategy_type = tp_strategy['strategy_type']
       
       print(f"\n[저항선 기반 익절 설정]")
       print(f"전략 유형: {strategy_type}")
       
       # 각 저항선 상세 정보 출력
       if key_levels:
           print("\n주요 저항/지지 레벨:")
           for i, level in enumerate(key_levels[:3]): # 상위 3개 레벨만 표시
               level_type = "저항선" if level['type'] == 'resistance' else "지지선"
               print(f"{i+1}. {level_type}: ${level['price']:.2f} (강도: {level['strength']:.2f})")
           
           # 익절 가격과 주요 레벨 간의 관계 설명
           print("\n익절 가격과 주요 레벨 관계:")
           for i, price in enumerate(tp_prices):
               nearest_level = min(key_levels, key=lambda x: abs(x['price'] - price))
               distance_pct = abs(price - nearest_level['price']) / price * 100
               if distance_pct < 1.0: # 1% 이내 거리
                   print(f"TP {i+1}: ${price:.2f} - {nearest_level['type']} 레벨 (${nearest_level['price']:.2f}) 근처에 설정됨")
               else:
                   print(f"TP {i+1}: ${price:.2f} - 가장 가까운 {nearest_level['type']} 레벨과 {distance_pct:.2f}% 차이")
       
       # AI가 제공한 분배 비율 사용 (없으면 기본값 사용)
       if hasattr(decision, 'take_profit_distribution') and decision.take_profit_distribution and len(decision.take_profit_distribution) >= 3:
           qty_distribution = decision.take_profit_distribution
           print(f"[정보] AI 제공 익절 분배 비율 사용: {decision.take_profit_distribution}")
       
       # 각 단계별 수량 계산
       qty_per_level = [round(qty * dist, 3) for dist in qty_distribution]
       
       print("\n[최종 익절 설정]")
       for i, (price, amount) in enumerate(zip(tp_prices, qty_per_level)):
           print(f"단계 {i+1}:")
           print(f"- 청산 가격: ${price:,.2f}")
           print(f"- 수량: {amount} BTC ({qty_distribution[i]*100:.0f}%)")
       
       # 기존 TP 주문 취소 (추가 매수가 아닌 경우에만)
       if not is_additional_entry:
           cancel_all_tp_orders(symbol, position_idx)
           time.sleep(1)  # API 레이트 리밋 방지
           
       # 첫 번째 TP 주문만 설정
       first_tp_price = tp_prices[0]
       first_tp_qty = qty_per_level[0]
       
       # 최소 수량 체크
       if first_tp_qty < MIN_QTY:
           print(f"[경고] TP 수량이 최소 거래량보다 작음: {first_tp_qty} < {MIN_QTY}")
           return False
       
       print(f"\n[TP 주문 설정]")
       print(f"1단계 TP: ${first_tp_price} (수량: {first_tp_qty} BTC)")
           
       response = bybit_client.set_trading_stop(
           category="linear",
           symbol=symbol,
           positionIdx=position_idx,
           takeProfit=str(first_tp_price),
           tpSize=str(first_tp_qty),
           tpTriggerBy="LastPrice",
           tpslMode="Partial"  # 부분 포지션 모드 활성화
       )
       
       if response["retCode"] == 0:
           print(f"[성공] 1단계 TP 설정 완료")
           order_id = response["result"].get("orderId")
           
           # DB에 모든 TP 계획 저장 (첫 번째는 Active, 나머지는 Pending)
           if conn and trade_id:
               # 기존 주문 초기화
               cursor = conn.cursor()
               cursor.execute("""
                   UPDATE take_profit_orders
                   SET status = 'Cancelled'
                   WHERE trade_id = ? AND status IN ('Active', 'Pending')
               """, (trade_id,))
               
               # 새로운 주문 계획 모두 저장
               created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
               
               for i, (price, amount) in enumerate(zip(tp_prices, qty_per_level)):
                   level = i + 1
                   status = 'Active' if i == 0 else 'Pending'
                   current_order_id = order_id if i == 0 else None
                   
                   cursor.execute("""
                       INSERT INTO take_profit_orders 
                       (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, order_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   """, (
                       trade_id,
                       position_type,
                       level,
                       price,
                       amount,
                       status,
                       created_at,
                       current_order_id
                   ))
               
               conn.commit()
               print(f"[성공] {len(tp_prices)}단계 저항선 기반 익절 계획이 DB에 저장되었습니다.")
               
           print(f"[성공] 저항선 기반 이익실현 설정 완료")
           return True
       else:
           print(f"[경고] TP 설정 실패: {response['retMsg']}")
           return False
           
   except Exception as e:
       print(f"[오류] 저항선 기반 익절 설정 중 문제 발생: {e}")
       print(traceback.format_exc())
       return False
   
def set_take_profit_with_dynamic_distribution(symbol: str, qty: float, entry_price: float, position_type: str, 
                                            market_data: Dict, leverage: int, conn=None, trade_id=None, 
                                            decision=None, is_additional_entry=False, additional_qty=None) -> bool:
    """
    시장 상황과 신호 강도에 따라 익절 분배 비율을 동적으로 조정하는 함수
    - 저항선/지지선 활용 익절 전략
    - 시장 구조별 최적화된 분배 비율
    - 신호 강도 기반 익절 타이밍 조정
    
    Args:
        symbol: 거래 심볼 (예: "BTCUSDT")
        qty: 전체 포지션 수량
        entry_price: 진입 가격
        position_type: 포지션 타입 ("Long" 또는 "Short")
        market_data: 시장 데이터 딕셔너리
        leverage: 레버리지 배수
        conn: 데이터베이스 연결 객체 (옵션)
        trade_id: 거래 ID (옵션)
        decision: 거래 결정 객체 (옵션)
        is_additional_entry: 추가 매수 여부
        additional_qty: 추가된 수량 (전체 수량이 아님)
    """
    try:
        # 로깅 추가 
        print("\n=== 동적 분배 익절 주문 설정 시작 ===")
        print(f"포지션 타입: {position_type}")
        print(f"진입가: ${entry_price:.2f}")
        print(f"수량: {qty} BTC")
        print(f"레버리지: {leverage}x")
        print(f"거래 ID: {trade_id}")
        print(f"추가 매수 여부: {is_additional_entry}")
        if is_additional_entry:
            print(f"추가 매수 수량: {additional_qty} BTC")
        
        position_idx = 1 if position_type == "Long" else 2
        
        # 시장 구조 분석
        market_structure = identify_market_structure(market_data)
        print(f"\n[시장 구조 분석]")
        print(f"시장 유형: {market_structure['type']}")
        print(f"방향성: {market_structure['direction']}")
        print(f"강도: {market_structure['strength']:.2f}")
        
        # 주요 저항/지지선 분석
        key_levels = identify_key_levels(market_data, position_type)
        
        # 신호 강도 평가
        signal_strength = evaluate_signal_strength(market_data, position_type)
        print(f"[정보] 신호 강도: {signal_strength:.2f}")
        
        # 시장 상태 분석 (추세 강도, 변동성 등)
        trend_strength = float(market_data['timeframes']["60"].iloc[-1]['ADX'])
        volatility = market_data['timeframes']["15"]['Close'].pct_change().rolling(14).std().iloc[-1] * 100
        
        # 타임프레임 간 일치도 계산
        tf_agreement = calculate_timeframe_agreement(market_data, position_type)
        print(f"[정보] 타임프레임 일치도: {tf_agreement:.2f}")
        
        # 기존 익절 주문 정보 조회
        existing_tp_orders = []
        if is_additional_entry:
            orders = bybit_client.get_open_orders(
                category="linear",
                symbol=symbol,
                positionIdx=position_idx,
                orderFilter="StopOrder"
            )
            
            if orders["retCode"] == 0:
                for order in orders["result"]["list"]:
                    if order["stopOrderType"] == "TakeProfit":
                        existing_tp_orders.append({
                            "order_id": order["orderId"],
                            "price": float(order["triggerPrice"]),
                            "qty": float(order["qty"])
                        })
                        
                print(f"[정보] 기존 익절 주문 {len(existing_tp_orders)}개 발견")
                for i, order in enumerate(existing_tp_orders):
                    print(f"  {i+1}. 가격: ${order['price']:.2f}, 수량: {order['qty']} BTC")
        
        # === 추가 매수 처리 로직 ===
        if is_additional_entry and existing_tp_orders and additional_qty:
            print(f"[정보] 추가 매수를 위한 익절 전략 적용")
            
            # 가장 가까운 저항/지지선을 기준으로 TP 설정
            if key_levels:
                # 롱은 위쪽 저항선, 숏은 아래쪽 지지선 관련 레벨 필터링
                relevant_levels = []
                current_price = float(market_data['timeframes']["15"].iloc[-1]['Close'])
                
                if position_type == "Long":
                    # 현재가 위의 저항선 찾기
                    relevant_levels = [level for level in key_levels 
                                      if level['price'] > current_price and
                                      level['type'] == 'resistance']
                else:
                    # 현재가 아래의 지지선 찾기
                    relevant_levels = [level for level in key_levels 
                                      if level['price'] < current_price and
                                      level['type'] == 'support']
                
                # 저항/지지선 기반 익절 가격 설정
                if relevant_levels:
                    if position_type == "Long":
                        # 가장 가까운 저항선에 약간 못 미치는 가격으로 설정
                        nearest_level = min(relevant_levels, key=lambda x: x['price'])
                        new_tp_price = round(nearest_level['price'] * 0.995, 1)
                        
                        # 저항 강도에 따른 비율 조정
                        strength_factor = nearest_level['strength']
                        if strength_factor > 0.8:  # 매우 강한 저항
                            new_tp_price = round(nearest_level['price'] * 0.99, 1)  # 더 일찍 익절
                    else:
                        # 가장 가까운 지지선보다 약간 높은 가격으로 설정
                        nearest_level = max(relevant_levels, key=lambda x: x['price'])
                        new_tp_price = round(nearest_level['price'] * 1.005, 1)
                        
                        # 지지 강도에 따른 비율 조정
                        strength_factor = nearest_level['strength']
                        if strength_factor > 0.8:  # 매우 강한 지지
                            new_tp_price = round(nearest_level['price'] * 1.01, 1)  # 더 일찍 익절
                    
                    print(f"[정보] 주요 레벨 (강도: {strength_factor:.2f}) 기반 추가 매수 익절 설정: ${new_tp_price:.2f}")
                else:
                    # 기존 TP와 유사한 비율로 설정
                    if existing_tp_orders:
                        existing_tp = existing_tp_orders[0]
                        
                        # 시장 구조에 따른 가격 조정
                        adjustment_factor = 1.0
                        if tf_agreement > 0.7:  # 강한 일치성
                            adjustment_factor = 1.05 if position_type == "Long" else 0.95
                        elif tf_agreement < 0.3:  # 약한 일치성
                            adjustment_factor = 0.95 if position_type == "Long" else 1.05
                        
                        # 시장 구조에 따른 추가 조정
                        if market_structure['type'] == "TREND" and market_structure['strength'] > 0.7:
                            # 강한 추세시 더 높은 목표
                            adjustment_factor *= 1.1 if position_type == "Long" else 0.9
                        elif market_structure['type'] == "RANGE":
                            # 횡보장에서는 더 보수적 목표
                            adjustment_factor *= 0.95 if position_type == "Long" else 1.05
                        
                        new_tp_price = round(existing_tp["price"] * adjustment_factor, 1)
                        print(f"[정보] 기존 TP 기반 익절 설정 (조정 계수: {adjustment_factor:.2f}): ${new_tp_price:.2f}")
                    else:
                        # 기본 익절 비율 사용
                        base_tp_pct = 0.02  # 기본 2%
                        
                        # 변동성에 따른 조정
                        if volatility > 2.0:  # 높은 변동성
                            vol_factor = 1.3
                        elif volatility < 0.8:  # 낮은 변동성
                            vol_factor = 0.8
                        else:
                            vol_factor = 1.0
                            
                        # 신호 강도에 따른 조정
                        if signal_strength > 0.8:  # 강한 신호
                            signal_factor = 1.2
                        elif signal_strength < 0.4:  # 약한 신호
                            signal_factor = 0.8
                        else:
                            signal_factor = 1.0
                            
                        adjusted_tp_pct = base_tp_pct * vol_factor * signal_factor
                        
                        if position_type == "Long":
                            new_tp_price = round(entry_price * (1 + adjusted_tp_pct), 1)
                        else:
                            new_tp_price = round(entry_price * (1 - adjusted_tp_pct), 1)
                            
                        print(f"[정보] 동적 비율({adjusted_tp_pct:.2%}) 기반 익절 설정: ${new_tp_price:.2f}")
            else:
                # key_levels가 없는 경우 기본 로직 사용
                if existing_tp_orders:
                    existing_tp = existing_tp_orders[0]
                    existing_tp_price = existing_tp["price"]
                    
                    # 시장 구조에 기반한 가격 조정
                    adjustment_factor = 1.0
                    if tf_agreement > 0.7:  # 강한 일치성
                        adjustment_factor = 1.05 if position_type == "Long" else 0.95
                    elif tf_agreement < 0.3:  # 약한 일치성
                        adjustment_factor = 0.95 if position_type == "Long" else 1.05
                    
                    new_tp_price = round(existing_tp_price * adjustment_factor, 1)
                else:
                    # 기본 익절 비율 사용
                    tp_pct = 0.02
                    if position_type == "Long":
                        new_tp_price = round(entry_price * (1 + tp_pct), 1)
                    else:
                        new_tp_price = round(entry_price * (1 - tp_pct), 1)
            
            # 추가 매수 수량에 대한 익절 물량 계산
            # 시장 구조와 신호 강도에 따른 수량 분배 최적화
            if market_structure['type'] == "TREND" and signal_strength > 0.7:
                # 강한 신호와 트렌드에서는 작은 부분만 익절
                additional_tp_qty = round(additional_qty * 0.6, 3)
                print("[전략] 강한 추세, 소량만 익절 (60%)")
            elif market_structure['type'] == "VOLATILE" or signal_strength < 0.4:
                # 변동성 큰 장이나 약한 신호에서는 많은 부분 익절
                additional_tp_qty = round(additional_qty * 0.85, 3)
                print("[전략] 높은 변동성 또는 약한 신호, 대부분 익절 (85%)")
            else:
                # 일반 상황에서는 중간 수준 익절
                additional_tp_qty = round(additional_qty * 0.75, 3)
                print("[전략] 일반 상황, 중간 수준 익절 (75%)")
            
            # 새로 추가: TP 가격 유효성 검증
            if position_type == "Long" and new_tp_price <= entry_price:
                print(f"[경고] 추가 매수의 TP 가격(${new_tp_price:.2f})이 진입가(${entry_price:.2f})보다 낮습니다. 조정 필요.")
                new_tp_price = round(entry_price * 1.005, 1)
            elif position_type == "Short" and new_tp_price >= entry_price:
                print(f"[경고] 추가 매수의 TP 가격(${new_tp_price:.2f})이 진입가(${entry_price:.2f})보다 높습니다. 조정 필요.")
                new_tp_price = round(entry_price * 0.995, 1)
                
            print(f"[정보] 추가 매수 익절 설정: 가격 ${new_tp_price:.2f}, 수량 {additional_tp_qty} BTC")
            
            # 익절 주문 설정
            response = bybit_client.set_trading_stop(
                category="linear",
                symbol=symbol,
                positionIdx=position_idx,
                takeProfit=str(new_tp_price),
                tpSize=str(additional_tp_qty),
                tpTriggerBy="LastPrice",
                tpslMode="Partial"  # 부분 포지션 모드 활성화
            )
            
            if response["retCode"] == 0:
                print(f"[성공] 추가 매수 익절 설정 완료: 가격 ${new_tp_price:.2f}, 수량 {additional_tp_qty} BTC")
                
                # DB에 추가 익절 주문 저장
                if conn and trade_id:
                    try:
                        cursor = conn.cursor()
                        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 새로운 주문 저장 (추가 매수 표시)
                        cursor.execute("""
                            INSERT INTO take_profit_orders 
                            (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, order_id, is_additional)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            trade_id,
                            position_type,
                            1,  # 항상 첫 번째 레벨에 추가
                            new_tp_price,
                            additional_tp_qty,
                            'Active',
                            created_at,
                            response["result"].get("orderId"),
                            1  # 추가 매수 표시
                        ))
                        
                        conn.commit()
                        print(f"[정보] 추가 매수에 대한 익절 계획이 DB에 저장되었습니다.")
                    except Exception as db_error:
                        print(f"[오류] DB 저장 중 문제 발생: {db_error}")
                        if 'cursor' in locals() and conn:
                            conn.rollback()
                
                return True
            else:
                print(f"[경고] 추가 매수 익절 설정 실패: {response['retMsg']}")
                # 실패 시 기존 전략 사용
        
        # === 상황에 따른 분배 비율 결정 ===
        if signal_strength > 0.8 and trend_strength > 25 and tf_agreement > 0.8:
            # 명확한 추세와 강한 신호 - 큰 TP에 더 많은 비중
            print("[전략] 강한 신호와 명확한 추세 감지 - 50/25/25 분배 적용")
            qty_distribution = [0.5, 0.25, 0.25]
        elif signal_strength > 0.7 and market_structure['type'] == "TREND":
            # 강한 신호와 트렌드 시장 - 큰 TP에 더 많은 비중
            print("[전략] 강한 신호와 트렌드 시장 - 30/30/40 분배 적용")
            qty_distribution = [0.3, 0.3, 0.4]
        elif signal_strength < 0.4 or volatility > 2.0:
            # 약한 신호나 높은 변동성 - 초기에 더 많이 청산
            print("[전략] 약한 신호 또는 높은 변동성 - 40/40/20 분배 적용")
            qty_distribution = [0.4, 0.4, 0.2]
        elif market_structure['type'] == "RANGE" and trend_strength < 20:
            # 횡보장에서는 보수적 접근
            print("[전략] 명확한 횡보장 감지 - 45/35/20 분배 적용")
            qty_distribution = [0.45, 0.35, 0.2]
        elif market_structure['type'] == "VOLATILE" or volatility > 1.5:
            # 변동성 장에서는 빠른 이익실현 중시
            print("[전략] 변동성 시장 감지 - 50/30/20 분배 적용")
            qty_distribution = [0.5, 0.3, 0.2]
        else:
            # 중간 강도의 신호, 일반적 상황 - 균형 잡힌 분배
            print("[전략] 중간 강도 신호 - 33/33/34 분배 적용")
            qty_distribution = [0.33, 0.33, 0.34]
        
        # AI가 제공한 분배 비율 사용 (없으면 위에서 결정한 비율 사용)
        if hasattr(decision, 'take_profit_distribution') and decision.take_profit_distribution and len(decision.take_profit_distribution) >= 3:
            qty_distribution = decision.take_profit_distribution
            print(f"[정보] AI 제공 익절 분배 비율 사용: {decision.take_profit_distribution}")
            
        # === 6. 손익비 기반 익절 계산 ===
        # 스탑로스 가격 확인 (손익비 계산에 필요)
        stop_loss_price = None
        try:
            orders = bybit_client.get_open_orders(
                category="linear",
                symbol=symbol,
                positionIdx=position_idx,
                orderFilter="StopOrder"
            )
            
            if orders["retCode"] == 0:
                for order in orders["result"]["list"]:
                    if order["stopOrderType"] == "StopLoss":
                        stop_loss_price = float(order["triggerPrice"])
                        print(f"[정보] 기존 스탑로스 가격 확인: ${stop_loss_price:.2f}")
                        break
                        
            # DB에서 조회 (API에서 찾지 못한 경우)
            if stop_loss_price is None and conn and trade_id:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT stop_loss_price FROM trades
                    WHERE trade_id = ?
                """, (trade_id,))
                
                result = cursor.fetchone()
                if result and result[0]:
                    stop_loss_price = float(result[0])
                    print(f"[정보] DB에서 스탑로스 가격 확인: ${stop_loss_price:.2f}")
        except Exception as e:
            print(f"[경고] 스탑로스 가격 조회 중 오류: {e}")
            
        # 스탑로스 가격이 없으면 기본값 계산
        if stop_loss_price is None:
            default_sl_pct = 2.0  # 기본 2%
            if market_data:
                atr = float(market_data['timeframes']["15"].iloc[-1]['ATR'])
                volatility = atr / entry_price * 100
                default_sl_pct = max(1.0, min(3.0, volatility * 2))
                
            # 레버리지 조정
            default_sl_pct = min(default_sl_pct, 5.0 / leverage)
            
            if position_type == "Long":
                stop_loss_price = round(entry_price * (1 - default_sl_pct/100), 1)
            else:
                stop_loss_price = round(entry_price * (1 + default_sl_pct/100), 1)
                
            print(f"[정보] 계산된 기본 스탑로스 가격: ${stop_loss_price:.2f} ({default_sl_pct:.2f}%)")
            
        # 리스크 금액 계산 (진입가-스탑로스 가격의 절대값)
        risk_amount = abs(entry_price - stop_loss_price)
        print(f"[정보] 리스크 금액: ${risk_amount:.2f}")
        
        # === 저항선 기반 익절 전략 결정 ===
        tp_strategy = determine_tp_strategy(
            market_structure, 
            position_type, 
            key_levels, 
            entry_price, 
            stop_loss_price
        )
        
        # 익절 전략 적용
        tp_prices = tp_strategy['tp_prices']
        strategy_type = tp_strategy['strategy_type']
        
        print(f"\n[저항선 기반 익절 설정]")
        print(f"전략 유형: {strategy_type}")
        
        # 각 저항선 상세 정보 출력
        if key_levels:
            print("\n주요 저항/지지 레벨:")
            for i, level in enumerate(key_levels[:3]): # 상위 3개 레벨만 표시
                level_type = "저항선" if level['type'] == 'resistance' else "지지선"
                print(f"{i+1}. {level_type}: ${level['price']:.2f} (강도: {level['strength']:.2f})")
            
            # 익절 가격과 주요 레벨 간의 관계 설명
            print("\n익절 가격과 주요 레벨 관계:")
            for i, price in enumerate(tp_prices):
                nearest_level = min(key_levels, key=lambda x: abs(x['price'] - price))
                distance_pct = abs(price - nearest_level['price']) / price * 100
                if distance_pct < 1.0: # 1% 이내 거리
                    print(f"TP {i+1}: ${price:.2f} - {nearest_level['type']} 레벨 (${nearest_level['price']:.2f}) 근처에 설정됨")
                else:
                    print(f"TP {i+1}: ${price:.2f} - 가장 가까운 {nearest_level['type']} 레벨과 {distance_pct:.2f}% 차이")
        
        # 각 단계별 수량 계산
        qty_per_level = [round(qty * dist, 3) for dist in qty_distribution]
        
        print("\n[최종 익절 설정]")
        for i, (price, amount) in enumerate(zip(tp_prices, qty_per_level)):
            print(f"단계 {i+1}:")
            print(f"- 청산 가격: ${price:,.2f}")
            print(f"- 수량: {amount} BTC ({qty_distribution[i]*100:.0f}%)")
        
        # 기존 TP 주문 취소 (추가 매수가 아닌 경우에만)
        if not is_additional_entry:
            cancel_all_tp_orders(symbol, position_idx)
            time.sleep(1)  # API 레이트 리밋 방지
            
        # 첫 번째 TP 주문만 설정
        first_tp_price = tp_prices[0]
        first_tp_qty = qty_per_level[0]
        
        # 최소 수량 체크
        if first_tp_qty < MIN_QTY:
            print(f"[경고] TP 수량이 최소 거래량보다 작음: {first_tp_qty} < {MIN_QTY}")
            return False
        
        print(f"\n[TP 주문 설정]")
        print(f"1단계 TP: ${first_tp_price} (수량: {first_tp_qty} BTC)")
            
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            takeProfit=str(first_tp_price),
            tpSize=str(first_tp_qty),
            tpTriggerBy="LastPrice",
            tpslMode="Partial"  # 부분 포지션 모드 활성화
        )
        
        if response["retCode"] == 0:
            print(f"[성공] 1단계 TP 설정 완료")
            order_id = response["result"].get("orderId")
            
            # DB에 모든 TP 계획 저장 (첫 번째는 Active, 나머지는 Pending)
            if conn and trade_id:
                # 기존 주문 초기화
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE take_profit_orders
                    SET status = 'Cancelled'
                    WHERE trade_id = ? AND status IN ('Active', 'Pending')
                """, (trade_id,))
                
                # 새로운 주문 계획 모두 저장
                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                for i, (price, amount) in enumerate(zip(tp_prices, qty_per_level)):
                    level = i + 1
                    status = 'Active' if i == 0 else 'Pending'
                    current_order_id = order_id if i == 0 else None
                    
                    cursor.execute("""
                        INSERT INTO take_profit_orders 
                        (trade_id, position_type, tp_level, tp_price, tp_quantity, status, created_at, order_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_id,
                        position_type,
                        level,
                        price,
                        amount,
                        status,
                        created_at,
                        current_order_id
                    ))
                
                conn.commit()
                print(f"[성공] {len(tp_prices)}단계 동적 분배 익절 계획이 DB에 저장되었습니다.")
                
            print(f"[성공] 동적 분배 이익실현 설정 완료")
            return True
        else:
            print(f"[경고] TP 설정 실패: {response['retMsg']}")
            return False
            
    except Exception as e:
        print(f"[오류] 동적 분배 익절 설정 중 문제 발생: {e}")
        print(traceback.format_exc())
        return False


def activate_next_tp_order(conn, trade_id: str, position_type: str, current_level: int) -> bool:
    """
    다음 단계 TP 주문을 활성화하는 함수
    """
    try:
        cursor = conn.cursor()
        next_level = current_level + 1
        
        # 다음 단계 TP 정보 조회
        cursor.execute("""
            SELECT tp_level, tp_price, tp_quantity 
            FROM take_profit_orders
            WHERE trade_id = ? AND position_type = ? AND tp_level = ? AND status = 'Pending'
        """, (trade_id, position_type, next_level))
        
        next_tp = cursor.fetchone()
        if not next_tp:
            print(f"[정보] {next_level}단계 TP 정보를 찾을 수 없음")
            return False
            
        next_level, next_price, next_qty = next_tp
        position_idx = 1 if position_type == "Long" else 2
        
        # 포지션이 여전히 존재하는지 확인
        position_info = get_position_info(bybit_client)
        if not position_info or position_info[position_type]["size"] <= 0:
            print(f"[정보] {position_type} 포지션이 더 이상 존재하지 않아 다음 단계 TP 설정 건너뜀")
            return False
        
        # 현재 가격 확인
        current_price = float(fetch_candle_data(interval="1", limit=1)['Close'].iloc[-1])
        
        # Short 포지션의 경우 TP 가격이 현재 가격보다 낮은지 확인
        if position_type == "Short" and next_price >= current_price:
            print(f"[경고] Short 포지션의 TP 가격({next_price})이 현재 가격({current_price})보다 높음")
            # 현재 가격보다 5% 낮게 조정
            next_price = round(current_price * 0.95, 1)
            print(f"[정보] TP 가격을 {next_price}로 조정")
            
        print(f"[정보] {next_level}단계 TP 활성화 중: 가격 ${next_price}, 수량 {next_qty}")
        
        # 기존 TP 주문 취소 (안전을 위해)
        cancel_all_tp_orders(SYMBOL, position_idx)
        time.sleep(1)  # API 제한 방지
        
        # 다음 단계 TP 설정
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=SYMBOL,
            positionIdx=position_idx,
            takeProfit=str(next_price),
            tpSize=str(next_qty),
            tpTriggerBy="LastPrice",
            tpslMode="Partial"  # 파라미터 추가
        )
        
        if response["retCode"] == 0:
            # 다음 단계 TP 상태 업데이트
            cursor.execute("""
                UPDATE take_profit_orders
                SET status = 'Active', order_id = ?
                WHERE trade_id = ? AND tp_level = ?
            """, (response["result"].get("orderId"), trade_id, next_level))
            
            conn.commit()
            print(f"[성공] {next_level}단계 TP 설정 완료")
            return True
        else:
            print(f"[경고] {next_level}단계 TP 설정 실패: {response['retMsg']}")
            return False
            
    except Exception as e:
        print(f"[오류] 다음 단계 TP 활성화 중 문제 발생: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def activate_next_tp_order_enhanced(conn, trade_id: str, position_type: str, current_level: int,
                                  market_data: Dict = None) -> bool:
    """
    다음 단계 TP 주문을 활성화하는 향상된 함수
    
    Args:
        conn: 데이터베이스 연결 객체
        trade_id: 거래 ID
        position_type: 포지션 유형 ("Long" 또는 "Short")
        current_level: 현재 TP 레벨
        market_data: 시장 데이터 (선택적)
        
    Returns:
        bool: 성공 여부
    """
    try:
        cursor = conn.cursor()
        next_level = current_level + 1
        
        # 1. 다음 단계 TP 정보 조회
        cursor.execute("""
            SELECT tp_level, tp_price, tp_quantity 
            FROM take_profit_orders
            WHERE trade_id = ? AND position_type = ? AND tp_level = ? AND status = 'Pending'
        """, (trade_id, position_type, next_level))
        
        next_tp = cursor.fetchone()
        if not next_tp:
            print(f"[정보] {next_level}단계 TP 정보를 찾을 수 없음")
            return False
            
        next_level, next_price, next_qty = next_tp
        position_idx = 1 if position_type == "Long" else 2
        
        print(f"[정보] {next_level}단계 TP 활성화 중: 가격 ${next_price}, 수량 {next_qty}")
        
        # 2. 시장 상황 변화 확인 및 TP 가격 동적 조정
        if market_data:
            current_price = float(market_data['timeframes']["15"].iloc[-1]['Close'])
            
            # 시장 구조 분석
            market_structure = identify_market_structure(market_data)
            
            # 주요 레벨 분석
            key_levels = identify_key_levels(market_data, position_type)
            
            # 2.1 시장 구조 변화에 따른 조정
            if market_structure:
                print(f"[정보] 현재 시장 구조: {market_structure['type']}, 방향: {market_structure['direction']}")
                
                # 추세 강도가 증가한 경우, TP 목표 확장
                if market_structure['type'] == "TREND" and market_structure['strength'] > 0.7:
                    adjust_factor = 1.1 if position_type == "Long" else 0.9
                    original_next_price = next_price
                    next_price = round(next_price * adjust_factor, 1)
                    print(f"[정보] 강한 추세 감지: TP 가격 조정 ${original_next_price:.2f} -> ${next_price:.2f}")
                
                # 추세와 포지션이 반대 방향인 경우, TP 목표 축소
                direction_opposite = (position_type == "Long" and market_structure['direction'] == "DOWN") or \
                                    (position_type == "Short" and market_structure['direction'] == "UP")
                if direction_opposite and market_structure['confidence'] > 0.6:
                    adjust_factor = 0.95 if position_type == "Long" else 1.05
                    original_next_price = next_price
                    next_price = round(next_price * adjust_factor, 1)
                    print(f"[정보] 반대 방향 추세 감지: TP 가격 조정 ${original_next_price:.2f} -> ${next_price:.2f}")
     
            # 2.2 주요 레벨을 고려한 조정
            if key_levels:
                # 롱 포지션은 위쪽 레벨, 숏 포지션은 아래쪽 레벨 찾기
                relevant_levels = []
                if position_type == "Long":
                    # 현재 가격보다 높고 계산된 TP 근처의 저항선
                    relevant_levels = [level for level in key_levels 
                                      if level['price'] > current_price and 
                                      abs(level['price'] - next_price) / next_price < 0.03 and  # 3% 내외
                                      level['type'] == 'resistance']
                else:
                    # 현재 가격보다 낮고 계산된 TP 근처의 지지선
                    relevant_levels = [level for level in key_levels 
                                      if level['price'] < current_price and 
                                      abs(level['price'] - next_price) / next_price < 0.03 and  # 3% 내외
                                      level['type'] == 'support']
                
                # 강한 레벨이 있으면 조정
                strong_levels = [level for level in relevant_levels if level['strength'] > 0.7]
                if strong_levels:
                    if position_type == "Long":
                        # 롱 포지션은 강한 저항선 직전에 TP 설정
                        nearest_level = min(strong_levels, key=lambda x: x['price'])
                        original_next_price = next_price
                        next_price = round(nearest_level['price'] * 0.995, 1)  # 저항선 바로 아래
                        print(f"[정보] 강한 저항선 감지: TP 가격 조정 ${original_next_price:.2f} -> ${next_price:.2f}")
                    else:
                        # 숏 포지션은 강한 지지선 직전에 TP 설정
                        nearest_level = max(strong_levels, key=lambda x: x['price'])
                        original_next_price = next_price
                        next_price = round(nearest_level['price'] * 0.995, 1)  # 지지선 바로 아래

                        print(f"[정보] 강한 지지선 감지: TP 가격 조정 ${original_next_price:.2f} -> ${next_price:.2f}")
        
        # 3. 기존 TP 주문 취소 (안전을 위해)
        cancel_all_tp_orders(SYMBOL, position_idx)
        time.sleep(1)  # API 제한 방지
        
        # 4. 다음 단계 TP 설정
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=SYMBOL,
            positionIdx=position_idx,
            takeProfit=str(next_price),
            tpSize=str(next_qty),
            tpTriggerBy="LastPrice",
            tpslMode="Partial"  # 부분 청산 모드
        )
        
        if response["retCode"] == 0:
            # 5. 다음 단계 TP 상태 업데이트
            cursor.execute("""
                UPDATE take_profit_orders
                SET status = 'Active', order_id = ?, tp_price = ?
                WHERE trade_id = ? AND tp_level = ?
            """, (response["result"].get("orderId"), next_price, trade_id, next_level))
            
            conn.commit()
            print(f"[성공] {next_level}단계 TP 설정 완료: ${next_price}")
            
            # 6. 이전 단계 TP 이후 스탑로스 조정 (선택적)
            if current_level == 1:
                # 첫 TP 이후 스탑로스를 진입가 근처로
                adjust_stop_loss_after_first_tp(conn, trade_id, position_type, 
                                               get_entry_price_from_db(conn, trade_id))
            elif current_level == 2:
                # 두 번째 TP 이후 스탑로스를 손익분기점으로
                set_breakeven_stop_loss(conn, trade_id, position_type,
                                       get_entry_price_from_db(conn, trade_id))
            
            return True
        else:
            print(f"[경고] {next_level}단계 TP 설정 실패: {response['retMsg']}")
            return False
            
    except Exception as e:
        print(f"[오류] 다음 단계 TP 활성화 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'conn' in locals():
            conn.rollback()
        return False

def cancel_all_tp_orders(symbol: str, position_idx: int) -> bool:
    """모든 TP 주문 취소"""
    try:
        orders = bybit_client.get_open_orders(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            orderFilter="StopOrder"
        )
        
        if orders["retCode"] == 0:
            cancelled = 0
            for order in orders["result"]["list"]:
                if order["stopOrderType"] == "TakeProfit":
                    response = bybit_client.cancel_order(
                        category="linear",
                        symbol=symbol,
                        orderId=order["orderId"]
                    )
                    if response["retCode"] == 0:
                        cancelled += 1
                        
            if cancelled > 0:
                print(f"[정보] {cancelled}개의 TP 주문 취소 완료")
            return True
        return False
    except Exception as e:
        print(f"[오류] TP 주문 취소 중 문제 발생: {e}")
        return False    

def cleanup_orphaned_tp_orders():
    """
    포지션이 없을 때 남아있는 활성 TP 주문을 정리하는 함수
    """
    try:
        # 현재 포지션 정보 가져오기
        positions = get_position_info(bybit_client)
        if not positions:
            print("[오류] 포지션 정보를 가져올 수 없습니다")
            return False
            
        # Long과 Short 포지션이 모두 없는 경우에만 실행
        if positions["Long"]["size"] <= 0 and positions["Short"]["size"] <= 0:
            print("[정보] 현재 포지션이 없음. 남아있는 TP 주문 정리 중...")
            
            # 데이터베이스에서 활성 상태의 모든 TP 주문을 비활성화
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE take_profit_orders
                    SET status = 'Cancelled'
                    WHERE status = 'Active' OR status = 'Pending'
                """)
                
                # Long 포지션의 모든 TP 주문 취소
                cancel_all_tp_orders(SYMBOL, 1)  # 1 = Long
                
                # Short 포지션의 모든 TP 주문 취소
                cancel_all_tp_orders(SYMBOL, 2)  # 2 = Short
                
                print("[성공] 고아 TP 주문 정리 완료")
            return True
        return False
    except Exception as e:
        print(f"[오류] 고아 TP 주문 정리 중 문제 발생: {e}")
        return False

import time
import logging

logger = logging.getLogger(__name__)

def fetch_real_execution_data(symbol: str, start_time: int = None, end_time: int = None, limit: int = 50):
    """
    바이비트에서 지정 심볼의 실제 체결 내역을 조회합니다.
    :param symbol: 예) 'BTCUSDT'
    :param start_time: 조회 시작 시간 (UNIX timestamp in ms)
    :param end_time: 조회 종료 시간 (UNIX timestamp in ms)
    :param limit: 조회할 체결 수 (기본 50개)
    :return: 체결 내역 리스트 (list of executions)
    """
    endpoint = "/v5/execution/list"
    params = {
        "category": "linear",  # USDT perpetual
        "symbol": symbol,
        "limit": limit,
    }

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    try:
        response = send_signed_request("GET", endpoint, params)
        if response and response.get("retCode") == 0:
            executions = response["result"]["list"]
            logger.info(f"[바이비트 체결] {len(executions)}건의 체결 정보 조회 완료")
            return executions
        else:
            logger.error(f"[오류] 체결 정보 조회 실패: {response}")
            return []
    except Exception as e:
        logger.exception(f"[예외] 체결 정보 요청 중 오류 발생: {e}")
        return []

def calculate_real_pnl(executions: list, position_side: str) -> dict:
    """
    바이비트 체결 데이터를 바탕으로 실제 진입/청산 평균가와 수익률을 계산합니다.
    
    :param executions: fetch_real_execution_data()의 결과 리스트
    :param position_side: 'Long' 또는 'Short'
    :return: {
        'entry_price': 평균 진입가,
        'exit_price': 평균 청산가,
        'realized_pnl': 실현 손익 (USDT),
        'pnl_pct': 수익률 (%),
        'total_fee': 총 수수료 (USDT)
    }
    """
    entry_value = 0
    exit_value = 0
    entry_qty = 0
    exit_qty = 0
    total_fee = 0

    for exe in executions:
        side = exe["side"]
        price = float(exe["price"])
        qty = float(exe["qty"])
        fee = float(exe["execFee"])
        total_fee += fee

        if position_side == "Long":
            if side == "Buy":
                entry_value += price * qty
                entry_qty += qty
            elif side == "Sell":
                exit_value += price * qty
                exit_qty += qty
        elif position_side == "Short":
            if side == "Sell":
                entry_value += price * qty
                entry_qty += qty
            elif side == "Buy":
                exit_value += price * qty
                exit_qty += qty

    # 진입과 청산이 모두 있어야 계산 가능
    if entry_qty == 0 or exit_qty == 0:
        return {
            'entry_price': 0,
            'exit_price': 0,
            'realized_pnl': 0,
            'pnl_pct': 0,
            'total_fee': total_fee
        }

    avg_entry_price = entry_value / entry_qty
    avg_exit_price = exit_value / exit_qty
    size = min(entry_qty, exit_qty)

    if position_side == "Long":
        realized_pnl = (avg_exit_price - avg_entry_price) * size - total_fee
    else:  # Short
        realized_pnl = (avg_entry_price - avg_exit_price) * size - total_fee

    pnl_pct = (realized_pnl / (avg_entry_price * size)) * 100

    return {
        'entry_price': round(avg_entry_price, 2),
        'exit_price': round(avg_exit_price, 2),
        'realized_pnl': round(realized_pnl, 4),
        'pnl_pct': round(pnl_pct, 2),
        'total_fee': round(total_fee, 4)
    }


def calculate_portfolio_performance(conn) -> Optional[Dict]:
    """포트폴리오 성과 계산 (초기 자산 대비 현재 잔고 및 수익률)"""
    INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "534.23"))
    try:
        c = conn.cursor()
        c.execute("SELECT wallet_balance FROM trades ORDER BY timestamp DESC LIMIT 1")
        result = c.fetchone()
        if result is None:
            logger.warning("[경고] 거래 내역이 없습니다. 포트폴리오 수익률 계산 불가.")
            return None
        latest_balance = result[0]
        total_return = ((latest_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        return {
            "initial_balance": round(INITIAL_BALANCE, 2),
            "current_balance": round(latest_balance, 2),
            "total_return_pct": round(total_return, 2)
        }
    except Exception as e:
        logger.exception(f"[오류] 포트폴리오 성과 계산 중 오류: {e}")
        return None



def update_trade_with_real_pnl(trade_id: str, symbol: str, position_side: str, start_time: int, end_time: int):
    """
    체결 데이터를 기반으로 실제 수익률을 계산하고 DB의 trade 레코드를 업데이트합니다.
    
    :param trade_id: 거래 ID (trades 테이블의 기본 키)
    :param symbol: 예) 'BTCUSDT'
    :param position_side: 'Long' 또는 'Short'
    :param start_time: 진입 시각 (UNIX timestamp in ms)
    :param end_time: 청산 시각 (UNIX timestamp in ms)
    """
    executions = fetch_real_execution_data(symbol, start_time, end_time)
    if not executions:
        print(f"[오류] 체결 데이터 없음 → Trade ID: {trade_id}")
        return False

    pnl_data = calculate_real_pnl(executions, position_side)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET
                    entry_price = ?,
                    exit_price = ?,
                    realized_pnl = ?,
                    pnl_pct = ?,
                    entry_fee = ?,
                    exit_fee = ?
                WHERE trade_id = ?
            """, (
                pnl_data['entry_price'],
                pnl_data['exit_price'],
                pnl_data['realized_pnl'],
                pnl_data['pnl_pct'],
                round(pnl_data['total_fee'] / 2, 4),  # 진입 수수료
                round(pnl_data['total_fee'] / 2, 4),  # 청산 수수료
                trade_id
            ))
            print(f"[성공] 거래 수익률 업데이트 완료 → Trade ID: {trade_id}")
            return True
    except Exception as e:
        print(f"[오류] 거래 수익률 업데이트 실패: {e}")
        return False


# 멀티 타임프레임 분석으로 익절 목표 최적화 함수
def optimize_tp_with_multiframe(df_15m, df_1h, df_4h, position_type):
    """
    멀티 타임프레임 분석을 통한 최적 익절 목표 도출
    """
    try:
        latest_15m = df_15m.iloc[-1]
        latest_1h = df_1h.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        
        # 각 타임프레임의 볼린저 밴드 폭 계산
        bb_width_15m = (float(latest_15m['BB_upper']) - float(latest_15m['BB_lower'])) / float(latest_15m['BB_mavg'])
        bb_width_1h = (float(latest_1h['BB_upper']) - float(latest_1h['BB_lower'])) / float(latest_1h['BB_mavg'])
        bb_width_4h = (float(latest_4h['BB_upper']) - float(latest_4h['BB_lower'])) / float(latest_4h['BB_mavg'])
        
        # 타임프레임별 가중치 산정
        # 변동성이 큰 타임프레임에 가중치 높게 부여
        total_width = bb_width_15m + bb_width_1h + bb_width_4h
        if total_width > 0:
            weights = {
                "15m": bb_width_15m / total_width,
                "1h": bb_width_1h / total_width,
                "4h": bb_width_4h / total_width
            }
        else:
            weights = {"15m": 0.5, "1h": 0.3, "4h": 0.2}
        
        # 타임프레임별 목표 이익 설정
        if position_type == "Long":
            tp_targets = {
                "15m": [
                    (latest_15m['BB_upper'] - latest_15m['Close']) / latest_15m['Close'],
                    (latest_15m['Resistance1'] - latest_15m['Close']) / latest_15m['Close'] if 'Resistance1' in latest_15m else 0.01,
                    (latest_15m['Resistance2'] - latest_15m['Close']) / latest_15m['Close'] if 'Resistance2' in latest_15m else 0.02
                ],
                "1h": [
                    (latest_1h['BB_upper'] - latest_1h['Close']) / latest_1h['Close'],
                    (latest_1h['Resistance1'] - latest_1h['Close']) / latest_1h['Close'] if 'Resistance1' in latest_1h else 0.015,
                    (latest_1h['Resistance2'] - latest_1h['Close']) / latest_1h['Close'] if 'Resistance2' in latest_1h else 0.03
                ],
                "4h": [
                    (latest_4h['BB_upper'] - latest_4h['Close']) / latest_4h['Close'],
                    (latest_4h['Resistance1'] - latest_4h['Close']) / latest_4h['Close'] if 'Resistance1' in latest_4h else 0.02,
                    (latest_4h['Resistance2'] - latest_4h['Close']) / latest_4h['Close'] if 'Resistance2' in latest_4h else 0.04
                ]
            }
        else:  # Short
            tp_targets = {
                "15m": [
                    (latest_15m['Close'] - latest_15m['BB_lower']) / latest_15m['Close'],
                    (latest_15m['Close'] - latest_15m['Support1']) / latest_15m['Close'] if 'Support1' in latest_15m else 0.01,
                    (latest_15m['Close'] - latest_15m['Support2']) / latest_15m['Close'] if 'Support2' in latest_15m else 0.02
                ],
                "1h": [
                    (latest_1h['Close'] - latest_1h['BB_lower']) / latest_1h['Close'],
                    (latest_1h['Close'] - latest_1h['Support1']) / latest_1h['Close'] if 'Support1' in latest_1h else 0.015,
                    (latest_1h['Close'] - latest_1h['Support2']) / latest_1h['Close'] if 'Support2' in latest_1h else 0.03
                ],
                "4h": [
                    (latest_4h['Close'] - latest_4h['BB_lower']) / latest_4h['Close'],
                    (latest_4h['Close'] - latest_4h['Support1']) / latest_4h['Close'] if 'Support1' in latest_4h else 0.02,
                    (latest_4h['Close'] - latest_4h['Support2']) / latest_4h['Close'] if 'Support2' in latest_4h else 0.04
                ]
            }
        
        # 최종 목표 계산 (가중 평균)
        final_targets = []
        for i in range(3):  # 3단계 익절
            target = 0
            for tf in ["15m", "1h", "4h"]:
                target += tp_targets[tf][i] * weights[tf]
            final_targets.append(max(0.005, target))  # 최소 0.5% 확보
        
        return final_targets
    except Exception as e:
        print(f"멀티 타임프레임 익절 최적화 중 오류: {e}")
        return [0.01, 0.02, 0.03]  # 기본값

def calculate_dynamic_stop_loss(entry_price: float, df: pd.DataFrame, 
                           position_type: str, leverage: int = 1, 
                           position_size_pct: float = None, 
                           wallet_balance: float = None) -> float:
    """
    지지/저항선 및 변동성 기반 동적 스탑로스 계산 함수
    
    Args:
        entry_price: 진입 가격
        df: 캔들스틱 데이터
        position_type: "Long" 또는 "Short"
        leverage: 레버리지
        position_size_pct: 포지션 크기 비율 (총자산 대비, 예: 25는 25% 의미)
        wallet_balance: 현재 총자산
        
    Returns:
        최적화된 스탑로스 가격
    """
    try:
        # 1. 변동성 계산 (ATR 활용)
        atr = float(df.iloc[-1]['ATR'])
        volatility_pct = (atr / entry_price) * 100
        
        # 2. 주요 지지/저항선 식별
        key_levels = []
        
        # 2.1 롱 포지션은 지지선 확인
        if position_type == "Long":
            # 진입가 아래의 주요 지지선 확인
            support_levels = [
                float(df.iloc[-1]['Support1']) if 'Support1' in df.iloc[-1] else None,
                float(df.iloc[-1]['Support2']) if 'Support2' in df.iloc[-1] else None,
                float(df.iloc[-1]['BB_lower']) if 'BB_lower' in df.iloc[-1] else None,
                float(df.iloc[-1]['EMA_50']) if 'EMA_50' in df.iloc[-1] and float(df.iloc[-1]['EMA_50']) < entry_price else None,
                float(df.iloc[-1]['Donchian_Low']) if 'Donchian_Low' in df.iloc[-1] else None
            ]
            
            # None 값 제거 및 진입가보다 낮은 지지선만 필터링
            key_levels = [level for level in support_levels if level is not None and level < entry_price]
            
            # 진입가에 가장 가까운 지지선 찾기 (내림차순 정렬 후 첫 번째 값)
            if key_levels:
                nearest_support = sorted(key_levels, reverse=True)[0]
                
                # 변동성에 따른 버퍼 크기 조정
                buffer_pct = min(2.0, max(1.0, volatility_pct))  # 1-2% 범위의 버퍼
                buffer = entry_price * (buffer_pct / 100)  # 백분율을 절대값으로 변환
                
                # 지지선 아래에 버퍼 추가한 스탑로스 설정
                stop_price = nearest_support - buffer
                
                # 최대 손실 제한 (총자산의 5% 이내로 제한)
                max_loss_pct = 5.0 / leverage if leverage > 0 else 5.0
                min_allowed_price = entry_price * (1 - max_loss_pct/100)
                
                # 스탑로스는 최소 손실 제한선 이상으로 설정
                stop_price = max(stop_price, min_allowed_price)
                
                return round(stop_price, 1)
        
        # 2.2 숏 포지션은 저항선 확인
        else:  # Short 포지션
            # 진입가 위의 주요 저항선 확인
            resistance_levels = [
                float(df.iloc[-1]['Resistance1']) if 'Resistance1' in df.iloc[-1] else None,
                float(df.iloc[-1]['Resistance2']) if 'Resistance2' in df.iloc[-1] else None,
                float(df.iloc[-1]['BB_upper']) if 'BB_upper' in df.iloc[-1] else None,
                float(df.iloc[-1]['EMA_50']) if 'EMA_50' in df.iloc[-1] and float(df.iloc[-1]['EMA_50']) > entry_price else None,
                float(df.iloc[-1]['Donchian_High']) if 'Donchian_High' in df.iloc[-1] else None
            ]
            
            # None 값 제거 및 진입가보다 높은 저항선만 필터링
            key_levels = [level for level in resistance_levels if level is not None and level > entry_price]
            
            # 진입가에 가장 가까운 저항선 찾기 (오름차순 정렬 후 첫 번째 값)
            if key_levels:
                nearest_resistance = sorted(key_levels)[0]
                
                # 변동성에 따른 버퍼 크기 조정
                buffer_pct = min(2.0, max(1.0, volatility_pct))  # 1-2% 범위의 버퍼
                buffer = entry_price * (buffer_pct / 100)  # 백분율을 절대값으로 변환
                
                # 저항선 위에 버퍼 추가한 스탑로스 설정
                stop_price = nearest_resistance + buffer
                
                # 최대 손실 제한 (총자산의 5% 이내로 제한)
                max_loss_pct = 5.0 / leverage if leverage > 0 else 5.0
                max_allowed_price = entry_price * (1 + max_loss_pct/100)
                
                # 스탑로스는 최대 손실 제한선 이하로 설정
                stop_price = min(stop_price, max_allowed_price)
                
                return round(stop_price, 1)
        
        # 3. 지지/저항선을 찾지 못한 경우 ATR 기반 스탑로스 계산
        atr_multiplier = 2.0  # 기본 ATR 승수
        
        # 레버리지에 따른 ATR 승수 조정
        if leverage > 10:
            atr_multiplier = 1.5  # 높은 레버리지에서는 더 타이트한 스탑
        elif leverage > 5:
            atr_multiplier = 1.8  # 중간 레버리지
            
        if position_type == "Long":
            stop_price = entry_price - (atr * atr_multiplier)
        else:  # Short
            stop_price = entry_price + (atr * atr_multiplier)
            
        # 4. 최대 손실 제한
        max_loss_pct = 5.0 / leverage if leverage > 0 else 5.0
        
        if position_type == "Long":
            min_price = entry_price * (1 - max_loss_pct/100)
            stop_price = max(stop_price, min_price)  # 스탑로스는 최소 허용 가격 이상
        else:  # Short
            max_price = entry_price * (1 + max_loss_pct/100)
            stop_price = min(stop_price, max_price)  # 스탑로스는 최대 허용 가격 이하
            
        return round(stop_price, 1)
            
    except Exception as e:
        print(f"[오류] 동적 스탑로스 계산 중 문제 발생: {e}")
        
        # 오류 발생 시 안전한 기본값 사용
        default_stop_pct = 5.0 / leverage if leverage > 0 else 5.0
        if position_type == "Long":
            return round(entry_price * (1 - default_stop_pct/100), 1)
        else:  # Short
            return round(entry_price * (1 + default_stop_pct/100), 1)

        

def calculate_ai_stop_loss(market_data: Dict, entry_price: float, position_type: str, 
                         leverage: int = 1, position_size_pct: float = None, 
                         wallet_balance: float = None) -> Tuple[float, float]:
    """
    지지/저항선 및 변동성 기반 손절가 계산 함수
    """
    try:
        # 데이터 추출
        df_15m = market_data['timeframes']["15"]
        df_1h = market_data['timeframes'].get("60", df_15m)
        
        latest_15m = df_15m.iloc[-1]
        latest_1h = df_1h.iloc[-1]
        
        # 변동성 계산
        atr_15m = float(latest_15m['ATR'])
        atr_1h = float(latest_1h['ATR'])
        weighted_atr = (atr_15m * 0.6) + (atr_1h * 0.4)
        
        # 변동성 백분율 계산
        volatility_pct = (weighted_atr / entry_price) * 100
        
        # 주요 지지/저항선 식별
        if position_type == "Long":
            # 롱 포지션의 경우 - 주요 지지선 확인 및 정렬
            key_support_levels = [
                float(latest_1h['Support1']) if 'Support1' in latest_1h else None,
                float(latest_1h['Support2']) if 'Support2' in latest_1h else None,
                float(latest_1h['BB_lower']) if 'BB_lower' in latest_1h else None,
                float(latest_1h['Donchian_Low']) if 'Donchian_Low' in latest_1h else None,
                float(latest_1h['EMA_50']) if 'EMA_50' in latest_1h and float(latest_1h['EMA_50']) < entry_price else None
            ]
            
            # None 값 제거 및 진입가보다 낮은 지지선만 필터링 후 내림차순 정렬 (높은 가격부터)
            key_support_levels = sorted([level for level in key_support_levels 
                                        if level is not None and level < entry_price], reverse=True)
            
            if key_support_levels:
                # 진입가에 가장 가까운 지지선 선택 (첫 번째 요소)
                nearest_support = key_support_levels[0]
                
                # 버퍼 크기를 변동성에 따라 조정 (1-2% 또는 ATR의 1-2배)
                buffer_pct = min(2.0, max(1.0, volatility_pct * 0.8))  # 변동성 기반 1-2% 버퍼
                buffer = entry_price * (buffer_pct / 100)  # 백분율을 절대값으로 변환
                
                # 지지선 아래에 버퍼 추가 설정
                stop_price = nearest_support - buffer
                
                # 최대 손실 제한 로직 유지
                max_loss_pct = 5.0 / leverage if leverage > 0 else 5.0
                min_stop_price = entry_price * (1 - max_loss_pct/100)
                
                # 스탑로스 가격은 최소 손실 제한선 이상이어야 함
                stop_price = max(stop_price, min_stop_price)
                
                # 손절 비율 계산
                stop_loss_pct = ((entry_price - stop_price) / entry_price) * 100
                
                print(f"[정보] 지지선 기반 스탑로스: 지지선 ${nearest_support:.2f} - 버퍼 ${buffer:.2f} = ${stop_price:.2f} ({stop_loss_pct:.2f}%)")
                
                return (round(float(stop_price), 1), stop_loss_pct)
            else:
                # 지지선 정보가 없을 경우 기본 ATR 기반 계산
                stop_loss_pct = min(volatility_pct * 2, 5.0 / leverage)
                stop_price = entry_price * (1 - stop_loss_pct/100)
        else:  # Short 포지션
            # 숏 포지션의 경우 - 주요 저항선 확인 및 정렬
            key_resistance_levels = [
                float(latest_1h['Resistance1']) if 'Resistance1' in latest_1h else None,
                float(latest_1h['Resistance2']) if 'Resistance2' in latest_1h else None,
                float(latest_1h['BB_upper']) if 'BB_upper' in latest_1h else None,
                float(latest_1h['Donchian_High']) if 'Donchian_High' in latest_1h else None,
                float(latest_1h['EMA_50']) if 'EMA_50' in latest_1h and float(latest_1h['EMA_50']) > entry_price else None
            ]
            
            # None 값 제거 및 진입가보다 높은 저항선만 필터링 후 오름차순 정렬 (낮은 가격부터)
            key_resistance_levels = sorted([level for level in key_resistance_levels 
                                           if level is not None and level > entry_price])
            
            if key_resistance_levels:
                # 진입가에 가장 가까운 저항선 선택 (첫 번째 요소)
                nearest_resistance = key_resistance_levels[0]
                
                # 버퍼 크기를 변동성에 따라 조정 (1-2% 또는 ATR의 1-2배)
                buffer_pct = min(2.0, max(1.0, volatility_pct * 0.8))  # 변동성 기반 1-2% 버퍼
                buffer = entry_price * (buffer_pct / 100)  # 백분율을 절대값으로 변환
                
                # 저항선 위에 버퍼 추가 설정
                stop_price = nearest_resistance + buffer
                
                # 최대 손실 제한 로직 유지
                max_loss_pct = 5.0 / leverage if leverage > 0 else 5.0
                max_stop_price = entry_price * (1 + max_loss_pct/100)
                
                # 스탑로스 가격은 최대 손실 제한선 이하여야 함
                stop_price = min(stop_price, max_stop_price)
                
                # 손절 비율 계산
                stop_loss_pct = ((stop_price - entry_price) / entry_price) * 100
                
                print(f"[정보] 저항선 기반 스탑로스: 저항선 ${nearest_resistance:.2f} + 버퍼 ${buffer:.2f} = ${stop_price:.2f} ({stop_loss_pct:.2f}%)")
                
                return (round(float(stop_price), 1), stop_loss_pct)
            else:
                # 저항선 정보가 없을 경우 기본 ATR 기반 계산
                stop_loss_pct = min(volatility_pct * 2, 5.0 / leverage)
                stop_price = entry_price * (1 + stop_loss_pct/100)
        
        return (round(float(stop_price), 1), stop_loss_pct)
        
    except Exception as e:
        print(f"[오류] 지지/저항선 기반 손절가 계산 중 문제 발생: {e}")
        print(traceback.format_exc())
        
        # 오류 발생 시 안전한 기본값 (7%) 사용    
        default_stop_pct = 7.0  # 기본 7% 손절폭
        if position_type == "Long":
            default_stop_price = round(float(entry_price) * (1 - default_stop_pct/100), 1)
        else:  # Short
            default_stop_price = round(float(entry_price) * (1 + default_stop_pct/100), 1)
        return (default_stop_price, default_stop_pct)
        

def calculate_support_based_stop_loss(market_data: Dict, entry_price: float, position_type: str, 
                                    leverage: int = 1, fallback_pct: float = 2.0) -> float:
    """
    지지/저항선 기반 스탑로스 가격 계산 함수
    - 포지션 방향에 맞는 가장 가까운 지지/저항선을 찾음
    - 버퍼를 추가하여 스탑로스 가격 계산
    - 적절한 지지/저항선이 없을 경우 기본값(fallback_pct) 사용
    
    Args:
        market_data: 시장 데이터 딕셔너리
        entry_price: 진입 가격
        position_type: 포지션 유형 ("Long" 또는 "Short")
        leverage: 레버리지 배수
        fallback_pct: 지지/저항선을 찾지 못할 경우 기본 손절 비율(%)
        
    Returns:
        float: 계산된 스탑로스 가격
    """
    try:
        # 현재 가격 정보 가져오기
        current_price = float(market_data['timeframes']["15"].iloc[-1]['Close'])
        
        # 1. 주요 지지/저항선 식별
        key_levels = identify_key_levels(market_data, position_type)
        
        # 2. 트렌드선 식별 (추가적인 지지/저항선 참고)
        trendline_level = detect_trendlines(
            market_data['timeframes']["15"], 
            lookback_period=20, 
            position_type=position_type
        )
        
        stop_price = None
        
        # 롱 포지션의 경우 - 진입가 아래의 지지선 찾기
        if position_type == "Long":
            # 유효한 지지선 필터링 (현재 가격 아래, 진입가 아래)
            valid_supports = [
                level for level in key_levels 
                if level['type'] == 'support' and level['price'] < current_price and level['price'] < entry_price
            ]
            
            # 지지선을 찾은 경우, 가장 강한(강도 높은) 지지선 선택
            if valid_supports:
                strongest_support = max(valid_supports, key=lambda x: x['strength'])
                support_price = strongest_support['price']
                support_strength = strongest_support['strength']
                
                # 지지선 강도에 따른 버퍼 조정 (강도가 높을수록 작은 버퍼)
                buffer_pct = max(0.5, 2.0 - support_strength)
                stop_price = support_price * (1 - buffer_pct/100)
                print(f"[지지선 스탑로스] 가격: ${support_price:.2f}, 강도: {support_strength:.2f}, 버퍼: {buffer_pct:.2f}%")
            
            # 트렌드선 활용 (지지선이 없거나 트렌드선이 더 높은 경우)
            if trendline_level and (stop_price is None or trendline_level > stop_price):
                # 트렌드선 아래에 버퍼 추가
                buffer_pct = 1.0  # 기본 1% 버퍼
                trendline_stop = trendline_level * (1 - buffer_pct/100)
                
                if stop_price is None or trendline_stop > stop_price:
                    stop_price = trendline_stop
                    print(f"[트렌드선 스탑로스] 가격: ${trendline_level:.2f}, 버퍼: {buffer_pct:.2f}%")
        
        # 숏 포지션의 경우 - 진입가 위의 저항선 찾기
        else:  # position_type == "Short"
            # 유효한 저항선 필터링 (현재 가격 위, 진입가 위)
            valid_resistances = [
                level for level in key_levels 
                if level['type'] == 'resistance' and level['price'] > current_price and level['price'] > entry_price
            ]
            
            # 저항선을 찾은 경우, 가장 강한(강도 높은) 저항선 선택
            if valid_resistances:
                strongest_resistance = max(valid_resistances, key=lambda x: x['strength'])
                resistance_price = strongest_resistance['price']
                resistance_strength = strongest_resistance['strength']
                
                # 저항선 강도에 따른 버퍼 조정 (강도가 높을수록 작은 버퍼)
                buffer_pct = max(0.5, 2.0 - resistance_strength)
                stop_price = resistance_price * (1 + buffer_pct/100)
                print(f"[저항선 스탑로스] 가격: ${resistance_price:.2f}, 강도: {resistance_strength:.2f}, 버퍼: {buffer_pct:.2f}%")
            
            # 트렌드선 활용 (저항선이 없거나 트렌드선이 더 낮은 경우) 
            if trendline_level and (stop_price is None or trendline_level < stop_price):
                # 트렌드선 위에 버퍼 추가
                buffer_pct = 1.0  # 기본 1% 버퍼
                trendline_stop = trendline_level * (1 + buffer_pct/100)
                
                if stop_price is None or trendline_stop < stop_price:
                    stop_price = trendline_stop
                    print(f"[트렌드선 스탑로스] 가격: ${trendline_level:.2f}, 버퍼: {buffer_pct:.2f}%")
        
        # 3. 적절한 지지/저항선을 찾지 못한 경우 기본값 사용
        if stop_price is None:
            if position_type == "Long":
                stop_price = entry_price * (1 - fallback_pct/100)
            else:
                stop_price = entry_price * (1 + fallback_pct/100)
            print(f"[기본 스탑로스] 가격: ${stop_price:.2f} (진입가에서 {fallback_pct:.2f}% 이동)")
        
        # 4. 레버리지 고려 및 안전 확인
        # 레버리지가 높을수록 더 타이트한 스탑로스
        if leverage > 10:
            # 높은 레버리지에서는 손실 제한 강화
            max_loss_pct = 5.0 / leverage  # 예: 레버리지 20x에서는 0.25% 가격 변동이 최대 5% 손실
            
            if position_type == "Long":
                max_allowed_stop = entry_price * (1 - max_loss_pct/100)
                # 계산된 스탑로스가 허용 범위를 넘으면 조정
                if stop_price < max_allowed_stop:
                    stop_price = max_allowed_stop
                    print(f"[레버리지 안전] 스탑로스 조정: ${stop_price:.2f} (최대 손실 {max_loss_pct:.2f}%)")
            else:
                max_allowed_stop = entry_price * (1 + max_loss_pct/100)
                # 계산된 스탑로스가 허용 범위를 넘으면 조정
                if stop_price > max_allowed_stop:
                    stop_price = max_allowed_stop
                    print(f"[레버리지 안전] 스탑로스 조정: ${stop_price:.2f} (최대 손실 {max_loss_pct:.2f}%)")
        
        return round(stop_price, 1)
        
    except Exception as e:
        print(f"[오류] 지지선 기반 스탑로스 계산 중 문제 발생: {e}")
        print(traceback.format_exc())
        
        # 오류 발생 시 기본 스탑로스 계산
        if position_type == "Long":
            return round(entry_price * (1 - fallback_pct/100), 1)
        else:
            return round(entry_price * (1 + fallback_pct/100), 1)

def adjust_position_dynamically(df: pd.DataFrame, position_info: Dict) -> Optional[Dict]:
    """포지션 동적 관리 (트레일링 스탑 및 부분 청산)"""
    try:
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        for position_type, info in position_info.items():
            if info["size"] > 0:
                entry_price = info["entry_price"]
                leverage = info["leverage"] or 1  # 레버리지 정보 가져오기
                
                # 레버리지를 고려한 수익률 계산
                if position_type == "Long":
                    profit_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
                else:
                    profit_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
                
                # 레버리지에 따른 트레일링 스탑 기준 조정
                activation_threshold = 2.0 / leverage  # 레버리지가 높을수록 더 일찍 활성화
                
                print(f"\n[트레일링 스탑 분석]")
                print(f"포지션 타입: {position_type}")
                print(f"진입가: {entry_price}")
                print(f"현재가: {current_price}")
                print(f"레버리지: {leverage}x")
                print(f"현재 수익률: {profit_pct:.2f}%")
                print(f"활성화 기준: {activation_threshold:.2f}%")
                
                # 트레일링 스탑 활성화
                if profit_pct > activation_threshold:
                    new_stop = calculate_dynamic_stop_loss(
                        current_price,
                        df,
                        position_type,
                        leverage  # 레버리지 파라미터 추가
                    )
                    
                    # 트레일링 스탑 설정
                    response = bybit_client.set_trading_stop(
                        category="linear",
                        symbol=SYMBOL,
                        positionIdx=1 if position_type == "Long" else 2,
                        stopLoss=str(new_stop),
                        slTriggerBy="LastPrice"
                    )
                    
                    if response["retCode"] == 0:
                        print(f"[성공] {position_type} 트레일링 스탑 조정: {new_stop}")
                    else:
                        print(f"[경고] 트레일링 스탑 설정 실패: {response['retMsg']}")
        
        return None
        
    except Exception as e:
        print(f"[오류] 포지션 동적 관리 중 문제 발생: {e}")
        return None

def fetch_multi_timeframe_data(timeframes: List[str] = ["5", "60", "240"]) -> Dict[str, pd.DataFrame]:
    """
    여러 시간대의 캔들스틱 데이터를 조회하고 기술적 지표를 추가
    
    Args:
        timeframes: 조회할 시간대 리스트 (기본값: ["5", "15", "30"])
        
    Returns:
        Dict[str, pd.DataFrame]: 각 시간대별 데이터프레임
    """
    try:
        data = {}
        # 각 시간대별 적절한 데이터 포인트 수 설정
        limits = {
            "15": 70,    # 15분봉 70개 (약 17.5시간)
            "60": 50,    # 1시간봉 50개 (약 50시간)
            "240": 30    # 4시간봉 30개 (약 120시간)
        }
        
        for tf in timeframes:
            response = bybit_client.get_kline(
                category="linear",
                symbol=SYMBOL,
                interval=tf,
                limit=limits.get(tf, 50)
            )
            
            if response["retCode"] != 0:
                print(f"{tf}분봉 데이터 조회 실패: {response['retMsg']}")
                continue
                
            df = pd.DataFrame(
                response['result']['list'],
                columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Value']
            )
            
# 'Time' 컬럼의 첫번째 값을 출력하여 단위를 확인합니다.
            print("Raw Time value:", df['Time'].iloc[0])
            df['Time'] = pd.to_datetime(df['Time'].astype(np.int64), unit='ms')
            df = df.astype({
                'Open': float,
                'High': float,
                'Low': float,
                'Close': float,
                'Volume': float
            })
            
            df.set_index('Time', inplace=True)
            df = add_technical_indicators(df)
            data[tf] = df
            
        return data
        
    except Exception as e:
        print(f"멀티 타임프레임 데이터 조회 중 오류 발생: {e}")
        return {}

# 9. 시장 데이터 수집 및 분석 함수들
def get_fear_greed_index() -> Optional[Dict]:
    """공포&탐욕 지수 조회"""
    try:
        response = requests.get("https://api.alternative.me/fng/")
        if response.status_code == 200:
            return response.json()['data'][0]
        return None
    except Exception as e:
        print(f"공포&탐욕 지수 조회 중 오류 발생: {e}")
        return None

# def get_crypto_news() -> List[Dict]:
#     """암호화폐 뉴스 조회 (상위 3건)"""
#     try:
#         response = requests.get(
#             "https://serpapi.com/search.json",
#             params={"engine": "google_news", "q": "bitcoin", "api_key": os.getenv("SERPAPI_API_KEY")}
#         )
#         if response.status_code == 200:
#             return response.json().get("news_results", [])[:3]
#         return []
#     except Exception as e:
#         print(f"뉴스 조회 중 오류 발생: {e}")
#         return []

def get_market_data() -> Optional[Dict]:
    """전체 시장 데이터 수집 및 분석"""
    try:
        multi_tf_data = fetch_multi_timeframe_data(["15", H1_INTERVAL, H4_INTERVAL])
        if not multi_tf_data or "15" not in multi_tf_data:
            raise Exception("멀티 타임프레임 데이터 조회 실패")
            
        for tf in multi_tf_data:
            multi_tf_data[tf] = add_technical_indicators(multi_tf_data[tf])
            
        return {
            "candlestick": multi_tf_data["15"],  # 주 트레이딩 타임프레임
            "timeframes": multi_tf_data,
            "fear_greed": get_fear_greed_index(),
            # "news": get_crypto_news()
        }
    except Exception as e:
        print(f"시장 데이터 수집 중 오류 발생: {e}")
        return None

def identify_market_structure(market_data: Dict) -> Dict:
    """
    시장 구조를 분석하여 현재 시장 유형과 방향성을 식별하는 함수
    
    Args:
        market_data: 시장 데이터 딕셔너리
        
    Returns:
        Dict: {
            'type': 'TREND'|'RANGE'|'VOLATILE'|'CONTRACTION', 
            'direction': 'UP'|'DOWN'|'NEUTRAL',
            'strength': float,
            'confidence': float
        }
    """
    try:
        # 다중 타임프레임 데이터 추출
        df_15m = market_data['timeframes']["15"]
        df_1h = market_data['timeframes'].get("60", df_15m)  # 1시간봉 (없으면 15분봉 사용)
        df_4h = market_data['timeframes'].get("240", df_15m)  # 4시간봉 (없으면 15분봉 사용)
        
        latest_15m = df_15m.iloc[-1]
        latest_1h = df_1h.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        
        # 1. 추세 강도 평가 (ADX 사용)
        adx_15m = float(latest_15m['ADX'])
        adx_1h = float(latest_1h['ADX'])
        adx_4h = float(latest_4h['ADX'])
        
        # 가중 평균 ADX (1시간봉에 가중치 높게)
        weighted_adx = (adx_15m * 0.2) + (adx_1h * 0.5) + (adx_4h * 0.3)
        
        # 2. 방향성 평가
        # MACD 신호
        macd_15m_bull = float(latest_15m['MACD']) > float(latest_15m['MACD_signal'])
        macd_1h_bull = float(latest_1h['MACD']) > float(latest_1h['MACD_signal'])
        macd_4h_bull = float(latest_4h['MACD']) > float(latest_4h['MACD_signal'])
        
        # EMA 배열
        ema_15m_bull = float(latest_15m['Close']) > float(latest_15m['EMA_20'])
        ema_1h_bull = float(latest_1h['Close']) > float(latest_1h['EMA_50'])
        ema_4h_bull = float(latest_4h['Close']) > float(latest_4h['EMA_100'])
        
        # 불리시 신호 개수
        bull_signals = sum([macd_15m_bull, macd_1h_bull, macd_4h_bull, ema_15m_bull, ema_1h_bull, ema_4h_bull])
        # 베어리시 신호 개수
        bear_signals = 6 - bull_signals
        
        # 3. 변동성 평가
        atr_15m = float(latest_15m['ATR'])
        atr_1h = float(latest_1h['ATR'])
        
        # 가격 대비 ATR 비율 (%)
        volatility_pct = (atr_15m / float(latest_15m['Close'])) * 100
        
        # 볼린저 밴드 폭 계산
        bb_width_15m = (float(latest_15m['BB_upper']) - float(latest_15m['BB_lower'])) / float(latest_15m['BB_mavg'])
        bb_width_1h = (float(latest_1h['BB_upper']) - float(latest_1h['BB_lower'])) / float(latest_1h['BB_mavg'])
        
        # 가중 평균 볼린저 밴드 폭
        weighted_bb_width = (bb_width_15m * 0.4) + (bb_width_1h * 0.6)
        
        # 4. 결과 판정
        # 기본값 초기화
        market_type = "RANGE"
        direction = "NEUTRAL"
        strength = 0.5
        confidence = 0.5
        
        # 추세 판정 (ADX > 25)
        if weighted_adx > 25:
            market_type = "TREND"
            strength = min(1.0, weighted_adx / 40)  # 40 이상이면 최대 강도 1.0
            
            if bull_signals >= 4:  # 6개 중 4개 이상 불리시
                direction = "UP"
                confidence = bull_signals / 6
            elif bear_signals >= 4:  # 6개 중 4개 이상 베어리시
                direction = "DOWN"
                confidence = bear_signals / 6
            else:
                # 방향성 불명확
                market_type = "VOLATILE"
                direction = "NEUTRAL"
                confidence = 0.5
                
        # 변동성 판정
        elif volatility_pct > 1.5:  # 1.5% 이상 변동성
            market_type = "VOLATILE"
            strength = min(1.0, volatility_pct / 3.0)  # 3% 이상이면 최대 강도 1.0
            
            if bull_signals >= 4:
                direction = "UP"
                confidence = bull_signals / 6
            elif bear_signals >= 4:
                direction = "DOWN"
                confidence = bear_signals / 6
            else:
                direction = "NEUTRAL"
                confidence = 0.5
                
        # 변동성 수축 판정
        elif weighted_bb_width < 0.03:  # 볼린저 밴드 폭이 좁을 때
            market_type = "CONTRACTION"
            strength = max(0.5, 1.0 - (weighted_bb_width / 0.06))  # 밴드 폭이 좁을수록 강도 높음
            direction = "NEUTRAL"
            confidence = 0.7  # 볼린저 밴드 수축은 비교적 명확한 패턴
                
        # 횡보장 판정 (기본값)
        else:
            market_type = "RANGE"
            strength = max(0.5, 1.0 - (weighted_adx / 25))  # ADX가 낮을수록 횡보 강도 높음
            
            if bb_width_15m > 0.04:  # 어느 정도 밴드 폭이 있어야 의미 있는 횡보
                confidence = 0.6
            else:
                confidence = 0.4
                
            # 횡보장에서도 약한 방향성은 있을 수 있음
            if bull_signals > bear_signals:
                direction = "UP"
            elif bear_signals > bull_signals:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
        
        # 키 레벨 데이터 생성
        key_levels_data = identify_key_levels(market_data, "Long" if direction == "UP" else "Short")
        
        # 모든 분석 데이터를 하나의 JSON 객체로 통합
        market_analysis_data = {
            'market_type': market_type,
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'adx': weighted_adx,
            'volatility': volatility_pct,
            'bb_width': weighted_bb_width,
            'bull_signals': bull_signals,
            'bear_signals': bear_signals,
            'key_levels': key_levels_data,
            'rsi': {
                '15m': float(latest_15m.get('RSI', 50)),
                '1h': float(latest_1h.get('RSI', 50)),
                '4h': float(latest_4h.get('RSI', 50))
            },
            'analysis_summary': f"{market_type} 시장, 방향성: {direction}, 강도: {strength:.2f}, 신뢰도: {confidence:.2f}"
        }
        
        # 현재 테이블 구조에 맞게 데이터 저장
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # update_db_schema 함수에서 사용된 스키마에 맞춰 데이터 저장
            cursor.execute('''
                INSERT INTO market_analysis_records
                (analysis_date, data)
                VALUES (?, ?)
            ''', (
                datetime.now().isoformat(),
                json.dumps(market_analysis_data)
            ))
            conn.commit()
        
        # 결과 반환
        return {
            'type': market_type,
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'adx': weighted_adx,
            'volatility': volatility_pct,
            'bb_width': weighted_bb_width,
            'bull_signals': bull_signals,
            'bear_signals': bear_signals
        }
        
    except Exception as e:
        print(f"[오류] 시장 구조 분석 중 문제 발생: {e}")
        print(traceback.format_exc())
        # 오류 발생 시 기본값 반환
        return {
            'type': 'RANGE',
            'direction': 'NEUTRAL',
            'strength': 0.5,
            'confidence': 0.3,
            'error': str(e)
        }


def identify_key_levels(market_data: Dict, position_type: str) -> List[Dict]:
    """
    주요 지지/저항선 분석 함수 (개선된 버전)
    여러 기술적 지표와 가격 행동을 종합하여 주요 가격 레벨과 그 강도를 분석
    
    Args:
        market_data: 시장 데이터 딕셔너리
        position_type: 포지션 유형 ("Long" 또는 "Short")
        
    Returns:
        List[Dict]: [{'price': float, 'strength': float, 'type': str, 'source': str, 'timeframe': str}, ...]
        - price: 주요 레벨 가격
        - strength: 강도 (0~1)
        - type: 'support'(지지) 또는 'resistance'(저항)
        - source: 레벨의 출처 (예: 'BB', 'EMA', 'Pivot', 'Historical', 'Fibonacci'...)
        - timeframe: 레벨이 발견된 타임프레임 (예: '15m', '1h', '4h')
    """
    try:
        # 다중 타임프레임 데이터 추출
        df_15m = market_data['timeframes']["15"]
        df_1h = market_data['timeframes'].get("60", df_15m)
        df_4h = market_data['timeframes'].get("240", df_15m)
        
        # 타임프레임별 가중치 설정 (중요도 기준)
        tf_weights = {
            "15m": 0.3,  # 단기 (30%)
            "1h": 0.5,   # 중기 (50%)
            "4h": 0.7    # 장기 (70%)
        }
        
        current_price = float(df_15m.iloc[-1]['Close'])
        key_levels = []
        
        # 롱 포지션은 위쪽 저항선, 숏 포지션은 아래쪽 지지선을 주로 분석
        is_long = position_type == "Long"
        
        print("\n=== 주요 저항/지지선 분석 시작 ===")
        print(f"현재 가격: ${current_price:.2f}")
        print(f"분석 방향: {'저항선 (위쪽)' if is_long else '지지선 (아래쪽)'}")
        
        # 최근 가격 변동폭 계산 (ATR 기반)
        atr_15m = float(df_15m.iloc[-1]['ATR'])
        atr_1h = float(df_1h.iloc[-1]['ATR'])
        atr_4h = float(df_4h.iloc[-1]['ATR'])
        
        # 가중 평균 ATR
        weighted_atr = (atr_15m * 0.3 + atr_1h * 0.5 + atr_4h * 0.2)
        
        # 1. 각 타임프레임별 주요 레벨 식별
        for tf, df, weight in [("15m", df_15m, tf_weights["15m"]), 
                              ("1h", df_1h, tf_weights["1h"]), 
                              ("4h", df_4h, tf_weights["4h"])]:
            latest = df.iloc[-1]
            
            # 1.1 볼린저 밴드 레벨
            if is_long:
                # 롱 포지션은 상단 밴드가 저항선
                if float(latest['BB_upper']) > current_price:
                    bb_level = {
                        'price': float(latest['BB_upper']),
                        'strength': 0.6 * weight,
                        'type': 'resistance',
                        'source': 'BB',
                        'timeframe': tf
                    }
                    key_levels.append(bb_level)
            else:
                # 숏 포지션은 하단 밴드가 지지선
                if float(latest['BB_lower']) < current_price:
                    bb_level = {
                        'price': float(latest['BB_lower']),
                        'strength': 0.6 * weight,
                        'type': 'support',
                        'source': 'BB',
                        'timeframe': tf
                    }
                    key_levels.append(bb_level)
            
            # 1.2 이동평균선 레벨
            ema_levels = []
            
            # 이동평균선 목록과 각각의 기본 강도
            ema_params = [
                ('EMA_20', 0.55),
                ('EMA_50', 0.65),
                ('EMA_100', 0.75),
                ('EMA_200', 0.85)
            ]
            
            for ema_name, base_strength in ema_params:
                if ema_name in latest:
                    ema_value = float(latest[ema_name])
                    
                    # 롱 포지션은 현재가 위의 EMA가 저항선
                    if is_long and ema_value > current_price:
                        ema_levels.append({
                            'price': ema_value,
                            'strength': base_strength * weight,
                            'type': 'resistance',
                            'source': ema_name,
                            'timeframe': tf
                        })
                    # 숏 포지션은 현재가 아래의 EMA가 지지선
                    elif not is_long and ema_value < current_price:
                        ema_levels.append({
                            'price': ema_value,
                            'strength': base_strength * weight,
                            'type': 'support',
                            'source': ema_name,
                            'timeframe': tf
                        })
            
            # EMA 레벨 추가
            key_levels.extend(ema_levels)
            
            # 1.3 피보나치 레벨
            fib_levels = []
            
            # 롱: 주요 저항 피보나치 레벨
            if is_long:
                fib_params = [
                    ('Fibo_0.382', 0.6),
                    ('Fibo_0.5', 0.65),
                    ('Fibo_0.618', 0.75),
                    ('Fibo_0.786', 0.8),
                    ('Fibo_1', 0.85)
                ]
            else:
                # 숏: 주요 지지 피보나치 레벨
                fib_params = [
                    ('Fibo_0', 0.85),
                    ('Fibo_0.236', 0.65),
                    ('Fibo_0.382', 0.7),
                    ('Fibo_0.5', 0.65),
                    ('Fibo_0.618', 0.6)
                ]
            
            for fib_name, base_strength in fib_params:
                if fib_name in latest:
                    fib_price = float(latest[fib_name])
                    
                    if is_long and fib_price > current_price:
                        fib_levels.append({
                            'price': fib_price,
                            'strength': base_strength * weight,
                            'type': 'resistance',
                            'source': 'Fibonacci',
                            'timeframe': tf
                        })
                    elif not is_long and fib_price < current_price:
                        fib_levels.append({
                            'price': fib_price,
                            'strength': base_strength * weight,
                            'type': 'support',
                            'source': 'Fibonacci',
                            'timeframe': tf
                        })
            
            # 피보나치 레벨 추가
            key_levels.extend(fib_levels)
            
            # 1.4 도치안 채널 레벨
            donchian_levels = []
            if 'Donchian_High' in latest and is_long:
                donchian_high = float(latest['Donchian_High'])
                if donchian_high > current_price:
                    donchian_levels.append({
                        'price': donchian_high,
                        'strength': 0.85 * weight,  # 강한 저항선
                        'type': 'resistance',
                        'source': 'Donchian',
                        'timeframe': tf
                    })
            
            if 'Donchian_Low' in latest and not is_long:
                donchian_low = float(latest['Donchian_Low'])
                if donchian_low < current_price:
                    donchian_levels.append({
                        'price': donchian_low,
                        'strength': 0.85 * weight,  # 강한 지지선
                        'type': 'support',
                        'source': 'Donchian',
                        'timeframe': tf
                    })
            
            # 도치안 중간선 (약한 지지/저항)
            if 'Donchian_Mid' in latest:
                donchian_mid = float(latest['Donchian_Mid'])
                if is_long and donchian_mid > current_price:
                    donchian_levels.append({
                        'price': donchian_mid,
                        'strength': 0.5 * weight,
                        'type': 'resistance',
                        'source': 'Donchian_Mid',
                        'timeframe': tf
                    })
                elif not is_long and donchian_mid < current_price:
                    donchian_levels.append({
                        'price': donchian_mid,
                        'strength': 0.5 * weight,
                        'type': 'support',
                        'source': 'Donchian_Mid',
                        'timeframe': tf
                    })
            
            # 도치안 채널 레벨 추가
            key_levels.extend(donchian_levels)
            
            # 1.5 피벗 포인트 레벨
            pivot_levels = []
            
            # 피벗 포인트 관련 필드
            pivot_fields = [
                ('Pivot', 0.7),
                ('Resistance1', 0.8),
                ('Resistance2', 0.75),
                ('Support1', 0.8),
                ('Support2', 0.75)
            ]
            
            for field, base_strength in pivot_fields:
                if field in latest:
                    pivot_value = float(latest[field])
                    
                    # 필드명에 따라 타입 결정
                    level_type = 'resistance' if 'Resistance' in field or field == 'Pivot' else 'support'
                    
                    # 롱 포지션은 위쪽 저항선, 숏 포지션은 아래쪽 지지선 필터링
                    if (is_long and level_type == 'resistance' and pivot_value > current_price) or \
                       (not is_long and level_type == 'support' and pivot_value < current_price):
                        pivot_levels.append({
                            'price': pivot_value,
                            'strength': base_strength * weight,
                            'type': level_type,
                            'source': 'Pivot',
                            'timeframe': tf
                        })
            
            # 피벗 레벨 추가
            key_levels.extend(pivot_levels)
            
            # 1.6 역사적 고점/저점 분석 (새로 추가)
            # 최근 200개 캔들에서 주요 고점/저점 찾기
            lookback = min(200, len(df))
            historical_highs = []
            historical_lows = []
            
            if is_long:
                # 롱 포지션은 과거 고점을 찾아 저항선으로 사용
                for i in range(5, lookback-5):
                    if df['High'].iloc[i] > df['High'].iloc[i-1] and \
                       df['High'].iloc[i] > df['High'].iloc[i-2] and \
                       df['High'].iloc[i] > df['High'].iloc[i+1] and \
                       df['High'].iloc[i] > df['High'].iloc[i+2]:
                        # 로컬 고점 발견
                        high_price = float(df['High'].iloc[i])
                        if high_price > current_price:
                            historical_highs.append(high_price)
            else:
                # 숏 포지션은 과거 저점을 찾아 지지선으로 사용
                for i in range(5, lookback-5):
                    if df['Low'].iloc[i] < df['Low'].iloc[i-1] and \
                       df['Low'].iloc[i] < df['Low'].iloc[i-2] and \
                       df['Low'].iloc[i] < df['Low'].iloc[i+1] and \
                       df['Low'].iloc[i] < df['Low'].iloc[i+2]:
                        # 로컬 저점 발견
                        low_price = float(df['Low'].iloc[i])
                        if low_price < current_price:
                            historical_lows.append(low_price)
            
            # 역사적 고점/저점 군집화 (비슷한 가격대 통합)
            if is_long and historical_highs:
                # 고점 정렬
                historical_highs.sort()
                
                # 군집화 (1% 이내 차이의 고점 그룹화)
                clusters = []
                current_cluster = [historical_highs[0]]
                
                for i in range(1, len(historical_highs)):
                    if (historical_highs[i] - current_cluster[-1]) / current_cluster[-1] < 0.01:
                        # 같은 군집에 추가
                        current_cluster.append(historical_highs[i])
                    else:
                        # 새 군집 시작
                        if current_cluster:
                            clusters.append(current_cluster)
                        current_cluster = [historical_highs[i]]
                
                if current_cluster:
                    clusters.append(current_cluster)
                
                # 각 군집의 평균 가격과 빈도(강도) 계산
                for cluster in clusters:
                    avg_price = sum(cluster) / len(cluster)
                    # 빈도가 높을수록 강한 저항선
                    strength_boost = min(0.3, 0.05 * len(cluster))  # 최대 0.3 보강
                    
                    key_levels.append({
                        'price': avg_price,
                        'strength': (0.7 + strength_boost) * weight,
                        'type': 'resistance',
                        'source': 'Historical_High',
                        'timeframe': tf
                    })
            
            elif not is_long and historical_lows:
                # 저점 정렬
                historical_lows.sort(reverse=True)
                
                # 군집화 (1% 이내 차이의 저점 그룹화)
                clusters = []
                current_cluster = [historical_lows[0]]
                
                for i in range(1, len(historical_lows)):
                    if (current_cluster[-1] - historical_lows[i]) / historical_lows[i] < 0.01:
                        # 같은 군집에 추가
                        current_cluster.append(historical_lows[i])
                    else:
                        # 새 군집 시작
                        if current_cluster:
                            clusters.append(current_cluster)
                        current_cluster = [historical_lows[i]]
                
                if current_cluster:
                    clusters.append(current_cluster)
                
                # 각 군집의 평균 가격과 빈도(강도) 계산
                for cluster in clusters:
                    avg_price = sum(cluster) / len(cluster)
                    # 빈도가 높을수록 강한 지지선
                    strength_boost = min(0.3, 0.05 * len(cluster))  # 최대 0.3 보강
                    
                    key_levels.append({
                        'price': avg_price,
                        'strength': (0.7 + strength_boost) * weight,
                        'type': 'support',
                        'source': 'Historical_Low',
                        'timeframe': tf
                    })
            
            # 1.7 거래량 프로파일 분석 (추가) - 거래량이 집중된 가격대를 주요 레벨로 식별
            if 'Volume' in df.columns:
                # 가격 범위를 10개 구간으로 나누기
                price_min = df['Low'].min()
                price_max = df['High'].max()
                bins = 10
                bin_size = (price_max - price_min) / bins
                
                volume_profile = {}
                
                # 가격대별 거래량 누적
                for i in range(lookback):
                    price = (df['High'].iloc[i] + df['Low'].iloc[i]) / 2  # 중간 가격
                    volume = df['Volume'].iloc[i]
                    
                    bin_idx = min(bins-1, int((price - price_min) / bin_size))
                    bin_price = price_min + bin_idx * bin_size + bin_size/2  # 구간 중간 가격
                    
                    if bin_price not in volume_profile:
                        volume_profile[bin_price] = 0
                    volume_profile[bin_price] += volume
                
                # 거래량이 많은 상위 3개 가격대 선택
                sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for price, volume in sorted_profile:
                    # 전체 거래량 대비 비율 계산
                    relative_volume = volume / df['Volume'].sum()
                    
                    # 롱 포지션은 현재가 위의 가격대를, 숏 포지션은 현재가 아래의 가격대를 선택
                    if (is_long and price > current_price) or (not is_long and price < current_price):
                        # 거래량 비중에 따른 강도 계산 (최소 0.6, 최대 0.9)
                        vol_strength = 0.6 + min(0.3, relative_volume * 10)
                        
                        key_levels.append({
                            'price': price,
                            'strength': vol_strength * weight,
                            'type': 'resistance' if is_long else 'support',
                            'source': 'Volume_Profile',
                            'timeframe': tf
                        })
        
        # 2. 주요 차트 패턴 식별 (새로 추가) - 헤드앤숄더, 이중 바닥/탑 등
        # 2.1 이중 바닥/탑 패턴 탐지
        if len(df_1h) >= 30:  # 충분한 데이터 필요
            if is_long:
                # 이중 탑 패턴 (Double Top) - 롱 포지션의 저항선
                for i in range(10, len(df_1h) - 5):
                    # 첫 번째 고점
                    if df_1h['High'].iloc[i] > df_1h['High'].iloc[i-1] and \
                       df_1h['High'].iloc[i] > df_1h['High'].iloc[i+1]:
                        peak1 = float(df_1h['High'].iloc[i])
                        peak1_idx = i
                        
                        # 두 번째 고점 검색 (5-15 캔들 간격)
                        for j in range(i + 5, min(i + 15, len(df_1h) - 1)):
                            if df_1h['High'].iloc[j] > df_1h['High'].iloc[j-1] and \
                               df_1h['High'].iloc[j] > df_1h['High'].iloc[j+1]:
                                peak2 = float(df_1h['High'].iloc[j])
                                
                                # 두 고점이 1% 이내로 비슷하면 이중 탑으로 간주
                                if abs(peak2 - peak1) / peak1 < 0.01 and peak1 > current_price:
                                    # 이중 탑 저항선 저장
                                    double_top_level = {
                                        'price': max(peak1, peak2),
                                        'strength': 0.9 * tf_weights["1h"],  # 강한 저항
                                        'type': 'resistance',
                                        'source': 'Double_Top',
                                        'timeframe': '1h'
                                    }
                                    key_levels.append(double_top_level)
                                    break
            else:
                # 이중 바닥 패턴 (Double Bottom) - 숏 포지션의 지지선
                for i in range(10, len(df_1h) - 5):
                    # 첫 번째 저점
                    if df_1h['Low'].iloc[i] < df_1h['Low'].iloc[i-1] and \
                       df_1h['Low'].iloc[i] < df_1h['Low'].iloc[i+1]:
                        bottom1 = float(df_1h['Low'].iloc[i])
                        bottom1_idx = i
                        
                        # 두 번째 저점 검색 (5-15 캔들 간격)
                        for j in range(i + 5, min(i + 15, len(df_1h) - 1)):
                            if df_1h['Low'].iloc[j] < df_1h['Low'].iloc[j-1] and \
                               df_1h['Low'].iloc[j] < df_1h['Low'].iloc[j+1]:
                                bottom2 = float(df_1h['Low'].iloc[j])
                                
                                # 두 저점이 1% 이내로 비슷하면 이중 바닥으로 간주
                                if abs(bottom2 - bottom1) / bottom1 < 0.01 and bottom1 < current_price:
                                    # 이중 바닥 지지선 저장
                                    double_bottom_level = {
                                        'price': min(bottom1, bottom2),
                                        'strength': 0.9 * tf_weights["1h"],  # 강한 지지
                                        'type': 'support',
                                        'source': 'Double_Bottom',
                                        'timeframe': '1h'
                                    }
                                    key_levels.append(double_bottom_level)
                                    break
        
        # 3. 레벨 정렬 및 중복 제거
        # 롱 포지션: 가격 오름차순 정렬(낮은 가격부터)
        # 숏 포지션: 가격 내림차순 정렬(높은 가격부터)
        key_levels = sorted(key_levels, key=lambda x: x['price'], reverse=not is_long)
        
        # 비슷한 가격대 레벨 통합 (0.5% 이내 차이)
        merged_levels = []
        i = 0
        while i < len(key_levels):
            current_level = key_levels[i]
            similar_levels = [current_level]
            
            j = i + 1
            while j < len(key_levels) and abs(key_levels[j]['price'] - current_level['price']) / current_level['price'] < 0.005:
                similar_levels.append(key_levels[j])
                j += 1
            
            # 여러 레벨이 비슷한 가격대에 있으면 강도 증가
            if len(similar_levels) > 1:
                avg_price = sum(level['price'] for level in similar_levels) / len(similar_levels)
                
                # 가중 평균으로 강도 계산 (중요도 높은 소스 우선)
                weighted_strength = 0
                total_weight = 0
                source_priority = {
                    'Double_Top': 2.0, 'Double_Bottom': 2.0,  # 차트 패턴 (최우선)
                    'Historical_High': 1.5, 'Historical_Low': 1.5,  # 과거 고점/저점 (높은 우선순위)
                    'Donchian': 1.3, 'Pivot': 1.3,  # 채널/피벗 (중간 우선순위)
                    'EMA_200': 1.3, 'Volume_Profile': 1.2,  # 주요 이평선/거래량 (중간 우선순위)
                    'Fibonacci': 1.1  # 피보나치 (표준 우선순위)
                }
                
                for level in similar_levels:
                    source = level['source']
                    priority = source_priority.get(source, 1.0)
                    weighted_strength += level['strength'] * priority
                    total_weight += priority
                
                # 강도 계산 및 제한 (최대 1.0)
                base_strength = weighted_strength / total_weight if total_weight > 0 else 0
                combined_strength = min(1.0, base_strength + 0.1 * (len(similar_levels) - 1))
                
                # 다중 타임프레임 확인 - 여러 타임프레임에서 발견되면 강도 추가 보너스
                timeframes = set(level['timeframe'] for level in similar_levels)
                if len(timeframes) > 1:
                    combined_strength = min(1.0, combined_strength + 0.1 * len(timeframes))
                
                # 소스 목록 생성
                sources = [level['source'] for level in similar_levels]
                source_count = {}
                for src in sources:
                    if src not in source_count:
                        source_count[src] = 0
                    source_count[src] += 1
                
                # 가장 많이 발견된 소스를 주요 소스로 설정
                primary_source = max(source_count.items(), key=lambda x: x[1])[0]
                
                merged_levels.append({
                    'price': avg_price,
                    'strength': combined_strength,
                    'type': current_level['type'],
                    'source': primary_source,
                    'timeframe': ','.join(timeframes)
                })
            else:
                merged_levels.append(current_level)
            
            i = j
        
        # 4. ATR 기반 최소 간격 필터링
        # 너무 가까운 레벨들은 통합 (ATR의 0.5배 이내)
        atr_min_gap = weighted_atr * 0.5
        filtered_levels = []
        
        for i, level in enumerate(merged_levels):
            if i == 0:
                filtered_levels.append(level)
            else:
                last_level = filtered_levels[-1]
                if abs(level['price'] - last_level['price']) > atr_min_gap:
                    filtered_levels.append(level)
                else:
                    # 더 강한 레벨을 유지
                    if level['strength'] > last_level['strength']:
                        filtered_levels[-1] = level
        
        # 5. 강도에 따른 최종 레벨 선택 (최대 5-7개)
        # 강도순으로 정렬 후 상위 레벨만 선택
        strongest_levels = sorted(filtered_levels, key=lambda x: x['strength'], reverse=True)
        
        # 최대 7개 레벨 선택 (단, 강도 0.5 이상)
        strongest_levels = [level for level in strongest_levels if level['strength'] >= 0.5][:7]
        
        # 다시 가격 순으로 정렬
        final_levels = sorted(strongest_levels, key=lambda x: x['price'], reverse=not is_long)
        
        # 6. 레벨 정보 출력
        print(f"\n탐지된 주요 {'저항선' if is_long else '지지선'}: {len(final_levels)}개")
        
        for i, level in enumerate(final_levels):
            print(f"{i+1}. 가격: ${level['price']:.2f}, 강도: {level['strength']:.2f}, "
                  f"출처: {level['source']}, 타임프레임: {level['timeframe']}")
        
        print("=== 주요 저항/지지선 분석 완료 ===\n")
        
        return final_levels
        
    except Exception as e:
        print(f"[오류] 주요 가격 레벨 분석 중 문제 발생: {e}")
        import traceback
        print(traceback.format_exc())
        # 기본값 반환
        return []

def analyze_market_data(market_data: Dict) -> None:
    """시장 데이터 분석 및 콘솔 출력"""
    try:
        latest_data = market_data['candlestick'].iloc[-1]
        print("\n=== 시장 데이터 분석 ===")
        print(f"현재 가격: ${float(latest_data['Close']):,.2f}")
        print("\n[모멘텀 지표]")
        print(f"RSI (14): {latest_data['RSI']:.2f}")
        print(f"MACD: {latest_data['MACD']:.2f}")
        print(f"MACD Signal: {latest_data['MACD_signal']:.2f}")
        print(f"MACD Diff: {latest_data['MACD_diff']:.2f}")
        print("\n[이동평균선]")
        print(f"SMA 20: ${float(latest_data['SMA_20']):,.2f}")
        print(f"EMA 10: ${float(latest_data['EMA_10']):,.2f}")
        print(f"EMA 20: ${float(latest_data['EMA_20']):,.2f}")
        print(f"EMA 50: ${float(latest_data['EMA_50']):,.2f}")
        print(f"EMA 100: ${float(latest_data['EMA_100']):,.2f}")
        print(f"EMA 200: ${float(latest_data['EMA_200']):,.2f}")
        print("\n[볼린저 밴드]")
        print(f"상단: ${float(latest_data['BB_upper']):,.2f}")
        print(f"중앙: ${float(latest_data['BB_mavg']):,.2f}")
        print(f"하단: ${float(latest_data['BB_lower']):,.2f}")
        print("\n[윌리엄스 엘리게이터]")
        print(f"Jaws: ${float(latest_data['Alligator_Jaws']):,.2f}")
        print(f"Teeth: ${float(latest_data['Alligator_Teeth']):,.2f}")
        print(f"Lips: ${float(latest_data['Alligator_Lips']):,.2f}")
        print("\n[돈치안 채널 (Donchian Channel)]")
        print("\n[돈치안 채널 (Donchian Channel)]")
        donchian_upper = max(market_data['candlestick']['High'].iloc[-50:])  # 최근 50봉 최고가
        donchian_lower = min(market_data['candlestick']['Low'].iloc[-50:])   # 최근 50봉 최저가
        donchian_middle = (donchian_upper + donchian_lower) / 2  # 중앙선 계산
        print(f"Upper Band (최고가): ${donchian_upper:,.2f}")
        print(f"Lower Band (최저가): ${donchian_lower:,.2f}")
        print(f"Middle Line (중앙선): ${donchian_middle:,.2f}")
        print("\n[피보나치 레벨]")
        print(f"0.236: ${float(latest_data['Fibo_0.236']):,.2f}")
        print(f"0.382: ${float(latest_data['Fibo_0.382']):,.2f}")
        print(f"0.500: ${float(latest_data['Fibo_0.5']):,.2f}")
        print(f"0.618: ${float(latest_data['Fibo_0.618']):,.2f}")
        print(f"0.786: ${float(latest_data['Fibo_0.786']):,.2f}")
        print("\n[기타 지표]")
        print(f"ATR: {latest_data['ATR']:.2f}")
        print(f"Fractal Up: {latest_data['Fractal_Up']}")
        print(f"Fractal Down: {latest_data['Fractal_Down']}")
        print("\n[시장 심리]")
        if market_data['fear_greed']:
            print(f"공포&탐욕 지수: {market_data['fear_greed']['value']} ({market_data['fear_greed']['value_classification']})")
        print("\n=====================")
    except Exception as e:
        print(f"시장 데이터 분석 중 오류 발생: {e}")

def identify_market_structure(market_data: Dict) -> Dict:
    """
    시장 구조를 분석하여 현재 시장 유형과 방향성을 식별하는 함수
    
    Args:
        market_data: 시장 데이터 딕셔너리
        
    Returns:
        Dict: {
            'type': 'TREND'|'RANGE'|'VOLATILE'|'CONTRACTION', 
            'direction': 'UP'|'DOWN'|'NEUTRAL',
            'strength': float,
            'confidence': float
        }
    """
    try:
        # 다중 타임프레임 데이터 추출
        df_15m = market_data['timeframes']["15"]
        df_1h = market_data['timeframes'].get("60", df_15m)  # 1시간봉 (없으면 15분봉 사용)
        df_4h = market_data['timeframes'].get("240", df_15m)  # 4시간봉 (없으면 15분봉 사용)
        
        latest_15m = df_15m.iloc[-1]
        latest_1h = df_1h.iloc[-1]
        latest_4h = df_4h.iloc[-1]
        
        # 1. 추세 강도 평가 (ADX 사용)
        adx_15m = float(latest_15m['ADX'])
        adx_1h = float(latest_1h['ADX'])
        adx_4h = float(latest_4h['ADX'])
        
        # 가중 평균 ADX (1시간봉에 가중치 높게)
        weighted_adx = (adx_15m * 0.2) + (adx_1h * 0.5) + (adx_4h * 0.3)
        
        # 2. 방향성 평가
        # MACD 신호
        macd_15m_bull = float(latest_15m['MACD']) > float(latest_15m['MACD_signal'])
        macd_1h_bull = float(latest_1h['MACD']) > float(latest_1h['MACD_signal'])
        macd_4h_bull = float(latest_4h['MACD']) > float(latest_4h['MACD_signal'])
        
        # EMA 배열
        ema_15m_bull = float(latest_15m['Close']) > float(latest_15m['EMA_20'])
        ema_1h_bull = float(latest_1h['Close']) > float(latest_1h['EMA_50'])
        ema_4h_bull = float(latest_4h['Close']) > float(latest_4h['EMA_100'])
        
        # 불리시 신호 개수
        bull_signals = sum([macd_15m_bull, macd_1h_bull, macd_4h_bull, ema_15m_bull, ema_1h_bull, ema_4h_bull])
        # 베어리시 신호 개수
        bear_signals = 6 - bull_signals
        
        # 3. 변동성 평가
        atr_15m = float(latest_15m['ATR'])
        atr_1h = float(latest_1h['ATR'])
        
        # 가격 대비 ATR 비율 (%)
        volatility_pct = (atr_15m / float(latest_15m['Close'])) * 100
        
        # 볼린저 밴드 폭 계산
        bb_width_15m = (float(latest_15m['BB_upper']) - float(latest_15m['BB_lower'])) / float(latest_15m['BB_mavg'])
        bb_width_1h = (float(latest_1h['BB_upper']) - float(latest_1h['BB_lower'])) / float(latest_1h['BB_mavg'])
        
        # 가중 평균 볼린저 밴드 폭
        weighted_bb_width = (bb_width_15m * 0.4) + (bb_width_1h * 0.6)
        
        # 4. 결과 판정
        # 기본값 초기화
        market_type = "RANGE"
        direction = "NEUTRAL"
        strength = 0.5
        confidence = 0.5
        
        # 추세 판정 (ADX > 25)
        if weighted_adx > 25:
            market_type = "TREND"
            strength = min(1.0, weighted_adx / 40)  # 40 이상이면 최대 강도 1.0
            
            if bull_signals >= 4:  # 6개 중 4개 이상 불리시
                direction = "UP"
                confidence = bull_signals / 6
            elif bear_signals >= 4:  # 6개 중 4개 이상 베어리시
                direction = "DOWN"
                confidence = bear_signals / 6
            else:
                # 방향성 불명확
                market_type = "VOLATILE"
                direction = "NEUTRAL"
                confidence = 0.5
                
        # 변동성 판정
        elif volatility_pct > 1.5:  # 1.5% 이상 변동성
            market_type = "VOLATILE"
            strength = min(1.0, volatility_pct / 3.0)  # 3% 이상이면 최대 강도 1.0
            
            if bull_signals >= 4:
                direction = "UP"
                confidence = bull_signals / 6
            elif bear_signals >= 4:
                direction = "DOWN"
                confidence = bear_signals / 6
            else:
                direction = "NEUTRAL"
                confidence = 0.5
                
        # 변동성 수축 판정
        elif weighted_bb_width < 0.03:  # 볼린저 밴드 폭이 좁을 때
            market_type = "CONTRACTION"
            strength = max(0.5, 1.0 - (weighted_bb_width / 0.06))  # 밴드 폭이 좁을수록 강도 높음
            direction = "NEUTRAL"
            confidence = 0.7  # 볼린저 밴드 수축은 비교적 명확한 패턴
                
        # 횡보장 판정 (기본값)
        else:
            market_type = "RANGE"
            strength = max(0.5, 1.0 - (weighted_adx / 25))  # ADX가 낮을수록 횡보 강도 높음
            
            if bb_width_15m > 0.04:  # 어느 정도 밴드 폭이 있어야 의미 있는 횡보
                confidence = 0.6
            else:
                confidence = 0.4
                
            # 횡보장에서도 약한 방향성은 있을 수 있음
            if bull_signals > bear_signals:
                direction = "UP"
            elif bear_signals > bull_signals:
                direction = "DOWN"
            else:
                direction = "NEUTRAL"
        
        # 키 레벨 데이터 생성
        key_levels_data = identify_key_levels(market_data, "Long" if direction == "UP" else "Short")
        key_levels_json = json.dumps(key_levels_data) if key_levels_data else None
        
        # 최종 결과 저장 (나중에 DB에 기록하기 위한 준비)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO market_analysis_records
                (timestamp, market_type, direction, strength, confidence, key_levels, volatility, adx, rsi, analysis_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                market_type,
                direction,
                strength,
                confidence,
                key_levels_json,  # key_levels 컬럼에 JSON 데이터 추가
                volatility_pct,
                weighted_adx,
                json.dumps({
                    '15m': float(latest_15m.get('RSI', 50)),
                    '1h': float(latest_1h.get('RSI', 50)),
                    '4h': float(latest_4h.get('RSI', 50))
                }),
                f"{market_type} 시장, 방향성: {direction}, 강도: {strength:.2f}, 신뢰도: {confidence:.2f}"
            ))
            conn.commit()
        
        # 결과 반환
        return {
            'type': market_type,
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'adx': weighted_adx,
            'volatility': volatility_pct,
            'bb_width': weighted_bb_width,
            'bull_signals': bull_signals,
            'bear_signals': bear_signals
        }
        
    except Exception as e:
        print(f"[오류] 시장 구조 분석 중 문제 발생: {e}")
        print(traceback.format_exc())
        # 오류 발생 시 기본값 반환
        return {
            'type': 'RANGE',
            'direction': 'NEUTRAL',
            'strength': 0.5,
            'confidence': 0.3,
            'error': str(e)
        }

def check_and_set_next_tp_order(conn, trade_id, position_type, entry_price, total_qty, tp_levels, tp_prices, current_tp_level, leverage=1):
    """다음 익절 주문 설정 로직"""
    try:
        next_level = current_tp_level + 1
        if next_level < len(tp_levels):
            # 안전한 가격 및 수량 계산
            next_tp_price = round(tp_prices[next_level], 1)
            next_tp_qty = round(total_qty * tp_levels[next_level], 3)
            
            # 최소 수량 체크
            if next_tp_qty < MIN_QTY:
                print(f"[경고] 다음 익절 주문 수량이 최소 거래 수량({MIN_QTY}) 미만입니다.")
                return False
            
            # 다음 익절 주문 설정
            response = bybit_client.set_trading_stop(
                category="linear",
                symbol=SYMBOL,
                positionIdx=1 if position_type == "Long" else 2,
                takeProfit=str(next_tp_price),
                tpSize=str(next_tp_qty),
                tpTriggerBy="LastPrice",
                tpslMode="Partial"  # 파라미터 추가
            )
            
            if response["retCode"] == 0:
                # DB에 다음 익절 주문 저장
                store_take_profit_orders(
                    conn, 
                    trade_id, 
                    position_type, 
                    [next_level + 1],  # 레벨은 1부터 시작
                    [next_tp_price], 
                    [next_tp_qty]
                )
                print(f"[성공] {next_level+1}차 익절 주문 설정: 가격 ${next_tp_price:.2f}, 수량 {next_tp_qty} BTC, 레버리지 {leverage}x")
                return True
            else:
                print(f"[오류] {next_level+1}차 익절 주문 설정 실패: {response['retMsg']}")
                return False
        return False
    except Exception as e:
        print(f"[오류] 다음 익절 주문 설정 중 문제 발생: {e}")
        return False

def request_ai_trade_analysis(trade_data):
    """
    AI에게 거래 분석을 요청하는 함수
    """
    try:
        # AI 분석 프롬프트 구성
        prompt = f"""
        거래 분석 요청:
        - 포지션 타입: {trade_data['position_type']}
        - 진입 가격: ${trade_data['entry_price']:.2f}
        - 익절 가격: ${trade_data['exit_price']:.2f}
        - 레버리지: {trade_data['leverage']}x
        - 수익률: {trade_data['profit_percentage']:.2f}%
        - 익절 단계: {trade_data['tp_level']}

        이 거래의 성공 요인과 개선 가능한 부분을 분석해주세요.
        """
        
        # AI 분석 요청
        analysis_result = get_ai_analysis(prompt)
        
        # 분석 결과 저장
        save_ai_trade_analysis(trade_data['trade_id'], analysis_result)
        
        return analysis_result
    
    except Exception as e:
        print(f"[오류] AI 거래 분석 요청 중 문제 발생: {e}")
        return None


def save_ai_trade_analysis(trade_id, analysis_result):
    """
    AI 분석 결과를 데이터베이스에 저장하는 함수 (재시도 로직 적용)
    """
    def _save_operation(conn):
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE trades
            SET trading_analysis_ai = ?
            WHERE trade_id = ?
        """, (analysis_result, trade_id))
        return True
    
    try:
        return execute_with_retry(_save_operation)
    except Exception as e:
        print(f"[오류] AI 분석 결과 저장 실패 (마지막 시도): {e}")
        return False        

# 4. 비동기 AI 분석 요청 함수
def request_ai_trade_analysis_async(trade_data):
    """
    AI 분석을 별도 스레드에서 비동기적으로 실행
    """
    def _async_analysis():
        try:
            analysis_result = get_ai_analysis(prompt)
            save_ai_trade_analysis(trade_data['trade_id'], analysis_result)
            print(f"[성공] 거래 ID {trade_data['trade_id']}의 AI 분석 저장 완료")
        except Exception as e:
            print(f"[오류] 비동기 AI 분석 중 문제 발생: {e}")
    
    # 분석 프롬프트 생성
    prompt = f"""
    거래 분석 요청:
    - 포지션 타입: {trade_data['position_type']}
    - 진입 가격: ${trade_data['entry_price']:.2f}
    - 익절 가격: ${trade_data['exit_price']:.2f}
    - 레버리지: {trade_data['leverage']}x
    - 수익률: {trade_data['profit_percentage']:.2f}%
    - 익절 단계: {trade_data['tp_level']}

    이 거래의 성공 요인과 개선 가능한 부분을 분석해주세요.
    """
    
    # 별도 스레드에서 비동기 실행
    Thread(target=_async_analysis, daemon=True).start()
    print(f"[정보] 거래 ID {trade_data['trade_id']}의 AI 분석 비동기 요청 시작")
    return True

def adjust_stop_loss_after_first_tp(conn, trade_id, position_type, entry_price):
    """
    첫 번째 익절 이후 스탑로스 조정
    """
    try:
        # 현재 포지션 정보 가져오기
        position_info = get_position_info(bybit_client)
        if not position_info or position_info[position_type]["size"] <= 0:
            print(f"[경고] {position_type} 포지션 정보를 찾을 수 없습니다.")
            return False
            
        # 현재 포지션 수량 가져오기
        current_size = position_info[position_type]["size"]
            
        # 손절가를 진입가 근처로 조정
        adjustment = 0.003  # 0.3% 조정
        if position_type == "Long":
            new_stop_price = entry_price * (1 - adjustment)
        else:
            new_stop_price = entry_price * (1 + adjustment)
            
        position_idx = 1 if position_type == "Long" else 2
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=SYMBOL,
            positionIdx=position_idx,
            stopLoss=str(new_stop_price),
            slSize=str(current_size),  # 전체 포지션 수량 사용
            slTriggerBy="LastPrice",
            tpslMode="Partial"
        )
        
        if response["retCode"] == 0:
            print(f"[성공] 첫 익절 이후 스탑로스 조정 완료: ${new_stop_price:.2f}")
            # DB에 스탑로스 조정 내역 기록
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET stop_loss_price = ?
                WHERE trade_id = ?
            """, (new_stop_price, trade_id))
            conn.commit()
            return True
        else:
            print(f"[경고] 스탑로스 조정 실패: {response['retMsg']}")
            return False
    except Exception as e:
        print(f"[오류] 스탑로스 조정 중 문제 발생: {e}")
        return False

def set_breakeven_stop_loss(conn, trade_id, position_type, entry_price):
    """
    두 번째 익절 이후 스탑로스를 손익분기점으로 조정
    """
    try:
        position_info = get_position_info(bybit_client)
        if not position_info or position_info[position_type]["size"] <= 0:
            print(f"[경고] {position_type} 포지션 정보를 찾을 수 없습니다.")
            return False
            
        # 현재 포지션 수량 가져오기
        current_size = position_info[position_type]["size"]
        
        # 약간의 마진을 추가하여 수수료를 커버
        fee_adjustment = 0.001  # 0.1% 수수료 고려
        if position_type == "Long":
            breakeven_price = entry_price * (1 + fee_adjustment)
        else:
            breakeven_price = entry_price * (1 - fee_adjustment)
            
        position_idx = 1 if position_type == "Long" else 2
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=SYMBOL,
            positionIdx=position_idx,
            stopLoss=str(breakeven_price),
            slSize=str(current_size),  # 전체 포지션 수량 사용
            slTriggerBy="LastPrice",
            tpslMode="Partial"
        )
        
        if response["retCode"] == 0:
            print(f"[성공] 손익분기점 스탑로스 설정 완료: ${breakeven_price:.2f}")
            # DB에 스탑로스 조정 내역 기록
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET stop_loss_price = ?
                WHERE trade_id = ?
            """, (breakeven_price, trade_id))
            conn.commit()
            return True
        else:
            print(f"[경고] 손익분기점 스탑로스 설정 실패: {response['retMsg']}")
            return False
    except Exception as e:
        print(f"[오류] 손익분기점 스탑로스 설정 중 문제 발생: {e}")
        return False

def close_remaining_position(conn, trade_id, position_type):
    """
    세 번째 익절 이후 남은 포지션 모두 청산
    """
    try:
        # 현재 포지션 정보 가져오기 (DB와 무관한 작업)
        position_info = get_position_info(bybit_client)
        if not position_info or position_info[position_type]["size"] <= 0:
            print(f"[경고] 청산할 {position_type} 포지션이 없습니다.")
            return False
            
        remaining_qty = position_info[position_type]["size"]
        position_idx = 1 if position_type == "Long" else 2
        order_type = "sell" if position_type == "Long" else "buy"
        
        # 남은 포지션 전부 청산 (DB와 무관한 작업)
        order_success = place_order(
            order_type=order_type, 
            symbol=SYMBOL, 
            qty=remaining_qty, 
            position_idx=position_idx
        )
        
        if order_success:
            print(f"[성공] 남은 {position_type} 포지션 {remaining_qty} BTC 청산 완료")
            
            # 현재 가격 가져오기 (DB와 무관한 작업)
            current_price = float(fetch_candle_data(interval="1", limit=1)['Close'].iloc[-1])
            
            # DB 작업 부분만 with 문으로 감싸기
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE trades
                    SET trade_status = 'Closed',
                        exit_price = ?,
                        profit_loss = ?
                    WHERE trade_id = ?
                """, (
                    current_price,
                    calculate_profit_loss(conn, trade_id, current_price),
                    trade_id
                ))
            return True
        else:
            print(f"[경고] 남은 포지션 청산 실패")
            return False
    except Exception as e:
        print(f"[오류] 남은 포지션 청산 중 문제 발생: {e}")
        return False

def check_for_liquidations():
    """
    스탑로스 청산 여부 확인하고 DB 업데이트
    """
    try:
        # 이전에 활성화된 포지션 정보 가져오기
        positions = get_position_info(bybit_client)
        if positions is None:
            return
            
        # 현재 가격 가져오기
        market_data = get_market_data()
        if not market_data or market_data['candlestick'].empty:
            return
        current_price = float(market_data['candlestick'].iloc[-1]['Close'])
            
        # DB에서 미청산 포지션 정보 확인
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        
        for position_type in ["Long", "Short"]:
            # 현재 해당 타입의 포지션이 없는지 확인
            if positions[position_type]["size"] == 0:
                # 가장 최근 미청산 스탑로스 설정 찾기
                cursor.execute("""
                    SELECT id, entry_price, stop_price
                    FROM stop_loss_records
                    WHERE position_type = ? AND liquidated = 0
                    ORDER BY set_time DESC
                    LIMIT 1
                """, (position_type,))
                
                record = cursor.fetchone()
                if record:
                    record_id, entry_price, stop_price = record
                    
                    # 손익 계산
                    leverage = positions[position_type].get("leverage", 1)
                    if not leverage or leverage <= 0:
                        leverage = 1
                        
                    if position_type == "Long":
                        profit_loss = ((stop_price - entry_price) / entry_price) * 100 * leverage
                    else:
                        profit_loss = ((entry_price - stop_price) / entry_price) * 100 * leverage
                    
                    # 청산 정보 업데이트
                    update_liquidation_info(
                        position_type=position_type,
                        liquidation_price=current_price,
                        profit_loss=profit_loss
                    )
                    
                    # trades 테이블 업데이트 (수정된 코드)
                    c = conn.cursor()
                    # 1단계: 대상 거래 ID 찾기
                    c.execute("""
                        SELECT trade_id FROM trades
                        WHERE position_type = ? AND trade_status = 'Open'
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (position_type,))
                    trade_record = c.fetchone()

                    if trade_record:
                        trade_id = trade_record[0]
                        # 2단계: 찾은 ID로 업데이트 수행
                        c.execute("""
                            UPDATE trades
                            SET is_stop_loss = 1, 
                                exit_price = ?,
                                profit_loss = ?,
                                trade_status = 'Closed'
                            WHERE trade_id = ?
                        """, (current_price, profit_loss, trade_id))
                        conn.commit()
                        print(f"[정보] 스탑로스 청산으로 인한 거래 상태 업데이트 완료 (TradeID: {trade_id})")
                    
                    print(f"[감지] 스탑로스 청산 발생! {position_type} 포지션")
                    print(f"진입가: ${entry_price:.2f}, 스탑가: ${stop_price:.2f}")
                    print(f"청산 손익: {profit_loss:.2f}%")
        
        conn.close()
    except Exception as e:
        print(f"[오류] 스탑로스 청산 확인 중 문제 발생: {e}")
        print(traceback.format_exc())

def check_liquidation_records():
    """청산 레코드 확인"""
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        
        # 청산된 레코드 조회
        cursor.execute("""
            SELECT id, position_type, entry_price, stop_price, 
                   liquidation_price, liquidation_time, profit_loss
            FROM stop_loss_records
            WHERE liquidated = 1
            ORDER BY liquidation_time DESC
            LIMIT 10
        """)
        
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            print("\n[정보] 청산된 레코드가 없습니다.")
            return
            
        print("\n=== 최근 청산된 포지션 ===")
        for record in records:
            id, pos_type, entry_price, stop_price, liq_price, liq_time, profit = record
            print(f"ID: {id}, 포지션: {pos_type}")
            print(f"진입가: ${entry_price:.2f}, 스탑가: ${stop_price:.2f}")
            print(f"청산가: ${liq_price if liq_price else 'N/A':.2f}")
            print(f"청산시간: {liq_time}")
            print(f"손익: {profit:.2f}%")
            print("-" * 40)
            
    except Exception as e:
        print(f"[오류] 청산 레코드 확인 중 문제 발생: {e}")

def check_max_loss_protection(conn):
    """
    최대 손실 보호 메커니즘 - 일정 비율 이상 손실 시 포지션 청산
    """
    try:
        MAX_LOSS_PCT = 25.0  # 최대 허용 손실률 (25%)
        
        # 포지션 정보 가져오기
        position_info = get_position_info(bybit_client)
        if not position_info:
            return
            
        # 현재 가격 확인
        current_price = float(fetch_candle_data(interval="1", limit=1)['Close'].iloc[-1])
        
        for position_type, info in position_info.items():
            if info["size"] <= 0:
                continue  # 포지션이 없으면 스킵
                
            # 오픈된 포지션과 관련된 거래 찾기
            cursor = conn.cursor()
            cursor.execute("""
                SELECT trade_id, entry_price, leverage
                FROM trades
                WHERE position_type = ? AND trade_status = 'Open'
                ORDER BY timestamp DESC
                LIMIT 1
            """, (position_type,))
            
            trade = cursor.fetchone()
            if not trade:
                continue  # 관련 거래 정보가 없으면 스킵
                
            trade_id, entry_price, leverage = trade
            
            # 손실률 계산
            if position_type == "Long":
                profit_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
            else:
                profit_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
                
            # 과도한 손실 확인
            if profit_pct < -MAX_LOSS_PCT:
                print(f"[경고] {position_type} 포지션 손실률({profit_pct:.2f}%)이 최대 허용치({-MAX_LOSS_PCT}%)를 초과함")
                print(f"[보호] 자동 손절 메커니즘 작동")
                
                # 포지션 청산
                position_idx = 1 if position_type == "Long" else 2
                order_type = "sell" if position_type == "Long" else "buy"
                
                order_success = place_order(
                    order_type=order_type, 
                    symbol=SYMBOL, 
                    qty=info["size"], 
                    position_idx=position_idx
                )
                
                if order_success:
                    # DB 업데이트
                    cursor.execute("""
                        UPDATE trades
                        SET trade_status = 'Closed',
                            exit_price = ?,
                            profit_loss = ?,
                            is_stop_loss = 1
                        WHERE trade_id = ?
                    """, (current_price, profit_pct, trade_id))
                    
                    conn.commit()
                    print(f"[성공] 최대 손실 보호 메커니즘에 의한 {position_type} 포지션 청산 완료")
                    
                    # 익절 주문 초기화
                    reset_take_profit_orders(conn, trade_id)
                
    except Exception as e:
        print(f"[오류] 최대 손실 보호 메커니즘 실행 중 문제 발생: {e}")
        if 'conn' in locals():
            conn.rollback()

def inspect_stop_loss_records():
    """데이터베이스의 스탑로스 레코드 직접 검사"""
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        
        # 모든 스탑로스 레코드 조회
        cursor.execute("SELECT * FROM stop_loss_records ORDER BY set_time DESC LIMIT 20")
        records = cursor.fetchall()
        
        print("\n=== 스탑로스 레코드 검사 ===")
        for record in records:
            print(record)
            
        # 최근 거래 관련 정보 조회
        cursor.execute("""
            SELECT trade_id, entry_price, stop_loss_price, stop_loss_pct
            FROM trades
            WHERE stop_loss_price IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        trades = cursor.fetchall()
        print("\n=== 거래의 스탑로스 정보 ===")
        for trade in trades:
            tid, entry, stop, pct = trade
            real_pct = ((entry - stop) / entry) * 100 if stop < entry else ((stop - entry) / entry) * 100
            print(f"거래 ID: {tid}")
            print(f"진입가: {entry}, 스탑가: {stop}")
            print(f"저장된 퍼센트: {pct}%, 실제 퍼센트: {real_pct}%")
            print("---")
            
        conn.close()
    except Exception as e:
        print(f"[오류] 레코드 검사 중 문제 발생: {e}")

# 10. GPT 데이터 준비 및 분석 함수들
def prepare_market_data_for_gpt(market_data: Dict) -> Optional[Dict]:
    """GPT 분석을 위한 시장 데이터 정리 (돈치안 채널 데이터 포함)"""
    try:
        timeframe_data = {}
        timeframe_candles = {}
        orderbook = get_orderbook_data(bybit_client)
        market_pressure = analyze_market_pressure(orderbook) if orderbook else None

        # 각 타임프레임별 데이터 처리
        for tf, df in market_data['timeframes'].items():
            latest_data = df.iloc[-1]
            candles = []
            for index, row in df.iterrows():
                candle = {
                    'timestamp': index.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'rsi': float(row['RSI']),
                    'macd': float(row['MACD']),
                    'macd_signal': float(row['MACD_signal']),
                    'macd_diff': float(row['MACD_diff']),
                    'ema_20': float(row['EMA_20']),
                    'ema_50': float(row['EMA_50']),
                    'bb_upper': float(row['BB_upper']),
                    'bb_lower': float(row['BB_lower']),
                    'bb_middle': float(row['BB_mavg']),
                    'alligator_jaws': float(row['Alligator_Jaws']),
                    'alligator_teeth': float(row['Alligator_Teeth']),
                    'alligator_lips': float(row['Alligator_Lips']),
                    'atr': float(row['ATR']),
                    'fractal_up': int(row['Fractal_Up']),
                    'fractal_down': int(row['Fractal_Down'])
                }
                # 돈치안 채널 값 추가 (열이 존재할 경우)
                if 'Donchian_High' in df.columns:
                    candle['donchian_high'] = float(row['Donchian_High'])
                    candle['donchian_low'] = float(row['Donchian_Low'])
                    candle['donchian_mid'] = float(row['Donchian_Mid'])
                    candle['support1'] = float(row['Support1'])
                    candle['resistance1'] = float(row['Resistance1'])
                    candle['support2'] = float(row['Support2'])
                    candle['resistance2'] = float(row['Resistance2'])
                    candle['pivot'] = float(row['Pivot'])
                candles.append(candle)
            timeframe_candles[tf] = candles

            # 기본 데이터 구성
            timeframe_data[tf] = {
                "current_price": float(latest_data['Close']),
                "daily_high": float(latest_data['High']),
                "daily_low": float(latest_data['Low']),
                "volume": float(latest_data['Volume']),
                "rsi": float(latest_data['RSI']),
                "macd": float(latest_data['MACD']),
                "macd_signal": float(latest_data['MACD_signal']),
                "macd_diff": float(latest_data['MACD_diff']),
                "sma_20": float(latest_data['SMA_20']),
                "ema_10": float(latest_data.get('EMA_10', latest_data['EMA_20'])),
                "ema_20": float(latest_data['EMA_20']),
                "ema_50": float(latest_data['EMA_50']),
                "ema_100": float(latest_data['EMA_100']),
                "ema_200": float(latest_data['EMA_200']),
                "bb_upper": float(latest_data['BB_upper']),
                "bb_lower": float(latest_data['BB_lower']),
                "bb_middle": float(latest_data['BB_mavg']),
                "atr": float(latest_data['ATR']),
                "adx": float(latest_data['ADX']),
                "+di": float(latest_data['+DI']),
                "-di": float(latest_data['-DI']),
                "alligator_jaws": float(latest_data['Alligator_Jaws']),
                "alligator_teeth": float(latest_data['Alligator_Teeth']),
                "alligator_lips": float(latest_data['Alligator_Lips']),
                "fibonacci_levels": {
                    "0.236": float(latest_data['Fibo_0.236']),
                    "0.382": float(latest_data['Fibo_0.382']),
                    "0.500": float(latest_data['Fibo_0.5']),
                    "0.618": float(latest_data['Fibo_0.618']),
                    "0.786": float(latest_data['Fibo_0.786'])
                },
                "fibonacci_levels_down": {
                    "-0.236": float(latest_data.get('Fibo_-0.236', 0)),
                    "-0.382": float(latest_data.get('Fibo_-0.382', 0)),
                    "-0.5": float(latest_data.get('Fibo_-0.5', 0)),
                    "-0.618": float(latest_data.get('Fibo_-0.618', 0)),
                    "-0.786": float(latest_data.get('Fibo_-0.786', 0)),
                    "-1": float(latest_data.get('Fibo_-1', 0))
                }
            }
            
            # 돈치안 채널 데이터 포함 (해당 열이 존재하는 경우)
            if 'Donchian_High' in df.columns:
                timeframe_data[tf]["donchian_high"] = float(latest_data['Donchian_High'])
                timeframe_data[tf]["donchian_low"] = float(latest_data['Donchian_Low'])
                timeframe_data[tf]["donchian_mid"] = float(latest_data['Donchian_Mid'])
                timeframe_data[tf]["pivot"] = float(latest_data['Pivot'])
                timeframe_data[tf]["support1"] = float(latest_data['Support1'])
                timeframe_data[tf]["resistance1"] = float(latest_data['Resistance1'])
                timeframe_data[tf]["support2"] = float(latest_data['Support2'])
                timeframe_data[tf]["resistance2"] = float(latest_data['Resistance2'])
        
        # 포지션 정보 수집 및 가공
        position_info = get_position_info(bybit_client)
        current_price_overall = market_data['timeframes']["15"].iloc[-1]['Close']
        orders = bybit_client.get_open_orders(
            category="linear",
            symbol=SYMBOL,
            orderFilter="StopOrder"
        )
        position_details = {
            "Long": {"active": False, "details": {}},
            "Short": {"active": False, "details": {}}
        }
        for side in ["Long", "Short"]:
            if position_info and position_info[side]["size"] > 0:
                entry_price = position_info[side]["entry_price"]
                leverage = position_info[side].get("leverage", 1)
                if side == "Long":
                    profit_pct = ((current_price_overall - entry_price) / entry_price) * 100 * leverage
                else:
                    profit_pct = ((entry_price - current_price_overall) / entry_price) * 100 * leverage
                position_details[side] = {
                    "active": True,
                    "details": {
                        "size": position_info[side]["size"],
                        "entry_price": entry_price,
                        "leverage": leverage,
                        "unrealized_pnl": position_info[side]["unrealized_pnl"],
                        "current_profit_pct": profit_pct,
                        "stop_loss": None,
                        "take_profits": []
                    }
                }
                if orders["retCode"] == 0:
                    for order in orders["result"]["list"]:
                        if order["stopOrderType"] == "TakeProfit":
                            position_details[side]["details"]["take_profits"].append(
                                float(order["triggerPrice"])
                            )
                        elif order["stopOrderType"] == "StopLoss":
                            position_details[side]["details"]["stop_loss"] = float(order["triggerPrice"])
        
        # 최종 데이터 구성
        gpt_data = {
            "price_data": {
                "timeframes": timeframe_data,
                "candles": timeframe_candles,
                "current_price": timeframe_data["15"]["current_price"]
            },
            "technical_indicators": {
                "timeframes": {
                    tf: {
                        "rsi": data["rsi"],
                        "macd": data["macd"],
                        "macd_signal": data["macd_signal"],
                        "macd_diff": data["macd_diff"],
                        "sma_20": data["sma_20"],
                        "ema_10": data["ema_10"],
                        "ema_20": data["ema_20"],
                        "ema_50": data["ema_50"],
                        "ema_100": data["ema_100"],
                        "ema_200": data["ema_200"],
                        "bollinger_bands": {
                            "upper": data["bb_upper"],
                            "middle": data["bb_middle"],
                            "lower": data["bb_lower"]
                        },
                        "williams_alligator": {
                            "jaws": data["alligator_jaws"],
                            "teeth": data["alligator_teeth"],
                            "lips": data["alligator_lips"]
                        },
                        "fibonacci_levels": data["fibonacci_levels"],
                        "fibonacci_levels_down": data["fibonacci_levels_down"],
                        "atr": data["atr"],
                        "adx": data["adx"],
                        "+di": data["+di"],
                        "-di": data["-di"],
                        # 돈치안 채널 데이터 추가
                        "donchian": {
                            "high": data.get("donchian_high"),
                            "low": data.get("donchian_low"),
                            "mid": data.get("donchian_mid")
                        },
                        "pivot": data.get("pivot"),
                        "support1": data.get("support1"),
                        "resistance1": data.get("resistance1"),
                        "support2": data.get("support2"),
                        "resistance2": data.get("resistance2")
                    } for tf, data in timeframe_data.items()
                }
            },
            "orderbook_analysis": {
                "buy_pressure": market_pressure["buy_pressure"] if market_pressure else None,
                "sell_pressure": market_pressure["sell_pressure"] if market_pressure else None,
                "pressure_ratio": market_pressure["pressure_ratio"] if market_pressure else None
            },
            "market_sentiment": {
                "fear_greed": market_data.get('fear_greed', None)
            },
            "position_management": {
                "current_positions": position_details,
                "current_price": timeframe_data["15"]["current_price"],
                "risk_summary": {
                    "active_positions": sum(1 for side in position_details.values() if side["active"]),
                    "has_stop_loss": any(side["details"].get("stop_loss") for side in position_details.values() if side["active"]),
                    "has_take_profit": any(side["details"].get("take_profits") for side in position_details.values() if side["active"])
                }
            }
        }
        return gpt_data
    except Exception as e:
        print(f"[오류] GPT 데이터 준비 중 문제 발생: {e}")
        return None


def get_ai_analysis(prompt: str) -> str:
    """OpenAI API를 통해 거래 데이터에 대한 AI 분석 수행"""
    try:
        system_message = (
            "You are a cryptocurrency futures trading strategy analysis expert. "
            "Based on the latest market data, analyze the market and provide an optimal trading decision according to the strategy outlined below.\n\n"
            "[Strategy Overview]\n"
            "1. Multi-Timeframe Analysis: Analyze 15-minute, 1-hour, and 4-hour charts to assess short-term momentum, intermediate trends, and long-term trends. "
            "A consistent bullish or bearish signal across all timeframes indicates a strong buy or sell signal; if signals conflict or the market is consolidating, adopt a cautious approach.\n"
            "2. Leverage Management: Use leverage between 10x and 15x for strong signals, and between 3x and 5x for weak signals or consolidating markets, with a default of around 7x.\n"
            "3. Stop Loss Setup: Implement an ATR-based dynamic stop loss. Under normal conditions, set the stop loss at approximately 2 ATR from the entry price; in high-leverage or consolidating markets, use about 1.5 ATR. "
            "Ensure that the stop loss does not conflict with major technical indicators (e.g., EMA 20 or key support/resistance levels).\n"
            "4. Trailing Stop: Once the position reaches a profit of 2–3%, gradually adjust the stop loss to secure about 50% of the profit.\n"
            "5. Take Profit Strategy: Exit the position in stages:\n"
            "   - First TP: Target a gain of approximately 1.5–2%.\n"
            "   - Second TP: Target a gain of approximately 3–4%.\n"
            "   - Third TP: Target a gain of 6% or more.\n"
            "   Adjust targets higher in trending markets and lower in consolidating markets.\n"
            "6. Additional Entry Conditions: If an existing position shows an unrealized profit exceeding 1% and all timeframes display strong signals, consider adding to the position. "
            "Ensure that the total risk does not exceed 3% of the account balance.\n"
            "7. Risk Management: Limit each trade’s maximum loss to 3% of your account balance. Adjust the stop_loss_pct based on leverage so that the actual price movement risk is properly reflected.\n\n"
            "Return your analysis result in the following JSON format:\n"
            "{\n"
            '  "decision": "Long/Short/Close Long Position/Close Short Position/Hold",\n'
            '  "reason": "A detailed explanation based on the analysis of the 15-minute, 1-hour, and 4-hour charts and risk management criteria.",\n'
            '  "position_size": (an integer between 0 and 100),\n'
            '  "leverage": (an integer between 1 and 20; omit if closing a position),\n'
            '  "stop_loss_pct": (a decimal between 0.1 and 5.0)\n'
            "}\n\n"
            "Please provide your analysis and decision based on the most current market data available."
        )

        response = openai_client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.7
        )
    
        response_text = response.choices[0].message.content.strip()
        # 디버깅을 위한 로그 추가
        print(f"[전체 AI 분석 응답] {response_text}")
        
        return response_text
    except Exception as e:
        print(f"[오류] AI 분석 중 문제 발생: {e}")
        return "AI 분석을 수행할 수 없습니다."


# 필요한 모듈 import와 전역 상수 선언 이후

def evaluate_signal_strength(market_data: Dict, new_position_type: str) -> float:
    """
    새 포지션 신호의 강도를 평가합니다.
    - 다중 타임프레임 분석
    - 여러 기술적 지표 통합
    - 시장 구조와 현재 가격 위치 고려
    
    Args:
        market_data: 시장 데이터 딕셔너리
        new_position_type: 신규 포지션 유형 ("Long" 또는 "Short")
        
    Returns:
        float: 0~1 사이의 값으로, 1에 가까울수록 강한 신호입니다.
    """
    try:
        # 타임프레임별 데이터
        tf_15m = market_data['timeframes']["15"].iloc[-1]
        tf_1h = market_data['timeframes'].get("60", market_data['timeframes']["15"]).iloc[-1]
        tf_4h = market_data['timeframes'].get("240", market_data['timeframes']["15"]).iloc[-1]
        
        # 여러 타임프레임 데이터
        df_15m = market_data['timeframes']["15"]
        df_1h = market_data['timeframes'].get("60", df_15m)
        df_4h = market_data['timeframes'].get("240", df_15m)
        
        # 각 타임프레임별 가중치
        weights = {"15m": 0.3, "1h": 0.4, "4h": 0.3}
        
        # 신호 점수 초기화
        score = 0.0
        
        # 롱 포지션 신호 평가
        if new_position_type == "Long":
            # 1. RSI 평가 (과매도 영역 = 더 강한 매수 신호)
            rsi_score = 0.0
            if float(tf_15m['RSI']) < 30: rsi_score += 1.0  # 강한 과매도
            elif float(tf_15m['RSI']) < 40: rsi_score += 0.7  # 중간 과매도
            elif float(tf_15m['RSI']) < 50: rsi_score += 0.3  # 약한 과매도
            
            if float(tf_1h['RSI']) < 30: rsi_score += 1.0
            elif float(tf_1h['RSI']) < 40: rsi_score += 0.7
            elif float(tf_1h['RSI']) < 50: rsi_score += 0.3
            
            if float(tf_4h['RSI']) < 30: rsi_score += 1.0
            elif float(tf_4h['RSI']) < 40: rsi_score += 0.7
            elif float(tf_4h['RSI']) < 50: rsi_score += 0.3
            
            # RSI 기울기 확인 (상승 반전 = 더 강한 매수 신호)
            if len(df_15m) > 3 and df_15m['RSI'].iloc[-1] > df_15m['RSI'].iloc[-2] > df_15m['RSI'].iloc[-3]:
                rsi_score += 0.5  # RSI 상승 반전
            
            score += rsi_score / 6  # 총 RSI 점수 정규화 (최대 3 + 0.5 = 3.5 점수)
            
            # 2. MACD 평가
            macd_score = 0.0
            # MACD 기준선 위치
            if float(tf_15m['MACD']) > float(tf_15m['MACD_signal']): macd_score += 0.3 * weights["15m"]
            if float(tf_1h['MACD']) > float(tf_1h['MACD_signal']): macd_score += 0.3 * weights["1h"]
            if float(tf_4h['MACD']) > float(tf_4h['MACD_signal']): macd_score += 0.3 * weights["4h"]
            
            # MACD 히스토그램 상승 추세
            if len(df_15m) > 2:
                macd_diff_15m = float(df_15m['MACD'].iloc[-1] - df_15m['MACD_signal'].iloc[-1])
                macd_diff_prev_15m = float(df_15m['MACD'].iloc[-2] - df_15m['MACD_signal'].iloc[-2])
                if macd_diff_15m > macd_diff_prev_15m: macd_score += 0.2 * weights["15m"]
            
            if len(df_1h) > 2:
                macd_diff_1h = float(df_1h['MACD'].iloc[-1] - df_1h['MACD_signal'].iloc[-1])
                macd_diff_prev_1h = float(df_1h['MACD'].iloc[-2] - df_1h['MACD_signal'].iloc[-2])
                if macd_diff_1h > macd_diff_prev_1h: macd_score += 0.2 * weights["1h"]
            
            score += macd_score
            
            # 3. 이동평균선 평가
            ma_score = 0.0
            # 가격이 주요 이동평균선 위에 있는지
            if float(tf_15m['Close']) > float(tf_15m['EMA_20']): ma_score += 0.2 * weights["15m"]
            if float(tf_15m['Close']) > float(tf_15m['EMA_50']): ma_score += 0.2 * weights["15m"]
            if float(tf_1h['Close']) > float(tf_1h['EMA_20']): ma_score += 0.2 * weights["1h"]
            if float(tf_1h['Close']) > float(tf_1h['EMA_50']): ma_score += 0.2 * weights["1h"]
            if float(tf_4h['Close']) > float(tf_4h['EMA_20']): ma_score += 0.2 * weights["4h"]
            if float(tf_4h['Close']) > float(tf_4h['EMA_50']): ma_score += 0.2 * weights["4h"]
            
            # 단기 이평선이 장기 이평선 위에 있는지 (황금 크로스 상태)
            if float(tf_15m['EMA_20']) > float(tf_15m['EMA_50']): ma_score += 0.3 * weights["15m"]
            if float(tf_1h['EMA_20']) > float(tf_1h['EMA_50']): ma_score += 0.3 * weights["1h"]
            if float(tf_4h['EMA_20']) > float(tf_4h['EMA_50']): ma_score += 0.3 * weights["4h"]
            
            score += ma_score
            
            # 4. 볼린저 밴드 평가
            bb_score = 0.0
            # 가격이 볼린저 밴드 하단에 가까운지 (롱 포지션에 유리)
            bb_pos_15m = (float(tf_15m['Close']) - float(tf_15m['BB_lower'])) / (float(tf_15m['BB_upper']) - float(tf_15m['BB_lower']))
            bb_pos_1h = (float(tf_1h['Close']) - float(tf_1h['BB_lower'])) / (float(tf_1h['BB_upper']) - float(tf_1h['BB_lower']))
            
            if bb_pos_15m < 0.2: bb_score += 0.5 * weights["15m"]  # 하단 밴드 근처
            elif bb_pos_15m < 0.5: bb_score += 0.2 * weights["15m"]  # 중간 밴드 아래
            
            if bb_pos_1h < 0.2: bb_score += 0.5 * weights["1h"]
            elif bb_pos_1h < 0.5: bb_score += 0.2 * weights["1h"]
            
            score += bb_score
            
            # 5. ADX (추세 강도) 평가
            adx_score = 0.0
            if float(tf_15m['ADX']) > 25: adx_score += 0.2 * weights["15m"]
            if float(tf_15m['ADX']) > 20 and float(tf_15m['+DI']) > float(tf_15m['-DI']): 
                adx_score += 0.3 * weights["15m"]  # ADX가 상승 추세 나타냄
            
            if float(tf_1h['ADX']) > 25: adx_score += 0.2 * weights["1h"]
            if float(tf_1h['ADX']) > 20 and float(tf_1h['+DI']) > float(tf_1h['-DI']): 
                adx_score += 0.3 * weights["1h"]
            
            if float(tf_4h['ADX']) > 25: adx_score += 0.2 * weights["4h"]
            if float(tf_4h['ADX']) > 20 and float(tf_4h['+DI']) > float(tf_4h['-DI']): 
                adx_score += 0.3 * weights["4h"]
            
            score += adx_score
            
            # 6. 도치안 채널 및 추가 지표 활용 (있는 경우)
            if 'Donchian_High' in tf_1h and 'Donchian_Low' in tf_1h:
                donchian_score = 0.0
                
                # 가격이 도치안 채널 하단에 가까운지 (롱 포지션에 유리)
                donchian_pos = (float(tf_1h['Close']) - float(tf_1h['Donchian_Low'])) / (float(tf_1h['Donchian_High']) - float(tf_1h['Donchian_Low']))
                
                if donchian_pos < 0.2: donchian_score += 0.4  # 채널 하단 근처
                elif donchian_pos < 0.5: donchian_score += 0.2  # 채널 중간 아래
                
                # 도치안 채널 폭이 확장 중인지 (상승 추세 가능성)
                if len(df_1h) > 2:
                    current_width = float(tf_1h['Donchian_High']) - float(tf_1h['Donchian_Low'])
                    prev_width = float(df_1h['Donchian_High'].iloc[-2]) - float(df_1h['Donchian_Low'].iloc[-2])
                    if current_width > prev_width: donchian_score += 0.2  # 채널 확장
                
                score += donchian_score * weights["1h"]
                
            # 7. 피보나치 레벨 활용 (있는 경우)
            if 'Fibo_0.236' in tf_1h and 'Fibo_0.786' in tf_1h:
                fibo_score = 0.0
                
                # 가격이 주요 피보나치 지지 레벨 근처인지
                current_price = float(tf_1h['Close'])
                
                # 0.786 레벨 (강한 지지)
                if abs(current_price - float(tf_1h['Fibo_0.786'])) / current_price < 0.01:
                    fibo_score += 0.4
                # 0.618 레벨 (중요 지지)
                elif abs(current_price - float(tf_1h['Fibo_0.618'])) / current_price < 0.01:
                    fibo_score += 0.3
                # 0.5 레벨 (중간 지지)
                elif abs(current_price - float(tf_1h['Fibo_0.5'])) / current_price < 0.01:
                    fibo_score += 0.2
                
                score += fibo_score * weights["1h"]
                
        # 숏 포지션 신호 평가
        else:  # Short
            # 1. RSI 평가 (과매수 영역 = 더 강한 매도 신호)
            rsi_score = 0.0
            if float(tf_15m['RSI']) > 70: rsi_score += 1.0  # 강한 과매수
            elif float(tf_15m['RSI']) > 60: rsi_score += 0.7  # 중간 과매수
            elif float(tf_15m['RSI']) > 50: rsi_score += 0.3  # 약한 과매수
            
            if float(tf_1h['RSI']) > 70: rsi_score += 1.0
            elif float(tf_1h['RSI']) > 60: rsi_score += 0.7
            elif float(tf_1h['RSI']) > 50: rsi_score += 0.3
            
            if float(tf_4h['RSI']) > 70: rsi_score += 1.0
            elif float(tf_4h['RSI']) > 60: rsi_score += 0.7
            elif float(tf_4h['RSI']) > 50: rsi_score += 0.3
            
            # RSI 기울기 확인 (하락 반전 = 더 강한 매도 신호)
            if len(df_15m) > 3 and df_15m['RSI'].iloc[-1] < df_15m['RSI'].iloc[-2] < df_15m['RSI'].iloc[-3]:
                rsi_score += 0.5  # RSI 하락 반전
            
            score += rsi_score / 6  # 총 RSI 점수 정규화 (최대 3 + 0.5 = 3.5 점수)
            
            # 2. MACD 평가
            macd_score = 0.0
            # MACD 기준선 위치
            if float(tf_15m['MACD']) < float(tf_15m['MACD_signal']): macd_score += 0.3 * weights["15m"]
            if float(tf_1h['MACD']) < float(tf_1h['MACD_signal']): macd_score += 0.3 * weights["1h"]
            if float(tf_4h['MACD']) < float(tf_4h['MACD_signal']): macd_score += 0.3 * weights["4h"]
            
            # MACD 히스토그램 하락 추세
            if len(df_15m) > 2:
                macd_diff_15m = float(df_15m['MACD'].iloc[-1] - df_15m['MACD_signal'].iloc[-1])
                macd_diff_prev_15m = float(df_15m['MACD'].iloc[-2] - df_15m['MACD_signal'].iloc[-2])
                if macd_diff_15m < macd_diff_prev_15m: macd_score += 0.2 * weights["15m"]
            
            if len(df_1h) > 2:
                macd_diff_1h = float(df_1h['MACD'].iloc[-1] - df_1h['MACD_signal'].iloc[-1])
                macd_diff_prev_1h = float(df_1h['MACD'].iloc[-2] - df_1h['MACD_signal'].iloc[-2])
                if macd_diff_1h < macd_diff_prev_1h: macd_score += 0.2 * weights["1h"]
            
            score += macd_score
            
            # 3. 이동평균선 평가
            ma_score = 0.0
            # 가격이 주요 이동평균선 아래에 있는지
            if float(tf_15m['Close']) < float(tf_15m['EMA_20']): ma_score += 0.2 * weights["15m"]
            if float(tf_15m['Close']) < float(tf_15m['EMA_50']): ma_score += 0.2 * weights["15m"]
            if float(tf_1h['Close']) < float(tf_1h['EMA_20']): ma_score += 0.2 * weights["1h"]
            if float(tf_1h['Close']) < float(tf_1h['EMA_50']): ma_score += 0.2 * weights["1h"]
            if float(tf_4h['Close']) < float(tf_4h['EMA_20']): ma_score += 0.2 * weights["4h"]
            if float(tf_4h['Close']) < float(tf_4h['EMA_50']): ma_score += 0.2 * weights["4h"]
            
            # 단기 이평선이 장기 이평선 아래에 있는지 (데드 크로스 상태)
            if float(tf_15m['EMA_20']) < float(tf_15m['EMA_50']): ma_score += 0.3 * weights["15m"]
            if float(tf_1h['EMA_20']) < float(tf_1h['EMA_50']): ma_score += 0.3 * weights["1h"]
            if float(tf_4h['EMA_20']) < float(tf_4h['EMA_50']): ma_score += 0.3 * weights["4h"]
            
            score += ma_score
            
            # 4. 볼린저 밴드 평가
            bb_score = 0.0
            # 가격이 볼린저 밴드 상단에 가까운지 (숏 포지션에 유리)
            bb_pos_15m = (float(tf_15m['Close']) - float(tf_15m['BB_lower'])) / (float(tf_15m['BB_upper']) - float(tf_15m['BB_lower']))
            bb_pos_1h = (float(tf_1h['Close']) - float(tf_1h['BB_lower'])) / (float(tf_1h['BB_upper']) - float(tf_1h['BB_lower']))
            
            if bb_pos_15m > 0.8: bb_score += 0.5 * weights["15m"]  # 상단 밴드 근처
            elif bb_pos_15m > 0.5: bb_score += 0.2 * weights["15m"]  # 중간 밴드 위
            
            if bb_pos_1h > 0.8: bb_score += 0.5 * weights["1h"]
            elif bb_pos_1h > 0.5: bb_score += 0.2 * weights["1h"]
            
            score += bb_score
            
            # 5. ADX (추세 강도) 평가
            adx_score = 0.0
            if float(tf_15m['ADX']) > 25: adx_score += 0.2 * weights["15m"]
            if float(tf_15m['ADX']) > 20 and float(tf_15m['+DI']) < float(tf_15m['-DI']): 
                adx_score += 0.3 * weights["15m"]  # ADX가 하락 추세 나타냄
            
            if float(tf_1h['ADX']) > 25: adx_score += 0.2 * weights["1h"]
            if float(tf_1h['ADX']) > 20 and float(tf_1h['+DI']) < float(tf_1h['-DI']): 
                adx_score += 0.3 * weights["1h"]
            
            if float(tf_4h['ADX']) > 25: adx_score += 0.2 * weights["4h"]
            if float(tf_4h['ADX']) > 20 and float(tf_4h['+DI']) < float(tf_4h['-DI']): 
                adx_score += 0.3 * weights["4h"]
            
            score += adx_score
            
            # 6. 도치안 채널 및 추가 지표 활용 (있는 경우)
            if 'Donchian_High' in tf_1h and 'Donchian_Low' in tf_1h:
                donchian_score = 0.0
                
                # 가격이 도치안 채널 상단에 가까운지 (숏 포지션에 유리)
                donchian_pos = (float(tf_1h['Close']) - float(tf_1h['Donchian_Low'])) / (float(tf_1h['Donchian_High']) - float(tf_1h['Donchian_Low']))
                
                if donchian_pos > 0.8: donchian_score += 0.4  # 채널 상단 근처
                elif donchian_pos > 0.5: donchian_score += 0.2  # 채널 중간 위
                
                # 도치안 채널 폭이 축소 중인지 (하락 추세 가능성)
                if len(df_1h) > 2:
                    current_width = float(tf_1h['Donchian_High']) - float(tf_1h['Donchian_Low'])
                    prev_width = float(df_1h['Donchian_High'].iloc[-2]) - float(df_1h['Donchian_Low'].iloc[-2])
                    if current_width < prev_width: donchian_score += 0.2  # 채널 축소
                
                score += donchian_score * weights["1h"]
                
            # 7. 피보나치 레벨 활용 (있는 경우)
            if 'Fibo_0.236' in tf_1h and 'Fibo_0.786' in tf_1h:
                fibo_score = 0.0
                
                # 가격이 주요 피보나치 저항 레벨 근처인지
                current_price = float(tf_1h['Close'])
                
                # 0.236 레벨 (강한 저항)
                if abs(current_price - float(tf_1h['Fibo_0.236'])) / current_price < 0.01:
                    fibo_score += 0.4
                # 0.382 레벨 (중요 저항)
                elif abs(current_price - float(tf_1h['Fibo_0.382'])) / current_price < 0.01:
                    fibo_score += 0.3
                # 0.5 레벨 (중간 저항)
                elif abs(current_price - float(tf_1h['Fibo_0.5'])) / current_price < 0.01:
                    fibo_score += 0.2
                
                score += fibo_score * weights["1h"]
        
        # 최종 점수 정규화 (0~1 범위로)
        # 이론적 최대 점수는 대략 4.0 정도
        normalized_score = min(1.0, score / 4.0)
        
        # 소수점 두 자리까지 반올림
        return round(normalized_score, 2)
        
    except Exception as e:
        print(f"[오류] 신호 강도 평가 중 문제 발생: {e}")
        print(traceback.format_exc())
        return 0.5  # 오류 발생 시 중간값 반환

def calculate_timeframe_agreement(market_data: Dict, position_type: str) -> float:
    """
    다중 타임프레임 간의 신호 일치도를 계산하는 함수
    - 타임프레임 간 지표 일치성 측정
    - 주요 지표 간 상관관계 분석
    - 시간 가중치 적용 (중요한 타임프레임에 더 높은 가중치)
    
    Args:
        market_data: 시장 데이터 딕셔너리
        position_type: 포지션 유형 ("Long" 또는 "Short")
        
    Returns:
        float: 0~1 사이 값으로, 1에 가까울수록 타임프레임 간 일치도가 높음
    """
    try:
        # 사용 가능한 타임프레임 확인
        timeframes = ["15", "60", "240"]
        available_timeframes = []
        
        for tf in timeframes:
            if tf in market_data['timeframes'] and not market_data['timeframes'][tf].empty:
                available_timeframes.append(tf)
        
        if not available_timeframes:
            print("[경고] 사용 가능한 타임프레임 데이터가 없습니다.")
            return 0.5
        
        # 각 타임프레임별 데이터 추출
        tf_data = {}
        for tf in available_timeframes:
            df = market_data['timeframes'][tf]
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # 각 타임프레임별 기술적 지표 값 저장
            tf_data[tf] = {
                "rsi": float(latest['RSI']),
                "rsi_prev": float(prev['RSI']) if 'RSI' in prev else float(latest['RSI']),
                "macd": float(latest['MACD']),
                "macd_signal": float(latest['MACD_signal']),
                "ema20": float(latest['EMA_20']),
                "ema50": float(latest['EMA_50']),
                "ema100": float(latest.get('EMA_100', latest['EMA_50'])),  # EMA_100이 없으면 EMA_50 사용
                "close": float(latest['Close']),
                "atr": float(latest['ATR']),
                "adx": float(latest['ADX']),
                "plus_di": float(latest['+DI']),
                "minus_di": float(latest['-DI']),
                "bb_upper": float(latest['BB_upper']),
                "bb_lower": float(latest['BB_lower']),
                "bb_mid": float(latest['BB_mavg'])
            }
            
            # 도치안 채널 (있는 경우)
            if 'Donchian_High' in latest and 'Donchian_Low' in latest:
                tf_data[tf]["donchian_high"] = float(latest['Donchian_High'])
                tf_data[tf]["donchian_low"] = float(latest['Donchian_Low'])
                tf_data[tf]["donchian_mid"] = float(latest['Donchian_Mid'])
        
        # 타임프레임별 가중치 설정 (중요도 기반)
        # 1시간봉이 중요도 높음, 4시간봉은 장기 추세, 15분봉은 단기 시그널
        weights = {
            "15": 0.25,  # 25%
            "60": 0.45,  # 45%
            "240": 0.30  # 30%
        }
        
        # 사용 가능한 타임프레임에 맞게 가중치 정규화
        total_weight = sum(weights[tf] for tf in available_timeframes)
        if total_weight > 0:
            normalized_weights = {tf: weights[tf] / total_weight for tf in available_timeframes}
        else:
            normalized_weights = {tf: 1.0 / len(available_timeframes) for tf in available_timeframes}
        
        # 지표별 확인 대상
        indicators = ["trend", "momentum", "volatility", "oscillator", "volume"]
        agreements = {indicator: 0.0 for indicator in indicators}
        
        # 1. 추세 지표 일치도 (EMA, MACD)
        trend_signals = []
        
        for tf in available_timeframes:
            data = tf_data[tf]
            if position_type == "Long":
                # 롱 포지션 긍정 신호: 가격 > EMA, 짧은 EMA > 긴 EMA, MACD > 시그널
                ema_signal = (data["close"] > data["ema20"] and data["ema20"] > data["ema50"])
                macd_signal = (data["macd"] > data["macd_signal"])
                adx_signal = (data["adx"] > 20 and data["plus_di"] > data["minus_di"])
            else:
                # 숏 포지션 긍정 신호: 가격 < EMA, 짧은 EMA < 긴 EMA, MACD < 시그널
                ema_signal = (data["close"] < data["ema20"] and data["ema20"] < data["ema50"])
                macd_signal = (data["macd"] < data["macd_signal"])
                adx_signal = (data["adx"] > 20 and data["plus_di"] < data["minus_di"])
            
            # 추세 강도까지 고려
            adx_strength = min(1.0, data["adx"] / 30.0)  # ADX를 0-1 범위로 정규화
            
            # 추세 신호 강도를 가중 평균으로 계산 (EMA 30%, MACD 40%, ADX 30%)
            trend_strength = 0.3 * int(ema_signal) + 0.4 * int(macd_signal) + 0.3 * (int(adx_signal) * adx_strength)
            trend_signals.append((trend_strength, normalized_weights[tf]))
        
        # 가중 평균 추세 신호 계산
        agreements["trend"] = sum(signal * weight for signal, weight in trend_signals)
        
        # 2. 모멘텀 지표 일치도 (RSI)
        momentum_signals = []
        
        for tf in available_timeframes:
            data = tf_data[tf]
            rsi = data["rsi"]
            rsi_prev = data["rsi_prev"]
            
            if position_type == "Long":
                # 롱 포지션 긍정 신호: RSI < 60 (과매수 아님), RSI 상승 중
                rsi_signal = 0.0
                if rsi < 30:  # 과매도 영역 (강한 매수 신호)
                    rsi_signal = 1.0
                elif rsi < 40:  # 저점 가능성 영역
                    rsi_signal = 0.8
                elif rsi < 50:  # 중간 이하 영역
                    rsi_signal = 0.6
                elif rsi < 60:  # 중간 영역
                    rsi_signal = 0.4
                
                # RSI 상승 추세 확인
                if rsi > rsi_prev:
                    rsi_signal += 0.2  # 상승 중인 RSI는 추가 점수
            else:
                # 숏 포지션 긍정 신호: RSI > 40 (과매도 아님), RSI 하락 중
                rsi_signal = 0.0
                if rsi > 70:  # 과매수 영역 (강한 매도 신호)
                    rsi_signal = 1.0
                elif rsi > 60:  # 고점 가능성 영역
                    rsi_signal = 0.8
                elif rsi > 50:  # 중간 이상 영역
                    rsi_signal = 0.6
                elif rsi > 40:  # 중간 영역
                    rsi_signal = 0.4
                
                # RSI 하락 추세 확인
                if rsi < rsi_prev:
                    rsi_signal += 0.2  # 하락 중인 RSI는 추가 점수
            
            # 모멘텀 신호 추가
            momentum_signals.append((min(1.0, rsi_signal), normalized_weights[tf]))
        
        # 가중 평균 모멘텀 신호 계산
        agreements["momentum"] = sum(signal * weight for signal, weight in momentum_signals)
        
        # 3. 변동성 지표 일치도 (볼린저 밴드, ATR)
        volatility_signals = []
        
        for tf in available_timeframes:
            data = tf_data[tf]
            
            # 볼린저 밴드 위치 계산 (0: 하단, 0.5: 중앙, 1: 상단)
            bb_pos = (data["close"] - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])
            
            if position_type == "Long":
                # 롱 포지션 긍정 신호: 가격이 볼린저 밴드 하단에 가까움
                if bb_pos < 0.2:  # 하단 20% 내
                    bb_signal = 1.0
                elif bb_pos < 0.4:  # 하단 40% 내
                    bb_signal = 0.7
                elif bb_pos < 0.5:  # 중앙 아래
                    bb_signal = 0.5
                else:
                    bb_signal = 0.3
            else:
                # 숏 포지션 긍정 신호: 가격이 볼린저 밴드 상단에 가까움
                if bb_pos > 0.8:  # 상단 20% 내
                    bb_signal = 1.0
                elif bb_pos > 0.6:  # 상단 40% 내
                    bb_signal = 0.7
                elif bb_pos > 0.5:  # 중앙 위
                    bb_signal = 0.5
                else:
                    bb_signal = 0.3
            
            # 볼린저 밴드 폭 (변동성 지표)
            bb_width = (data["bb_upper"] - data["bb_lower"]) / data["bb_mid"]
            
            # ATR 기반 변동성 평가 (정규화된 ATR)
            norm_atr = data["atr"] / data["close"] * 100  # ATR를 현재가 대비 백분율로 계산
            
            # 변동성 신호: 적당한 변동성이 좋음
            if norm_atr > 0.5 and norm_atr < 2.0:  # 적당한 변동성
                vol_signal = 0.8
            elif norm_atr > 2.0:  # 높은 변동성
                vol_signal = 0.5
            else:  # 낮은 변동성
                vol_signal = 0.3
            
            # 전체 변동성 신호 (볼린저 밴드 위치 70%, ATR 30%)
            volatility_strength = 0.7 * bb_signal + 0.3 * vol_signal
            volatility_signals.append((volatility_strength, normalized_weights[tf]))
        
        # 가중 평균 변동성 신호 계산
        agreements["volatility"] = sum(signal * weight for signal, weight in volatility_signals)
        
        # 4. 진동 지표 일치도 (도치안 채널)
        oscillator_signals = []
        
        for tf in available_timeframes:
            data = tf_data[tf]
            
            # 도치안 채널 확인 (있는 경우)
            if "donchian_high" in data and "donchian_low" in data:
                # 도치안 채널 내 위치 계산 (0: 하단, 0.5: 중앙, 1: 상단)
                donchian_pos = (data["close"] - data["donchian_low"]) / (data["donchian_high"] - data["donchian_low"])
                
                if position_type == "Long":
                    # 롱 포지션: 도치안 채널 하단 근처가 유리
                    if donchian_pos < 0.2:
                        donchian_signal = 1.0
                    elif donchian_pos < 0.4:
                        donchian_signal = 0.7
                    elif donchian_pos < 0.5:
                        donchian_signal = 0.5
                    else:
                        donchian_signal = 0.3
                else:
                    # 숏 포지션: 도치안 채널 상단 근처가 유리
                    if donchian_pos > 0.8:
                        donchian_signal = 1.0
                    elif donchian_pos > 0.6:
                        donchian_signal = 0.7
                    elif donchian_pos > 0.5:
                        donchian_signal = 0.5
                    else:
                        donchian_signal = 0.3
                        
                oscillator_signals.append((donchian_signal, normalized_weights[tf]))
            else:
                # 도치안 채널 데이터가 없으면 RSI 추가 활용
                # 이미 모멘텀에서 RSI를 다뤘으므로 간단하게 처리
                if position_type == "Long" and data["rsi"] < 50:
                    osc_signal = 0.7
                elif position_type == "Short" and data["rsi"] > 50:
                    osc_signal = 0.7
                else:
                    osc_signal = 0.3
                    
                oscillator_signals.append((osc_signal, normalized_weights[tf]))
        
        # 가중 평균 진동 지표 신호 계산
        agreements["oscillator"] = sum(signal * weight for signal, weight in oscillator_signals)
        
        # 5. 거래량 지표 일치도 (간단 구현)
        # 볼륨 데이터가 제한적이므로 기본 신호 사용
        agreements["volume"] = 0.7  # 기본값
        
        # 전체 지표 통합 (각 범주별 가중치 적용)
        indicator_weights = {
            "trend": 0.35,       # 추세 지표 35%
            "momentum": 0.25,    # 모멘텀 지표 25%
            "volatility": 0.20,  # 변동성 지표 20%
            "oscillator": 0.15,  # 진동 지표 15%
            "volume": 0.05       # 거래량 지표 5%
        }
        
        overall_agreement = sum(agreements[ind] * indicator_weights[ind] for ind in indicators)
        
        # 최종 일치도 점수 (0-1 범위)
        return round(overall_agreement, 2)
        
    except Exception as e:
        print(f"[오류] 타임프레임 일치도 계산 중 문제 발생: {e}")
        print(traceback.format_exc())
        return 0.5  # 오류 발생 시 중간값 반환

# 11. AI 트레이딩 결정 함수
def get_ai_trading_decision(market_data: Dict, video_id: Optional[str] = None) -> Optional[TradingDecision]:
    """GPT 기반 AI 트레이딩 결정 (멀티 타임프레임 분석 포함)"""
    try:
        position_info = get_position_info(bybit_client)
        if not position_info:
            position_info = {
                "Long": {"size": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0},
                "Short": {"size": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0}
            }
        
        gpt_data = prepare_market_data_for_gpt(market_data)
        if not gpt_data:
            raise Exception("GPT 데이터 준비 실패")

        gpt_data["current_positions"] = {
            "Long": {
                "size": position_info["Long"]["size"],
                "entry_price": position_info["Long"]["entry_price"],
                "unrealized_pnl": position_info["Long"]["unrealized_pnl"],
                "leverage": position_info["Long"].get("leverage", 0)
            },
            "Short": {
                "size": position_info["Short"]["size"],
                "entry_price": position_info["Short"]["entry_price"],
                "unrealized_pnl": position_info["Short"]["unrealized_pnl"],
                "leverage": position_info["Short"].get("leverage", 0)
            }
        }

        system_prompt = """
        You are a Bitcoin futures trading expert. Based on the latest market data, please provide an optimal trading decision following the strategy below.

        [Strategy Overview]
        1. **Multi-Timeframe Analysis:**
        - Analyze 15-minute, 1-hour, and 4-hour charts to evaluate short-term momentum, intermediate trends, and long-term trends.
        - A consistent bullish or bearish signal across all charts indicates a strong buy or sell; if signals conflict or the market is consolidating, adopt a more cautious approach.

        2. **Leverage Management:**
        - Use 10x to 15x leverage for strong signals and 3x to 5x for weak signals or consolidating markets, with a default of around 7x.

        3. **Stop Loss Setup:**
        - Implement an ATR-based dynamic stop loss—approximately 2 ATR from the entry price under normal conditions, and about 1.5 ATR in high-leverage or consolidating situations.
        - Ensure the stop loss does not conflict with major technical levels (e.g., EMA 20 or key support/resistance).

        4. **Trailing Stop:**
        - Once the position gains 2–3%, gradually adjust the stop loss to secure about 50% of the profit.

        5. **Take Profit Strategy:**
        - Exit the position in stages:
            - First TP: Target roughly 1.5–2% gain.
            - Second TP: Target roughly 3–4% gain.
            - Third TP: Target 6% or higher.
        - Adjust targets higher in trending markets and lower in consolidating markets.

        6. **Additional Entry Conditions:**
        - If an existing position shows an unrealized profit over 1% and all timeframes confirm a strong signal, consider adding to the position.
        - Total risk must not exceed 3% of the account balance.

        7. **Risk Management:**
        - Limit each trade's maximum loss to 3% of your account balance. Adjust the stop_loss_pct based on leverage so that the actual price movement percentage reflects the leverage used.

        [Final JSON Format]
        Return your trading decision as JSON in the following format:
        {
        "decision": "Long/Short/Close Long Position/Close Short Position/Hold",
        "reason": "A detailed explanation based on the analysis of the 15-minute, 1-hour, and 4-hour charts and risk management criteria.",
        "position_size": (an integer between 0 and 100),
        "leverage": (an integer between 1 and 20; omit if closing a position),
        "stop_loss_pct": (a decimal between 0.1 and 5.0)
        }

        Please base your analysis and decision on the most current market data available.
        """
        response = openai_client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(gpt_data, default=str)}
            ],
            max_completion_tokens=4096
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # 디버깅을 위한 로그 추가
        print(f"[전체 AI 응답] {response_text}")
        
        # JSON 파싱 전에 응답 텍스트 정리
        if "```json" in response_text:
            response_text = response_text[response_text.find("{"):response_text.rfind("}")+1]
        
        # JSON 파싱 부분을 아래 코드로 교체
        try:
            # JSON 부분만 추출하는 로직 추가
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                decision_data = json.loads(json_text)
            else:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[오류] JSON 파싱 실패: {e}")
            print(f"원본 응답: {response_text[:500]}...")  # 앞부분 500자만 출력
            
            # 기본 결정 반환 (오류 발생 시)
            decision_data = {
                "decision": "Hold",
                "reason": "AI 응답 파싱 실패로 안전 모드 진입",
                "position_size": 0,
                "leverage": None,
                "stop_loss_pct": 0.1
            }
        
        if "stop_loss_pct" in decision_data and decision_data["stop_loss_pct"] == 0:
            decision_data["stop_loss_pct"] = 0.1
        if "position_size" not in decision_data:
            decision_data["position_size"] = 0
        if "decision" not in decision_data:
            decision_data["decision"] = "Hold"
        if decision_data["decision"] == "Hold":
            decision_data["leverage"] = None
            
        decision = TradingDecision.model_validate(decision_data)
        
        print("\n=== AI 트레이딩 결정 (멀티 타임프레임) ===")
        print(f"결정: {decision.decision}")
        print(f"이유: {decision.reason}")
        print(f"포지션 크기: {decision.position_size}%")
        if decision.leverage:
            print(f"레버리지: {decision.leverage}x")
        else:
            print("레버리지: 없음 (Hold 상태)")
        print("=====================")
        
        return decision
        
    except Exception as e:
        print(f"[오류] AI 트레이딩 결정 중 오류 발생: {e}")
        if 'response_text' in locals():
            print(f"GPT 응답: {response_text[:200]}")
        return None

# 12. 청산 결정 및 리스크 관리 함수     
def get_liquidation_decision(conn, latest_data, market_data, current_position_type=None, new_position_type=None) -> Optional[str]:
    try:
        # 실시간 포지션 정보 사용: Bybit API에서 직접 데이터를 받아와 계산
        realtime_positions = get_position_info(bybit_client)
        if realtime_positions and current_position_type in realtime_positions and realtime_positions[current_position_type]["size"] > 0:
            pos_info = realtime_positions[current_position_type]
            entry_price = pos_info["entry_price"]
            current_price = float(latest_data['Close'])
            if current_position_type == "Long":
                current_profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                current_profit_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            current_profit_pct = 0.0

        print("\n=== 청산 결정 분석 ===")
        print(f"현재 미실현 손익: {current_profit_pct:.2f}%")        
        position_context = ""
        if current_position_type and new_position_type:
            position_context = f"\n현재 {current_position_type} 포지션 보유 중이며, {new_position_type}으로 전환 신호 감지됨."
        print("\n[현재 시장 상황]")
        print(f"- 현재 가격: ${float(latest_data['Close']):,.2f}")
        print(f"- RSI: {float(latest_data['RSI']):.2f}")
        print(f"- MACD: {float(latest_data['MACD']):.2f}")
        print(f"- MACD Signal: {float(latest_data['MACD_signal']):.2f}")
        prompt = f"""
                Your current position is in a loss.
                The current unrealized profit/loss is {current_profit_pct:.2f}%{position_context}.
                
                Based on the following market data, please determine whether it is advantageous to liquidate (Liquidate) the position or to maintain (Hold) it.
                
                Market Data:
                - Current Price: ${float(latest_data['Close']):,.2f}
                - RSI: {float(latest_data['RSI']):.2f}
                - MACD: {float(latest_data['MACD']):.2f}
                - MACD Signal: {float(latest_data['MACD_signal']):.2f}
                - Bollinger Band Upper: ${float(latest_data['BB_upper']):,.2f}
                - Bollinger Band Lower: ${float(latest_data['BB_lower']):,.2f}
                - Recent News: {json.dumps(market_data.get('news', []), ensure_ascii=False)}
                
                Please provide your answer in the following JSON format:
                {{
                    "decision": "Liquidate/Hold",
                    "reason": "Explanation for the decision",
                    "confidence": "Confidence level (0-100)"
                }}
                """

        response = openai_client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are an expert in cryptocurrency trading strategy analysis."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.3,
            max_completion_tokens=600  # max_tokens 대신 max_completion_tokens 사용
        )

        response_text = response.choices[0].message.content.strip()
        # 디버깅을 위한 로그 추가
        print(f"[전체 AI 응답] {response_text}")

        try:
            # JSON 부분만 추출하는 로직 추가
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                decision_data = json.loads(json_text)
            else:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[오류] JSON 파싱 실패: {e}")
            print(f"원본 응답: {response_text[:500]}...")  # 앞부분 500자만 출력
            
            # 기본 결정 반환 (오류 발생 시)
            return "Hold"  # 오류 시 안전하게 포지션 유지

        print("\n[AI 판단 결과]")
        print(f"결정: {'청산' if decision_data['decision'] == 'Liquidate' else '유지'}")
        print(f"이유: {decision_data['reason']}")
        print(f"확신도: {decision_data['confidence']}%")
        print("=====================")
        return decision_data['decision']
    except Exception as e:
        print(f"[오류] 청산 여부 결정 AI 호출 중 문제 발생: {e}")
        return None
        

def analyze_position_risk(position_info: Dict, market_data: Dict) -> Optional[Dict]:
    """현재 포지션의 리스크 분석 (진입가 대비 현재가, ATR 및 변동성 반영)"""
    try:
        current_price = float(market_data['candlestick'].iloc[-1]['Close'])
        atr = float(market_data['candlestick'].iloc[-1]['ATR'])
        # 최근 20개 종가 데이터로 변동성 계산
        volatility = market_data['candlestick']['Close'].tail(20).pct_change().std()
        
        risk_analysis = {
            "Long": {
                "distance_to_entry": ((current_price - position_info["Long"]["entry_price"]) / position_info["Long"]["entry_price"]) * 100 if position_info["Long"]["size"] > 0 else 0,
                "risk_to_atr_ratio": (position_info["Long"]["size"] * current_price * atr) / 100,
                "volatility_risk": volatility * 100,
                "total_risk_score": 0  # 초기화
            },
            "Short": {
                "distance_to_entry": ((position_info["Short"]["entry_price"] - current_price) / position_info["Short"]["entry_price"]) * 100 if position_info["Short"]["size"] > 0 else 0,
                "risk_to_atr_ratio": (position_info["Short"]["size"] * current_price * atr) / 100,
                "volatility_risk": volatility * 100,
                "total_risk_score": 0  # 초기화
            }
        }
        
        # 각 포지션의 총 리스크 점수 계산
        for position_type in ["Long", "Short"]:
            if position_info[position_type]["size"] > 0:
                # 레버리지 가져오기
                leverage = position_info[position_type].get("leverage", 1)
                
                # 기본 리스크 점수 계산
                base_risk_score = (
                    abs(risk_analysis[position_type]["distance_to_entry"]) * 0.4 +
                    risk_analysis[position_type]["risk_to_atr_ratio"] * 0.3 +
                    risk_analysis[position_type]["volatility_risk"] * 0.3
                )
                
                # 레버리지에 따른 리스크 가중치 계산
                leverage_multiplier = 1.0
                if leverage > 10:
                    leverage_multiplier = 2.0
                elif leverage > 5:
                    leverage_multiplier = 1.5
                
                # 최종 리스크 점수 계산
                risk_analysis[position_type]["total_risk_score"] = base_risk_score * leverage_multiplier
                
                # 디버깅을 위한 로깅 추가
                print(f"\n[{position_type} 포지션 리스크 분석]")
                print(f"레버리지: {leverage}x")
                print(f"기본 리스크 점수: {base_risk_score:.2f}")
                print(f"레버리지 가중치: {leverage_multiplier:.1f}")
                print(f"최종 리스크 점수: {risk_analysis[position_type]['total_risk_score']:.2f}")

        return risk_analysis
        
    except Exception as e:
        print(f"[오류] 포지션 리스크 분석 중 문제 발생: {e}")
        print(f"상세 오류: {str(e)}")
        return None
    
def update_strategy_parameters_from_reflection(conn) -> Optional[Dict]:
    """
    DB에 저장된 최근의 반성 데이터를 기반으로 전략 파라미터 업데이트
    """
    c = conn.cursor()
    c.execute("""
        SELECT reflection 
        FROM trades 
        WHERE reflection IS NOT NULL 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    result = c.fetchone()
    if result:
        reflection_text = result[0]
        win_rate = 0.5
        avg_leverage = 1
        max_drawdown = 0
        try:
            if "승률:" in reflection_text:
                win_rate = float(reflection_text.split("승률: ")[1].split("%")[0]) / 100
        except:
            pass
        try:
            if "평균 레버리지:" in reflection_text:
                avg_leverage = float(reflection_text.split("평균 레버리지: ")[1].split("x")[0])
        except:
            pass
        try:
            if "최대 손실폭:" in reflection_text:
                max_drawdown = float(reflection_text.split("최대 손실폭: ")[1].split("%")[0])
        except:
            pass
        new_params = {
            "position_size_multiplier": 1.0,
            "leverage_multiplier": 1.0,
            "stop_loss_pct": 2.0
        }
        if win_rate < 0.5:
            new_params["position_size_multiplier"] *= 0.8
        if avg_leverage > 8:
            new_params["leverage_multiplier"] *= 0.9
        if max_drawdown > 5:
            new_params["stop_loss_pct"] = min(new_params["stop_loss_pct"], 1.0)
        print("자동 전략 업데이트된 파라미터:", new_params)
        return new_params
    else:
        print("최근 반성 결과가 없습니다.")
        return None    
    
# 13. 거래 실행 관련 함수들
def place_order(order_type: str, symbol: str, qty: float, position_idx: int) -> bool:
    """
    주문 실행 (실제 거래 주문)
    order_type: "buy" 또는 "sell"
    symbol: 거래 심볼 (예: "BTCUSDT")
    qty: 주문 수량
    position_idx: 포지션 인덱스 (롱: 1, 숏: 2)
    """
    MIN_QTY = 0.001
    qty = round(qty, 3)
    if qty < MIN_QTY:
        print(f"[오류] 최소 거래량 미달: {qty} < {MIN_QTY}")
        return False
    try:
        order_params = {
            "category": "linear",
            "symbol": symbol,
            "side": "Buy" if order_type.lower() == "buy" else "Sell",
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "positionIdx": position_idx
        }
        response = bybit_client.place_order(**order_params)
        if response["retCode"] == 0:
            order_id = response['result']['orderId']
            print(f"[성공] 주문 완료! 주문 ID: {order_id}")
            return True
        else:
            print(f"[오류] 주문 실패: {response['retMsg']} (ErrCode: {response['retCode']})")
            return False
    except Exception as e:
        print(f"[오류] 주문 실행 중 문제 발생: {e}")
        return False

def execute_close_position(position_type: str, qty: float, conn, market_data: Dict, decision: TradingDecision) -> bool:
    """포지션 청산 실행 및 거래 종료 상태 업데이트"""
    try:
        position_idx = 1 if position_type == "Long" else 2
        latest_data = market_data['candlestick'].iloc[-1]
        current_price = float(latest_data['Close'])
               
        # 모든 조건부 주문(스탑로스, 테이크프로핏) 취소
        cancel_all_conditional_orders(SYMBOL, position_idx)
        print(f"[정보] {position_type} 포지션 청산 전 모든 조건부 주문 취소 완료")
        
        # 청산 주문 실행
        close_success = place_order("sell" if position_type == "Long" else "buy", SYMBOL, qty, position_idx=position_idx)
        
        if close_success and conn:
            try:
                c = conn.cursor()
                
                # 최근 거래 ID 조회
                c.execute("""
                    SELECT trade_id, entry_price 
                    FROM trades 
                    WHERE position_type = ? AND trade_status = 'Open'
                    ORDER BY timestamp DESC LIMIT 1
                """, (position_type,))
                
                trade_data = c.fetchone()
                if trade_data:
                    trade_id, entry_price = trade_data
                    
                    
                    # 모든 관련 TP 주문 상태 취소로 변경 (DB)
                    c.execute("""
                        UPDATE take_profit_orders
                        SET status = 'Cancelled'
                        WHERE trade_id = ? AND (status = 'Active' OR status = 'Pending')
                    """, (trade_id,))
                    
                    # 수익률 계산
                    if position_type == "Long":
                        profit_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        profit_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    # 거래 상태 업데이트
                    c.execute("""
                        UPDATE trades 
                        SET trade_status = 'Closed',
                            exit_price = ?,
                            profit_loss = ?,
                            success = ?
                        WHERE trade_id = ? AND trade_status = 'Open'
                    """, (current_price, profit_pct, 1 if profit_pct > 0 else 0, trade_id))
                    
                    conn.commit()
                    print(f"[성공] 거래 {trade_id} 청산 완료 (수익률: {profit_pct:.2f}%)")
                    
                    # 익절 주문 초기화 (예: stop loss에 의해 청산된 경우)
                    reset_take_profit_orders(conn, trade_id)
                    
                    # 새로운 거래 기록 저장
                    estimated_fee = qty * current_price * FEE_RATE
                    close_decision = TradingDecision(
                        decision=f"Close {position_type} Position",
                        reason=f"포지션 청산: {decision.reason}",
                        position_size=0,
                        leverage=None
                    )
                    save_trade_to_db(conn, market_data, close_decision, get_wallet_balance(), estimated_fee, success=1)
                    
                return True
            except Exception as e:
                print(f"[경고] 청산 데이터 저장 실패: {e}")
                conn.rollback()
                return False
        return close_success
    except Exception as e:
        print(f"[오류] 포지션 청산 중 문제 발생: {e}")
        return False

def sync_positions_with_exchange(conn, exchange_client, symbol: str) -> bool:
    """
    교환소의 포지션 정보와 로컬 DB를 동기화하는 함수
    
    Args:
        conn: 데이터베이스 연결 객체
        exchange_client: 교환소 API 클라이언트
        symbol: 거래 심볼
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 교환소에서 현재 포지션 정보 가져오기
        open_positions = exchange_client.get_positions(symbol=symbol)
        
        # DB에서 현재 열린 포지션 가져오기
        cursor = conn.cursor()
        cursor.execute('''
            SELECT trade_id, position_type, entry_price, size, stop_loss, take_profit 
            FROM trades 
            WHERE status = 'OPEN' AND symbol = ?
        ''', (symbol,))
        db_positions = {row['trade_id']: dict(row) for row in cursor.fetchall()}
        
        # 교환소에는 없지만 DB에 있는 포지션 처리 (이미 닫힌 포지션)
        exchange_position_ids = [pos['trade_id'] for pos in open_positions]
        for trade_id, db_pos in db_positions.items():
            if trade_id not in exchange_position_ids:
                # 해당 포지션이 교환소에 없음 -> 이미 닫혔을 가능성
                print(f"[정보] 교환소에 없는 포지션 감지: {trade_id}, DB 상태 업데이트 필요")
                
                # 포지션 종료 처리 (exchange_client에서 포지션 이력 조회 필요)
                position_history = exchange_client.get_position_history(trade_id)
                
                if position_history and 'close_time' in position_history:
                    # 포지션 종료 처리
                    cursor.execute('''
                        UPDATE trades
                        SET status = 'CLOSED',
                            exit_price = ?,
                            pnl = ?,
                            close_time = ?
                        WHERE trade_id = ?
                    ''', (
                        position_history.get('exit_price', 0),
                        position_history.get('pnl', 0),
                        position_history.get('close_time'),
                        trade_id
                    ))
                    
                    # 종료 원인 기록
                    close_reason = position_history.get('close_reason', 'unknown')
                    track_position_close_reason(conn, trade_id, close_reason)
        
        # 교환소에는 있지만 DB에 없는 포지션 처리 (외부에서 추가된 포지션)
        for pos in open_positions:
            if pos['trade_id'] not in db_positions:
                print(f"[정보] DB에 없는 포지션 감지: {pos['trade_id']}, 포지션 추가 필요")
                # DB에 포지션 추가 로직 구현 필요
                cursor.execute('''
                    INSERT INTO trades (
                        trade_id, symbol, position_type, entry_price, size, 
                        stop_loss, take_profit, status, entry_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                ''', (
                    pos['trade_id'],
                    symbol,
                    pos['position_type'],
                    pos['entry_price'],
                    pos['size'],
                    pos.get('stop_loss', 0),
                    pos.get('take_profit', 0),
                    pos.get('entry_time', datetime.now().isoformat())
                ))
        
        # 변경 사항 저장
        conn.commit()
        print(f"[정보] 포지션 동기화 완료: {symbol}")
        return True
        
    except Exception as e:
        print(f"[오류] 포지션 동기화 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'conn' in locals():
            conn.rollback()
        return False


def evaluate_additional_position_entry(current_positions, market_data, decision):
    """기존 포지션이 있을 때 추가 진입 여부를 평가"""
    try:
        if decision.decision not in ["Long", "Short"]:
            return False, "추가 진입 신호 없음", None
            
        position_type = decision.decision
        current_size = current_positions[position_type]["size"]
        
        if current_size == 0:
            return True, "신규 포지션 진입", None
            
        # 1. 현재 포지션 수익 확인
        entry_price = current_positions[position_type]["entry_price"]
        current_price = float(market_data['candlestick'].iloc[-1]['Close'])
        leverage = current_positions[position_type].get("leverage", 1)
        
        if position_type == "Long":
            profit_pct = ((current_price - entry_price) / entry_price) * 100 * leverage
        else:
            profit_pct = ((entry_price - current_price) / entry_price) * 100 * leverage
        
        # 2. 멀티 타임프레임 신호 일치 확인
        frames = ["15", "60", "240"]
        signals = {tf: [] for tf in frames}
        
        for tf in frames:
            latest = market_data['timeframes'][tf].iloc[-1]
            # 롱 포지션 신호 확인
            if position_type == "Long":
                if latest['RSI'] < 70 and latest['RSI'] > 40:
                    signals[tf].append(True)
                if latest['MACD'] > latest['MACD_signal']:
                    signals[tf].append(True)
                if latest['Close'] > latest['EMA_20']:
                    signals[tf].append(True)
            # 숏 포지션 신호 확인
            else:
                if latest['RSI'] > 30 and latest['RSI'] < 60:
                    signals[tf].append(True)
                if latest['MACD'] < latest['MACD_signal']:
                    signals[tf].append(True)
                if latest['Close'] < latest['EMA_20']:
                    signals[tf].append(True)
        
        # 각 타임프레임별 긍정적 신호 비율 계산
        tf_scores = {tf: sum(signals[tf]) / 3 for tf in frames}
        avg_score = sum(tf_scores.values()) / len(tf_scores)
        
        # 신호 강도에 따른 레버리지 최적화
        optimal_leverage = decide_optimal_leverage(market_data, avg_score)
        
        # 3. 변동성 확인
        volatility = market_data['candlestick']['Close'].pct_change().rolling(14).std().iloc[-1] * 100
        
        # 4. 판단 로직
        if profit_pct > 1.0 and avg_score > 0.6 and volatility < 2.0:
            # 최대 3개 포지션으로 제한
            if current_size < 3 * MIN_QTY:
                # 포지션 크기 조정 (기존 대비 50-80% 크기로 제한)
                adjusted_size = min(current_size * 0.8, decision.position_size / 100 * get_wallet_balance() / current_price)
                if adjusted_size >= MIN_QTY:
                    return True, f"추가 진입 조건 충족: 수익률 {profit_pct:.1f}%, 신호 점수 {avg_score:.1f}, 변동성 {volatility:.1f}%", optimal_leverage
        
        # 5. 특별 케이스: 매우 강한 신호 (모든 타임프레임 일치 + 매우 낮은 변동성)
        if all(score > 0.8 for score in tf_scores.values()) and volatility < 1.0:
            return True, "매우 강한 신호로 인한 추가 진입", optimal_leverage
            
        return False, f"추가 진입 조건 불충족: 수익률 {profit_pct:.1f}%, 신호 점수 {avg_score:.1f}", None
        
    except Exception as e:
        print(f"[오류] 추가 포지션 평가 중 문제 발생: {e}")
        return False, "오류로 인한 추가 진입 중단", None

def execute_trading_logic(decision: TradingDecision, position_size: int, usdt_balance: float, leverage: Optional[int], conn, market_data: Dict) -> bool:
    """실제 거래 실행 로직 (AI 판단 중심)"""
    try:
        print("\n=== 트레이딩 로직 시작 ===")
        print(f"결정: {decision.decision}")
        print(f"포지션 크기: {position_size}%")
        print(f"레버리지: {leverage}x")
        latest_data = market_data['candlestick'].iloc[-1]
        print(f"\n[잔액 확인] USDT 잔액: {usdt_balance}")
        
        if usdt_balance <= 0:
            print("[오류] 잔액 부족")
            return False
            
        print("\n[포지션 조회 중...]")
        current_positions = get_current_positions(bybit_client)
        if current_positions is None:
            print("[오류] 현재 포지션 조회 실패")
            return False
            
        print("\n=== 포지션 분석 ===")
        for pos_type, size in current_positions.items():
            if size > 0:
                print(f"- {pos_type} 포지션: {size} BTC")
                
        if decision.decision in ["Long", "Short"]:
            position_type = decision.decision
            # execute_trading_logic 함수 내 추가 진입 평가 부분 (일부만 표시)
            if current_positions[position_type] > 0:
                position_info = get_position_info(bybit_client)
                should_add, reason, dynamic_leverage = evaluate_additional_position_entry(
                    position_info, market_data, decision
                )
                
                if should_add:
                    print(f"[정보] 기존 {position_type} 포지션에 추가 진입: {reason}")
                    # 최적 레버리지 사용 (없으면 기존 decision의 레버리지 사용)
                    actual_leverage = dynamic_leverage if dynamic_leverage else decision.leverage
                    return execute_entry_logic(decision, position_size, usdt_balance, actual_leverage, conn, market_data, latest_data)
                else:
                    print(f"[정보] 이미 {position_type} 포지션 보유 중입니다. 추가 진입 없이 유지합니다: {reason}")
                    return True
                
            opposite_position = "Short" if decision.decision == "Long" else "Long"
            if current_positions[opposite_position] > 0:
                position_info = get_position_info(bybit_client)
                
                # 신호 강도 평가
                signal_strength = evaluate_signal_strength(market_data, decision.decision)
                
                # 현재 이익률 계산
                current_profit = 0
                if position_info and position_info[opposite_position]["unrealized_pnl"] > 0:
                    entry_price = position_info[opposite_position]["entry_price"]
                    current_price = float(latest_data['Close'])
                    leverage = position_info[opposite_position].get("leverage", 1)
                    
                    if opposite_position == "Long":
                        current_profit = ((current_price - entry_price) / entry_price) * 100 * leverage
                    else:
                        current_profit = ((entry_price - current_price) / entry_price) * 100 * leverage
                    
                    # 거래 비용 (진입+청산 수수료)
                    trading_cost = FEE_RATE * 2 * 100  # 예: 0.12%를 백분율로
                    
                    # 강한 반전 신호이고 충분한 이익이 있으면 전환
                    if signal_strength > 0.8 and current_profit > trading_cost * 1.5:
                        print(f"[정보] 강한 {decision.decision} 신호 감지: {signal_strength:.2f}")
                        print(f"[정보] 현재 {opposite_position} 이익({current_profit:.2f}%)이 거래비용({trading_cost:.2f}%)보다 충분히 높음")
                        print(f"[결정] {opposite_position} 포지션 청산 후 {decision.decision} 전환")
                        
                        # 기존 포지션 청산
                        close_success = execute_close_position(opposite_position, current_positions[opposite_position], conn, market_data, decision)
                        if close_success:
                            time.sleep(1)
                            # 새 포지션 진입
                            return execute_entry_logic(decision, position_size, usdt_balance, leverage, conn, market_data, latest_data)
                    else:
                        print(f"[정보] {opposite_position} 포지션 이익 중({current_profit:.2f}%), 신호 강도({signal_strength:.2f})로 유지")
                        return True
                else:
                    # 손실 중이면 기존 로직대로 AI에게 청산 결정 요청
                    liquidation_decision = get_liquidation_decision(
                        conn, latest_data, market_data,
                        current_position_type=opposite_position,
                        new_position_type=decision.decision
                    )
                    if liquidation_decision == "Liquidate":
                        print(f"[정보] AI 판단: {opposite_position} 포지션 청산 후 {decision.decision} 전환")
                        close_success = execute_close_position(opposite_position, current_positions[opposite_position], conn, market_data, decision)
                        if close_success:
                            time.sleep(1)
                            print(f"\n[신규 {decision.decision} 진입 시도]")
                            return execute_entry_logic(decision, position_size, usdt_balance, leverage, conn, market_data, latest_data)
                    else:
                        print(f"[정보] AI 판단: 현재 {opposite_position} 포지션 유지")
                        return True

            print(f"\n[신규 {decision.decision} 포지션 진입 시도]")
            print(f"- 포지션 크기: {position_size}%")
            print(f"- 레버리지: {leverage}x")
            print(f"- 잔액: {usdt_balance} USDT")
            print(f"- 현재가: {float(latest_data['Close'])} USDT")
            return execute_entry_logic(decision, position_size, usdt_balance, leverage, conn, market_data, latest_data)
            
        elif decision.decision in ["Close Long Position", "Close Short Position"]:
            position_type = "Long" if decision.decision == "Close Long Position" else "Short"
            qty = current_positions[position_type]
            if qty > 0:
                print(f"\n[포지션 청산 시도] {position_type} {qty} BTC")
                return execute_close_position(position_type, qty, conn, market_data, decision)
            else:
                print(f"[정보] 청산할 {position_type} 포지션이 없습니다")
                return True
                
        elif decision.decision == "Hold":
            print("[정보] 현재 포지션 유지")
            if conn:
                try:
                    hold_decision = TradingDecision(
                        decision="Hold",
                        reason="시장 상황에 따라 현재 포지션 유지",
                        position_size=0,
                        leverage=None
                    )
                    save_trade_to_db(conn, market_data, hold_decision, usdt_balance, estimated_fee=0, success=1)
                except Exception as e:
                    print(f"[경고] Hold 데이터 저장 실패: {e}")
            return True
            
        print("\n=== 트레이딩 로직 종료 ===")
        return True
        
    except Exception as e:
        print(f"[오류] 거래 실행 중 문제 발생: {e}")
        print(traceback.format_exc())
        if conn:
            try:
                save_trade_to_db(conn, market_data, decision, usdt_balance, estimated_fee=0, success=0)
            except Exception as db_error:
                print(f"[경고] 실패 데이터 저장 중 오류: {db_error}")
        return False


def adjust_position_size(position_size: int, volatility: float, trend_strength: float, stop_loss_pct: float = None) -> int:
    """
    시장 상황과 손절폭에 따른 포지션 크기 동적 조정
    """
    try:
        # 기본 사이즈는 입력값 유지
        adjusted_size = position_size
        
        print(f"\n[포지션 크기 조정]")
        print(f"기존 크기: {position_size}%")
        print(f"현재 변동성: {volatility:.2f}%")
        print(f"트렌드 강도: {trend_strength:.2f}")
        if stop_loss_pct:
            print(f"설정된 손절폭: {stop_loss_pct:.2f}%")
        
        # 변동성에 따른 조정
        if volatility > 2.0:  # 매우 높은 변동성
            adjusted_size = int(position_size * 0.7)  # 30% 감소
            print(f"매우 높은 변동성: 크기 30% 감소 -> {adjusted_size}%")
        elif volatility > 1.5:  # 높은 변동성
            adjusted_size = int(position_size * 0.8)  # 20% 감소
            print(f"높은 변동성: 크기 20% 감소 -> {adjusted_size}%")
        elif volatility < 0.5:  # 매우 낮은 변동성
            adjusted_size = int(position_size * 1.2)  # 20% 증가
            print(f"낮은 변동성: 크기 20% 증가 -> {adjusted_size}%")
            
        # 트렌드 강도에 따른 조정
        if trend_strength > 40:  # 강한 트렌드
            prev_size = adjusted_size
            adjusted_size = int(adjusted_size * 1.1)  # 10% 증가
            print(f"강한 트렌드: 크기 10% 증가 ({prev_size}% -> {adjusted_size}%)")
        elif trend_strength < 20:  # 약한 트렌드
            prev_size = adjusted_size
            adjusted_size = int(adjusted_size * 0.9)  # 10% 감소
            print(f"약한 트렌드: 크기 10% 감소 ({prev_size}% -> {adjusted_size}%)")
            
        # 손절폭에 따른 추가 조정
        if stop_loss_pct:
            # 손절폭이 클수록 포지션 크기를 줄임
            if stop_loss_pct > 2.0:
                size_modifier = 2.0 / stop_loss_pct  # 손절폭이 2% 이상이면 비례해서 감소
                prev_size = adjusted_size
                adjusted_size = int(adjusted_size * size_modifier)
                print(f"큰 손절폭({stop_loss_pct:.2f}%): 크기 조정 ({prev_size}% -> {adjusted_size}%)")
            
        # 최소/최대 제한
        adjusted_size = max(10, min(adjusted_size, 50))  # 10~50% 범위 제한
        print(f"최종 조정된 크기: {adjusted_size}%")
        
        return adjusted_size
        
    except Exception as e:
        print(f"[오류] 포지션 크기 조정 중 문제 발생: {e}")
        return position_size  # 오류 발생 시 원래 크기 반환

def decide_optimal_leverage(market_data, signals_strength):
    """신호 강도에 따른 최적 레버리지 결정"""
    try:
        base_leverage = 5  # 기본 레버리지
        adx = float(market_data['timeframes']["60"].iloc[-1]['ADX'])
        volatility = market_data['timeframes']["15"]['Close'].pct_change().rolling(14).std().iloc[-1] * 100
        
        # 신호 강도에 따른 레버리지 조정
        if signals_strength > 0.8:  # 매우 강한 신호 (80% 이상의 지표가 일치)
            optimal_leverage = min(15, base_leverage * 2)  # 최대 15x로 제한
        elif signals_strength > 0.6:  # 강한 신호
            optimal_leverage = min(10, int(base_leverage * 1.5))
        elif signals_strength < 0.4:  # 약한 신호
            optimal_leverage = max(2, int(base_leverage * 0.5))
        else:  # 보통 신호
            optimal_leverage = base_leverage
        
        # 변동성에 따른 추가 조정
        if volatility > 2.0:  # 높은 변동성
            optimal_leverage = max(2, int(optimal_leverage * 0.7))
        elif volatility < 0.5:  # 낮은 변동성
            optimal_leverage = min(15, int(optimal_leverage * 1.2))
        
        # ADX(추세 강도)에 따른 조정
        if adx > 30:  # 강한 추세
            optimal_leverage = min(15, int(optimal_leverage * 1.2))
        elif adx < 15:  # 약한 추세
            optimal_leverage = max(2, int(optimal_leverage * 0.8))
        
        print(f"\n[레버리지 최적화]")
        print(f"- 신호 강도: {signals_strength:.2f}")
        print(f"- 변동성: {volatility:.2f}%")
        print(f"- ADX(추세 강도): {adx:.2f}")
        print(f"- 최적 레버리지: {max(1, min(20, optimal_leverage))}x")
        
        return max(1, min(20, optimal_leverage))  # 1-20x 범위 내로 제한
    except Exception as e:
        print(f"[오류] 레버리지 최적화 중 문제 발생: {e}")
        return 5  # 오류 발생 시 기본 레버리지 반환

def execute_entry_logic(decision: TradingDecision, position_size: int, usdt_balance: float, leverage: Optional[int], conn, market_data: Dict, latest_data) -> bool:
    """진입(Long/Short) 로직 실행 - 향상된 익절 전략 사용"""
    try:
        print("\n=== 진입 로직 시작 ===")
        position_idx = 1 if decision.decision == "Long" else 2

        # 트레이드 ID 생성 (추가)
        trade_id = str(uuid.uuid4())
        print(f"[정보] 생성된 거래 ID: {trade_id}")

        # 레버리지 설정
        if leverage and not set_leverage(symbol=SYMBOL, leverage=leverage):
            print("[오류] 레버리지 설정 실패")
            return False

        # 기존 스탑로스 주문 취소
        cancel_stop_orders(SYMBOL, position_idx)

        # 변동성과 트렌드 강도에 따른 포지션 크기 조정
        volatility = latest_data['ATR'] / latest_data['Close'] * 100
        trend_strength = float(latest_data['ADX'])
        adjusted_position_size = adjust_position_size(position_size, volatility, trend_strength)
        print(f"[정보] 조정된 포지션 크기: {adjusted_position_size}% (원래: {position_size}%)")

        # 기본 할당액 계산 (수수료 반영)
        fee_adjusted_balance = usdt_balance * (1 - FEE_RATE)
        allocation = (adjusted_position_size / 100) * fee_adjusted_balance * (leverage or 1)

        # [추가] 재투자 전략 적용: 현재 포지션 미실현 수익의 일부를 추가 할당
        current_profit_allocation = 0
        position_info = get_position_info(bybit_client)
        if position_info and position_info[decision.decision]["size"] > 0:
            current_profit = position_info[decision.decision]["unrealized_pnl"]
            # 재투자 금액은 미실현 수익 * 재투자 비율
            current_profit_allocation = current_profit * REINVESTMENT_RATIO
            print(f"[정보] 재투자 할당액: {current_profit_allocation:.2f} USDT (미실현 수익의 {REINVESTMENT_RATIO*100:.0f}%)")
        allocation += current_profit_allocation

        current_price = float(latest_data['Close'])
        qty = round(allocation / current_price, 3)

        print(f"\n[주문 정보]")
        print(f"현재가: ${current_price:,.2f}")
        print(f"총 할당액: {allocation:.4f} USDT")
        print(f"수량: {qty} BTC")
        print(f"레버리지: {leverage}x")

        if qty < MIN_QTY:
            print(f"[오류] 최소 거래량 미달: {qty} < {MIN_QTY}")
            return False

        order_type = "buy" if decision.decision == "Long" else "sell"
        order_success = place_order(order_type, SYMBOL, qty, position_idx=position_idx)

        if order_success:
            print("\n[정보] 주문 성공, 포지션 확인 중...")
            time.sleep(1)

            position_info = get_position_info(bybit_client)
            actual_stop_loss = None
            
            if position_info and position_info[decision.decision]["size"] > 0:
                entry_price = position_info[decision.decision]["entry_price"]
                
                # 여기서 실제 포지션의 레버리지를 확인하고 사용
                actual_leverage = position_info[decision.decision].get("leverage", leverage)
                if not actual_leverage or actual_leverage <= 0:
                    actual_leverage = leverage or 1
                
                print(f"[정보] 포지션 레버리지 확인: {actual_leverage}x")
                
                # 동적 스탑로스 계산
                dynamic_stop_price = calculate_dynamic_stop_loss(
                    entry_price, 
                    market_data['timeframes']["15"],
                    decision.decision,
                    actual_leverage,
                    position_size_pct=adjusted_position_size,
                    wallet_balance=usdt_balance
                )
                
                print(f"[정보] 동적 손절매 가격: ${dynamic_stop_price:,.2f}")

                stop_loss_pct = abs((entry_price - dynamic_stop_price) / entry_price * 100)
                if set_stop_loss(SYMBOL, qty, entry_price, current_price, stop_loss_pct, decision.decision):
                    print(f"[성공] 손절매 설정 완료 ({stop_loss_pct:.2f}%)")
                    
                    # API에서 실제 설정된 스탑로스 가격 조회
                    time.sleep(1)  # API 응답을 위한 대기
                    actual_stop_loss = get_stop_loss_price(SYMBOL, position_idx)
                    if actual_stop_loss:
                        print(f"[정보] API에서 확인된 실제 스탑로스 가격: ${actual_stop_loss:,.2f}")
                    else:
                        print("[경고] API에서 스탑로스 가격을 확인할 수 없습니다. 계산된 가격을 사용합니다.")
                        actual_stop_loss = dynamic_stop_price

                # 거래 정보 DB에 저장
                if conn:
                    estimated_fee = allocation * FEE_RATE
                    
                    # 거래 정보 저장
                    save_trade_to_db(conn, market_data, decision, usdt_balance, estimated_fee, 
                                   success=1, actual_stop_price=actual_stop_loss, trade_id=trade_id)
                    
                    print(f"[정보] 거래 ID 생성: {trade_id}")
                    
                # 추가 매수 여부 확인
                is_additional_entry = False
                additional_qty = None

                if position_info and position_info[decision.decision]["size"] > 0:
                    is_additional_entry = True
                    # 실제 추가된 수량만 계산
                    previous_size = position_info[decision.decision]["size"]
                    additional_qty = qty  # 새로 추가되는 수량
                    
                    print(f"[정보] 추가 매수 감지: 기존 포지션 {previous_size} BTC, 추가 수량 {additional_qty} BTC")

                # 향상된 익절 설정 함수 호출
                # 비교: 기존 코드는 set_take_profit_enhanced를 호출했지만,
                # 이제 set_take_profit_enhanced_improved를 호출하여 향상된 기능 사용
                if set_take_profit_enhanced_improved(
                        SYMBOL, qty, entry_price, decision.decision, 
                        market_data, leverage, conn, trade_id, 
                        decision, is_additional_entry, additional_qty):
                    print("[성공] 향상된 이익실현 설정 완료")    

            else:
                # 포지션 생성 실패 시에도 거래 정보는 저장
                if conn:
                    estimated_fee = allocation * FEE_RATE
                    save_trade_to_db(conn, market_data, decision, usdt_balance, estimated_fee, success=1, trade_id=trade_id)

            print("\n=== 진입 완료 ===")
            return True
        else:
            print("[오류] 주문 실행 실패")
            if conn:
                save_trade_to_db(conn, market_data, decision, usdt_balance, 0, success=0, trade_id=trade_id)
            return False

    except Exception as e:
        print(f"[오류] 진입 로직 실행 중 예외 발생: {e}")
        print(traceback.format_exc())
        if conn:
            try:
                save_trade_to_db(conn, market_data, decision, usdt_balance, 0, success=0, trade_id=trade_id)
            except:
                pass
        return False

##### 이 이후에는 GPT를 통한 거래 결과 기록, 분석 및 반성 함수들 #####
def save_trade_to_db(conn, market_data: dict, decision: TradingDecision, wallet_balance: float, 
                    estimated_fee: float = 0, success: int = 1, 
                    actual_stop_price: Optional[float] = None, 
                    trade_id: Optional[str] = None) -> bool:
    """거래 데이터를 DB에 저장"""

    try:
        # trade_id가 None일 경우에만 새로 생성 (기존 인자 그대로 사용)
        if trade_id is None:
            trade_id = str(uuid.uuid4())    
        
        latest_data = market_data['candlestick'].iloc[-1]
        c = conn.cursor()
        current_price = float(latest_data['Close'])
        
        def calculate_confidence_score(data):
            """AI 신뢰도 점수 계산"""
            indicators = [
                abs(float(data['RSI']) - 50),
                abs(float(data['MACD'])),
                abs(float(data['MACD_signal']))
            ]
            return max(0, min(100, 100 - sum(indicators)))
        
        def get_entry_conditions(data):
            """진입 조건 데이터 생성"""
            return json.dumps({
                'rsi': float(data['RSI']),
                'macd': float(data['MACD']),
                'ema_20': float(data['EMA_20']),
                'ema_50': float(data['EMA_50'])
            })
        
        def get_exit_conditions(data):
            """청산 조건 데이터 생성"""
            return json.dumps({
                'rsi': float(data['RSI']),
                'macd': float(data['MACD']),
                'bb_upper': float(data['BB_upper']),
                'bb_lower': float(data['BB_lower'])
            })
        
        # 기본 데이터 초기화
        leverage_value = 1  # 기본값 설정
        position_type = ""
        entry_price = 0.0
        exit_price = 0.0
        trade_status = ""
        profit_loss = 0.0
        position_duration = 0
        entry_conditions = ""
        exit_conditions = ""
        trading_analysis_ai = ""
        reflection = ""
        stop_loss_price = None
        stop_loss_pct = 0.0
        
        # 거래 유형에 따른 데이터 설정
        if decision.decision in ["Long", "Short"]:
            leverage_value = decision.leverage if decision.leverage is not None else 1
            position_type = decision.decision
            entry_price = current_price
            trade_status = "Open"
            entry_conditions = get_entry_conditions(latest_data)
            
            # 스탑로스 정보 설정
            if actual_stop_price:
                stop_loss_price = actual_stop_price
                # 실제 스탑로스 가격으로 비율 역산
                if entry_price > 0:
                    if position_type == "Long":
                        stop_loss_pct = ((entry_price - stop_loss_price) / entry_price) * 100
                    else:  # Short
                        stop_loss_pct = ((stop_loss_price - entry_price) / entry_price) * 100
            else:
                # 설정된 스탑로스 비율 사용
                stop_loss_pct = decision.stop_loss_pct or 0.1
                # 비율로 스탑로스 가격 계산
                if position_type == "Long":
                    stop_loss_price = round(entry_price * (1 - stop_loss_pct/100), 1)
                else:  # Short
                    stop_loss_price = round(entry_price * (1 + stop_loss_pct/100), 1)
            
        elif decision.decision in ["Close Long Position", "Close Short Position"]:
            position_type = "Long" if decision.decision == "Close Long Position" else "Short"
            
            # 기존 포지션 정보 조회
            c.execute("""
                SELECT leverage, entry_price, timestamp, stop_loss_price, stop_loss_pct
                FROM trades 
                WHERE position_type = ? AND trade_status = 'Open'
                ORDER BY timestamp DESC LIMIT 1
            """, (position_type,))
            result = c.fetchone()
            
            if result:
                leverage_value = result[0] if result[0] is not None else 1
                entry_price = result[1]
                exit_price = current_price
                entry_time = datetime.strptime(result[2], "%Y-%m-%d %H:%M:%S")
                stop_loss_price = result[3]
                stop_loss_pct = result[4] or 0.0
                position_duration = int((datetime.now() - entry_time).total_seconds() / 60)
                
                # 수익률 계산
                if position_type == "Long":
                    profit_loss = ((exit_price - entry_price) / entry_price) * 100 * leverage_value
                else:
                    profit_loss = ((entry_price - exit_price) / entry_price) * 100 * leverage_value
                    
            trade_status = "Closed"
            exit_conditions = get_exit_conditions(latest_data)
        
        else:  # Hold 또는 기타 결정
            position_type = "None"
            trade_status = "Hold"
            leverage_value = 0
        
        # 신뢰도 점수 및 시장 심리 데이터
        ai_confidence_score = calculate_confidence_score(latest_data)
        social_sentiment = market_data.get('fear_greed', {}).get('value', 'N/A')
        
        # DB 저장용 values 튜플 생성
        values = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
            current_price,                                  # current_price
            decision.decision,                              # decision
            decision.reason,                                # reason
            decision.position_size,                         # position_size
            leverage_value,                                 # leverage
            wallet_balance,                                 # wallet_balance
            estimated_fee,                                  # estimated_fee
            float(latest_data['RSI']),                     # rsi
            float(latest_data['MACD']),                    # macd
            float(latest_data['MACD_signal']),             # macd_signal
            float(latest_data['EMA_20']),                  # ema_20
            float(latest_data['EMA_50']),                  # ema_50
            float(latest_data['BB_upper']),                # bb_upper
            float(latest_data['BB_lower']),                # bb_lower
            market_data.get('fear_greed', {}).get('value', 'N/A'),  # fear_greed_value
            success,                                        # success
            reflection,                                     # reflection
            position_type,                                  # position_type
            entry_price,                                    # entry_price
            exit_price,                                     # exit_price
            trade_status,                                   # trade_status
            profit_loss,                                    # profit_loss
            trade_id,                                       # trade_id
            ai_confidence_score,                            # ai_confidence_score
            position_duration,                              # position_duration
            entry_conditions,                               # entry_conditions
            exit_conditions,                                # exit_conditions
            social_sentiment,                               # social_sentiment
            "",                                             # news_keywords
            trading_analysis_ai,                            # trading_analysis_ai
            0.0,                                            # funding_fee
            estimated_fee,                                  # total_fee
            profit_loss,                                    # roi_percentage
            stop_loss_price,                                # stop_loss_price (API에서 가져온 실제 값)
            0,                                              # is_stop_loss
            stop_loss_pct                                   # stop_loss_pct (실제 가격 기준으로 계산된 비율)
        )
        
        # DB에 데이터 삽입
        c.execute('''INSERT INTO trades 
                   (timestamp, current_price, decision, reason, position_size, leverage,
                    wallet_balance, estimated_fee, rsi, macd, macd_signal, ema_20, ema_50,
                    bb_upper, bb_lower, fear_greed_value, success, reflection, position_type,
                    entry_price, exit_price, trade_status, profit_loss, trade_id,
                    ai_confidence_score, position_duration, entry_conditions, exit_conditions,
                    social_sentiment, news_keywords, trading_analysis_ai, funding_fee, total_fee,
                    roi_percentage, stop_loss_price, is_stop_loss, stop_loss_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', values)
        
        conn.commit()
        print(f"[성공] 거래 데이터가 DB에 저장되었습니다. (TradeID: {trade_id}, Leverage: {leverage_value}x)")
        if stop_loss_price:
            print(f"[정보] 저장된 스탑로스 정보: 가격 ${stop_loss_price:,.2f}, 비율 {stop_loss_pct:.2f}%")
        return True
        
    except Exception as e:
        print(f"[오류] DB 저장 중 문제 발생: {e}")
        print(traceback.format_exc())
        conn.rollback()
        return False

def handle_exit(conn, market_data, decision: TradingDecision, wallet_balance: float) -> bool:
    """
    포지션 청산 처리 함수 (익절 또는 손절일 경우)
    - 거래 DB 저장
    - 실제 체결 내역 기반 수익률 보정
    """

    print(f"[처리] 청산 처리 시작: {decision.decision}")

    # 1. DB에 청산 정보 저장
    trade_saved = save_trade_to_db(
        conn=conn,
        market_data=market_data,
        decision=decision,
        wallet_balance=wallet_balance,
        estimated_fee=decision.fee if hasattr(decision, "fee") else 0
    )

    if not trade_saved:
        print("[오류] 청산 저장 실패")
        return False

    # 2. 청산된 거래일 경우 → 실 수익률 보정
    if decision.decision in ["Close Long Position", "Close Short Position"]:
        try:
            trade_id = get_latest_trade_id(conn)
            entry_time, exit_time = get_trade_time_range(conn, trade_id)
            position_type = "Long" if decision.decision == "Close Long Position" else "Short"

            update_trade_with_real_pnl(
                trade_id=trade_id,
                symbol="BTCUSDT",  # 필요 시 파라미터로 symbol 받아도 됨
                position_side=position_type,
                start_time=entry_time,
                end_time=exit_time
            )

        except Exception as e:
            print(f"[경고] 실 수익률 보정 실패: {e}")
            return False

    print(f"[완료] 청산 처리 완료: {decision.decision}")
    return True

def get_latest_trade_id(conn):
    c = conn.cursor()
    c.execute("SELECT trade_id FROM trades ORDER BY timestamp DESC LIMIT 1")
    return c.fetchone()[0]

def get_trade_time_range(conn, trade_id: str):
    c = conn.cursor()
    c.execute("SELECT timestamp FROM trades WHERE trade_id = ?", (trade_id,))
    ts = c.fetchone()[0]
    entry_time = int(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    exit_time = int(datetime.now().timestamp() * 1000)
    return entry_time, exit_time

    
def calculate_profit_loss(conn, trade_id: str, exit_price: float) -> Optional[float]:
    """포지션 수익률 계산 (수수료, 레버리지 반영)"""
    FEE_RATE_LOCAL = 0.0006  # 0.06%
    try:
        c = conn.cursor()
        c.execute("""
            SELECT position_type, entry_price, position_size, leverage, wallet_balance, timestamp,
                   stop_loss_price, is_stop_loss
            FROM trades
            WHERE trade_id = ? AND trade_status = 'Open'
        """, (trade_id,))
        trade = c.fetchone()
        
        if trade:
            position_type, entry_price, position_size, leverage, wallet_balance, entry_time, stop_loss_price, is_stop_loss = trade
            
            # 1. 포지션 규모 계산
            position_value = wallet_balance * (position_size / 100)  # 원금 투자액
            leveraged_value = position_value * leverage  # 레버리지 적용된 거래 금액
            
            # 2. 진입 수수료
            entry_fee = leveraged_value * FEE_RATE_LOCAL
            
            # 3. 가격 변화 계산
            if position_type == "Long":
                price_change_ratio = (exit_price - entry_price) / entry_price
                # 레버리지 적용 전 원금 기준 수익
                raw_pnl = position_value * price_change_ratio * leverage  
            else:  # Short
                price_change_ratio = (entry_price - exit_price) / entry_price
                raw_pnl = position_value * price_change_ratio * leverage
            
            # 4. 청산 수수료 (가격 변동 후 포지션 가치에 적용)
            exit_position_value = leveraged_value * (1 + (price_change_ratio if position_type == "Long" else -price_change_ratio))
            exit_fee = exit_position_value * FEE_RATE_LOCAL
            
            # 5. 펀딩비는 제외 (요청에 따라)
            funding_fee = 0
            
            # 6. 최종 순이익 계산
            total_fees = entry_fee + exit_fee + funding_fee
            net_pnl = raw_pnl - total_fees
            
            # 7. 투자 원금 대비 ROI 계산
            roi_percentage = (net_pnl / position_value) * 100
            
            # 8. 손절매 여부 기록
            if stop_loss_price is not None:
                if (position_type == "Long" and exit_price <= stop_loss_price) or \
                   (position_type == "Short" and exit_price >= stop_loss_price):
                    c.execute("""
                        UPDATE trades 
                        SET is_stop_loss = 1
                        WHERE trade_id = ?
                    """, (trade_id,))
                    conn.commit()
            
            # 9. 거래 DB에 수수료 및 ROI 세부 정보 업데이트
            c.execute("""
                UPDATE trades
                SET total_fee = ?,
                    entry_fee = ?,
                    exit_fee = ?,
                    funding_fee = ?,
                    roi_percentage = ?
                WHERE trade_id = ?
            """, (total_fees, entry_fee, exit_fee, funding_fee, roi_percentage, trade_id))
            conn.commit()
            
            print(f"[정보] 거래 ID {trade_id} 수익률 계산:")
            print(f"  레버리지: {leverage}x, 포지션 가치: ${leveraged_value:.2f}")
            print(f"  진입 수수료: ${entry_fee:.2f}, 청산 수수료: ${exit_fee:.2f}")
            print(f"  원시 수익: ${raw_pnl:.2f}, 총 수수료: ${total_fees:.2f}")
            print(f"  순수익: ${net_pnl:.2f}, ROI: {roi_percentage:.2f}%")
            
            return roi_percentage
        return None
        
    except Exception as e:
        print(f"[오류] 수익률 계산 중 문제 발생: {e}")
        print(traceback.format_exc())
        return None

def save_trading_summary(conn, metrics: dict, analysis: str, suggestions: str) -> bool:
    """거래 요약 데이터 저장"""
    try:
        c = conn.cursor()
        
        values = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metrics["initial_balance"],
            metrics["current_balance"],
            metrics["total_return_pct"],
            metrics["total_trades"],
            metrics["win_rate"],
            metrics["avg_profit_pct"],
            metrics["max_profit_pct"],
            metrics["max_loss_pct"],
            metrics["avg_leverage"],
            metrics["max_consecutive_streak"],
            metrics["current_streak"],
            metrics["trades_per_hour"],
            metrics["high_leverage_trades"],
            metrics["medium_leverage_trades"],
            metrics["low_leverage_trades"],
            metrics["high_leverage_winrate"],
            metrics["medium_leverage_winrate"],
            metrics["low_leverage_winrate"],
            analysis,
            suggestions
        )
        
        c.execute('''
            INSERT INTO trading_summary 
            (timestamp, initial_balance, current_balance, total_return_pct,
             total_trades, win_rate, avg_profit_pct, max_profit_pct, max_loss_pct,
             avg_leverage, max_consecutive_streak, current_streak, trades_per_hour,
             high_leverage_trades, medium_leverage_trades, low_leverage_trades,
             high_leverage_winrate, medium_leverage_winrate, low_leverage_winrate,
             ai_analysis, strategy_suggestions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)
        
        conn.commit()
        print("[성공] 거래 요약 데이터가 저장되었습니다.")
        return True
        
    except Exception as e:
        print(f"[오류] 거래 요약 데이터 저장 중 문제 발생: {e}")
        conn.rollback()
        return False

def update_trading_summary() -> bool:
    """
    trades 테이블의 거래 데이터를 집계하여 trading_summary 테이블을 업데이트하는 함수.
    
    - trades 테이블에서 'Closed' 상태의 거래를 대상으로,
      총 거래 건수, 승률, 평균 수익률, 최대 수익률, 최대 손실률, 평균 레버리지,
      거래 당 시간 및 레버리지별 거래 성과 등을 계산한 후,
      trading_summary 테이블에 최신 거래 성과 레코드를 삽입합니다.
    
    Returns:
        bool: 업데이트 성공 여부.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                       AVG(profit_loss) as avg_profit,
                       MAX(profit_loss) as max_profit,
                       MIN(profit_loss) as max_loss,
                       AVG(leverage) as avg_leverage
                FROM trades
                WHERE trade_status = 'Closed'
            """)
            result = cursor.fetchone()
            if result is None:
                logger.warning("[경고] trades 테이블에서 데이터를 조회할 수 없습니다.")
                return False
            
            total_trades = result["total_trades"]
            wins = result["wins"] or 0
            avg_profit = result["avg_profit"] or 0.0
            max_profit = result["max_profit"] or 0.0
            max_loss = result["max_loss"] or 0.0
            avg_leverage = result["avg_leverage"] or 1.0
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
            
            cursor.execute("""
                SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
                FROM trades
                WHERE trade_status = 'Closed'
            """)
            time_result = cursor.fetchone()
            trades_per_hour = 0.0
            if time_result and time_result["min_time"] and time_result["max_time"]:
                min_time = datetime.strptime(time_result["min_time"], "%Y-%m-%d %H:%M:%S")
                max_time = datetime.strptime(time_result["max_time"], "%Y-%m-%d %H:%M:%S")
                hours = (max_time - min_time).total_seconds() / 3600
                if hours > 0:
                    trades_per_hour = total_trades / hours
            
            initial_balance = 10000.0
            current_balance = initial_balance * (1 + avg_profit / 100)
            total_return_pct = ((current_balance - initial_balance) / initial_balance) * 100
            
            high_leverage_threshold = 10
            medium_leverage_threshold = 5
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_status = 'Closed' AND leverage >= ?", (high_leverage_threshold,))
            high_leverage_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_status = 'Closed' AND leverage < ? AND leverage >= ?", (high_leverage_threshold, medium_leverage_threshold))
            medium_leverage_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_status = 'Closed' AND leverage < ?", (medium_leverage_threshold,))
            low_leverage_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_status = 'Closed' AND leverage >= ? AND profit_loss > 0", (high_leverage_threshold,))
            high_wins = cursor.fetchone()[0]
            high_leverage_winrate = (high_wins / high_leverage_trades * 100) if high_leverage_trades > 0 else 0.0
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_status = 'Closed' AND leverage < ? AND leverage >= ? AND profit_loss > 0", (high_leverage_threshold, medium_leverage_threshold))
            medium_wins = cursor.fetchone()[0]
            medium_leverage_winrate = (medium_wins / medium_leverage_trades * 100) if medium_leverage_trades > 0 else 0.0
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE trade_status = 'Closed' AND leverage < ? AND profit_loss > 0", (medium_leverage_threshold,))
            low_wins = cursor.fetchone()[0]
            low_leverage_winrate = (low_wins / low_leverage_trades * 100) if low_leverage_trades > 0 else 0.0
            
            max_consecutive_streak = 0
            current_streak = 0
            
            ai_analysis = ""
            strategy_suggestions = ""
            
            insert_sql = """
                INSERT INTO trading_summary (
                    timestamp,
                    initial_balance,
                    current_balance,
                    total_return_pct,
                    total_trades,
                    win_rate,
                    avg_profit_pct,
                    max_profit_pct,
                    max_loss_pct,
                    avg_leverage,
                    max_consecutive_streak,
                    current_streak,
                    trades_per_hour,
                    high_leverage_trades,
                    medium_leverage_trades,
                    low_leverage_trades,
                    high_leverage_winrate,
                    medium_leverage_winrate,
                    low_leverage_winrate,
                    ai_analysis,
                    strategy_suggestions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(insert_sql, (current_time_str, initial_balance, current_balance, total_return_pct, total_trades,
                                          win_rate, avg_profit, max_profit, max_loss, avg_leverage, max_consecutive_streak,
                                          current_streak, trades_per_hour, high_leverage_trades, medium_leverage_trades,
                                          low_leverage_trades, high_leverage_winrate, medium_leverage_winrate, low_leverage_winrate,
                                          ai_analysis, strategy_suggestions))
            conn.commit()
            logger.info("[정보] trading_summary 테이블 업데이트 완료.")
        return True
    except Exception as e:
        logger.error(f"[오류] trading_summary 업데이트 중 문제 발생: {e}")
        return False


def calculate_portfolio_performance(conn) -> Optional[Dict]:
    """포트폴리오 성과 계산 (초기 자산 대비 현재 잔고 및 수익률)"""
    INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "534.23"))
    try:
        c = conn.cursor()
        c.execute("SELECT wallet_balance FROM trades ORDER BY timestamp DESC LIMIT 1")
        result = c.fetchone()
        if result is None:
            print("[오류] 거래 내역이 없습니다.")
            return None
        latest_balance = result[0]
        total_return = ((latest_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        return {"initial_balance": INITIAL_BALANCE, "current_balance": latest_balance, "total_return_pct": total_return}
    except Exception as e:
        print(f"[오류] 포트폴리오 성과 계산 중 오류: {e}")
        return None

def cleanup_duplicate_trades():
    """중복된 오픈 거래 정리"""
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        c = conn.cursor()
        
        # 각 포지션 타입별로 가장 최근 오픈 거래를 제외한 나머지 거래 닫기
        for position_type in ["Long", "Short"]:
            # 가장 최근 오픈 거래 ID 찾기
            c.execute("""
                SELECT trade_id FROM trades
                WHERE position_type = ? AND trade_status = 'Open'
                ORDER BY timestamp DESC
                LIMIT 1
            """, (position_type,))
            
            latest = c.fetchone()
            if latest:
                latest_id = latest[0]
                
                # 그 외 모든 오픈 거래 닫기
                c.execute("""
                    UPDATE trades
                    SET trade_status = 'Closed',
                        exit_price = (SELECT current_price FROM trades WHERE trade_id = ?),
                        profit_loss = 0,
                        success = 0
                    WHERE position_type = ? AND trade_status = 'Open' AND trade_id != ?
                """, (latest_id, position_type, latest_id))
                
                cleaned = c.rowcount
                if cleaned > 0:
                    print(f"[정보] {position_type} 포지션의 중복 거래 {cleaned}개 정리 완료")
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[오류] 중복 거래 정리 중 문제 발생: {e}")
        if 'conn' in locals():
            conn.close()
        return False

def check_trades_data(conn):
    try:
        c = conn.cursor()

        # 전체 거래 통계 출력
        c.execute("SELECT COUNT(*) FROM trades")
        total_count = c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        print(f"\n=== 전체 거래 통계 ===")
        print(f"총 거래 수: {total_count}")
        
        # 현재 포지션 정보 출력
        print("\n=== 현재 포지션 정보 ===")
        positions = get_position_info(bybit_client)
        if positions:
            for side, pos in positions.items():
                if pos["size"] > 0:
                    print(f"\n[{side} 포지션]")
                    print(f"   진입가: ${pos['entry_price']:,.2f}")
                    print(f"   포지션 크기: {pos['size']} BTC")
                    print(f"   미실현 손익: {pos['unrealized_pnl']} USD")
                    print(f"   레버리지: {pos['leverage']}x")
                    
                    # 스탑로스와 테이크프로핏 주문 정보 조회
                    stop_orders = bybit_client.get_open_orders(
                        category="linear",
                        symbol=SYMBOL,
                        positionIdx=1 if side == "Long" else 2,
                        orderFilter="StopOrder"
                    )
                    if stop_orders["retCode"] == 0:
                        sl_value = None
                        tp_values = []
                        for order in stop_orders["result"]["list"]:
                            if order["stopOrderType"] == "StopMarket":
                                sl_value = order.get("triggerPrice")
                            elif order["stopOrderType"] == "TakeProfit":
                                tp_values.append(order.get("triggerPrice"))
                        if sl_value:
                            print(f"   설정된 손절가: ${float(sl_value):,.2f}")
                        else:
                            print(f"   설정된 손절가: 미설정")
                        if tp_values:
                            tp_str = ", ".join([f"${float(tp):,.2f}" for tp in tp_values])
                            print(f"   설정된 익절가: {tp_str}")
                        else:
                            print(f"   설정된 익절가: 미설정")
                    else:
                        print("   스탑 주문 정보를 가져올 수 없습니다.")
                else:
                    print(f"\n현재 {side} 포지션은 보유하고 있지 않습니다.")
        else:
            print("현재 포지션 정보를 가져올 수 없습니다.")
        
        # 최근 거래 내역 출력
        c.execute("""
            SELECT trade_id, timestamp, position_type, leverage, 
                   entry_price, exit_price, profit_loss, trade_status,
                   stop_loss_price
            FROM trades 
            WHERE (position_type = 'Long' OR position_type = 'Short')
            AND trade_status IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 3
        """)
        recent_trades = c.fetchall()
        
        if recent_trades:
            print("\n=== 최근 주요 거래 ===")
            for trade in recent_trades:
                print(f"\n거래 ID: {trade[0]}")
                print(f"시간: {trade[1]}")
                print(f"포지션: {trade[2]}")
                print(f"레버리지: {trade[3]}x")
                print(f"진입가: ${float(trade[4]):,.2f}" if trade[4] else "진입가: 미설정")
                print(f"청산가: ${float(trade[5]):,.2f}" if trade[5] else "청산가: 진행중")
                print(f"손익: {float(trade[6]):,.2f}%" if trade[6] else "손익: 진행중")
                print(f"상태: {trade[7]}")
                print(f"손절가: ${float(trade[8]):,.2f}" if trade[8] else "손절가: 미설정")
    
                # 현재 거래가 Open 상태이면 실시간 수익률도 표시
                if trade[7] == 'Open':
                    positions = get_position_info(bybit_client)
                    if positions and positions[trade[2]]["size"] > 0:
                        entry_price = float(trade[4])
                        # 15분봉 캔들 데이터로 현재가 조회
                        current_candle = fetch_candle_data(interval="15", limit=1)
                        if not current_candle.empty:
                            current_price = float(current_candle['Close'].iloc[-1])
                            leverage = int(trade[3])
                            
                            # 거래 수수료 (진입+청산) 적용 전후 수익률 계산 (예: 총 0.12% 수수료)
                            total_fee_rate = FEE_RATE * 2
                            if trade[2] == "Long":
                                gross_pnl = ((current_price - entry_price) / entry_price) * 100 * leverage
                                net_pnl = gross_pnl - (total_fee_rate * 100 * leverage)
                            else:
                                gross_pnl = ((entry_price - current_price) / entry_price) * 100 * leverage
                                net_pnl = gross_pnl - (total_fee_rate * 100 * leverage)
                                
                            print(f"현재 총 수익률: {gross_pnl:.2f}%")
                            print(f"수수료 차감 후 순수익률: {net_pnl:.2f}%")
    
                print("-" * 40)
        else:
            print("\n최근 실행된 거래가 없습니다.")
            
        print("\n=== 데이터 확인 완료 ===")
    
    except Exception as e:
        print(f"\n[오류] 거래 데이터 확인 중 문제 발생: {e}")
        import traceback
        print(traceback.format_exc())

        
def analyze_and_reflect(conn) -> None:
    """매매 결과 분석 및 요약 데이터 저장"""
    try:
        performance = calculate_portfolio_performance(conn)
        if not performance:
            print("[오류] 포트폴리오 성과 데이터를 가져올 수 없습니다.")
            return
        
        c = conn.cursor()
        
        # 완료된 거래 조회
        c.execute("""
            SELECT trade_id, profit_loss, trade_status, success, leverage, timestamp,
                   position_type, entry_price, exit_price, position_size
            FROM trades 
            WHERE trade_status = 'Closed'
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        closed_trades = c.fetchall()
        
        # 최근 거래 조회
        c.execute("""
            SELECT trade_id, profit_loss, trade_status, success, leverage, timestamp,
                   position_type, entry_price, exit_price, position_size
            FROM trades 
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        recent_trades = c.fetchall()

        # 핵심 메트릭 계산
        total_trades = len(closed_trades)
        if total_trades > 0:
            # 승률 계산
            winning_trades = sum(1 for trade in closed_trades 
                               if trade[1] is not None and float(trade[1]) > 0)
            win_rate = (winning_trades / total_trades) * 100
            
            # 수익률 통계
            profits = [float(trade[1]) for trade in closed_trades if trade[1] is not None]
            avg_profit = sum(profits) / len(profits) if profits else 0
            max_profit = max(profits) if profits else 0
            max_loss = min(profits) if profits else 0
            
            # 레버리지 분석
            leverages = [float(trade[4]) for trade in closed_trades if trade[4] is not None]
            avg_leverage = sum(leverages) / len(leverages) if leverages else 0
            
            high_leverage = sum(1 for l in leverages if l >= 8)
            medium_leverage = sum(1 for l in leverages if 4 <= l < 8)
            low_leverage = sum(1 for l in leverages if 0 < l < 4)
            
            # 레버리지별 승률
            high_lev_wins = sum(1 for trade in closed_trades 
                              if trade[4] is not None and float(trade[4]) >= 8 
                              and trade[1] is not None and float(trade[1]) > 0)
            med_lev_wins = sum(1 for trade in closed_trades 
                             if trade[4] is not None and 4 <= float(trade[4]) < 8 
                             and trade[1] is not None and float(trade[1]) > 0)
            low_lev_wins = sum(1 for trade in closed_trades 
                             if trade[4] is not None and 0 < float(trade[4]) < 4 
                             and trade[1] is not None and float(trade[1]) > 0)
            
            high_lev_winrate = (high_lev_wins / high_leverage * 100) if high_leverage > 0 else 0
            med_lev_winrate = (med_lev_wins / medium_leverage * 100) if medium_leverage > 0 else 0
            low_lev_winrate = (low_lev_wins / low_leverage * 100) if low_leverage > 0 else 0
            
            # 연속 스트릭 계산
            current_streak = 0
            max_streak = 0
            for trade in closed_trades:
                if trade[1] is not None:
                    if float(trade[1]) > 0:
                        if current_streak < 0:
                            current_streak = 1
                        else:
                            current_streak += 1
                    else:
                        if current_streak > 0:
                            current_streak = -1
                        else:
                            current_streak -= 1
                    max_streak = max(abs(current_streak), max_streak)
            
            # 시간당 거래 횟수 계산
            if len(recent_trades) >= 2:
                first_time = datetime.strptime(recent_trades[-1][5], "%Y-%m-%d %H:%M:%S")
                last_time = datetime.strptime(recent_trades[0][5], "%Y-%m-%d %H:%M:%S")
                hours_diff = (last_time - first_time).total_seconds() / 3600
                trades_per_hour = len(recent_trades) / hours_diff if hours_diff > 0 else 0
            else:
                trades_per_hour = 0
            
            # 메트릭 데이터 구성
            metrics = {
                "initial_balance": performance['initial_balance'],
                "current_balance": performance['current_balance'],
                "total_return_pct": performance['total_return_pct'],
                "total_trades": total_trades,
                "win_rate": win_rate,
                "avg_profit_pct": avg_profit,
                "max_profit_pct": max_profit,
                "max_loss_pct": max_loss,
                "avg_leverage": avg_leverage,
                "max_consecutive_streak": max_streak,
                "current_streak": current_streak,
                "trades_per_hour": trades_per_hour,
                "high_leverage_trades": high_leverage,
                "medium_leverage_trades": medium_leverage,
                "low_leverage_trades": low_leverage,
                "high_leverage_winrate": high_lev_winrate,
                "medium_leverage_winrate": med_lev_winrate,
                "low_leverage_winrate": low_lev_winrate
            }
            
            # AI 분석 요청
            analysis_prompt = f"""
거래 데이터 분석:
- 전체 거래 수: {total_trades}
- 전체 승률: {win_rate:.2f}%
- 평균 수익률: {avg_profit:.2f}%
- 초기 자산: ${performance['initial_balance']:.2f}
- 현재 자산: ${performance['current_balance']:.2f}
- 총 수익률: {performance['total_return_pct']:.2f}%
- 평균 레버리지: {avg_leverage:.1f}x
- 레버리지별 승률:
  * 높은 레버리지: {high_lev_winrate:.2f}%
  * 중간 레버리지: {med_lev_winrate:.2f}%
  * 낮은 레버리지: {low_lev_winrate:.2f}%
- 시간당 거래 횟수: {trades_per_hour:.2f}
- 현재 연속 스트릭: {current_streak}
"""
            
            ai_analysis = get_ai_analysis(analysis_prompt)
            
            # 성과 요약 및 제안 생성
            suggestions = ""
            if win_rate < 50:
                suggestions += "\n- 승률이 50% 미만입니다. 진입 조건을 더 엄격하게 설정하는 것을 고려하세요."
            if trades_per_hour > 4:
                suggestions += "\n- 과거래가 감지되었습니다. 거래 빈도를 줄이고 더 명확한 시그널을 기다리세요."
            if max_loss < -5:
                suggestions += f"\n- 최대 손실이 {max_loss:.2f}%로 큽니다. 손절매 전략을 재검토하세요."
            if high_leverage > (medium_leverage + low_leverage):
                suggestions += "\n- 높은 레버리지 사용이 많습니다. 리스크 관리를 강화하세요."
            
            # 거래 요약 데이터 저장
            save_trading_summary(conn, metrics, ai_analysis, suggestions)
            
            # 분석 결과 출력
            print("\n=== 매매 결과 분석 ===")
            print(f"""
거래 성과 분석 리포트
====================

포트폴리오 성과:
- 초기 투자금: ${performance['initial_balance']:.2f}
- 현재 잔고: ${performance['current_balance']:.2f}
- 총 수익률: {performance['total_return_pct']:.2f}%

전체 거래 분석 (총 {total_trades}건):
- 승률: {win_rate:.2f}%
- 평균 수익률: {avg_profit:.2f}%
- 최대 수익: {max_profit:.2f}%
- 최대 손실: {max_loss:.2f}%

최근 거래 분석:
- 평균 레버리지: {avg_leverage:.1f}x
- 시간당 거래 횟수: {trades_per_hour:.2f}

레버리지 분석:
- 높은 레버리지(8-20x): {high_leverage}건, 승률: {high_lev_winrate:.2f}%
- 중간 레버리지(4-7x): {medium_leverage}건, 승률: {med_lev_winrate:.2f}%
- 낮은 레버리지(1-3x): {low_leverage}건, 승률: {low_lev_winrate:.2f}%

거래 패턴:
- 최대 연속 스트릭: {max_streak}
- 현재 스트릭: {current_streak}
- 과거래 여부: {'예' if trades_per_hour > 4 else '아니오'}

AI 분석 및 제안:
{ai_analysis}

전략 개선 제안:
{suggestions}
""")
            
        else:
            print("[정보] 분석할 거래 데이터가 없습니다.")
        
    except Exception as e:
        print(f"[오류] 매매 결과 분석 중 오류 발생: {e}")
        print(traceback.format_exc())
        if conn:
            conn.rollback()

def analyze_trading_performance(conn, period_days=7):
    """
    지정된 기간 동안의 거래 성과를 분석하여 파라미터 조정에 필요한 지표를 반환합니다.
    
    Args:
        conn: 데이터베이스 연결 객체
        period_days: 분석할 기간(일)
        
    Returns:
        Dict: 분석된 지표들
    """
    try:
        c = conn.cursor()
        
        # 지정된 기간 내 거래 데이터 조회
        period_start = (datetime.now() - timedelta(days=period_days)).strftime("%Y-%m-%d %H:%M:%S")
        
        c.execute("""
            SELECT position_type, entry_price, exit_price, profit_loss, leverage, 
                   is_stop_loss, stop_loss_price, stop_loss_pct
            FROM trades
            WHERE timestamp >= ? AND trade_status = 'Closed'
            ORDER BY timestamp
        """, (period_start,))
        
        trades = c.fetchall()
        
        if not trades:
            print(f"[정보] 분석 기간({period_days}일) 내 거래 데이터가 없습니다.")
            return None
        
        # 성과 지표 계산
        total_trades = len(trades)
        stop_loss_count = sum(1 for trade in trades if trade[5] == 1)  # is_stop_loss가 1인 거래 수
        profitable_trades = sum(1 for trade in trades if trade[3] > 0)  # profit_loss > 0인 거래 수
        
        # 승률 및 손절률 계산
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        stop_loss_rate = stop_loss_count / total_trades if total_trades > 0 else 0
        
        # 평균 수익률 및 손실률 계산
        profits = [trade[3] for trade in trades if trade[3] > 0]
        losses = [trade[3] for trade in trades if trade[3] <= 0]
        
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # 평균 레버리지 및 스탑로스 비율 계산
        avg_leverage = sum(trade[4] for trade in trades if trade[4]) / total_trades if total_trades > 0 else 0
        avg_stop_loss_pct = sum(trade[7] for trade in trades if trade[7]) / total_trades if total_trades > 0 else 0
        
        # 포지션 타입별 성과 분석
        long_trades = [trade for trade in trades if trade[0] == "Long"]
        short_trades = [trade for trade in trades if trade[0] == "Short"]
        
        long_win_rate = sum(1 for trade in long_trades if trade[3] > 0) / len(long_trades) if long_trades else 0
        short_win_rate = sum(1 for trade in short_trades if trade[3] > 0) / len(short_trades) if short_trades else 0
        
        # 포지션 타입별 스탑로스 발생률
        long_sl_rate = sum(1 for trade in long_trades if trade[5] == 1) / len(long_trades) if long_trades else 0
        short_sl_rate = sum(1 for trade in short_trades if trade[5] == 1) / len(short_trades) if short_trades else 0
        
        # 최종 분석 결과 구성
        analysis = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "stop_loss_rate": stop_loss_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "avg_leverage": avg_leverage,
            "avg_stop_loss_pct": avg_stop_loss_pct,
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "long_sl_rate": long_sl_rate,
            "short_sl_rate": short_sl_rate,
            "period_days": period_days
        }
        
        print(f"\n=== 거래 성과 분석 결과 ({period_days}일 기준) ===")
        print(f"총 거래 수: {total_trades}")
        print(f"승률: {win_rate:.2f}")
        print(f"손절률: {stop_loss_rate:.2f}")
        print(f"평균 수익률: {avg_profit:.2f}%")
        print(f"평균 손실률: {avg_loss:.2f}%")
        print(f"롱 포지션 승률: {long_win_rate:.2f}")
        print(f"숏 포지션 승률: {short_win_rate:.2f}")
        
        return analysis
        
    except Exception as e:
        print(f"[오류] 거래 성과 분석 중 문제 발생: {e}")
        print(traceback.format_exc())
        return None

def adjust_strategy_parameters(analysis):
    """
    분석 결과를 기반으로 전략 파라미터를 동적으로 조정합니다.
    
    Args:
        analysis: analyze_trading_performance에서 반환된 분석 결과
        
    Returns:
        Dict: 조정된 파라미터 값들
    """
    try:
        if not analysis:
            print("[경고] 파라미터 조정을 위한 분석 결과가 없습니다.")
            return None
        
        # 글로벌 변수 참조
        global RSI_HIGH_THRESHOLD, RSI_LOW_THRESHOLD, ATR_MULTIPLIER, TP_LEVELS
        global VOLATILITY_THRESHOLD, ADX_THRESHOLD
        
        # 현재 파라미터 값 저장
        current_params = {
            "rsi_high": RSI_HIGH_THRESHOLD,
            "rsi_low": RSI_LOW_THRESHOLD,
            "atr_multiplier": ATR_MULTIPLIER,
            "tp_levels": TP_LEVELS,
            "volatility_threshold": VOLATILITY_THRESHOLD,
            "adx_threshold": ADX_THRESHOLD
        }
        
        # 새 파라미터 초기화 (현재 값으로)
        new_params = current_params.copy()
        
        # 조정 로직 시작
        # 1. 손절률 기반 조정
        if analysis["stop_loss_rate"] > 0.4:  # 손절 비율이 40% 이상인 경우
            # 스탑로스 완화 (버퍼 증가)
            new_params["atr_multiplier"] = min(4.0, current_params["atr_multiplier"] * 1.2)
            print(f"[조정] 손절률이 높아 ATR 승수 증가: {current_params['atr_multiplier']} -> {new_params['atr_multiplier']}")
            
            # RSI 임계값 완화
            new_params["rsi_high"] = min(80, current_params["rsi_high"] + 2)
            new_params["rsi_low"] = max(20, current_params["rsi_low"] - 2)
            print(f"[조정] RSI 임계값 완화: {current_params['rsi_high']}/{current_params['rsi_low']} -> {new_params['rsi_high']}/{new_params['rsi_low']}")
        
        # 2. 승률 기반 조정
        if analysis["win_rate"] < 0.4:  # 승률이 40% 미만인 경우
            # 변동성 임계값 증가 (필터링 강화)
            new_params["volatility_threshold"] = min(1.5, current_params["volatility_threshold"] * 1.1)
            print(f"[조정] 승률이 낮아 변동성 임계값 증가: {current_params['volatility_threshold']} -> {new_params['volatility_threshold']}")
            
            # ADX 임계값 증가 (추세 신호 강화)
            new_params["adx_threshold"] = min(25, current_params["adx_threshold"] + 2)
            print(f"[조정] 추세 신호 강화: ADX 임계값 {current_params['adx_threshold']} -> {new_params['adx_threshold']}")
        elif analysis["win_rate"] > 0.7:  # 승률이 높은 경우 (70% 이상)
            # 수익 목표 상향 조정 (더 큰 이익 추구)
            tp_levels_adjusted = [level * 1.1 for level in current_params["tp_levels"]]
            new_params["tp_levels"] = [min(5.0, level) for level in tp_levels_adjusted]  # 최대 5.0으로 제한
            print(f"[조정] 승률이 높아 수익 목표 상향: {current_params['tp_levels']} -> {new_params['tp_levels']}")
        
        # 3. 포지션 타입별 조정
        # 롱 포지션 성과가 좋지 않은 경우
        if analysis["long_win_rate"] < 0.3 and analysis["long_sl_rate"] > 0.5:
            # 롱 진입 조건 강화
            new_params["rsi_low"] = min(45, current_params["rsi_low"] + 5)  # RSI 하단 임계값 상향
            print(f"[조정] 롱 포지션 성과가 낮아 진입 조건 강화: RSI 하단 {current_params['rsi_low']} -> {new_params['rsi_low']}")
        
        # 숏 포지션 성과가 좋지 않은 경우
        if analysis["short_win_rate"] < 0.3 and analysis["short_sl_rate"] > 0.5:
            # 숏 진입 조건 강화
            new_params["rsi_high"] = max(55, current_params["rsi_high"] - 5)  # RSI 상단 임계값 하향
            print(f"[조정] 숏 포지션 성과가 낮아 진입 조건 강화: RSI 상단 {current_params['rsi_high']} -> {new_params['rsi_high']}")
        
        # 변경 사항 요약
        changes = {param: (current_params[param], new_params[param]) 
                  for param in current_params 
                  if current_params[param] != new_params[param]}
        
        if changes:
            print("\n=== 파라미터 조정 결과 ===")
            for param, (old_val, new_val) in changes.items():
                print(f"{param}: {old_val} -> {new_val}")
        else:
            print("\n[정보] 파라미터 조정 불필요")
        
        return new_params
        
    except Exception as e:
        print(f"[오류] 파라미터 조정 중 문제 발생: {e}")
        print(traceback.format_exc())
        return None

def apply_dynamic_parameters(params):
    """
    조정된 파라미터를 전역 변수에 적용합니다.
    
    Args:
        params: 조정된 파라미터 딕셔너리
    """
    try:
        if not params:
            return False
        
        global RSI_HIGH_THRESHOLD, RSI_LOW_THRESHOLD, ATR_MULTIPLIER, TP_LEVELS
        global VOLATILITY_THRESHOLD, ADX_THRESHOLD
        
        # 파라미터 적용
        RSI_HIGH_THRESHOLD = params["rsi_high"]
        RSI_LOW_THRESHOLD = params["rsi_low"]
        ATR_MULTIPLIER = params["atr_multiplier"]
        TP_LEVELS = params["tp_levels"]
        VOLATILITY_THRESHOLD = params["volatility_threshold"]
        ADX_THRESHOLD = params["adx_threshold"]
        
        # DB에 파라미터 변경 기록 저장
        save_parameter_history(params)
        
        print("[성공] 조정된 파라미터가 적용되었습니다.")
        return True
        
    except Exception as e:
        print(f"[오류] 파라미터 적용 중 문제 발생: {e}")
        return False

def save_parameter_history(params):
    """
    파라미터 변경 이력을 DB에 저장합니다.
    
    Args:
        params: 적용된 파라미터 딕셔너리
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # parameters_history 테이블이 없으면 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameters_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    rsi_high REAL,
                    rsi_low REAL,
                    atr_multiplier REAL,
                    tp_levels TEXT,
                    volatility_threshold REAL,
                    adx_threshold REAL,
                    reason TEXT
                )
            ''')
            
            # 파라미터 저장
            cursor.execute('''
                INSERT INTO parameters_history 
                (timestamp, rsi_high, rsi_low, atr_multiplier, tp_levels, 
                 volatility_threshold, adx_threshold, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                params["rsi_high"],
                params["rsi_low"],
                params["atr_multiplier"],
                json.dumps(params["tp_levels"]),  # 리스트를 JSON 문자열로 변환
                params["volatility_threshold"],
                params["adx_threshold"],
                "자동 파라미터 조정에 의한 변경"
            ))
            
            conn.commit()
            print("[정보] 파라미터 변경 이력이 저장되었습니다.")
            return True
            
    except Exception as e:
        print(f"[오류] 파라미터 이력 저장 중 문제 발생: {e}")
        print(traceback.format_exc())
        if 'conn' in locals():
            conn.rollback()
        return False

def ai_trading() -> None:
    """단일 거래 사이클 실행 함수"""
    print(f"\n🚀 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 트레이딩 봇 실행 중...")
    try:
        conn = init_db()
        if not conn:
            print("❌ 데이터베이스 연결 실패!")
            return
        main()
        print(f"✅ [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 거래 성공! 15분 후 다시 실행.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("⏳ 1분 후 다시 시도...")
        time.sleep(60)
    finally:
        if 'conn' in locals():
            conn.close()

def run_trading_bot() -> None:
    """15분 간격으로 트레이딩 사이클 실행 (정각 실행)"""
    while True:
        current_time = datetime.now()
        minutes_until_next = 15 - (current_time.minute % 15)
        seconds_until_next = minutes_until_next * 60 - current_time.second
        if seconds_until_next > 0:
            print(f"다음 실행 시간까지 대기 중... ({minutes_until_next}분 {current_time.second}초)")
            time.sleep(seconds_until_next)
        current_time = datetime.now()
        print(f"\n=== 현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        try:
            ai_trading()
        except Exception as e:
            print(f"오류 발생: {e}")

def optimize_parameters(conn) -> Dict[str, Any]:
    """
    거래 이력을 분석하여 파라미터를 최적화하는 함수
    """
    try:
        c = conn.cursor()
        
        # 최근 100개 거래 데이터 조회
        c.execute("""
            SELECT *
            FROM trades
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        trades = c.fetchall()
        
        if not trades:
            return None
            
        # 성과 지표 계산
        win_rate = sum(1 for trade in trades if trade[23] and float(trade[23]) > 0) / len(trades)
        avg_profit = sum(float(trade[23]) for trade in trades if trade[23]) / len(trades)
        
        # 변동성 계산
        volatility = np.std([float(trade[2]) for trade in trades])  # current_price 기준
        
        # 파라미터 최적화
        optimized_params = {
            # RSI 임계값 최적화
            "rsi_high": min(80, 70 + (win_rate * 10)),  # 승률이 높을수록 더 aggressive하게
            "rsi_low": max(20, 30 - (win_rate * 10)),
            
            # ATR 승수 최적화
            "atr_multiplier": max(1.0, min(2.5, 1.5 + (volatility * 0.5))),
            
            # 레버리지 최적화
            "max_leverage": int(max(5, min(20, 10 + (win_rate * 10)))),
            
            # 이익실현 단계 최적화
            "tp_levels": [
                1.5 + (avg_profit * 0.1),  # 첫 번째 TP
                2.5 + (avg_profit * 0.15),  # 두 번째 TP
                4.0 + (avg_profit * 0.2)    # 세 번째 TP
            ],
            
            # 변동성 기준 최적화
            "volatility_threshold": max(1.0, min(2.0, volatility)),
            
            # ADX 임계값 최적화
            "adx_threshold": max(20, min(40, 25 + (win_rate * 15)))
        }
        
        # DB에 최적화된 파라미터 저장
        c.execute("""
            INSERT INTO parameter_history (timestamp, parameters, reason)
            VALUES (?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            json.dumps(optimized_params),
            f"Win Rate: {win_rate:.2%}, Avg Profit: {avg_profit:.2f}%, Volatility: {volatility:.2f}"
        ))
        conn.commit()
        
        print("\n=== 파라미터 최적화 결과 ===")
        for param, value in optimized_params.items():
            print(f"{param}: {value}")
        print("========================")
        
        return optimized_params
        
    except Exception as e:
        print(f"[오류] 파라미터 최적화 중 문제 발생: {e}")
        return None

def ai_optimize_parameters(conn) -> Optional[Dict[str, Any]]:
    """
    AI를 활용하여 거래 성과 및 이력을 기반으로 전략 파라미터를 최적화하는 함수.
    OpenAI API를 통해 최적화된 파라미터를 JSON 형식으로 도출합니다.
    """
    try:
        # 포트폴리오 성과를 가져와서 프롬프트에 활용합니다.
        performance = calculate_portfolio_performance(conn)
        if performance is None:
            raise Exception("포트폴리오 성과 데이터가 없습니다.")

        prompt = f"""
        당신은 암호화폐 트레이딩 전략 전문가입니다.
        최근 거래 이력 및 포트폴리오 성과를 바탕으로 다음 전략 파라미터들을 최적화해 주세요.
        
        현재 파라미터:
        - RSI_HIGH_THRESHOLD: {RSI_HIGH_THRESHOLD}
        - RSI_LOW_THRESHOLD: {RSI_LOW_THRESHOLD}
        - ATR_MULTIPLIER: {ATR_MULTIPLIER}
        - MAX_LEVERAGE: 10
        - TP_LEVELS: {TP_LEVELS}
        - VOLATILITY_THRESHOLD: {VOLATILITY_THRESHOLD}
        - ADX_THRESHOLD: {ADX_THRESHOLD}
        
        포트폴리오 성과:
        - 초기 투자금: ${performance['initial_balance']:.2f}
        - 현재 잔액: ${performance['current_balance']:.2f}
        - 총 수익률: {performance['total_return_pct']:.2f}%
        
        거래 승률와 평균 수익률 등도 고려하여, 보다 안정적이고 효율적인 파라미터 값을 제안해 주세요.
        반드시 아래와 같은 JSON 형식으로 답변해 주세요:
        
        {{
            "rsi_high": (예: 75.0),
            "rsi_low": (예: 25.0),
            "atr_multiplier": (예: 2.0),
            "max_leverage": (예: 8),
            "tp_levels": [예: 1.8, 2.8, 4.2],
            "volatility_threshold": (예: 1.8),
            "adx_threshold": (예: 28.0)
        }}
        """
        ai_response = get_ai_analysis(prompt)
        # AI 응답이 코드 블록 등으로 감싸져 있을 수 있으므로 JSON 부분만 추출합니다.
        if "```json" in ai_response:
            start = ai_response.find("{")
            end = ai_response.rfind("}") + 1
            ai_response = ai_response[start:end]
        optimized_params = json.loads(ai_response)
        print("\n=== AI 최적화 파라미터 결과 ===")
        for key, value in optimized_params.items():
            print(f"{key}: {value}")
        print("===============================")
        return optimized_params
    except Exception as e:
        print(f"[오류] AI 파라미터 최적화 중 문제 발생: {e}")
        return None

def apply_optimized_parameters(optimized_params: Dict[str, Any]) -> None:
    """
    최적화된 파라미터를 전역 변수 및 전략에 적용하는 함수.
    """
    try:
        global RSI_HIGH_THRESHOLD, RSI_LOW_THRESHOLD, ATR_MULTIPLIER, TP_LEVELS, VOLATILITY_THRESHOLD, ADX_THRESHOLD
        RSI_HIGH_THRESHOLD = optimized_params.get("rsi_high", RSI_HIGH_THRESHOLD)
        RSI_LOW_THRESHOLD = optimized_params.get("rsi_low", RSI_LOW_THRESHOLD)
        ATR_MULTIPLIER = optimized_params.get("atr_multiplier", ATR_MULTIPLIER)
        # TP_LEVELS가 리스트 형식인지 확인합니다.
        TP_LEVELS = optimized_params.get("tp_levels", TP_LEVELS)
        VOLATILITY_THRESHOLD = optimized_params.get("volatility_threshold", VOLATILITY_THRESHOLD)
        ADX_THRESHOLD = optimized_params.get("adx_threshold", ADX_THRESHOLD)
        print("[정보] 최적화된 파라미터 적용 완료")
    except Exception as e:
        print(f"[오류] 파라미터 적용 중 문제 발생: {e}")

# main() 함수 내에서 AI 최적화 및 적용 예시
# 10. 데이터베이스 연결 재사용 개선을 위한 메인 함수 수정
def main():
    """
    메인 실행 함수 (자동 파라미터 조정 기능 추가)
    """
    db_conn = None
    try:
        print("\n🚀 비트코인 트레이딩 봇 시작...")
        
        # 진단 실행 (문제 해결 후 이 줄 제거)
        run_entry_price_diagnostics()
        
        # 1. 데이터베이스 연결 (컨텍스트 매니저 사용)
        with get_db_connection() as db_conn:
            
            # 진입가 불일치 확인 (문제 해결 후 제거)
            check_entry_price_discrepancy()
            
            # 여기에 동기화 함수 호출 추가
            sync_positions_with_exchange(db_conn)
            
            # 포지션이 없을 때 남아있는 TP 주문 정리 (추가된 부분)
            cleanup_orphaned_tp_orders()
            
            # 2. 오래된 거래 데이터 정리
            clean_old_trades(db_conn, 2000)

            # 3. 거래 테이블 스키마 업데이트
            update_trades_table_schema()
            update_db_schema()
            update_tp_table_schema()
            
            # 4. 중복 거래 정리
            cleanup_duplicate_trades()
            
            # 5. 스탑로스 청산 확인 및 익절 주문 실행 확인
            check_for_liquidations()
            check_max_loss_protection(db_conn)  # 이 위치에 배치
            check_take_profit_executions(db_conn)
            
            # 6. 청산 레코드 및 거래 분석
            check_liquidation_records()
            check_trades_data(db_conn)
            
            # 7. 자동 파라미터 조정 기능 추가 (새로운 부분)
            # 일정 조건에서만 실행 (예: 10회 거래마다, 또는 하루에 한 번)
            current_time = datetime.now()
            should_adjust_params = False
            
            # 매일 00시 또는 12시에 파라미터 조정 실행
            if current_time.hour in [0, 12] and current_time.minute < 15:
                should_adjust_params = True
            
            # 또는 최근 10개 거래가 완료된 후 실행
            cursor = db_conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE timestamp > (SELECT MAX(timestamp) FROM parameters_history)
            """)
            trades_since_last_adjustment = cursor.fetchone()[0] or 0
            if trades_since_last_adjustment >= 10:
                should_adjust_params = True
            
            if should_adjust_params:
                print("[정보] 자동 파라미터 조정 실행 중...")
                # 최근 7일간의 거래 데이터 분석
                performance_analysis = analyze_trading_performance(db_conn, period_days=7)
                if performance_analysis:
                    # 분석 결과에 따라 파라미터 조정
                    adjusted_params = adjust_strategy_parameters(performance_analysis)
                    if adjusted_params:
                        # 조정된 파라미터 적용
                        apply_dynamic_parameters(adjusted_params)
            
            # 기존 반성 기반 파라미터 최적화 (유지)
            reflection_params = update_strategy_parameters_from_reflection(db_conn)
            if reflection_params:
                apply_optimized_parameters(reflection_params)
            
            # 8. 현재 잔고 확인
            balance = get_wallet_balance()
            if balance <= 0:
                print("[오류] 거래 가능한 잔액이 없습니다.")
                return

            # 9. 시장 데이터 수집 및 분석
            market_data = get_market_data()
            if not market_data:
                print("[오류] 시장 데이터 수집 실패")
                return
            
            analyze_market_data(market_data)
            
            # 10. 포지션 리스크 분석
            position_info = get_position_info(bybit_client)
            if position_info:
                position_risk = analyze_position_risk(position_info, market_data)
            
            # 11. AI 매매 결정 및 실행
            decision = get_final_trading_decision(market_data)
            if not decision:
                print("[오류] AI 분석 실패")
                return
            
            success = execute_trading_logic(decision, decision.position_size, balance, decision.leverage, db_conn, market_data)
            
            # 12. 거래 기록 저장
            save_trading_history(decision)
            estimated_fee = 0
            if decision.decision in ["Long", "Short"]:
                allocation = (decision.position_size / 100) * balance * (decision.leverage or 1)
                estimated_fee = allocation * 0.00055
            
            # 별도 함수 호출로 DB 트랜잭션 분리
            def _save_trade_operation(conn):
                return save_trade_to_db(conn, market_data, decision, balance, estimated_fee, success=1 if success else 0)
            
            execute_with_retry(_save_trade_operation)
        
        print("\n✅ 거래 완료!")
        
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        print(traceback.format_exc())


def save_trading_history(decision: TradingDecision) -> None:
    """거래 이력 파일(JSONL) 저장"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history = {
            "timestamp": timestamp,
            "decision": decision.decision,
            "reason": decision.reason,
            "position_size": decision.position_size,
            "leverage": decision.leverage if decision.leverage else 0
        }
        # 파일명만 .jsonl로 변경
        with open("trading_history.jsonl", "a", encoding='utf-8') as f:
            f.write(json.dumps(history, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[오류] 거래 이력 저장 중 문제 발생: {e}")

def get_wallet_balance(coin: str = "USDT") -> float:
    """지갑 잔액 조회"""
    try:
        response = bybit_client.get_wallet_balance(coin=coin, accountType="UNIFIED")
        print(f"[디버깅] 전체 응답: {response}")
        if response["retCode"] == 0:
            try:
                coin_info = response["result"]["list"][0]["coin"][0]
                print(f"[디버깅] 코인 정보: {coin_info}")
                wallet_balance = coin_info.get("walletBalance", "0")
                if not wallet_balance:
                    wallet_balance = "0"
                balance = float(wallet_balance)
                print(f"[정보] 현재 잔액: {balance:.2f} {coin}")
                return balance
            except (IndexError, KeyError, TypeError) as e:
                print(f"[오류] 잔액 데이터 파싱 중 오류: {e}")
                return 0.0
        else:
            print(f"[오류] 잔액 조회 실패: {response['retMsg']}")
            return 0.0
    except Exception as e:
        print(f"[오류] 잔액 조회 중 문제 발생: {e}")
        return 0.0

def get_current_positions(client: HTTP, symbol: str = "BTCUSDT") -> Optional[Dict[str, float]]:
    """
    현재 모든 포지션 정보 조회 (롱/숏)
    반환 예시: {"Long": 0.5, "Short": 0.0}
    """
    try:
        response = client.get_positions(category="linear", symbol=symbol)
        if response["retCode"] == 0:
            positions = response["result"]["list"]
            current_positions = {"Long": 0.0, "Short": 0.0}
            for position in positions:
                size = float(position.get("size", 0.0))
                if position["side"] == "Buy":
                    current_positions["Long"] = size
                elif position["side"] == "Sell":
                    current_positions["Short"] = size
            return current_positions
        else:
            print(f"[오류] 포지션 조회 실패: {response.get('retMsg', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"[오류] 포지션 조회 중 문제 발생: {e}")
        return None

def get_position_info(client: HTTP, symbol: str = SYMBOL) -> Optional[Dict]:
    """상세 포지션 정보 조회 (진입가격, 미실현 손익, 레버리지 등)"""
    try:
        response = client.get_positions(category="linear", symbol=symbol)
        if response["retCode"] == 0:
            positions = response["result"]["list"]
            position_info = {
                "Long": {"size": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0, "leverage": 0},
                "Short": {"size": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0, "leverage": 0}
            }
            for position in positions:
                if float(position.get("size", "0.0")) > 0:
                    side = "Long" if position["side"] == "Buy" else "Short"
                    position_info[side] = {
                        "size": float(position.get("size", "0.0")),
                        "entry_price": float(position.get("avgPrice", "0.0")),
                        "unrealized_pnl": float(position.get("unrealisedPnl", "0.0")),
                        "leverage": int(position.get("leverage", "0"))
                    }
            return position_info
        return None
    except Exception as e:
        print(f"[오류] 포지션 정보 조회 중 문제 발생: {e}")
        return None

def check_entry_price_discrepancy():
    position_info = get_position_info(bybit_client)
    if position_info and position_info["Long"]["size"] > 0:
        exchange_entry_price = position_info["Long"]["entry_price"]
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT trade_id, entry_price FROM trades 
            WHERE position_type = 'Long' AND trade_status = 'Open'
            ORDER BY timestamp DESC
        """)
        db_positions = cursor.fetchall()
        conn.close()
        print("\n=== 진입가 불일치 확인 ===")
        print(f"거래소 실제 진입가: ${exchange_entry_price:.2f}")
        print("데이터베이스 저장 진입가:")
        for trade_id, entry_price in db_positions:
            print(f"- 거래 ID: {trade_id}, 진입가: ${entry_price:.2f}")
        return exchange_entry_price, db_positions
    else:
        # 포지션이 없을 경우 기본값 반환
        return None, None

def analyze_trading_history(trade_id):
    conn = sqlite3.connect('bitcoin_trades.db')
    cursor = conn.cursor()
    
    # 특정 trade_id의 모든 관련 정보 조회
    cursor.execute("""
        SELECT timestamp, entry_price, current_price, position_size, leverage, reason
        FROM trades 
        WHERE trade_id = ?
    """, (trade_id,))
    
    trade_details = cursor.fetchone()
    if trade_details:
        timestamp, entry_price, current_price, position_size, leverage, reason = trade_details
        print(f"\n=== 거래 ID {trade_id} 상세 정보 ===")
        print(f"시간: {timestamp}")
        print(f"진입가: ${entry_price:.2f}")
        print(f"당시 시장가: ${current_price:.2f}")
        print(f"포지션 크기: {position_size}%")
        print(f"레버리지: {leverage}x")
        print(f"진입 이유: {reason}")
    
    # 해당 trade_id의 익절 주문 정보 조회
    cursor.execute("""
        SELECT tp_level, tp_price, tp_quantity, status, created_at
        FROM take_profit_orders 
        WHERE trade_id = ?
        ORDER BY tp_level
    """, (trade_id,))
    
    tp_orders = cursor.fetchall()
    if tp_orders:
        print("\n관련 익절 주문:")
        for level, price, qty, status, created_at in tp_orders:
            print(f"- 레벨 {level}: 가격 ${price:.2f}, 수량 {qty} BTC, 상태: {status}, 생성 시간: {created_at}")
    
    conn.close()
    return trade_details, tp_orders

def check_additional_entries():
    """
    추가 매수(additional entries) 이력 확인
    """
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        cursor = conn.cursor()
        
        # is_additional 필드가 있는 take_profit_orders 테이블에서 추가 매수 확인
        cursor.execute("""
            SELECT trade_id, tp_price, tp_quantity, created_at
            FROM take_profit_orders 
            WHERE is_additional = 1
            ORDER BY created_at DESC
        """)
        
        additional_entries = cursor.fetchall()
        if additional_entries:
            print("\n=== 추가 매수 이력 ===")
            for trade_id, price, qty, created_at in additional_entries:
                print(f"거래 ID: {trade_id}, 가격: ${price:.2f}, 수량: {qty} BTC, 시간: {created_at}")
        else:
            print("\n추가 매수 이력이 없습니다.")
        
        conn.close()
        return additional_entries
    except Exception as e:
        print(f"[오류] 추가 매수 이력 확인 중 문제 발생: {e}")
        if 'conn' in locals():
            conn.close()
        return []

def run_entry_price_diagnostics():
    """
    진입가 불일치 및 관련 문제 진단을 위한 함수
    """
    print("\n========== 진입가 불일치 진단 시작 ==========")
    
    # 진입가 불일치 확인
    exchange_entry_price, db_positions = check_entry_price_discrepancy()
    
    # None 값 체크 추가
    if exchange_entry_price is not None and db_positions is not None:
        # 불일치가 발견되면, 각 거래 ID에 대한 상세 정보 분석
        if db_positions:
            for trade_id, _ in db_positions:
                analyze_trading_history(trade_id)
    else:
        print("현재 활성화된 Long 포지션이 없습니다.")
    
    # 추가 매수 이력 확인
    additional_entries = check_additional_entries()
    
    print("\n========== 진입가 불일치 진단 완료 ==========")

def sync_positions_with_exchange(db_conn):
    """
    실제 거래소의 포지션과 DB의 포지션 상태를 동기화
    """
    try:
        # 실제 거래소 포지션 확인
        exchange_positions = get_position_info(bybit_client)
        if not exchange_positions:
            exchange_positions = {
                "Long": {"size": 0, "entry_price": 0},
                "Short": {"size": 0, "entry_price": 0}
            }
        
        cursor = db_conn.cursor()
        
        # DB에서 Open 상태인 거래 조회
        cursor.execute("""
            SELECT trade_id, position_type, entry_price
            FROM trades
            WHERE trade_status = 'Open'
        """)
        
        open_trades = cursor.fetchall()
        
        for trade in open_trades:
            trade_id, position_type, entry_price = trade
            
            # 거래소에 해당 포지션이 없다면 DB 업데이트
            if exchange_positions[position_type]["size"] <= 0:
                current_price = float(fetch_candle_data(interval="1", limit=1)['Close'].iloc[-1])
                
                # 수익률 계산
                if position_type == "Long":
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    profit_pct = ((entry_price - current_price) / entry_price) * 100
                
                cursor.execute("""
                    UPDATE trades
                    SET trade_status = 'Closed',
                        exit_price = ?,
                        profit_loss = ?
                    WHERE trade_id = ?
                """, (current_price, profit_pct, trade_id))
                
                print(f"[정보] DB와 거래소 동기화: {position_type} 포지션 {trade_id}를 Closed로 업데이트")
        
        db_conn.commit()
        print("[정보] 거래소와 DB 동기화 완료")
        
    except Exception as e:
        print(f"[오류] 포지션 동기화 중 문제 발생: {e}")
        if 'db_conn' in locals():
            db_conn.rollback()

def get_stop_loss_price(symbol: str, position_idx: int) -> Optional[float]:
    """거래소 API를 통해 현재 설정된 스탑로스 가격 조회"""
    try:
        response = bybit_client.get_open_orders(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            orderFilter="StopOrder"
        )
        
        if response["retCode"] == 0:
            for order in response["result"]["list"]:
                if order["stopOrderType"] == "StopLoss" or order["orderType"] == "StopMarket":
                    return float(order["triggerPrice"])
        return None
    except Exception as e:
        print(f"[오류] 스탑로스 가격 조회 중 문제 발생: {e}")
        return None


def get_stop_loss_from_exchange_or_db(symbol: str, position_type: str, conn=None, trade_id=None) -> Optional[float]:
    """
    현재 설정된 스탑로스 가격을 거래소 API 또는 DB에서 가져오는 함수
    
    Args:
        symbol: 거래 심볼
        position_type: 포지션 유형 ("Long" 또는 "Short")
        conn: DB 연결 객체 (선택적)
        trade_id: 거래 ID (선택적)
        
    Returns:
        float 또는 None: 설정된 스탑로스 가격, 없으면 None
    """
    try:
        position_idx = 1 if position_type == "Long" else 2
        stop_loss = None
        
        # 1. 거래소 API에서 스탑로스 확인
        response = bybit_client.get_open_orders(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            orderFilter="StopOrder"
        )
        
        if response["retCode"] == 0:
            for order in response["result"]["list"]:
                if order["stopOrderType"] == "StopLoss" or (order["orderType"] == "Market" and order.get("triggerDirection") == "Fall" if position_type == "Long" else "Rise"):
                    stop_loss = float(order["triggerPrice"])
                    print(f"[정보] API에서 {position_type} 포지션의 스탑로스 가격 확인: ${stop_loss:.2f}")
                    return stop_loss
        
        # 2. DB에서 스탑로스 확인 (API에서 찾지 못한 경우)
        if conn and trade_id:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT stop_loss_price 
                FROM trades
                WHERE trade_id = ?
            """, (trade_id,))
            result = cursor.fetchone()
            if result and result[0]:
                stop_loss = float(result[0])
                print(f"[정보] DB에서 {position_type} 포지션의 스탑로스 가격 확인: ${stop_loss:.2f}")
                return stop_loss
        
        # 3. 현재 포지션 정보에서 확인 (마지막 시도)
        if not stop_loss:
            positions = get_position_info(bybit_client)
            if positions and positions[position_type]["size"] > 0:
                stop_loss = positions[position_type].get("stop_loss")
                if stop_loss:
                    print(f"[정보] 포지션 정보에서 {position_type} 포지션의 스탑로스 가격 확인: ${float(stop_loss):.2f}")
                    return float(stop_loss)
        
        print(f"[경고] {position_type} 포지션의 스탑로스 가격을 찾을 수 없습니다.")
        return None
    
    except Exception as e:
        print(f"[오류] 스탑로스 가격 조회 중 문제 발생: {e}")
        return None


def get_entry_price_from_db(conn, trade_id: str) -> Optional[float]:
    """
    DB에서 거래의 진입 가격을 조회하는 함수
    
    Args:
        conn: 데이터베이스 연결 객체
        trade_id: 거래 ID
        
    Returns:
        float 또는 None: 진입 가격. 실패 시 None 반환
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entry_price FROM trades
            WHERE trade_id = ?
        """, (trade_id,))
        
        result = cursor.fetchone()
        if result and result[0]:
            return float(result[0])
        
        print(f"[경고] 거래 ID {trade_id}의 진입 가격을 찾을 수 없습니다.")
        return None
    
    except Exception as e:
        print(f"[오류] DB에서 진입 가격 조회 중 문제 발생: {e}")
        return None

def get_position_size(symbol: str = SYMBOL, side: str = "Buy") -> float:
    """포지션 크기 조회"""
    try:
        response = bybit_client.get_positions(category="linear", symbol=symbol)
        if response["retCode"] == 0:
            positions = response["result"]["list"]
            for position in positions:
                if position["side"].lower() == side.lower():
                    return float(position.get("size", 0.0))
        return 0.0
    except Exception as e:
        print(f"[오류] 포지션 크기 조회 중 문제 발생: {e}")
        return 0.0

def get_actual_entry_price(symbol: str) -> float:
    """
    바이비트에서 현재 활성화된 포지션의 실제 진입가(entry price)를 불러옵니다.
    :param symbol: 거래 심볼 (예: 'BTCUSDT')
    :return: 진입가 (float) 또는 실패 시 None
    """
    try:
        response = session.get('/v5/position/list', params={
            'category': 'linear',
            'symbol': symbol
        })
        response.raise_for_status()
        data = response.json()

        if data['retCode'] != 0:
            logger.error(f"[오류] 진입가 조회 실패 - Bybit API 응답 오류: {data['retMsg']}")
            return None

        position_list = data['result'].get('list', [])
        if not position_list:
            logger.warning(f"[정보] 활성 포지션 없음 - symbol: {symbol}")
            return None

        entry_price = float(position_list[0]['entryPrice'])
        logger.info(f"[조회 완료] 실제 진입가 (symbol: {symbol}) → {entry_price}")
        return entry_price

    except Exception as e:
        logger.exception(f"[예외 발생] 진입가 조회 중 오류: {e}")
        return None


def cancel_stop_orders(symbol: str, position_idx: int) -> bool:
    """기존 스탑로스 주문 취소"""
    try:
        active_orders = bybit_client.get_open_orders(
            category="linear", 
            symbol=symbol, 
            positionIdx=position_idx,
            orderFilter="StopOrder"
        )
        if active_orders["retCode"] != 0:
            print(f"[경고] 활성 주문 조회 실패: {active_orders['retMsg']}")
            return False
        stop_orders = [order for order in active_orders["result"]["list"] if order["orderType"] in ["StopMarket", "StopLimit"]]
        if not stop_orders:
            print("[정보] 취소할 스탑로스 주문이 없습니다")
            return True
        for order in stop_orders:
            cancel_response = bybit_client.cancel_order(category="linear", symbol=symbol, orderId=order["orderId"])
            if cancel_response["retCode"] != 0:
                print(f"[경고] 주문 ID {order['orderId']} 취소 실패: {cancel_response['retMsg']}")
                return False
        print(f"[성공] {len(stop_orders)}개의 스탑로스 주문 취소 완료")
        return True
    except Exception as e:
        print(f"[오류] 스탑로스 주문 취소 중 문제 발생: {e}")
        return False

def cancel_all_conditional_orders(symbol: str, position_idx: int) -> bool:
    """스탑로스 및 테이크프로핏 등 모든 조건부 주문 취소"""
    try:
        active_orders = bybit_client.get_open_orders(
            category="linear", 
            symbol=symbol, 
            positionIdx=position_idx,
            orderFilter="StopOrder"
        )
        if active_orders["retCode"] != 0:
            print(f"[경고] 활성 주문 조회 실패: {active_orders['retMsg']}")
            return False
            
        conditional_orders = active_orders["result"]["list"]
        if not conditional_orders:
            print("[정보] 취소할 조건부 주문이 없습니다")
            return True
            
        for order in conditional_orders:
            cancel_response = bybit_client.cancel_order(category="linear", symbol=symbol, orderId=order["orderId"])
            if cancel_response["retCode"] != 0:
                print(f"[경고] 주문 ID {order['orderId']} 취소 실패: {cancel_response['retMsg']}")
                
        print(f"[성공] {len(conditional_orders)}개의 조건부 주문 취소 완료")
        return True
    except Exception as e:
        print(f"[오류] 조건부 주문 취소 중 문제 발생: {e}")
        return False


def set_stop_loss(symbol: str, qty: float, entry_price: float, current_price: float, 
                  stop_loss_pct: float, position_type: str, leverage: int = 1, 
                  order_id: Optional[str] = None, decision=None, market_data=None) -> bool:
    """
    AI 기반 손절가 설정 함수 (개선된 버전)
    - 단순 고정 비율 대신 AI 기반으로 기술적 지표와 지지/저항선 분석
    - 리스크 관리 요소도 고려
    - 최소 4% 이상의 손절가 보장
    
    Args:
        symbol: 심볼 (예: "BTCUSDT")
        qty: 포지션 수량
        entry_price: 진입 가격
        current_price: 현재 가격
        stop_loss_pct: 기본 손절 백분율 (AI가 활용하지만 대체될 수 있음)
        position_type: "Long" 또는 "Short"
        leverage: 레버리지
        order_id: 기존 주문 ID (수정 시 사용)
        decision: 거래 결정 객체
        market_data: 시장 데이터 (AI 분석용)
    """
    try:
        print("\n=== 스탑로스 설정 시작 ===")
        print(f"- position_type: {position_type}")
        print(f"- entry_price: {entry_price}")
        print(f"- current_price: {current_price}")
        print(f"- 기본 stop_loss_pct: {stop_loss_pct}")
        print(f"- leverage: {leverage}")
        
        # 포지션 크기 및 지갑 잔액 확인
        position_size_pct = decision.position_size if decision and hasattr(decision, 'position_size') else None
        current_balance = get_wallet_balance()
        
        # AI 기반 손절가 계산 (market_data가 있을 경우)
        if market_data:
            stop_price = calculate_dynamic_stop_loss(
                entry_price, 
                market_data['timeframes']["15"],
                position_type,
                leverage,
                position_size_pct=position_size_pct,
                wallet_balance=current_balance
            )
            print(f"[동적 스탑로스] 계산된 손절가: ${stop_price:.2f}")
            
            # 실제 백분율 차이 계산 (필요시)
            adjusted_stop_loss_pct = abs((entry_price - stop_price) / entry_price) * 100
            print(f"[동적 스탑로스] 백분율: {adjusted_stop_loss_pct:.2f}%")
        
        else:
            # market_data가 없으면 기존 로직 사용
            adjusted_stop_loss_pct = adjust_stop_loss_for_leverage(
                stop_loss_pct, 
                leverage, 
                position_size_pct=position_size_pct,
                wallet_balance=current_balance
            )
            print(f"[기존 로직] 조정된 stop_loss_pct: {adjusted_stop_loss_pct}")
            
            # 손절가 계산 - 최소 4% 이상 보장
            MIN_STOP_LOSS_PCT = 4.0
            adjusted_stop_loss_pct = max(MIN_STOP_LOSS_PCT, adjusted_stop_loss_pct)
            
            if position_type == "Long":
                stop_price = round(entry_price * (1 - adjusted_stop_loss_pct/100), 1)
            else:  # Short
                stop_price = round(entry_price * (1 + adjusted_stop_loss_pct/100), 1)
            print(f"[기존 로직] 계산된 stop_price: {stop_price}")
        
        # 실제 백분율 차이 계산 및 확인
        real_pct = abs((entry_price - stop_price) / entry_price) * 100
        print(f"- 최종 손절가: ${stop_price:.2f}")
        print(f"- 실제 백분율 차이: {real_pct:.2f}%")
        
        position_idx = 1 if position_type == "Long" else 2
        
        # API를 통한 손절가 설정
        response = bybit_client.set_trading_stop(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            stopLoss=str(stop_price),
            slSize=str(qty),
            tpslMode="Partial",
            slTriggerBy="LastPrice"
        )
        
        print(f"[디버깅] Bybit API 응답: {response}")
        if response["retCode"] == 0:
            print(f"[성공] 스탑로스 설정 완료 ({real_pct:.2f}% @ ${stop_price:.2f})")
            
            # 스탑로스 설정 성공 시, DB에 기록
            store_stop_loss_db(position_type, entry_price, stop_price)
            
            time.sleep(1)
            return True
        else:
            print(f"[경고] 스탑로스 설정 실패: {response['retMsg']}")
            return False
    except Exception as e:
        print(f"[오류] 스탑로스 설정 중 문제 발생: {e}")
        print(traceback.format_exc())
        return False

def check_stop_loss_status(symbol: str, position_idx: int) -> Tuple[bool, Optional[float]]:
    """스탑로스 주문 상태 확인 및 가격 반환"""
    try:
        response = bybit_client.get_open_orders(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            orderFilter="StopOrder"
        )
        if response["retCode"] == 0:
            stop_orders = [order for order in response["result"]["list"] if order["orderType"] == "StopMarket"]
            if stop_orders:
                stop_price = float(stop_orders[0]['stopPrice'])
                print("[정보] 스탑로스 주문 확인됨:")
                for order in stop_orders:
                    print(f"- 주문 ID: {order['orderId']}, 스탑로스 가격: {order['stopPrice']}")
                return True, stop_price
            else:
                print("[경고] 활성화된 스탑로스 주문이 없습니다")
                return False, None
        else:
            print(f"[오류] 주문 조회 실패: {response['retMsg']}")
            return False, None
    except Exception as e:
        print(f"[오류] 스탑로스 상태 확인 중 문제 발생: {e}")
        return False, None

def get_latest_trade_id(conn) -> Optional[str]:
    """가장 최근 거래 ID 조회"""
    try:
        if not conn:
            print("[오류] 데이터베이스 연결이 없습니다")
            return None
        c = conn.cursor()
        c.execute("SELECT trade_id FROM trades ORDER BY timestamp DESC LIMIT 1")
        result = c.fetchone()
        if result is None:
            print("[오류] 거래 내역이 없습니다.")
            return None
        return result[0]
    except Exception as e:
        print(f"[오류] 최근 거래 ID 조회 중 문제 발생: {e}")
        return None


def adjust_stop_loss_for_leverage(stop_loss_pct: float, leverage: int, position_size_pct: float = None, wallet_balance: float = None) -> float:
    """
    총자산 대비 리스크 관리를 위한 스탑로스 비율 조정 함수 (개선된 버전)
    
    Args:
        stop_loss_pct: 기본 손절 퍼센트 (예: 2.0은 2% 의미)
        leverage: 현재 설정된 레버리지
        position_size_pct: 포지션 크기 비율 (총자산 대비, 예: 25는 25% 의미)
        wallet_balance: 현재 총자산
        
    Returns:
        리스크 조정된 최종 손절 퍼센트
    """
    # 상수 정의
    MAX_ACCOUNT_RISK_PCT = 5.0  # 계좌 최대 리스크 비율 (5%)
    
    # 최소 손절폭 설정 (4% 이상으로 증가)
    MIN_STOP_LOSS_PCT = 4.0

    # 레버리지에 따른 기본 조정 
    if leverage > 0:
        # 총자산의 MAX_ACCOUNT_RISK_PCT%를 최대 손실로 제한, 하지만 최소한 더 넓게
        max_price_movement = max(MIN_STOP_LOSS_PCT, MAX_ACCOUNT_RISK_PCT / leverage)
    else:
        max_price_movement = MAX_ACCOUNT_RISK_PCT  # 레버리지가 0이면 기본값 사용
    
    # 포지션 크기와 총자산이 제공되었다면 총자산 대비 리스크 계산
    if position_size_pct is not None and wallet_balance is not None:
        # 총자산 중 실제 투자되는 금액 계산
        invested_amount = (position_size_pct / 100) * wallet_balance
        
        # 허용 가능한 최대 손실액 (총자산의 MAX_ACCOUNT_RISK_PCT%)
        max_loss_amount = (MAX_ACCOUNT_RISK_PCT / 100) * wallet_balance
        
        # 투자 금액 대비 최대 손실 비율 계산
        max_investment_loss_pct = (max_loss_amount / invested_amount) * 100
        
        # 레버리지를 고려한 가격 변동 비율 계산
        max_price_movement = max_investment_loss_pct / leverage
    
    # 최종 반환 시 최소값 상향 조정
    return max(MIN_STOP_LOSS_PCT, min(stop_loss_pct, max_price_movement))


def print_current_profit_rate():
    # 최신 시장 데이터(캔들 데이터)를 가져옵니다.
    market_data = get_market_data()
    if market_data is None:
        print("시장 데이터를 가져올 수 없습니다.")
        return

    latest_data = market_data['candlestick'].iloc[-1]
    current_price = float(latest_data['Close'])

    # 실시간 포지션 정보를 가져옵니다.
    positions = get_position_info(bybit_client)
    if positions is None:
        print("포지션 정보를 가져올 수 없습니다.")
        return

    # Long 포지션이 있는 경우, 레버리지 포함 수익률 계산 및 출력
    if positions["Long"]["size"] > 0:
        entry_price = positions["Long"]["entry_price"]
        leverage = positions["Long"].get("leverage", 1)
        profit_pct = (((current_price - entry_price) / entry_price) * 100) * leverage
        print(f"현재 Long 포지션의 (레버리지 포함) 미실현 수익률: {profit_pct:.2f}%")
    else:
        print("현재 Long 포지션이 없습니다.")

    # Short 포지션이 있는 경우, 레버리지 포함 수익률 계산 및 출력
    if positions["Short"]["size"] > 0:
        entry_price = positions["Short"]["entry_price"]
        leverage = positions["Short"].get("leverage", 1)
        profit_pct = (((entry_price - current_price) / entry_price) * 100) * leverage
        print(f"현재 Short 포지션의 (레버리지 포함) 미실현 수익률: {profit_pct:.2f}%")
    else:
        print("현재 Short 포지션이 없습니다.")

if __name__ == "__main__":
    print("비트코인 트레이딩 봇 시작...")
    print("15분 간격으로 실행됩니다 (00분, 15분, 30분, 45분)")
    run_trading_bot() 

    # 테스트를 위해 현재 포지션의 미실현 수익률을 출력하고 싶다면:
    print_current_profit_rate() 
