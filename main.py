# 파일: backend_aiohttp/main.py
import argparse
import asyncio
import json
import os
import logging
import time # 추가: time 모듈 import for expiry
from aiohttp import web, ClientError
import aiohttp_cors
from openai import AsyncOpenAI
import google.genai as genai # 변경: google-genai import (latest SDK)
from xai_sdk import AsyncClient
from xai_sdk.chat import user
from collections import defaultdict
from base64 import b64decode
from hashlib import sha256, pbkdf2_hmac
from Crypto.Cipher import AES
from graphlib import TopologicalSorter
from cachetools import TTLCache # 추가: cachetools import for TTLCache
import uuid # 추가: 세션 ID 생성을 위해
import secrets  # 추가: 랜덤 salt 생성을 위해
# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# 환경 변수
API_SECRET = os.environ.get("API_SECRET", "default_secret")
# 수정: salt 길이 확대 (기본 32바이트 랜덤 값, 환경 변수 우선)
API_SALT = os.environ.get("API_SALT")
if API_SALT:
    API_SALT = API_SALT.encode()
else:
    API_SALT = secrets.token_bytes(32)  # 32바이트 랜덤 salt
args = type("Args", (), {})()
args.serve_static = True  # Always serve static
app = web.Application()
cors = aiohttp_cors.setup(
    app,
    defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, expose_headers="*", allow_headers="*"
        )
    },
)
for route in list(app.router.routes()):
    cors.add(route)
# 파일 기반 글로벌 키 저장 (영속화)
API_KEYS_FILE = 'api_keys.json'
def load_global_api_keys():
    try:
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'openai': None, 'google': None, 'xai': None}
def save_global_api_keys(keys):
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(keys, f)
def decrypt(enc, secret):
    try:
        enc = b64decode(enc)
        iv = enc[:16]
        cipher_text = enc[16:]
        # PBKDF2로 키 유도 (강화: iterations 600000으로 증가)
        key = pbkdf2_hmac('sha256', secret.encode(), API_SALT, 600000, dklen=32)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(cipher_text)
        pad = decrypted[-1]
        decrypted_str = decrypted[:-pad].decode("utf-8")
        # 추가 검증: 길이 (32~128자)와 형식 (알파벳, 숫자, -, _ 만 허용)
        if len(decrypted_str) < 32 or len(decrypted_str) > 128:
            raise ValueError("Invalid key length (must be 32-128 characters)")
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', decrypted_str):
            raise ValueError("Invalid key format (only alphanumeric, -, _ allowed)")
        return decrypted_str
    except Exception as e:
        logger.error(f"복호화 오류: {str(e)}")
        raise ValueError("API 키 복호화 실패")
def get_provider_from_model(model):
    if model.startswith("openai/"):
        return "openai"
    elif model.startswith("google/"):
        return "google"
    elif model.startswith("xai/"):
        return "xai"
    else:
        raise ValueError("지원되지 않는 모델입니다.")
def get_effective_api_key(node_data, global_api_keys, secret, model):
    api_key = node_data.get("apiKey", "")
    provider = get_provider_from_model(model)
    if not api_key and global_api_keys.get(provider):
        try:
            # 추가: 만료 시간 검사
            expiry = global_api_keys[provider].get('expiry', 0)
            if time.time() > expiry:
                raise ValueError("API key has expired")
            api_key = decrypt(global_api_keys[provider]['enc'], secret)
            # 추가 검증: provider별 형식 체크 (강화)
            if provider == 'openai' and not api_key.startswith('sk-'):
                raise ValueError("Invalid OpenAI key format (must start with 'sk-')")
            elif provider == 'google' and not api_key.startswith('AIza'):
                raise ValueError("Invalid Google key format (must start with 'AIza')")
            elif provider == 'xai' and len(api_key) < 40:
                raise ValueError("Invalid xAI key length (must be at least 40 characters)")
        except ValueError:
            return None
    return api_key
# 글로벌 캐시: TTLCache로 변경 (LRU maxsize=100, TTL=3600초=1시간)
client_cache = TTLCache(maxsize=100, ttl=3600)
# 추가: 모델별 응답 캐시 (동적 TTL)
response_caches = {
    "openai": TTLCache(maxsize=100, ttl=1800),
    "google": TTLCache(maxsize=100, ttl=3600),
    "xai": TTLCache(maxsize=100, ttl=3600)
}
# 추가: 환경 변수로 캐싱 비활성화 옵션 (구름 IDE에서 export DISABLE_CACHE=1 로 설정 가능)
DISABLE_CACHE = os.environ.get("DISABLE_CACHE", "0") == "1"
# 중앙 에러 핸들러 클래스
class ErrorHandler:
    async def handle(self, e, node_id=None, model=None, retry_func=None, retries_left=5):
        error_msg = str(e).lower()
        context = f"(노드 ID: {node_id or '없음'}, 모델: {model or '없음'})"
        if "401" in error_msg or "unauthorized" in error_msg or "invalid api key" in error_msg:
            error_type = "유효하지 않은 API 키 (401 Unauthorized)"
        elif "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg or "빈 응답" in error_msg:
            error_type = "할당량 제한 초과 (429 Rate Limit)"
            if retry_func and retries_left > 0:
                wait_time = 2 ** (5 - retries_left) # exponential backoff: 1, 2, 4, 8, 16초
                logger.warning(f"{error_type} {context}: {str(e)} - {wait_time}초 후 재시도 (남은 횟수: {retries_left})")
                await asyncio.sleep(wait_time)
                return await retry_func(retries_left - 1)
            else:
                logger.error(f"최대 재시도 횟수 초과: {error_type} {context}: {str(e)}", exc_info=True)
                return f"최대 재시도 횟수 초과: {error_type}: {str(e)}"
        elif "network" in error_msg or "connection" in error_msg or isinstance(e, ClientError):
            error_type = "네트워크 오류"
        else:
            error_type = "알 수 없는 오류"
        logger.error(f"{error_type} {context}: {str(e)}", exc_info=True) # 상세 로그 기록 (스택 트레이스 포함)
        return f"{error_type}: {str(e)}"
error_handler = ErrorHandler()
async def validate_client(api_key, model, cache_key, node_id=None):
    try:
        if model.startswith("openai/"):
            client = AsyncOpenAI(api_key=api_key)
            await client.models.list()
            client_cache[cache_key] = client
        elif model.startswith("google/"):
            client = genai.Client(api_key=api_key)
            await client.aio.models.list()
            client_cache[cache_key] = client
        elif model.startswith("xai/"):
            client = AsyncClient(api_key=api_key)
            chat = client.chat.create(model=model.split("/")[-1])
            del chat
            client_cache[cache_key] = client
        else:
            raise ValueError("지원되지 않는 모델입니다.")
    except Exception as e:
        error_msg = await error_handler.handle(e, node_id, model)
        raise ValueError(error_msg)
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    session_id = str(uuid.uuid4()) # 추가: 세션 ID 생성
    logger.info(f"WebSocket 연결 수신, session_id: {session_id}")
    global_api_keys = load_global_api_keys()
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)
            msg_type = data.get("type")
            # 추가: heartbeat 핸들러 (ping 받으면 pong 응답)
            if msg_type == 'ping':
                await ws.send_json({"type": "pong"})
                continue
            if msg_type == "setGlobalApiKey":
                provider = data.get("provider")
                enc_key = data.get("key")
                received_hash = data.get("hash")
                if provider in global_api_keys:
                    try:
                        # 복호화 및 해시 검증
                        dec_key = decrypt(enc_key, API_SECRET)
                        computed_hash = sha256(dec_key.encode()).hexdigest()
                        if computed_hash != received_hash:
                            raise ValueError("Hash mismatch: Key integrity check failed")
                        # 수정: 만료 시간 추가 (현재 시간 + 3600초)
                        expiry = time.time() + 3600
                        global_api_keys[provider] = {'enc': enc_key, 'expiry': expiry} # 암호화된 키와 만료 시간 저장
                        save_global_api_keys(global_api_keys)
                        await ws.send_json({"type": "globalKeySet", "provider": provider})
                    except Exception as e:
                        logger.error(f"Set key error for {provider}: {str(e)}")
                        await ws.send_json({"type": "error", "message": str(e)})
                continue
            elif msg_type == "getGlobalApiKey":
                # 수정: get 시 만료 시간 검사 (만료된 키는 null로 반환)
                for provider in global_api_keys:
                    if global_api_keys[provider] and time.time() > global_api_keys[provider].get('expiry', 0):
                        global_api_keys[provider] = None
                await ws.send_json({"type": "globalApiKeys", "keys": {p: g['enc'] if g else None for p, g in global_api_keys.items()}})
                continue
            if msg_type == "executeGraph":
                try:
                    graph = data.get("graph")
                    await process_graph(ws, graph, global_api_keys, session_id) # 세션 ID 전달
                except Exception as e:
                    error_msg = await error_handler.handle(e)
                    try:
                        await ws.send_json({"type": "error", "message": error_msg})
                    except ConnectionResetError:
                        logger.warning("Connection reset while sending error message, ignoring.")
            else:
                logger.warning(f"알 수 없는 메시지 타입: {msg_type}")
        elif msg.type == web.WSMsgType.ERROR:
            logger.error(f"웹소켓 연결 오류: {ws.exception()}")
    return ws
async def process_model(ws, model_id, prompt, node_data, model_name, output_ids, global_api_keys, session_id, execution_id):
    api_key = get_effective_api_key(node_data, global_api_keys, API_SECRET, model_name)
    if not api_key:
        for output_id in output_ids:
            await ws.send_json({"nodeId": output_id, "type": "error", "message": "API 키가 제공되지 않았습니다."})
        return None
    salted_api_key = api_key + API_SECRET # 추가: 캐시 키에 salt 적용 (강화)
    cache_key = (salted_api_key, model_name) # 변경: salted 캐시 키
    if cache_key in client_cache:
        client = client_cache[cache_key] # fresh cache 재사용 (TTL로 만료됨)
    else:
        try:
            await validate_client(api_key, model_name, cache_key, model_id)
            client = client_cache[cache_key]
        except ValueError as ve:
            for output_id in output_ids:
                await ws.send_json({"nodeId": output_id, "type": "error", "message": str(ve)})
            return None
    for output_id in output_ids:
        await ws.send_json({"nodeId": output_id, "type": "processing"})
    full_response = ""
    # 추가: 응답 캐싱 체크
    provider = get_provider_from_model(model_name)
    response_cache = response_caches[provider]
    prompt_hash = sha256(prompt.encode()).hexdigest()
    response_key = (prompt_hash, session_id, execution_id)
    if DISABLE_CACHE:
        logger.info(f"Cache disabled for response_key: {response_key} (model: {model_name})")
    else:
        if response_key in response_cache:
            logger.info(f"Cache hit for response_key: {response_key} (model: {model_name})")
            full_response = response_cache[response_key]
            for output_id in output_ids:
                await ws.send_json({"nodeId": output_id, "type": "complete", "content": full_response})
            return full_response
        else:
            logger.info(f"Cache miss for response_key: {response_key} (model: {model_name})")
    async def perform_stream():
        nonlocal full_response
        try:
            if model_name.startswith("openai/"):
                stream = await client.chat.completions.create(
                    model=model_name.split("/")[-1],
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        for output_id in output_ids:
                            await ws.send_json({"nodeId": output_id, "type": "chunk", "content": content})
            elif model_name.startswith("google/"):
                stream = await client.aio.models.generate_content_stream(
                    model=model_name.split("/")[-1], contents=[prompt]
                )
                async for chunk in stream:
                    if chunk.text:
                        content = chunk.text
                        full_response += content
                        for output_id in output_ids:
                            await ws.send_json({"nodeId": output_id, "type": "chunk", "content": content})
                if not full_response:
                    raise ValueError("빈 응답 수신")
                logger.info(f"Gemini 응답 길이: {len(full_response)}")
            elif model_name.startswith("xai/"):
                chat = client.chat.create(model=model_name.split("/")[-1])
                chat.append(user(prompt))
                async for response, chunk in chat.stream():
                    if chunk.content is not None:
                        content = chunk.content
                        full_response += content
                        for output_id in output_ids:
                            await ws.send_json({"nodeId": output_id, "type": "chunk", "content": content})
            # 추가: 응답 캐싱
            if not DISABLE_CACHE:
                response_cache[response_key] = full_response
            for output_id in output_ids:
                await ws.send_json({"nodeId": output_id, "type": "complete", "content": full_response})
        except Exception as e:
            raise e
    async def retry_wrapper(retries_left=5):
        try:
            await perform_stream()
            return full_response
        except Exception as e:
            error_msg = await error_handler.handle(e, model_id, model_name, lambda r=retries_left: retry_wrapper(r), retries_left)
            if error_msg:
                for output_id in output_ids:
                    await ws.send_json({"nodeId": output_id, "type": "error", "message": error_msg})
            return None
    return await retry_wrapper()
# 그래프 빌드 함수 분리
def build_graph(nodes, edges):
    outgoing = defaultdict(set)
    incoming = defaultdict(lambda: defaultdict(list))
    predecessors = {node_id: set() for node_id in nodes}
    for edge in edges:
        outgoing[edge["source"]].add(edge["target"])
        incoming[edge["target"]][edge["targetHandle"]].append(edge["source"])
        predecessors[edge["target"]].add(edge["source"])
    return outgoing, incoming, predecessors
# 사이클 감지 함수 분리
async def detect_cycle(ws, predecessors):
    ts = TopologicalSorter(predecessors)
    try:
        ts.prepare()
    except ValueError:
        error_msg = "순환 그래프가 감지되었습니다. 그래프를 수정해주세요."
        logger.error(error_msg)
        await ws.send_json({"type": "error", "message": error_msg})
        return None
    return ts
# 모든 모델 검증 함수 분리
async def validate_all_models(ws, nodes, global_api_keys):
    unique_pairs = set()
    for node_id, node in nodes.items():
        if node["type"] == "model":
            model = node["data"]["selectedModel"]
            api_key = get_effective_api_key(node["data"], global_api_keys, API_SECRET, model)
            if not api_key:
                await ws.send_json({"type": "error", "message": "API 키가 제공되지 않았습니다."})
                return False
            unique_pairs.add((api_key, model))
    for api_key, model in unique_pairs:
        salted_api_key = api_key + API_SECRET # 추가: 검증 시 salt 적용
        cache_key = (salted_api_key, model) # 변경: salted 캐시 키
        if cache_key in client_cache:
            continue
        try:
            await validate_client(api_key, model, cache_key)
        except ValueError as ve:
            await ws.send_json({"type": "error", "message": str(ve)})
            return False
    return True
async def process_graph(ws, graph, global_api_keys, session_id):
    # 추가: execution_id 추출 (프론트에서 보낸 executionId)
    execution_id = graph.get("executionId", str(uuid.uuid4())) # 없으면 새로 생성
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    outgoing, incoming, predecessors = build_graph(nodes, edges)
    ts = await detect_cycle(ws, predecessors)
    if ts is None:
        return
    if not await validate_all_models(ws, nodes, global_api_keys):
        return
    node_outputs = {}
    while ts.is_active():
        ready = list(ts.get_ready())
        if not ready:
            break
        current_tasks = []
        for node_id in ready:
            current = nodes.get(node_id)
            if not current:
                ts.done(node_id)
                continue
            if not current["data"].get("active", True):
                ts.done(node_id)
                continue
            if current["type"] == "group":
                ts.done(node_id)
                continue
            port_order = [p["id"] for p in current["data"].get("inputPorts", [])]
            priority_order = current["data"].get("priorityOrder", [])
            inputs_vals = []
            for port_id in port_order:
                sources = incoming[node_id].get(port_id, [])
                sources = sorted(sources, key=lambda s: priority_order.index(s) if s in priority_order else len(priority_order))
                vals = []
                for source in sources:
                    edge = next((e for e in edges if e["source"] == source and e["target"] == node_id and e["targetHandle"] == port_id), None)
                    if edge:
                        source_node = nodes.get(source)
                        if source_node["type"] in ["condition", "split"]:
                            if source_node["type"] == "split":
                                val = node_outputs.get(node_id, "")
                            else:
                                val = node_outputs.get(source + '_' + edge["sourceHandle"], "")
                        else:
                            val = node_outputs.get(source, "")
                        vals.append(str(val))
                combined = " ".join(vals) if vals else ""
                inputs_vals.append(combined)
            if current["type"] == "input":
                node_outputs[node_id] = current["data"].get("input", "")
                ts.done(node_id)
            elif current["type"] == "file":
                node_outputs[node_id] = current["data"].get("content", "")
                ts.done(node_id)
            elif current["type"] == "merge":
                all_sources = set()
                for port_id in port_order:
                    sources = incoming[node_id].get(port_id, [])
                    all_sources.update(sources)
                sorted_sources = sorted(all_sources, key=lambda s: priority_order.index(s) if s in priority_order else len(priority_order))
                vals = [str(node_outputs.get(s, "")) for s in sorted_sources]
                merged = " ".join(vals)
                node_outputs[node_id] = merged
                await ws.send_json({"nodeId": node_id, "type": "output_log", "logs": merged})
                logger.info(f"Merge 노드 {node_id}에 합쳐진 결과 저장: {merged[:50]}...")
                ts.done(node_id)
            elif current["type"] == "condition":
                combined = " ".join(inputs_vals)
                condition_vals = [str(node_outputs.get(selection["sourceId"], "")) for selection in current["data"].get("conditionSelections", [])]
                condition_val = " ".join(condition_vals) or combined
                op = current["data"].get("operator", "==")
                value = current["data"].get("value", "")
                try:
                    if op in ['>', '>=', '<', '<=']:
                        c = float(condition_val)
                        v = float(value)
                    else:
                        c = condition_val
                        v = value
                    if op == '==':
                        is_true = c == v
                    elif op == '>':
                        is_true = c > v
                    elif op == '>=':
                        is_true = c >= v
                    elif op == '<':
                        is_true = c < v
                    elif op == '<=':
                        is_true = c <= v
                    elif op == 'contains':
                        is_true = v in c
                    else:
                        is_true = False
                except ValueError:
                    logger.warning(f"Condition evaluation error in node {node_id}: invalid numeric value")
                    is_true = False
                true_selections = current["data"].get("trueSelections", [])
                false_selections = current["data"].get("falseSelections", [])
                true_value = " ".join([str(node_outputs.get(s["sourceId"], "")) for s in true_selections])
                false_value = " ".join([str(node_outputs.get(s["sourceId"], "")) for s in false_selections])
                output_ports = current["data"].get("outputPorts", [])
                if len(output_ports) >= 2:
                    true_port = output_ports[0]["id"]
                    false_port = output_ports[1]["id"]
                    node_outputs[node_id + '_' + true_port] = true_value if is_true else ""
                    node_outputs[node_id + '_' + false_port] = false_value if not is_true else ""
                ts.done(node_id)
            elif current["type"] == "split":
                combined = " ".join(inputs_vals) # 입력 텍스트 결합
                mode = current["data"].get("mode", "fixed")
                size = int(current["data"].get("size", 100))
                unit = current["data"].get("unit", "chars") if mode == "fixed" else None
                overlap_enabled = current["data"].get("overlapEnabled", False)
                overlap = int(current["data"].get("overlap", 0)) if overlap_enabled else 0
                max_chunks = int(current["data"].get("maxChunks", 5))
                delimiter = current["data"].get("delimiter", "") if mode == "delimiter" else None
                # 프론트에서 설정한 selectedOutputIds 추출 (출력 노드 ID 목록)
                selected_output_ids = [chunk.get("selectedOutputId") for chunk in current["data"].get("chunks", []) if chunk.get("selectedOutputId")]
                # 추가: 연결된 output에 processing 메시지 보내기 (로그 초기화 트리거)
                output_ids = list(outgoing[node_id])
                for output_id in output_ids:
                    await ws.send_json({"nodeId": output_id, "type": "processing"})
                    node_outputs[output_id] = "" # 백엔드 node_outputs 초기화 (condition 평가용)
                # 변동 길이 처리: 총 길이 계산
                total_length = len(combined.split()) if unit == "words" else len(combined)
                chunks = []
                start = 0
                if mode == "delimiter" and delimiter:
                    # delimiter 모드: 구분자 기준 분할 (공백 제거)
                    chunk_texts = [chunk.strip() for chunk in combined.split(delimiter) if chunk.strip()]
                    chunks = chunk_texts
                else:
                    while start < total_length:
                        if mode == "percent":
                            chunk_size = int(total_length * (size / 100))
                            overlap_amount = int(chunk_size * (overlap / 100)) if overlap_enabled else 0
                        else:
                            chunk_size = size
                            overlap_amount = overlap if overlap_enabled else 0
                        end = min(start + chunk_size, total_length)
                     
                        if unit == "words":
                            words = combined.split()
                            chunk_text = " ".join(words[start:end])
                        else:
                            chunk_text = combined[start:end]
                     
                        chunks.append(chunk_text)
                        start = end - overlap_amount
                # max_chunks 초과 시 병합 (fixed와 delimiter 모드에서 적용)
                if (mode == "fixed" or mode == "delimiter") and len(chunks) > max_chunks:
                    merged_chunks = []
                    chunk_per_group = len(chunks) // max_chunks
                    remainder = len(chunks) % max_chunks
                    idx = 0
                    for i in range(max_chunks):
                        group_size = chunk_per_group + (1 if i < remainder else 0)
                        merged = " ".join(chunks[idx:idx + group_size])
                        merged_chunks.append(merged)
                        idx += group_size
                    chunks = merged_chunks
                # 청크를 selected_output_ids에 따라 라우팅 (selected_output_ids가 있으면 우선 사용, 없으면 순환 라우팅)
                output_ids = list(outgoing[node_id]) # 연결된 출력 노드 ID들
                for i, chunk in enumerate(chunks):
                    if selected_output_ids:
                        # 프론트 설정 우선: selected_output_ids 순환
                        output_id = selected_output_ids[i % len(selected_output_ids)]
                    else:
                        # 기본 순환 라우팅
                        output_id = output_ids[i % len(output_ids)]
                    # 수정: "chunk" 메시지 전송 제거 (중복 문제 방지)
                    node_outputs[output_id] += chunk # 백엔드 node_outputs 누적 (condition 평가용)
             
                # 완료 시 (unique output_id에만 보내기, content 포함)
                unique_output_ids = set(output_ids) | set(selected_output_ids) # 중복 제거
                for output_id in unique_output_ids:
                    await ws.send_json({"nodeId": output_id, "type": "complete", "content": node_outputs[output_id]}) # content 포함
             
                ts.done(node_id)
            elif current["type"] == "model":
                prompt = " ".join(inputs_vals)
                model_name = current["data"]["selectedModel"]
                output_ids = list(outgoing[node_id])
                task = asyncio.create_task(
                    process_model(ws, node_id, prompt, current["data"], model_name, output_ids, global_api_keys, session_id, execution_id) # 세션 ID 전달
                )
                current_tasks.append((task, node_id))
            elif current["type"] == "output":
                all_sources = set()
                for port_id in port_order:
                    sources = incoming[node_id].get(port_id, [])
                    all_sources.update(sources)
                # 수정: split 입력 시 스킵 제거, node_outputs에서 logs 가져옴 (split에서 이미 설정됨)
                logs = node_outputs.get(node_id, "") # split 입력 시 node_outputs 사용
                if not logs: # 일반 입력 시 (split 아닌 경우)
                    logs = "".join(inputs_vals)
                node_outputs[node_id] = logs
                await ws.send_json({"nodeId": node_id, "type": "output_log", "logs": logs})
                logger.info(f"Display 노드 {node_id}에 로그 저장: {logs[:50]}...")
                ts.done(node_id)
        if current_tasks:
            await asyncio.wait([t for t, _ in current_tasks], return_when=asyncio.ALL_COMPLETED)
            for task, node_id in current_tasks:
                try:
                    result = await task
                except Exception as e:
                    result = None
                    logger.error(f"모델 {node_id} 처리 중 오류: {str(e)}")
                node_outputs[node_id] = result if result is not None else ""
                ts.done(node_id)
app.router.add_get("/ws", websocket_handler)
if True:
    dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
    if os.path.exists(dist_path):
        async def redirect_to_index(request):
            raise web.HTTPFound('/index.html') # 추가: 리다이렉트 함수
        app.router.add_get('/', redirect_to_index) # 추가: / 에 리다이렉트 연결
        app.router.add_static("/", path=dist_path, name="static", show_index=True) # 기존 static (True 유지 OK)
        logger.info(f"Static 파일 서빙 중: {dist_path}")
    else:
        logger.warning("dist 폴더가 없음, static 서빙 비활성화")
port = int(os.environ.get("PORT", 8080))
if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=port)
