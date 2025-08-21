# backend_aiohttp/main.py
import argparse
import asyncio
import json
import os
import logging
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
from cachetools import TTLCache  # 추가: cachetools import for TTLCache
# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# 환경 변수
API_SECRET = os.environ.get("API_SECRET", "default_secret")
API_SALT = os.environ.get("API_SALT", "default_salt").encode()  # 추가: 환경 변수로 salt 관리 (강화)
parser = argparse.ArgumentParser()
parser.add_argument("--serve-static", action="store_true")
args = parser.parse_args()
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
def decrypt(enc, secret):
    try:
        enc = b64decode(enc)
        iv = enc[:16]
        cipher_text = enc[16:]
        # PBKDF2로 키 유도 (강화: salt 추가)
        key = pbkdf2_hmac('sha256', secret.encode(), API_SALT, 100000, dklen=32)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted = cipher.decrypt(cipher_text)
        pad = decrypted[-1]
        return decrypted[:-pad].decode("utf-8")
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
            api_key = decrypt(global_api_keys[provider], secret)
        except ValueError:
            return None
    return api_key
# 글로벌 캐시: TTLCache로 변경 (LRU maxsize=100, TTL=3600초=1시간)
client_cache = TTLCache(maxsize=100, ttl=3600)
async def handle_error(e, node_id=None, model=None, retry_func=None, retries_left=3):
    error_msg = str(e).lower()
    context = f"(노드 ID: {node_id or '없음'}, 모델: {model or '없음'})"
    if "401" in error_msg or "unauthorized" in error_msg or "invalid api key" in error_msg:
        error_type = "유효하지 않은 API 키 (401 Unauthorized)"
    elif "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg or "빈 응답" in error_msg:
        error_type = "할당량 제한 초과 (429 Rate Limit)"
        if retry_func and retries_left > 0:
            wait_time = 2 ** (3 - retries_left)  # exponential backoff: 1, 2, 4초
            logger.warning(f"{error_type} {context}: {str(e)} - {wait_time}초 후 재시도 (남은 횟수: {retries_left})")
            await asyncio.sleep(wait_time)
            return await retry_func(retries_left - 1)
    elif "network" in error_msg or "connection" in error_msg or isinstance(e, ClientError):
        error_type = "네트워크 오류"
    else:
        error_type = "알 수 없는 오류"
    logger.error(f"{error_type} {context}: {str(e)}")
    return f"{error_type}: {str(e)}"
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
        error_msg = await handle_error(e, node_id, model)
        raise ValueError(error_msg)
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info("WebSocket 연결 수신")
    global_api_keys = {'openai': None, 'google': None, 'xai': None}
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            data = json.loads(msg.data)
            msg_type = data.get("type")
            if msg_type == "setGlobalApiKey":
                provider = data.get("provider")
                if provider in global_api_keys:
                    global_api_keys[provider] = data.get("key")
                    await ws.send_json({"type": "globalKeySet", "provider": provider})
                continue
            elif msg_type == "getGlobalApiKey":
                await ws.send_json({"type": "globalApiKeys", "keys": global_api_keys})
                continue
            if msg_type == "executeGraph":
                try:
                    graph = data.get("graph")
                    await process_graph(ws, graph, global_api_keys)
                except Exception as e:
                    error_msg = await handle_error(e)
                    await ws.send_json({"type": "error", "message": error_msg})
            else:
                logger.warning(f"알 수 없는 메시지 타입: {msg_type}")
        elif msg.type == web.WSMsgType.ERROR:
            logger.error(f"웹소켓 연결 오류: {ws.exception()}")
    return ws
async def process_model(ws, model_id, prompt, node_data, model_name, output_ids, global_api_keys):
    api_key = get_effective_api_key(node_data, global_api_keys, API_SECRET, model_name)
    if not api_key:
        for output_id in output_ids:
            await ws.send_json({"nodeId": output_id, "type": "error", "message": "API 키가 제공되지 않았습니다."})
        return None
    salted_api_key = api_key + API_SECRET  # 추가: 캐시 키에 salt 적용 (강화)
    cache_key = (salted_api_key, model_name)  # 변경: salted 캐시 키
    if cache_key in client_cache:
        client = client_cache[cache_key]  # fresh cache 재사용 (TTL로 만료됨)
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
            for output_id in output_ids:
                await ws.send_json({"nodeId": output_id, "type": "complete", "content": full_response})
        except Exception as e:
            raise e
    async def retry_wrapper(retries_left=3):
        try:
            await perform_stream()
            return full_response
        except Exception as e:
            error_msg = await handle_error(e, model_id, model_name, lambda r=retries_left: retry_wrapper(r), retries_left)
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
        salted_api_key = api_key + API_SECRET  # 추가: 검증 시 salt 적용
        cache_key = (salted_api_key, model)  # 변경: salted 캐시 키
        if cache_key in client_cache:
            continue
        try:
            await validate_client(api_key, model, cache_key)
        except ValueError as ve:
            await ws.send_json({"type": "error", "message": str(ve)})
            return False
    return True
async def process_graph(ws, graph, global_api_keys):
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
                        if source_node["type"] == "condition":
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
                merged = " ".join(inputs_vals)
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
            elif current["type"] == "model":
                prompt = " ".join(inputs_vals)
                model_name = current["data"]["selectedModel"]
                output_ids = list(outgoing[node_id])
                task = asyncio.create_task(
                    process_model(ws, node_id, prompt, current["data"], model_name, output_ids, global_api_keys)
                )
                current_tasks.append((task, node_id))
            elif current["type"] == "output":
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
if args.serve_static:
    dist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
    if os.path.exists(dist_path):
        async def redirect_to_index(request):
            raise web.HTTPFound('/index.html')  # 추가: 리다이렉트 함수

        app.router.add_get('/', redirect_to_index)  # 추가: / 에 리다이렉트 연결
        app.router.add_static("/", path=dist_path, name="static", show_index=True)  # 기존 static (True 유지 OK)
        logger.info(f"Static 파일 서빙 중: {dist_path}")
    else:
        logger.warning("dist 폴더가 없음, static 서빙 비활성화")
port = int(os.environ.get("PORT", 8080))
if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=port)
