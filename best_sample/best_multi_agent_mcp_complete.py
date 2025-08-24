"""
Agent Patterns — Constrained ReAct vs FSM (Production-Style Demo, v5)

이번 업데이트(요청 반영)
- ✅ **MCP 스키마 추가**: Jira/GitHub/Drive 툴 스키마 & 아규먼트 빌더(휴리스틱 → LLM JSON 폴백)
- ✅ **Qdrant RAG 확장**: 필터(payload), 네임스페이스(컬렉션 접미사), 하이브리드(Vector+Keyword), 간단 rerank
- ✅ 기존 기능 유지: LangGraph Supervisor, 비용/타임아웃 캡, 휴먼 핸드오프, 실제 API(Open‑Meteo/sqlite3), 싱글/멀티 에이전트

설치/환경
  pip install "langgraph>=0.2" "langchain-openai>=0.2" python-dotenv requests qdrant-client
  # (선택) 임베딩 로컬모델: pip install sentence-transformers
  # (선택) PDF: pip install pypdf
  # (선택) MCP: pip install mcp

환경변수 예시
  export OPENAI_API_KEY=...
  export USE_REAL_APIS=true
  export USE_MCP=true
  export MCP_SERVER_CMD="node"
  export MCP_SERVER_ARGS="/path/to/mcp_server.js"
  export QDRANT_URL="http://localhost:6333"
  export QDRANT_API_KEY=""           # 로컬이면 비워도 됨
  export RAG_COLLECTION="demo_rag"
  export EMBED_PROVIDER="hf"          # 'hf' or 'openai'
  export EMBED_MODEL="BAAI/bge-m3"    # HF용 기본값

실행 예
  # 1) 인덱싱(청킹→임베딩→Qdrant) — 프로젝트 네임스페이스 포함
  uv run python best_sample/best_multi_agent_mcp_complete.py --index_dir ./docs --chunk 800 --overlap 120 --namespace projectA --project projectA

  # 2) 에이전트(멀티 + MCP + 필터된 RAG 하이브리드)
  uv run python best_sample/best_multi_agent_mcp_complete.py --mode react --q "GitHub에서 repo:acme/app 이슈 중 label:bug 관련 최신 근거 찾아 요약" \
    --agents multi --namespace projectA --filter "project:projectA,label:bug" --topk 8 --hybrid 1 --alpha 0.7

  # 3) FSM(결정적 플로우)
  uv run python best_sample/best_multi_agent_mcp_complete.py --mode fsm --city seoul --item umbrella
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import argparse, os, time, sqlite3, asyncio, json, glob, re
import requests

# =====( 옵션: LangGraph / LangChain )=====
try:
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    _LG = True
except Exception:
    _LG = False
    class TypedDict(dict):
        pass
    END = "__END__"

try:
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
    _LC = True
except Exception:
    _LC = False

# =====( 옵션: MCP )=====
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP = True
except Exception:
    _MCP = False

# =====( 옵션: Qdrant )=====
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    _QDR = True
except Exception:
    _QDR = False

# =====( 옵션: Sentence-Transformers 임베딩 )=====
try:
    from sentence_transformers import SentenceTransformer
    _ST = True
except Exception:
    _ST = False

# =====( 글로벌 설정 )=====
USE_REAL_APIS = os.getenv("USE_REAL_APIS", "false").lower() in ("1","true","yes")
USE_MCP = os.getenv("USE_MCP", "false").lower() in ("1","true","yes")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
RAG_COLLECTION = os.getenv("RAG_COLLECTION", "demo_rag")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "hf")  # 'hf' or 'openai'
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

@dataclass
class Caps:
    cost_cap: float = 0.03
    step_cap: int = 10
    deadline_sec: float = 8.0

COST_TABLE = {
    "llm_call": 0.002,
    "tool_search": 0.0005,
    "tool_db": 0.0008,
    "tool_weather": 0.0005,
    "tool_mcp": 0.0010,
    "embed": 0.0009,
    "qdrant_upsert": 0.0012,
    "qdrant_search": 0.0007,
}

# =====( 공통 유틸 )=====
def deadline_exceeded(state) -> bool:
    return (time.perf_counter() - state["meta"]["start_ts"]) > state["meta"]["caps"].deadline_sec

def now_exceeded(start_ts: float, sec: float) -> bool:
    return (time.perf_counter() - start_ts) > sec

# =====( 도구: mock + real )=====
class Tool(Enum):
    SEARCH = "search"
    DB = "db"
    WEATHER = "weather"

CITY_TO_LATLON = {
    "seoul": (37.5665, 126.9780),
    "busan": (35.1796, 129.0756),
    "incheon": (37.4563, 126.7052),
}
OPEN_METEO = (
    "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m&current_weather=true"
)

def tool_search(query: str) -> Dict[str, Any]:
    time.sleep(0.03)
    return {"tool": Tool.SEARCH.value, "items": [
        {"title": "서울 날씨 가이드", "source": "docs/weather_guide.md"},
        {"title": f"키워드('{query}') 관련 문서", "source": "kb/related.md"},
    ]}


def tool_db(sql: str) -> Dict[str, Any]:
    if not USE_REAL_APIS:
        time.sleep(0.02)
        return {"tool": Tool.DB.value, "rows": [{"id": 1, "name": "홍길동"}]}
    try:
        conn = sqlite3.connect(os.getenv("DEMO_DB_PATH", ":memory:"), timeout=0.5)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT)")
        cur.execute("INSERT INTO users VALUES (1, '홍길동')")
        conn.commit()
        cur.execute("SELECT * FROM users LIMIT 1")
        rows = cur.fetchall()
        conn.close()
        return {"tool": Tool.DB.value, "rows": [{"id": rows[0][0], "name": rows[0][1]}]}
    except Exception as e:
        return {"tool": Tool.DB.value, "error": str(e)}


def tool_weather(city_or_query: str) -> Dict[str, Any]:
    if not USE_REAL_APIS:
        time.sleep(0.03)
        return {"tool": Tool.WEATHER.value, "city": city_or_query, "temp_c": 29, "status": "Sunny"}
    city = city_or_query.lower().strip()
    latlon = CITY_TO_LATLON.get(city)
    if not latlon:
        return {"tool": Tool.WEATHER.value, "error": f"city '{city}' not supported in demo"}
    lat, lon = latlon
    url = OPEN_METEO.format(lat=lat, lon=lon)
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        js = r.json()
        cw = js.get("current_weather", {})
        return {
            "tool": Tool.WEATHER.value,
            "city": city.title(),
            "temp_c": cw.get("temperature"),
            "status": cw.get("weathercode", "OK"),
        }
    except Exception as e:
        return {"tool": Tool.WEATHER.value, "error": str(e)}

# =====( Qdrant + 임베딩 )=====
class Embeddings:
    def __init__(self):
        self.provider = EMBED_PROVIDER
        self.model = EMBED_MODEL
        self.st_model = None
        self.openai = None
        if self.provider == "hf" and _ST:
            self.st_model = SentenceTransformer(self.model)
        elif self.provider == "openai" and _LC and os.getenv("OPENAI_API_KEY"):
            self.openai = LCOpenAIEmbeddings(model="text-embedding-3-large")
        else:
            self.st_model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.st_model:
            return [list(map(float, v)) for v in self.st_model.encode(texts, normalize_embeddings=True)]
        if self.openai:
            return self.openai.embed_documents(texts)
        return [[(hash(t) % 997)/997.0 for _ in range(384)] for t in texts]

class QdrantRAG:
    def __init__(self, base_collection: str = RAG_COLLECTION, namespace: str = ""):
        self.base = base_collection
        self.ns = namespace.strip()
        self.collection = f"{self.base}__{self.ns}" if self.ns else self.base
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if _QDR else None
        self.emb = Embeddings()

    def ensure_collection(self, dim: int):
        if not self.client:
            return
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )

    def chunk(self, text: str, chunk: int = 800, overlap: int = 120) -> List[str]:
        out = []
        i = 0
        n = len(text)
        while i < n:
            out.append(text[i:i+chunk])
            i += max(1, chunk - overlap)
        return out

    def load_docs(self, path: str) -> List[Tuple[str, str]]:
        files = []
        if os.path.isdir(path):
            for p in glob.glob(os.path.join(path, "**", "*"), recursive=True):
                if os.path.isfile(p) and any(p.lower().endswith(ext) for ext in [".txt",".md",".pdf"]):
                    files.append(p)
        else:
            files = [path]
        docs = []
        for fp in files:
            try:
                if fp.lower().endswith(".pdf"):
                    try:
                        from pypdf import PdfReader
                        txt = "\n".join(page.extract_text() or "" for page in PdfReader(fp).pages)
                    except Exception:
                        continue
                else:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                if txt.strip():
                    docs.append((fp, txt))
            except Exception:
                continue
        return docs

    def index(self, path: str, project: str = "", chunk: int = 800, overlap: int = 120) -> Dict[str, Any]:
        start = time.perf_counter()
        docs = self.load_docs(path)
        if not docs:
            return {"indexed": 0, "msg": "no docs"}
        chunks, payloads = [], []
        for fp, txt in docs:
            for idx, c in enumerate(self.chunk(txt, chunk, overlap)):
                chunks.append(c)
                payloads.append({"source": fp, "chunk_index": idx, "project": project, "text": c[:1000]})
        vecs = self.emb.embed(chunks)
        dim = len(vecs[0]) if vecs else 0
        self.ensure_collection(dim)
        if not self.client:
            return {"indexed": len(chunks), "msg": "qdrant not available (client None)"}
        points = [qmodels.PointStruct(id=i, vector=vecs[i], payload=payloads[i]) for i in range(len(chunks))]
        self.client.upsert(collection_name=self.collection, points=points)
        return {"indexed": len(chunks), "sec": round(time.perf_counter()-start,2), "collection": self.collection}

    def _filter(self, kv: Dict[str, Any]) -> Optional[qmodels.Filter]:
        if not kv:
            return None
        must = []
        for k, v in kv.items():
            must.append(qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)))
        return qmodels.Filter(must=must)

    def _keyword_score(self, text: str, query: str) -> float:
        # 아주 간단한 키워드 스코어(토이): 중복 제외 매칭 비율
        qs = set([t for t in re.split(r"\W+", query.lower()) if t and len(t) > 2])
        ts = set([t for t in re.split(r"\W+", (text or "").lower()) if t and len(t) > 2])
        if not qs:
            return 0.0
        inter = len(qs & ts)
        return inter / max(1, len(qs))

    def search(self, query: str, top_k: int = 8, filter_kv: Optional[Dict[str, Any]] = None,
               hybrid: bool = True, alpha: float = 0.7) -> List[Dict[str, Any]]:
        # alpha: dense 가중치(0~1). 하이브리드일 때 최종 = alpha*dense + (1-alpha)*keyword
        qv = self.emb.embed([query])[0]
        if not self.client:
            return []
        flt = self._filter(filter_kv)
        # 1) 우선 벡터 topN 넓게 가져오기
        baseN = max(top_k*4, 32)
        res = self.client.search(collection_name=self.collection, query_vector=qv, limit=baseN, query_filter=flt)
        # 2) rerank/hybrid
        scored = []
        for r in res:
            pl = r.payload or {}
            dense = float(r.score)
            kw = self._keyword_score(pl.get("text", ""), query)
            final = (alpha * dense + (1 - alpha) * kw) if hybrid else dense
            scored.append({
                "text": pl.get("text", ""),
                "source": pl.get("source", ""),
                "score": final,
                "dense": dense,
                "kw": kw,
                "project": pl.get("project", ""),
                "chunk_index": pl.get("chunk_index", -1),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

# =====( MCP 클라이언트: 타이트 스키마 + 빌더 )=====
class MCPClient:
    def __init__(self, cmd: Optional[str], args: Optional[str]):
        self.cmd = cmd
        self.args = (args or "").split() if args else []
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._conn = None

    async def start(self):
        if not (_MCP and self.cmd):
            return
        params = StdioServerParameters(command=self.cmd, args=self.args)
        self._conn = await stdio_client(params)
        self.session = self._conn.session
        await self.session.initialize()
        try:
            caps = await self.session.list_tools()
            self.tools = {t.name: {"schema": getattr(t, 'input_schema', None)} for t in getattr(caps, 'tools', [])}
        except Exception:
            self.tools = {}

    async def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session:
            raise RuntimeError("MCP not started")
        res = await self.session.call_tool(tool_name, args)
        return {"tool": f"mcp:{tool_name}", "content": getattr(res, 'content', []), "error": getattr(res, 'is_error', False)}

    async def aclose(self):
        try:
            if self._conn:
                await self._conn.close()
        except Exception:
            pass

# --- 휴리스틱 파서들 ---
def _extract_path(q: str) -> str:
    m = re.search(r"/(?:[\w\-./]+)", q)
    return m.group(0) if m else "/var/log/system.log"

def _extract_date_iso(q: str) -> str:
    from datetime import date
    return str(date.today())

def _find_repo(q: str) -> Tuple[str,str]:
    m = re.search(r"repo:([\w\-]+)/([\w\-\.]+)", q)
    if m: return m.group(1), m.group(2)
    m2 = re.search(r"([\w\-]+)/([\w\-\.]+)", q)
    return (m2.group(1), m2.group(2)) if m2 else ("acme","app")

def _find_label(q: str) -> str:
    m = re.search(r"label:([\w\-]+)", q)
    return m.group(1) if m else "bug"

def _find_project(q: str) -> str:
    m = re.search(r"project:([A-Za-z0-9_\-]+)", q)
    return m.group(1) if m else "PROJECT"

# MCP 스키마-기반 아규먼트 빌더
MCP_SCHEMA_HINTS = {
    # 파일시스템
    "filesystem.read": {
        "required": ["path"],
        "builder": lambda q: {"path": _extract_path(q)},
    },
    # Slack
    "slack.postMessage": {
        "required": ["channel", "text"],
        "builder": lambda q: {"channel": (re.search(r"#([\w\-]+)", q).group(1) if re.search(r"#([\w\-]+)", q) else "general"),
                                "text": re.sub(r"^.*?:\s*", "", q).strip() or q},
    },
    # Google Calendar
    "calendar.listEvents": {
        "required": ["date"],
        "builder": lambda q: {"date": _extract_date_iso(q)},
    },
    # === 추가: Jira / GitHub / Drive ===
    "jira.searchIssues": {
        "required": ["jql", "maxResults"],
        "builder": lambda q: {"jql": f"project = {_find_project(q)} AND text ~ '" + re.sub(r".*?\s", "", q) + "'", "maxResults": 20},
    },
    "github.searchIssues": {
        "required": ["q"],
        "builder": lambda q: (lambda owner, repo, label: {"q": f"repo:{owner}/{repo} label:{label} state:open {q}"})(*_find_repo(q), _find_label(q)),
    },
    "drive.searchFiles": {
        "required": ["query"],
        "builder": lambda q: {"query": f"name contains '{q.strip()[:40]}' and trashed=false"},
    },
}

# LLM 보조(JSON) — 빌더 실패 시 호출

def _llm_struct(llm: 'LLM', tool_name: str, required: List[str], question: str) -> Dict[str, Any]:
    if not llm or not llm._impl:
        return {}
    prompt = (
        f"Tool: {tool_name}\n"
        f"Required fields: {required}\n"
        f"Question: {question}\n"
        "Return a STRICT JSON with exactly these keys."
    )
    try:
        raw = llm._call(prompt)
        data = json.loads(raw) if raw and raw.strip().startswith('{') else {}
        if all(k in data for k in required):
            return data
    except Exception:
        pass
    return {}

# =====( LLM/Agent )=====
class LLM:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.0, role: str = ""):
        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self.role = role
        self._impl = None
        if _LC and os.getenv("OPENAI_API_KEY"):
            self._impl = ChatOpenAI(model=self.model, temperature=self.temperature)

    def _call(self, prompt: str) -> str:
        if self._impl:
            try:
                sys = f"You are {self.role}." if self.role else ""
                out = self._impl.invoke(sys + "\n" + prompt)
                return (out.content or "").strip()
            except Exception:
                return ""
        return ""

    def choose_tool(self, question: str, allowed_tools: List[str], context_snippets: List[str]) -> str:
        q = question.lower()
        if any(k in q for k in ["날씨", "weather", "기온", "미세먼지"]) and any("weather" in t for t in allowed_tools):
            return next((t for t in allowed_tools if "weather" in t), allowed_tools[0])
        if any(k in q for k in ["sql", "db", "고객", "회원"]) and any("db" in t or t=="db" for t in allowed_tools):
            return next((t for t in allowed_tools if "db" in t or t=="db"), allowed_tools[0])
        # MCP 관련 키워드 힌트
        if any(k in q for k in ["jira", "github", "drive", "캘린더", "슬랙", "repo:", "label:"]):
            mcps = [n for n in allowed_tools if n.startswith("mcp:")]
            if mcps:
                return mcps[0]
        options = ", ".join(allowed_tools[:12])
        hint = self._call(f"Choose one tool from [{options}] for the question: {question}.\nAnswer with EXACT tool name only.")
        return hint if hint in allowed_tools else (allowed_tools[0] if allowed_tools else "search")

    def polish(self, text: str) -> str:
        out = self._call(f"Polish concisely in Korean: {text}")
        return out or text

@dataclass
class Agent:
    name: str
    llm: LLM

# =====( 1) Constrained ReAct + Supervisor + MCP + Qdrant RAG )=====
class ReActState(TypedDict):
    question: str
    route: Literal["rag", "tool", "direct", "human", "mcp"]
    ctx: List[Dict[str, Any]]
    answer: str
    citations: List[str]
    tries: int
    grounded: float
    trace: List[str]
    budget: Dict[str, Any]
    meta: Dict[str, Any]
    handoff: Dict[str, Any]

MAX_INTERNAL_STEPS = 2

BASE_TOOLS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    Tool.SEARCH.value: tool_search,
    Tool.DB.value: lambda q: tool_db("SELECT 1"),
    Tool.WEATHER.value: tool_weather,
}

# Router

def router(state: ReActState) -> ReActState:
    q = state["question"].lower()
    if any(k in q for k in ["슬랙", "calendar", "drive", "파일", "notion", "filesystem", "github", "jira", "k8s", "kubernetes", "s3", "gcs"]):
        state["route"] = "mcp"
    elif any(k in q for k in ["근거", "출처", "검색", "docs", "kb", "어디", "언제", "법령", "레퍼런스"]):
        state["route"] = "rag"
    elif any(k in q for k in ["날씨", "weather", "db", "sql", "고객", "회원"]):
        state["route"] = "tool"
    else:
        state["route"] = "direct"
    state["trace"].append(f"Router→{state['route']}")
    return state

# Supervisor 게이트

def supervisor(state: ReActState) -> ReActState:
    caps: Caps = state["meta"]["caps"]
    if state["budget"]["cost"] >= caps.cost_cap or state["budget"]["steps"] >= caps.step_cap or deadline_exceeded(state):
        state["route"] = "human"
        state["handoff"] = {"required": True, "reason": "caps_exceeded", "payload": {"question": state["question"], "ctx": state.get("ctx", [])[:3]}}
        state["trace"].append("Supervisor→Human (caps)")
        return state
    q = state["question"].lower()
    if any(k in q for k in ["카드번호", "cvc", "주민번호", "social security"]):
        state["route"] = "human"
        state["handoff"] = {"required": True, "reason": "policy_violation_risk", "payload": {"question": state["question"]}}
        state["trace"].append("Supervisor→Human (policy)")
        return state
    state["trace"].append("Supervisor→Proceed")
    return state

# RAG(청킹+Qdrant) 준비 — 지연 초기화
_RAG: Optional[QdrantRAG] = None

def get_rag(namespace: str = "") -> Optional[QdrantRAG]:
    global _RAG
    if _RAG is None and _QDR:
        _RAG = QdrantRAG(RAG_COLLECTION, namespace)
    return _RAG

# RAG 노드

def rag_node(state: ReActState, agent: Agent, allowed_tools: Dict[str, Callable[[str], Dict[str, Any]]]) -> ReActState:
    ns = state["meta"].get("namespace", "")
    rag = get_rag(ns)
    if not rag:
        state["ctx"] = [{"text":"fallback", "source":"mock", "score":0.1}]
    else:
        state["trace"].append("RAG: qdrant_search")
        state["budget"]["cost"] += COST_TABLE["qdrant_search"]
        filt = state["meta"].get("filter_kv", {})
        alpha = state["meta"].get("alpha", 0.7)
        topk = state["meta"].get("topk", 8)
        state["ctx"] = rag.search(state["question"], top_k=topk, filter_kv=filt, hybrid=True, alpha=alpha)
    tool_name = agent.llm.choose_tool(state["question"], list(allowed_tools.keys()), [c.get("text","") for c in state["ctx"]])
    if tool_name in allowed_tools:
        state["trace"].append(f"RAG Action={tool_name}")
        state["budget"]["cost"] += COST_TABLE.get("tool_mcp" if tool_name.startswith("mcp:") else f"tool_{tool_name}", 0.0004)
        obs = allowed_tools[tool_name](state["question"])  # 실행
    else:
        obs = {"note": "no external tool used"}
    state["grounded"] = 0.88 if state["ctx"] else 0.5
    cites = []
    for c in state["ctx"][:3]:
        src = c.get("source")
        if src:
            cites.append(src)
    draft = (
        f"질문: {state['question']}\n"
        f"컨텍스트 수: {len(state['ctx'])}\n"
        f"샘플 근거: {cites}\n"
        f"외부관측: {str(obs)[:300]}\n"
    )
    state["budget"]["cost"] += COST_TABLE["llm_call"]
    state["answer"] = agent.llm.polish(draft)
    state["citations"] = cites
    return state

# MCP 노드 — 타이트 스키마 빌더 포함

def mcp_node(state: ReActState, agent: Agent, mcp: 'MCPClient') -> ReActState:
    if not (mcp and mcp.tools):
        state["answer"] = "사용 가능한 MCP 도구가 없습니다."
        return state
    names = list(mcp.tools.keys())
    preferred = [t for t in names if t in MCP_SCHEMA_HINTS]
    candidates = preferred or names
    tool_name = agent.llm.choose_tool(state["question"], candidates, [])
    required = MCP_SCHEMA_HINTS.get(tool_name, {}).get("required", [])
    builder = MCP_SCHEMA_HINTS.get(tool_name, {}).get("builder")
    args = builder(state["question"]) if builder else {}
    if (not args or not all(k in args for k in required)):
        assist = _llm_struct(agent.llm, tool_name, required, state["question"]) if required else {}
        args.update({k:v for k,v in assist.items() if v is not None})
    if required and not all(k in args for k in required):
        state["answer"] = f"필수 인자 누락: {required}"
        return state
    try:
        state["budget"]["cost"] += COST_TABLE["tool_mcp"]
        res = asyncio.get_event_loop().run_until_complete(mcp.call(tool_name, args))
    except RuntimeError:
        res = asyncio.run(mcp.call(tool_name, args))
    draft = f"MCP '{tool_name}' 호출 args={args} 결과: {str(res)[:400]}"
    state["budget"]["cost"] += COST_TABLE["llm_call"]
    state["answer"] = agent.llm.polish(draft)
    state["grounded"] = 0.8
    return state

# 일반 Tool/Direct/Human 노드

def tool_node(state: ReActState, agent: Agent, allowed_tools: Dict[str, Callable[[str], Dict[str, Any]]]) -> ReActState:
    tool_name = agent.llm.choose_tool(state["question"], list(allowed_tools.keys()), [])
    if tool_name not in allowed_tools:
        state["answer"] = "허용되지 않은 도구"
        return state
    state["budget"]["cost"] += COST_TABLE.get("tool_mcp" if tool_name.startswith("mcp:") else f"tool_{tool_name}", 0.0004)
    obs = allowed_tools[tool_name](state["question"])  # 결정적
    state["grounded"] = 0.85
    draft = f"요청을 '{tool_name}' 도구로 처리했습니다. 결과 요약: {str(obs)[:400]}"
    state["budget"]["cost"] += COST_TABLE["llm_call"]
    state["answer"] = agent.llm.polish(draft)
    return state


def direct_node(state: ReActState, agent: Agent) -> ReActState:
    draft = f"질문: {state['question']}\n직접 답변 초안: 컨텍스트 및 정책 내에서 응답합니다."
    state["budget"]["cost"] += COST_TABLE["llm_call"]
    state["answer"] = agent.llm.polish(draft)
    state["grounded"] = 0.4
    return state


def to_human(state: ReActState, reason: str) -> ReActState:
    state["route"] = "human"
    state["handoff"] = {"required": True, "reason": reason, "payload": {"question": state["question"], "ctx": state.get("ctx", [])[:2]}}
    state["trace"].append(f"→ Human ({reason})")
    return state


def human_review(state: ReActState) -> ReActState:
    state["answer"] = ("상담원 연결이 필요합니다. 질문과 컨텍스트를 전달했어요. " f"사유: {state['handoff'].get('reason')}")
    return state

# Orchestrator

def run_react_graph(question: str, caps: Optional[Caps] = None, agents_mode: str = "single",
                    namespace: str = "", filter_kv: Optional[Dict[str,Any]] = None, topk: int = 8, alpha: float = 0.7) -> Dict[str, Any]:
    caps = caps or Caps()
    # MCP
    mcp = None
    if USE_MCP and _MCP and os.getenv("MCP_SERVER_CMD"):
        mcp = MCPClient(os.getenv("MCP_SERVER_CMD"), os.getenv("MCP_SERVER_ARGS"))
        try:
            asyncio.get_event_loop().run_until_complete(mcp.start())
        except RuntimeError:
            asyncio.run(mcp.start())

    allowed_tools: Dict[str, Callable[[str], Dict[str, Any]]] = dict(BASE_TOOLS)

    # 에이전트 구성(싱글/멀티)
    if agents_mode == "multi":
        planner = Agent("Planner", LLM(role="Planner/Supervisor", temperature=0.0))
        rag_agent = Agent("RAG Agent", LLM(role="Retriever/Analyst", temperature=0.0))
        tool_agent = Agent("Tool Agent", LLM(role="Tool Runner", temperature=0.0))
        mcp_agent = Agent("MCP Agent", LLM(role="MCP Integrator", temperature=0.0))
        writer = Agent("Writer", LLM(role="Answer Writer", temperature=0.2))
    else:
        planner = rag_agent = tool_agent = mcp_agent = writer = Agent("Agent", LLM(temperature=0.0))

    init: ReActState = {
        "question": question,
        "route": "direct",
        "ctx": [],
        "answer": "",
        "citations": [],
        "tries": 0,
        "grounded": 0.0,
        "trace": [],
        "budget": {"cost": 0.0, "steps": 0},
        "meta": {"start_ts": time.perf_counter(), "caps": caps, "namespace": namespace, "filter_kv": filter_kv or {}, "topk": topk, "alpha": alpha},
        "handoff": {"required": False, "reason": "", "payload": {}},
    }

    if _LG:
        sg = StateGraph(ReActState)
        def _router(s: ReActState): return router(s)
        def _supervisor(s: ReActState): return supervisor(s)
        def _rag(s: ReActState): return rag_node(s, rag_agent, allowed_tools)
        def _tool(s: ReActState): return tool_node(s, tool_agent, allowed_tools)
        def _mcp(s: ReActState): return mcp_node(s, mcp_agent, mcp)
        def _direct(s: ReActState): return direct_node(s, writer)
        def _human(s: ReActState): return human_review(s)

        sg.add_node("router", _router)
        sg.add_node("supervisor", _supervisor)
        sg.add_node("rag", _rag)
        sg.add_node("tool", _tool)
        sg.add_node("mcp", _mcp)
        sg.add_node("direct", _direct)
        sg.add_node("human", _human)
        sg.set_entry_point("router")

        sg.add_edge("router", "supervisor")
        sg.add_conditional_edges("supervisor", lambda s: s["route"], {"rag": "rag", "tool": "tool", "mcp": "mcp", "direct": "direct", "human": "human"})
        for node in ("rag","tool","mcp","direct","human"):
            sg.add_edge(node, END)

        app = sg.compile()
        out = app.invoke(init)
        if mcp:
            try:
                asyncio.get_event_loop().run_until_complete(mcp.aclose())
            except RuntimeError:
                asyncio.run(mcp.aclose())
        return out
    else:
        st = router(init)
        st = supervisor(st)
        if st["route"] == "rag":
            st = rag_node(st, rag_agent, allowed_tools)
        elif st["route"] == "tool":
            st = tool_node(st, tool_agent, allowed_tools)
        elif st["route"] == "mcp":
            st = mcp_node(st, mcp_agent, mcp)
        elif st["route"] == "human":
            st = human_review(st)
        else:
            st = direct_node(st, writer)
        if mcp:
            try:
                asyncio.get_event_loop().run_until_complete(mcp.aclose())
            except RuntimeError:
                asyncio.run(mcp.aclose())
        return st

# =====( 2) FSM (결정적) — 유지 )=====
@dataclass
class OrderPayload:
    user_id: int
    city: str
    item: str

@dataclass
class FsmState:
    payload: OrderPayload
    valid: bool = False
    normalized: Dict[str, Any] = None
    api_result: Dict[str, Any] = None
    message: str = ""
    budget: Dict[str, Any] = None
    meta: Dict[str, Any] = None
    handoff: Dict[str, Any] = None


def fsm_validate(st: FsmState) -> FsmState:
    p = st.payload
    st.valid = bool(p.user_id and p.city and p.item)
    if not st.valid:
        st.message = "필수 입력 누락"
    return st

def fsm_normalize(st: FsmState) -> FsmState:
    if not st.valid:
        return st
    st.normalized = {
        "user_id": int(st.payload.user_id),
        "city": st.payload.city.strip().title(),
        "item": st.payload.item.strip().lower(),
    }
    return st

def fsm_call_api(st: FsmState) -> FsmState:
    if not st.valid:
        return st
    if now_exceeded(st.meta["start_ts"], st.meta["caps"].deadline_sec):
        return fsm_to_human(st, "deadline_fsm")
    st.budget["cost"] += COST_TABLE["tool_weather"]
    w = tool_weather(st.normalized["city"])  # 결정적
    st.budget["cost"] += COST_TABLE["tool_db"]
    d = tool_db("SELECT 1")
    st.api_result = {"weather": w, "db": d}
    return st

def fsm_render(st: FsmState, writer: Optional[Agent] = None) -> FsmState:
    if not st.valid:
        return st
    draft = (
        f"주문 접수: 사용자#{st.normalized['user_id']} / 도시={st.normalized['city']} / 품목={st.normalized['item']}\n"
        f"현재 날씨: {st.api_result['weather'].get('status')} {st.api_result['weather'].get('temp_c')}℃\n"
        f"처리 결과: OK"
    )
    if writer:
        st.budget["cost"] += COST_TABLE["llm_call"]
        st.message = writer.llm.polish(draft)
    else:
        st.message = draft
    return st

def fsm_to_human(st: FsmState, reason: str) -> FsmState:
    st.handoff = {"required": True, "reason": reason, "payload": st.normalized}
    st.message = f"상담원 연결 필요: {reason}"
    return st


def run_fsm_example(user_id: int = 1, city: str = "seoul", item: str = "umbrella", caps: Optional[Caps] = None, agents_mode: str = "single") -> Dict[str, Any]:
    caps = caps or Caps()
    writer = Agent("Writer", LLM(role="Writer", temperature=0.1)) if agents_mode == "multi" else Agent("Agent", LLM())
    st = FsmState(
        payload=OrderPayload(user_id=user_id, city=city, item=item),
        budget={"cost": 0.0, "steps": 0},
        meta={"start_ts": time.perf_counter(), "caps": caps},
        handoff={"required": False},
    )
    st = fsm_validate(st)
    st = fsm_normalize(st)
    st = fsm_call_api(st)
    if st.handoff and st.handoff.get("required"):
        return as_dict(st)
    st = fsm_render(st, writer)
    return as_dict(st)


def as_dict(st: FsmState) -> Dict[str, Any]:
    return {
        "valid": st.valid,
        "normalized": st.normalized,
        "api_result": st.api_result,
        "message": st.message,
        "budget": st.budget,
        "handoff": st.handoff,
    }

# =====( CLI )=====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["react", "fsm"], default="react")
    parser.add_argument("--q", default="서울 내일 날씨 알려줘.")
    parser.add_argument("--city", default="seoul")
    parser.add_argument("--item", default="umbrella")
    parser.add_argument("--cost_cap", type=float, default=0.03)
    parser.add_argument("--step_cap", type=int, default=10)
    parser.add_argument("--deadline", type=float, default=8.0)
    parser.add_argument("--agents", choices=["single","multi"], default="single")
    # RAG/Qdrant
    parser.add_argument("--index_dir", default="")
    parser.add_argument("--chunk", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--filter", default="")  # key:value[,key:value]
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hybrid", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.7)

    args = parser.parse_args()

    caps = Caps(cost_cap=args.cost_cap, step_cap=args.step_cap, deadline_sec=args.deadline)

    # 인덱싱 모드
    if args.index_dir:
        if not _QDR:
            print("Qdrant client not installed. pip install qdrant-client")
        else:
            rag = QdrantRAG(RAG_COLLECTION, args.namespace)
            out = rag.index(args.index_dir, project=args.project, chunk=args.chunk, overlap=args.overlap)
            print("Indexed:", out)
        raise SystemExit(0)

    # 필터 파싱
    filter_kv: Dict[str,Any] = {}
    if args.filter:
        for pair in args.filter.split(','):
            if ':' in pair:
                k,v = pair.split(':',1)
                filter_kv[k.strip()] = v.strip()

    if args.mode == "react":
        out = run_react_graph(
            args.q,
            caps,
            args.agents,
            namespace=args.namespace,
            filter_kv=filter_kv,
            topk=args.topk,
            alpha=args.alpha,
        )
        print("=== Constrained ReAct + Supervisor + MCP + Qdrant RAG (hybrid/rerank/filters) ===")
        print("route:", out.get("route"))
        print("answer:\n", out.get("answer"))
        print("citations:", out.get("citations"))
        print("grounded:", out.get("grounded"))
        print("budget:", out.get("budget"))
        print("handoff:", out.get("handoff"))
        print("trace:", out.get("trace"))
    else:
        out = run_fsm_example(args.cost_cap, args.city, args.item, caps, args.agents)  # 파라미터 순서 주의
        print("=== FSM (결정적) ===")
        for k, v in out.items():
            print(f"{k}: {v}")
