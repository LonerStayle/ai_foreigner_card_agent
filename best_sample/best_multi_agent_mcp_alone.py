"""
Agent Patterns — Constrained ReAct vs FSM (Production-Style Demo, v3: +MCP, Single/Multi-Agent)

추가된 것
- ✅ LangGraph **Supervisor** 분기 (정책/예산/시간 가드)
- ✅ **비용/타임아웃 캡**: cost/steps/deadline 관리
- ✅ **휴먼 핸드오프(HITL)** 노드
- ✅ **실제 API 연동**: Open‑Meteo(무키) + sqlite3 샘플
- ✅ **MCP 통합**: stdio 서버에 연결해 동적 툴 등록 및 호출
- ✅ **싱글/멀티 에이전트 토글**: `--agents single|multi`

설치/환경
  pip install "langgraph>=0.2" "langchain-openai>=0.2" python-dotenv requests
  # MCP(선택)
  pip install mcp

환경변수 예시
  export OPENAI_API_KEY=...
  export USE_REAL_APIS=true
  export USE_MCP=true
  export MCP_SERVER_CMD="node"               # 예: node
  export MCP_SERVER_ARGS="/path/to/server.js" # 예: mcp 파일시스템/슬랙 등

실행 예
  uv run python best_sample/best_multi_agent_mcp_alone.py --mode react --q "서울 내일 날씨" --agents multi
  uv run python best_sample/best_multi_agent_mcp_alone.py --mode fsm --city seoul --item umbrella --agents single
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Optional, Callable,TypedDict
from enum import Enum
import argparse, os, time, sqlite3, asyncio
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

# =====( 글로벌 설정 )=====
USE_REAL_APIS = os.getenv("USE_REAL_APIS", "false").lower() in ("1","true","yes")
USE_MCP = os.getenv("USE_MCP", "false").lower() in ("1","true","yes")

@dataclass
class Caps:
    cost_cap: float = 0.02
    step_cap: int = 8
    deadline_sec: float = 8.0

COST_TABLE = {
    "llm_call": 0.002,
    "tool_search": 0.0005,
    "tool_db": 0.0008,
    "tool_weather": 0.0005,
    "tool_mcp": 0.0010,
}

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

# 과금/시간 체크
def charge(state, key: str):
    state["budget"]["cost"] += COST_TABLE.get(key, 0.0002)
    state["budget"]["steps"] += 1

def deadline_exceeded(state) -> bool:
    return (time.perf_counter() - state["meta"]["start_ts"]) > state["meta"]["caps"].deadline_sec

# 기본 툴

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

# =====( MCP 클라이언트 )=====
class MCPClient:
    def __init__(self, cmd: Optional[str], args: Optional[str]):
        self.cmd = cmd
        self.args = (args or "").split() if args else []
        self.session: Optional[ClientSession] = None
        self.tools: List[str] = []
        self._conn = None

    async def start(self):
        if not (_MCP and self.cmd):
            return
        params = StdioServerParameters(command=self.cmd, args=self.args)
        self._conn = await stdio_client(params)
        self.session = self._conn.session
        await self.session.initialize()
        # list tools
        try:
            caps = await self.session.list_tools()
            # caps.tools -> [{name, description, input_schema, ...}]
            self.tools = [t.name for t in getattr(caps, 'tools', [])]
        except Exception:
            self.tools = []

    async def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session:
            raise RuntimeError("MCP not started")
        res = await self.session.call_tool(tool_name, args)
        # res -> {content: [...], is_error: bool}
        return {"tool": f"mcp:{tool_name}", "content": getattr(res, 'content', []), "error": getattr(res, 'is_error', False)}

    async def aclose(self):
        try:
            if self._conn:
                await self._conn.close()
        except Exception:
            pass

# 동기 헬퍼
async def _mcp_start_and_register(client: MCPClient) -> Dict[str, Callable[[str], Dict[str, Any]]]:
    await client.start()
    m: Dict[str, Callable[[str], Dict[str, Any]]] = {}
    for name in client.tools[:8]:  # 데모: 최대 8개 등록
        async def _call(q: str, _name=name):
            # 일반적으로는 schema에 맞춰 args를 구성해야 함. 데모로 query 전달
            return await client.call(_name, {"query": q})
        # 동기 wrapper 반환
        def wrapper(q: str, _name=name):
            return asyncio.get_event_loop().run_until_complete(_call(q, _name))
        m[f"mcp:{name}"] = wrapper
    return m

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
                out = self._impl.invoke(sys + " " + prompt)
                return (out.content or "").strip()
            except Exception:
                return ""
        return ""

    def choose_tool(self, question: str, allowed_tools: List[str], context_snippets: List[str]) -> str:
        # 간단 휴리스틱
        q = question.lower()
        if any(k in q for k in ["날씨", "weather", "기온", "미세먼지"]) and any("weather" in t for t in allowed_tools):
            return next(t for t in allowed_tools if "weather" in t)
        if any(k in q for k in ["sql", "db", "고객", "회원"]) and any("db" in t or t=="db" for t in allowed_tools):
            return next((t for t in allowed_tools if "db" in t or t=="db"), allowed_tools[0])
        # LLM로 선택
        options = ", ".join(allowed_tools[:12])
        hint = self._call(
            f"Choose one tool from [{options}] for the question: {question}."
            f"Answer with EXACT tool name only."
        )
        return hint if hint in allowed_tools else (allowed_tools[0] if allowed_tools else "search")

    def polish(self, text: str) -> str:
        out = self._call(f"Polish concisely in Korean: {text}")
        return out or text

@dataclass
class Agent:
    name: str
    llm: LLM

# =====( 1) Constrained ReAct + Supervisor + MCP )=====
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

# 동적 툴 레지스트리(기본 + MCP)
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
    elif any(k in q for k in ["근거", "출처", "검색", "docs", "kb", "어디", "언제"]):
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

# Retrieval

def retrieve_ctx(question: str) -> List[Dict[str, Any]]:
    res = tool_search(question)
    return [{"text": i["title"], "source": i["source"], "score": 0.7} for i in res["items"]]

# ReAct-lite 노드들

def rag_node(state: ReActState, agent: Agent, allowed_tools: Dict[str, Callable[[str], Dict[str, Any]]]) -> ReActState:
    if deadline_exceeded(state):
        return to_human(state, "deadline_before_rag")
    state["ctx"] = retrieve_ctx(state["question"])  # 결정적
    steps = 0
    while steps < MAX_INTERNAL_STEPS:
        steps += 1
        tool_name = agent.llm.choose_tool(state["question"], list(allowed_tools.keys()), [c["text"] for c in state["ctx"]])
        if tool_name not in allowed_tools:
            state["trace"].append(f"RAG Step{steps}: invalid_tool={tool_name}")
            break
        state["trace"].append(f"RAG Step{steps}: Action={tool_name}")
        charge(state, "tool_mcp" if tool_name.startswith("mcp:") else f"tool_{tool_name}")
        obs = allowed_tools[tool_name](state["question"])  # 실행
        state["trace"].append(f"RAG Step{steps}: ObsKeys={list(obs.keys())}")
        state["grounded"] = 0.85 if (tool_name == Tool.SEARCH.value or tool_name.startswith("mcp:")) else 0.7
        draft = f"""질문: {state['question']}
도구: {tool_name}
관측: {str(obs)[:400]}
"""
        charge(state, "llm_call")
        state["answer"] = agent.llm.polish(draft)
        state["citations"] = [c["source"] for c in state["ctx"][:2]]
        return state
    return state


def tool_node(state: ReActState, agent: Agent, allowed_tools: Dict[str, Callable[[str], Dict[str, Any]]]) -> ReActState:
    if deadline_exceeded(state):
        return to_human(state, "deadline_before_tool")
    tool_name = agent.llm.choose_tool(state["question"], list(allowed_tools.keys()), [])
    if tool_name not in allowed_tools:
        state["answer"] = "허용되지 않은 도구"
        return state
    charge(state, "tool_mcp" if tool_name.startswith("mcp:") else f"tool_{tool_name}")
    obs = allowed_tools[tool_name](state["question"])  # 결정적
    state["grounded"] = 0.85
    draft = f"요청을 '{tool_name}' 도구로 처리했습니다. 결과 요약: {str(obs)[:400]}"
    charge(state, "llm_call")
    state["answer"] = agent.llm.polish(draft)
    return state


def mcp_node(state: ReActState, agent: Agent, mcp_tools: Dict[str, Callable[[str], Dict[str, Any]]]) -> ReActState:
    if deadline_exceeded(state):
        return to_human(state, "deadline_before_mcp")
    if not mcp_tools:
        state["answer"] = "사용 가능한 MCP 도구가 없습니다."
        return state
    tool_name = agent.llm.choose_tool(state["question"], list(mcp_tools.keys()), [])
    charge(state, "tool_mcp")
    obs = mcp_tools[tool_name](state["question"])  # 실행
    draft = f"MCP 도구 '{tool_name}' 호출 결과: {str(obs)[:400]}"
    charge(state, "llm_call")
    state["answer"] = agent.llm.polish(draft)
    state["grounded"] = 0.8
    return state


def direct_node(state: ReActState, agent: Agent) -> ReActState:
    if deadline_exceeded(state):
        return to_human(state, "deadline_before_direct")
    draft = f"질문: {state['question']} 직접 답변 초안(데모): 관련 지식을 요약해 응답합니다."
    charge(state, "llm_call")
    state["answer"] = agent.llm.polish(draft)
    state["grounded"] = 0.4
    return state

# Human

def to_human(state: ReActState, reason: str) -> ReActState:
    state["route"] = "human"
    state["handoff"] = {"required": True, "reason": reason, "payload": {"question": state["question"], "ctx": state.get("ctx", [])[:2]}}
    state["trace"].append(f"→ Human ({reason})")
    return state


def human_review(state: ReActState) -> ReActState:
    state["answer"] = (
        "상담원 연결이 필요합니다. 질문과 컨텍스트를 전달했어요. "
        f"사유: {state['handoff'].get('reason')}"
    )
    return state

# Orchestrator

def run_react_graph(question: str, caps: Optional[Caps] = None, agents_mode: str = "single") -> Dict[str, Any]:
    caps = caps or Caps()
    # MCP 준비
    mcp_tools: Dict[str, Callable[[str], Dict[str, Any]]] = {}
    mcp = None
    if USE_MCP and _MCP and os.getenv("MCP_SERVER_CMD"):
        mcp = MCPClient(os.getenv("MCP_SERVER_CMD"), os.getenv("MCP_SERVER_ARGS"))
        try:
            mcp_tools = asyncio.get_event_loop().run_until_complete(_mcp_start_and_register(mcp))
        except RuntimeError:
            # 일부 환경에서 이미 루프가 있는 경우
            mcp_tools = asyncio.run(_mcp_start_and_register(mcp))

    # 동적 툴 합치기
    allowed_tools: Dict[str, Callable[[str], Dict[str, Any]]] = {**BASE_TOOLS, **mcp_tools}

    # 에이전트 구성(싱글/멀티)
    if agents_mode == "multi":
        planner = Agent("Planner", LLM(role="Planner/Supervisor", temperature=0.0))
        rag_agent = Agent("RAG Agent", LLM(role="Retriever/Analyst", temperature=0.0))
        tool_agent = Agent("Tool Agent", LLM(role="Tool Runner", temperature=0.0))
        mcp_agent = Agent("MCP Agent", LLM(role="MCP Integrator", temperature=0.0))
        writer = Agent("Writer", LLM(role="Answer Writer", temperature=0.2))
    else:
        # 모든 역할을 하나로
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
        "meta": {"start_ts": time.perf_counter(), "caps": caps},
        "handoff": {"required": False, "reason": "", "payload": {}},
    }

    if _LG:
        sg = StateGraph(ReActState)
        # 노드 래퍼
        def _router(s: ReActState):
            return router(s)
        def _supervisor(s: ReActState):
            return supervisor(s)
        def _rag(s: ReActState):
            return rag_node(s, rag_agent, allowed_tools)
        def _tool(s: ReActState):
            return tool_node(s, tool_agent, allowed_tools)
        def _mcp(s: ReActState):
            return mcp_node(s, mcp_agent, mcp_tools)
        def _direct(s: ReActState):
            return direct_node(s, writer)
        def _human(s: ReActState):
            return human_review(s)

        sg.add_node("router", _router)
        sg.add_node("supervisor", _supervisor)
        sg.add_node("rag", _rag)
        sg.add_node("tool", _tool)
        sg.add_node("mcp", _mcp)
        sg.add_node("direct", _direct)
        sg.add_node("human", _human)
        sg.set_entry_point("router")

        sg.add_edge("router", "supervisor")
        sg.add_conditional_edges(
            "supervisor",
            lambda s: s["route"],
            {"rag": "rag", "tool": "tool", "mcp": "mcp", "direct": "direct", "human": "human"},
        )
        for node in ("rag","tool","mcp","direct","human"):
            sg.add_edge(node, END)

        app = sg.compile()
        out = app.invoke(init)
        # MCP 종료
        if mcp:
            try:
                asyncio.get_event_loop().run_until_complete(mcp.aclose())
            except RuntimeError:
                asyncio.run(mcp.aclose())
        return out
    else:
        # LangGraph 없이 간이 실행
        st = router(init)
        st = supervisor(st)
        if st["route"] == "rag":
            st = rag_node(st, rag_agent, allowed_tools)
        elif st["route"] == "tool":
            st = tool_node(st, tool_agent, allowed_tools)
        elif st["route"] == "mcp":
            st = mcp_node(st, mcp_agent, mcp_tools)
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

# =====( 2) FSM (결정적) 그대로 유지: 예산/타임아웃/HITL 포함 )=====
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

# 결정적 단계

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
    if (time.perf_counter() - st.meta["start_ts"]) > st.meta["caps"].deadline_sec:
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
        f"주문 접수: 사용자#{st.normalized['user_id']} / 도시={st.normalized['city']} / 품목={st.normalized['item']}"
        f"현재 날씨: {st.api_result['weather'].get('status')} {st.api_result['weather'].get('temp_c')}℃"
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
    parser.add_argument("--cost_cap", type=float, default=0.02)
    parser.add_argument("--step_cap", type=int, default=8)
    parser.add_argument("--deadline", type=float, default=8.0)
    parser.add_argument("--agents", choices=["single","multi"], default="single")
    args = parser.parse_args()

    caps = Caps(cost_cap=args.cost_cap, step_cap=args.step_cap, deadline_sec=args.deadline)

    if args.mode == "react":
        out = run_react_graph(args.q, caps, args.agents)
        print("=== Constrained ReAct + Supervisor + MCP ===")
        print("route:", out.get("route"))
        print("answer:", out.get("answer"))
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

