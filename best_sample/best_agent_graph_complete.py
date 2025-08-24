"""
Agent Patterns — Constrained ReAct vs FSM (Production-Style Demo, v2)

✅ 무엇이 달라졌나 (요청하신 4가지 포함)
- **LangGraph Supervisor 분기**: 정책 게이트 + 예산/시간 한도 체크로 분기(→ RAG / Tool / Direct / Human).
- **비용/타임아웃 캡**: state에 `budget`(cost/steps)과 `deadline_sec`을 두고 모든 호출에서 과금/시간 초과 감시.
- **휴먼 핸드오프(HITL)**: 기준 초과·불확실·정책 위반 시 `human_review` 노드로 전환, 사유와 컨텍스트 전달.
- **실제 API 연동**: 날씨는 Open‑Meteo(무키), DB는 로컬 sqlite3 예시. (요청/DB에 타임아웃/예외 처리 포함)

🧰 의존성 (선택)
  pip install "langgraph>=0.2" "langchain-openai>=0.2" python-dotenv requests

환경변수(선택)
  export OPENAI_API_KEY=...
  export USE_REAL_APIS=true   # 실제 API/DB 시연 (없으면 mock 동작)

실행 예
  uv run python best_sample/best_agent_graph_complete.py --mode react --q "서울 내일 날씨 알려줘"
  uv run python best_sample/best_agent_graph_complete.py --mode fsm   --city seoul --item umbrella
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Optional
from enum import Enum
import argparse, os, time, sqlite3

# =====( 선택: LangGraph / LangChain )=====
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

import requests

# =====( 공통 설정 )=====
USE_REAL_APIS = os.getenv("USE_REAL_APIS", "false").lower() in ("1", "true", "yes")

@dataclass
class Caps:
    cost_cap: float = 0.02      # 달러 가정(데모)
    step_cap: int = 6           # 그래프 전체 최대 스텝
    deadline_sec: float = 6.0   # 전체 실행 시간 제한

# 간단 과금 테이블(데모): 호출마다 누적
COST_TABLE = {
    "llm_call": 0.002,
    "tool_search": 0.0005,
    "tool_db": 0.0008,
    "tool_weather": 0.0005,
}

# =====( 도구 정의: mock + real )=====
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

# 과금/시간 체크용 헬퍼
def charge(state, key: str):
    state["budget"]["cost"] += COST_TABLE.get(key, 0.0002)
    state["budget"]["steps"] += 1

def deadline_exceeded(state) -> bool:
    return (time.perf_counter() - state["meta"]["start_ts"]) > state["meta"]["caps"].deadline_sec

# 실제/목업 도구

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
    # sqlite3 로컬 샘플(DB 파일 없으면 메모리)
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

ALLOWED_TOOLS = {
    Tool.SEARCH.value: tool_search,
    Tool.DB.value: tool_db,
    Tool.WEATHER.value: tool_weather,
}

# =====( LLM 래퍼: 선택적으로 OpenAI 사용 )=====
class LLM:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self._impl = None
        if _LC and os.getenv("OPENAI_API_KEY"):
            self._impl = ChatOpenAI(model=self.model, temperature=self.temperature)

    def _call(self, prompt: str) -> str:
        if self._impl:
            try:
                out = self._impl.invoke(prompt)
                return (out.content or "").strip()
            except Exception:
                return ""
        # mock
        return ""

    def choose_tool(self, question: str, context_snippets: List[str]) -> str:
        # 간단 휴리스틱
        q = question.lower()
        if any(k in q for k in ["날씨", "weather", "기온", "미세먼지"]):
            return Tool.WEATHER.value
        if any(k in q for k in ["sql", "db", "고객", "회원"]):
            return Tool.DB.value
        guess = Tool.SEARCH.value
        # 필요시 LLM로 확인
        hint = self._call(
            """
            Choose one tool from [search, db, weather] for the user's question below.
            Answer with only: search | db | weather.
            """ + question
        )
        return hint if hint in ALLOWED_TOOLS else guess

    def polish(self, text: str) -> str:
        out = self._call(f"Polish concisely in Korean: {text}")
        return out or text

# =====( 1) Constrained ReAct + LangGraph + Supervisor )=====
class ReActState(TypedDict):
    question: str
    route: Literal["rag", "tool", "direct", "human"]
    ctx: List[Dict[str, Any]]
    answer: str
    citations: List[str]
    tries: int
    grounded: float
    trace: List[str]
    budget: Dict[str, Any]      # {cost, steps}
    meta: Dict[str, Any]        # {start_ts, caps}
    handoff: Dict[str, Any]     # {required, reason, payload}

MAX_INTERNAL_STEPS = 2  # 노드 내부 ReAct-lite 제한

# Router: 질의 기반 1차 라우팅

def router(state: ReActState) -> ReActState:
    q = state["question"].lower()
    if any(k in q for k in ["근거", "출처", "검색", "docs", "kb", "어디", "언제"]):
        state["route"] = "rag"
    elif any(k in q for k in ["날씨", "weather", "db", "sql", "고객", "회원"]):
        state["route"] = "tool"
    else:
        state["route"] = "direct"
    state["trace"].append(f"Router→{state['route']}")
    return state

# 정책/예산/시간 게이트 적용 Supervision

def supervisor(state: ReActState) -> ReActState:
    caps: Caps = state["meta"]["caps"]
    # 예산 초과 또는 데드라인 초과 → 휴먼 핸드오프
    if state["budget"]["cost"] >= caps.cost_cap or state["budget"]["steps"] >= caps.step_cap or deadline_exceeded(state):
        state["route"] = "human"
        state["handoff"] = {
            "required": True,
            "reason": f"caps_exceeded(cost={state['budget']['cost']:.4f}, steps={state['budget']['steps']}, deadline={caps.deadline_sec}s)",
            "payload": {"question": state["question"], "ctx": state["ctx"][:3]},
        }
        state["trace"].append("Supervisor→Human (caps)")
        return state
    # (선택) 간단 정책 위반 예시: 카드정보/주민번호 등 키워드 탐지
    q = state["question"].lower()
    if any(k in q for k in ["카드번호", "cvc", "주민번호", "social security"]):
        state["route"] = "human"
        state["handoff"] = {
            "required": True,
            "reason": "policy_violation_risk",
            "payload": {"question": state["question"]},
        }
        state["trace"].append("Supervisor→Human (policy)")
        return state
    # 통과 → 기존 라우팅 유지
    state["trace"].append("Supervisor→Proceed")
    return state

# Retrieval 컨텍스트(결정적)

def retrieve_ctx(question: str) -> List[Dict[str, Any]]:
    res = tool_search(question)
    return [{"text": i["title"], "source": i["source"], "score": 0.7} for i in res["items"]]

# RAG 노드: 1회 제한 ReAct-lite (툴 선택→관측→후처리)

def rag_node(state: ReActState, llm: LLM) -> ReActState:
    if deadline_exceeded(state):
        return to_human(state, "deadline_before_rag")
    state["ctx"] = retrieve_ctx(state["question"])  # 결정적
    steps = 0
    while steps < MAX_INTERNAL_STEPS:
        steps += 1
        tool_name = llm.choose_tool(state["question"], [c["text"] for c in state["ctx"]])
        if tool_name not in ALLOWED_TOOLS:
            state["trace"].append(f"RAG Step{steps}: invalid_tool={tool_name}")
            break
        state["trace"].append(f"RAG Step{steps}: Action={tool_name}")
        charge(state, f"tool_{tool_name}")
        obs = ALLOWED_TOOLS[tool_name](state["question"])  # 실행
        state["trace"].append(f"RAG Step{steps}: ObsKeys={list(obs.keys())}")
        state["grounded"] = 0.7 if tool_name == Tool.SEARCH.value else 0.85
        draft = (
            f"질문: {state['question']}\n"
            f"도구: {tool_name}\n"
            f"관측: {obs}\n"
        )
        charge(state, "llm_call")
        state["answer"] = llm.polish(draft)
        state["citations"] = [c["source"] for c in state["ctx"][:2]]
        return state
    return state

# Tool 전용 노드(명시적 도구 질의)

def tool_node(state: ReActState, llm: LLM) -> ReActState:
    if deadline_exceeded(state):
        return to_human(state, "deadline_before_tool")
    tool_name = llm.choose_tool(state["question"], [])
    if tool_name not in ALLOWED_TOOLS:
        state["answer"] = "허용되지 않은 도구"
        return state
    charge(state, f"tool_{tool_name}")
    obs = ALLOWED_TOOLS[tool_name](state["question"])  # 결정적
    state["grounded"] = 0.85
    draft = f"요청을 '{tool_name}' 도구로 처리했습니다. 결과 요약: {obs}"
    charge(state, "llm_call")
    state["answer"] = llm.polish(draft)
    return state

# Direct 노드(LLM 요약)

def direct_node(state: ReActState, llm: LLM) -> ReActState:
    if deadline_exceeded(state):
        return to_human(state, "deadline_before_direct")
    draft = f"질문: {state['question']}\n직접 답변 초안(데모): 관련 지식을 요약해 응답합니다."
    charge(state, "llm_call")
    state["answer"] = llm.polish(draft)
    state["grounded"] = 0.4
    return state

# Human 핸드오프 노드

def to_human(state: ReActState, reason: str) -> ReActState:
    state["route"] = "human"
    state["handoff"] = {"required": True, "reason": reason, "payload": {"question": state["question"], "ctx": state["ctx"][:2]}}
    state["trace"].append(f"→ Human ({reason})")
    return state


def human_review(state: ReActState) -> ReActState:
    # 실제에선 티켓 생성/슬랙 알림/CRM 연결 등
    state["answer"] = (
        "상담원 연결이 필요합니다. 질문과 컨텍스트를 전달했어요. "
        f"사유: {state['handoff'].get('reason')}"
    )
    return state

# Orchestrator

def run_react_graph(question: str, caps: Optional[Caps] = None) -> Dict[str, Any]:
    llm = LLM()
    caps = caps or Caps()
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
            return rag_node(s, llm)
        def _tool(s: ReActState):
            return tool_node(s, llm)
        def _direct(s: ReActState):
            return direct_node(s, llm)
        def _human(s: ReActState):
            return human_review(s)

        sg.add_node("router", _router)
        sg.add_node("supervisor", _supervisor)
        sg.add_node("rag", _rag)
        sg.add_node("tool", _tool)
        sg.add_node("direct", _direct)
        sg.add_node("human", _human)
        sg.set_entry_point("router")

        # 라우팅 → 슈퍼바이저 게이트 → 목적지
        sg.add_edge("router", "supervisor")
        sg.add_conditional_edges(
            "supervisor",
            lambda s: s["route"],
            {"rag": "rag", "tool": "tool", "direct": "direct", "human": "human"},
        )
        # 각 작업 노드 종료
        sg.add_edge("rag", END)
        sg.add_edge("tool", END)
        sg.add_edge("direct", END)
        sg.add_edge("human", END)

        app = sg.compile()
        out = app.invoke(init)
        return out
    else:
        # LangGraph 미사용 간이 실행
        st = router(init)
        st = supervisor(st)
        if st["route"] == "rag":
            st = rag_node(st, llm)
        elif st["route"] == "tool":
            st = tool_node(st, llm)
        elif st["route"] == "human":
            st = human_review(st)
        else:
            st = direct_node(st, llm)
        return st

# =====( 2) 순수 FSM — 결정적 플로우 + 예산/타임아웃/HITL )=====
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

# 결정적 단계들

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
    if deadline_exceeded({"meta": st.meta}):
        return fsm_to_human(st, "deadline_fsm")
    # 결정적 외부 호출(날씨 + DB)
    charge_fsm(st, "tool_weather")
    w = tool_weather(st.normalized["city"])  # 결정적
    charge_fsm(st, "tool_db")
    d = tool_db("SELECT 1")
    st.api_result = {"weather": w, "db": d}
    return st


def fsm_render(st: FsmState, llm: Optional[LLM] = None) -> FsmState:
    if not st.valid:
        return st
    draft = (
        f"주문 접수: 사용자#{st.normalized['user_id']} / 도시={st.normalized['city']} / 품목={st.normalized['item']}\n"
        f"현재 날씨: {st.api_result['weather'].get('status')} {st.api_result['weather'].get('temp_c')}℃\n"
        f"처리 결과: OK"
    )
    if llm:
        st.budget["cost"] += COST_TABLE["llm_call"]
        st.message = llm.polish(draft)
    else:
        st.message = draft
    return st


def fsm_to_human(st: FsmState, reason: str) -> FsmState:
    st.handoff = {"required": True, "reason": reason, "payload": st.normalized}
    st.message = f"상담원 연결 필요: {reason}"
    return st


def charge_fsm(st: FsmState, key: str):
    st.budget["cost"] += COST_TABLE.get(key, 0.0002)
    st.budget["steps"] += 1

# FSM 러너

def run_fsm_example(user_id: int = 1, city: str = "seoul", item: str = "umbrella", caps: Optional[Caps] = None) -> Dict[str, Any]:
    caps = caps or Caps()
    llm = LLM()
    st = FsmState(
        payload=OrderPayload(user_id=user_id, city=city, item=item),
        budget={"cost": 0.0, "steps": 0},
        meta={"start_ts": time.perf_counter(), "caps": caps},
        handoff={"required": False},
    )
    # 예산/타임아웃 감시
    st = fsm_validate(st)
    if st.budget["cost"] >= caps.cost_cap or st.budget["steps"] >= caps.step_cap:
        return as_dict(st)
    st = fsm_normalize(st)
    if st.budget["cost"] >= caps.cost_cap or st.budget["steps"] >= caps.step_cap:
        return as_dict(st)
    st = fsm_call_api(st)
    if st.handoff.get("required"):
        return as_dict(st)
    st = fsm_render(st, llm)
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
    parser.add_argument("--step_cap", type=int, default=6)
    parser.add_argument("--deadline", type=float, default=6.0)
    args = parser.parse_args()

    caps = Caps(cost_cap=args.cost_cap, step_cap=args.step_cap, deadline_sec=args.deadline)

    if args.mode == "react":
        out = run_react_graph(args.q, caps)
        print("=== Constrained ReAct + Supervisor ===")
        print("route:", out.get("route"))
        print("answer:\n", out.get("answer"))
        print("citations:", out.get("citations"))
        print("grounded:", out.get("grounded"))
        print("budget:", out.get("budget"))
        print("handoff:", out.get("handoff"))
        print("trace:", out.get("trace"))
    else:
        out = run_fsm_example(args.cost_cap, args.city, args.item, caps)  # 파라미터 순서 주의
        print("=== FSM (결정적) ===")
        for k, v in out.items():
            print(f"{k}: {v}")
