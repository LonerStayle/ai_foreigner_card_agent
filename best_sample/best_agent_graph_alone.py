"""
Agent Patterns — Constrained ReAct vs FSM (Production-Style Demo)

✅ 목적
- 현업식 구조로 "Constrained ReAct + Graph" 와 "순수 FSM(Graph-only)" 두 가지 패턴을 한 파일에서 비교 실행할 수 있게 예제 제공합니다.
- LangGraph를 쓰는 버전(프로덕션 지향)과, 의존성 줄인 순수 파이썬 FSM 버전 둘 다 포함.

🧰 의존성 (선택)
- langgraph>=0.2
- langchain-openai>=0.2
- python-dotenv (선택)

설치 예:
  pip install "langgraph>=0.2" "langchain-openai>=0.2" python-dotenv
환경변수:
  export OPENAI_API_KEY=...

실행 예:
  uv run python best_sample/best_agent_graph_alone.py --mode react   # Constrained ReAct + Graph
  uv run python best_sample/best_agent_graph_alone.py --mode fsm     # 순수 FSM(Graph-only)

메시지 흐름
- Constrained ReAct: 노드 내부에서 단일/소수 스텝의 Reason→Act→Obs를 허용(툴은 enum 제한), 상위 Graph가 분기/중단/가드레일 제어
- FSM: 전 단계 결정적 실행(검증→정규화→API→템플릿), LLM은 문장 다듬기 정도로만 사용
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Optional
from enum import Enum
import argparse
import os
import time

# =====( 선택: LangGraph / LangChain )=====
try:
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    _LG = True
except Exception:
    # LangGraph 미설치 환경에서도 파일이 import는 되도록
    _LG = False
    class TypedDict(dict):
        pass
    END = "__END__"

try:
    from langchain_openai import ChatOpenAI
    _LC = True
except Exception:
    _LC = False

# =====( 공통: 도구 정의 )=====
class Tool(Enum):
    SEARCH = "search"
    DB = "db"
    WEATHER = "weather"

# Mock 도구 구현 (실무에선 실제 API/DB 연동)
def tool_search(query: str) -> Dict[str, Any]:
    time.sleep(0.05)
    return {"tool": Tool.SEARCH.value, "items": [
        {"title": "서울 날씨 가이드", "source": "docs/weather_guide.md"},
        {"title": f"키워드('{query}') 관련 문서", "source": "kb/related.md"},
    ]}

def tool_db(sql: str) -> Dict[str, Any]:
    time.sleep(0.05)
    return {"tool": Tool.DB.value, "rows": [{"id": 1, "name": "홍길동"}]}

def tool_weather(city: str) -> Dict[str, Any]:
    time.sleep(0.05)
    # 실제에선 외부 API 호출
    return {"tool": Tool.WEATHER.value, "city": city, "temp_c": 29, "status": "Sunny"}

ALLOWED_TOOLS = {
    Tool.SEARCH.value: tool_search,
    Tool.DB.value: tool_db,
    Tool.WEATHER.value: tool_weather,
}

# =====( 공통: LLM 선택 — 실무는 OpenAI, 데모는 Mock )=====
class LLM:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.0):
        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self._impl = None
        if _LC and os.getenv("OPENAI_API_KEY"):
            self._impl = ChatOpenAI(model=self.model, temperature=self.temperature)

    def choose_tool(self, question: str, context_snippets: List[str]) -> str:
        """제약된 툴 중 하나를 선택. 프로덕션에서는 함수콜/스키마로 강제.
        여기선 간단 휴리스틱 + (가능하면) LLM 백업.
        """
        # 우선 간단 휴리스틱
        q = question.lower()
        if any(k in q for k in ["날씨", "weather", "기온", "미세먼지"]):
            return Tool.WEATHER.value
        if any(k in q for k in ["sql", "db", "고객", "회원"]):
            return Tool.DB.value
        # 기본값
        guess = Tool.SEARCH.value

        # 가능하면 LLM에 재확인(툴 이름만 추출 유도)
        if self._impl:
            prompt = f"""
            You are a tool selector. Choose one tool name from this list: [search, db, weather].
            Question: {question}
            Context: {context_snippets[:3]}
            Answer with ONLY one of: search | db | weather
            """
            try:
                out = self._impl.invoke(prompt)
                text = (out.content or "").strip().lower()
                if text in ALLOWED_TOOLS:
                    return text
            except Exception:
                pass
        return guess

    def polish(self, text: str) -> str:
        if self._impl:
            try:
                out = self._impl.invoke(f"Polish this text concisely in Korean: {text}")
                return out.content
            except Exception:
                pass
        return text  # mock fallback

# =====( 1) Constrained ReAct + Graph )=====
class ReActState(TypedDict):
    question: str
    route: Literal["rag", "tool", "direct"]
    ctx: List[Dict[str, Any]]
    answer: str
    citations: List[str]
    tries: int
    grounded: float
    trace: List[str]

MAX_STEPS = 2  # ReAct-lite: 내부 단계 제한

# --- Router (간단 휴리스틱) ---
def router(state: ReActState) -> ReActState:
    q = state["question"].lower()
    if any(k in q for k in ["근거", "출처", "어디", "언제", "무엇", "검색", "docs", "kb"]):
        state["route"] = "rag"
    elif any(k in q for k in ["날씨", "weather", "db", "sql", "고객", "회원"]):
        state["route"] = "tool"
    else:
        state["route"] = "direct"
    state["trace"].append(f"Router→{state['route']}")
    return state

# --- RAG 준비 (데모: 간단 검색 컨텍스트) ---
def retrieve_ctx(question: str) -> List[Dict[str, Any]]:
    res = tool_search(question)
    return [{"text": i["title"], "source": i["source"], "score": 0.7} for i in res["items"]]

# --- ReAct-lite 노드: 1~2회 툴 선택/관찰/후처리 ---
def rag_node(state: ReActState, llm: LLM) -> ReActState:
    state["ctx"] = retrieve_ctx(state["question"])  # 결정적
    steps = 0
    obs: Dict[str, Any] = {}
    while steps < MAX_STEPS:
        steps += 1
        ctx_snips = [c["text"] for c in state["ctx"]]
        tool_name = llm.choose_tool(state["question"], ctx_snips)
        if tool_name not in ALLOWED_TOOLS:
            state["trace"].append(f"Step{steps}: invalid_tool={tool_name}")
            break
        state["trace"].append(f"Step{steps}: Action={tool_name}")
        # Observation
        obs = ALLOWED_TOOLS[tool_name](state["question"])
        state["trace"].append(f"Step{steps}: Observation keys={list(obs.keys())}")
        # 간단 groundedness 스코어(데모)
        state["grounded"] = 0.6 if tool_name == Tool.SEARCH.value else 0.8
        # 간단 후처리: 답 생성
        draft = f"질문: {state['question']}\n도구: {tool_name}\n관측: {obs}\n"
        state["answer"] = llm.polish(draft)
        state["citations"] = [c["source"] for c in state["ctx"][:2]]
        break  # lite: 한 번이면 충분
    return state

# --- Tool-only 노드 (질의가 명확하게 툴 지칭) ---
def tool_node(state: ReActState, llm: LLM) -> ReActState:
    tool_name = llm.choose_tool(state["question"], [])
    if tool_name not in ALLOWED_TOOLS:
        state["answer"] = "허용되지 않은 도구 요청입니다."
        return state
    obs = ALLOWED_TOOLS[tool_name](state["question"])  # 결정적
    state["grounded"] = 0.85
    draft = f"요청을 '{tool_name}' 도구로 처리했습니다. 결과 요약: {obs}"
    state["answer"] = llm.polish(draft)
    return state

# --- Direct 노드 (LLM만 요약/답변) ---
def direct_node(state: ReActState, llm: LLM) -> ReActState:
    draft = f"질문: {state['question']}\n직접 답변 초안(데모): 관련 지식을 요약해 응답합니다."
    state["answer"] = llm.polish(draft)
    state["grounded"] = 0.4
    return state

# --- Orchestrator (LangGraph 사용) ---
def run_react_graph(question: str) -> Dict[str, Any]:
    llm = LLM()
    init: ReActState = {
        "question": question,
        "route": "direct",
        "ctx": [],
        "answer": "",
        "citations": [],
        "tries": 0,
        "grounded": 0.0,
        "trace": [],
    }

    if _LG:
        sg = StateGraph(ReActState)
        # 노드 래퍼 (LangGraph 노드는 state만 받음)
        def _router(s: ReActState):
            return router(s)
        def _rag(s: ReActState):
            return rag_node(s, llm)
        def _tool(s: ReActState):
            return tool_node(s, llm)
        def _direct(s: ReActState):
            return direct_node(s, llm)

        sg.add_node("router", _router)
        sg.add_node("rag", _rag)
        sg.add_node("tool", _tool)
        sg.add_node("direct", _direct)
        sg.set_entry_point("router")
        sg.add_conditional_edges(
            "router",
            lambda s: s["route"],
            {"rag": "rag", "tool": "tool", "direct": "direct"},
        )
        sg.add_edge("rag", END)
        sg.add_edge("tool", END)
        sg.add_edge("direct", END)
        app = sg.compile()
        out = app.invoke(init)
        return out
    else:
        # LangGraph 미사용 간이 실행(동일 로직)
        st = router(init)
        if st["route"] == "rag":
            st = rag_node(st, llm)
        elif st["route"] == "tool":
            st = tool_node(st, llm)
        else:
            st = direct_node(st, llm)
        return st

# =====( 2) 순수 FSM(Graph-only) — 결정적 플로우 )=====
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
    # 결정적 외부 호출(데모: 날씨 + DB)
    w = tool_weather(st.normalized["city"])  # 결정적
    d = tool_db("SELECT 1")
    st.api_result = {"weather": w, "db": d}
    return st

def fsm_render(st: FsmState, llm: Optional[LLM] = None) -> FsmState:
    if not st.valid:
        return st
    draft = (
        f"주문 접수: 사용자#{st.normalized['user_id']} / 도시={st.normalized['city']} / 품목={st.normalized['item']}\n"
        f"현재 날씨: {st.api_result['weather']['status']} {st.api_result['weather']['temp_c']}℃\n"
        f"처리 결과: OK"
    )
    st.message = llm.polish(draft) if llm else draft
    return st

# 간이 FSM 러너 (결정적)
def run_fsm_example(user_id: int = 1, city: str = "seoul", item: str = "umbrella") -> Dict[str, Any]:
    llm = LLM()
    st = FsmState(payload=OrderPayload(user_id=user_id, city=city, item=item))
    st = fsm_validate(st)
    st = fsm_normalize(st)
    st = fsm_call_api(st)
    st = fsm_render(st, llm)
    return {
        "valid": st.valid,
        "normalized": st.normalized,
        "api_result": st.api_result,
        "message": st.message,
    }

# =====( CLI )=====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["react", "fsm"], default="react")
    parser.add_argument("--q", default="서울 내일 날씨 알려줘.")
    args = parser.parse_args()

    if args.mode == "react":
        out = run_react_graph(args.q)
        print("=== Constrained ReAct + Graph ===")
        print("route:", out.get("route"))
        print("answer:\n", out.get("answer"))
        print("citations:", out.get("citations"))
        print("grounded:", out.get("grounded"))
        print("trace:", out.get("trace"))
    else:
        out = run_fsm_example()
        print("=== FSM (Graph-only, 결정적) ===")
        for k, v in out.items():
            print(f"{k}: {v}")
