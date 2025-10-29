import os
import getpass
import json
import re
import math
import string
from typing import List, Optional, Literal, TypedDict, Dict, Any, Tuple
from urllib.parse import urlencode, quote
from urllib.request import urlopen, Request

from supabase import create_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# ===========
# ENV (OpenAI only; Supabase comes from .env/local shell)
# ===========
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Keep if you still use the LLM relevancy gate. Otherwise, remove this line.
_set_env("OPENAI_API_KEY")

# ===========
# Supabase helpers ("tool" code)
# ===========
def _sb_client():
    """Create a Supabase client from env. Uses ANON by default; falls back to SERVICE_ROLE if present."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL and/or SUPABASE_ANON_KEY (or SERVICE_ROLE) in environment.")
    return create_client(url, key)

def load_qa_from_supabase() -> Tuple[List[str], List[str]]:
    """Load all Q/A pairs from public.knowledge into in-memory lists."""
    sb = _sb_client()
    res = sb.table("knowledge").select("q,a").execute()
    rows = res.data or []
    qs = [r["q"] for r in rows if r.get("q") and r.get("a")]
    ans = [r["a"] for r in rows if r.get("q") and r.get("a")]
    return qs, ans

def refresh_knowledge() -> None:
    """Refresh global in-memory cache from Supabase (callable at runtime if needed)."""
    global KNOWLEDGE_Q, KNOWLEDGE_A
    KNOWLEDGE_Q, KNOWLEDGE_A = load_qa_from_supabase()

def add_qa(q: str, a: str) -> None:
    """(Dev helper) Insert or upsert a new Q/A into Supabase."""
    sb = _sb_client()
    # If you added UNIQUE(q) you can use upsert; otherwise use insert.
    sb.table("knowledge").upsert({"q": q, "a": a}, on_conflict="q").execute()
    refresh_knowledge()

# ===========
# Knowledge cache (from Supabase)
# ===========
KNOWLEDGE_Q, KNOWLEDGE_A = load_qa_from_supabase()

# ===========
# Models (LLM relevancy gate)
# ===========
LLM_RELEVANCY = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===========
# State
# ===========
class State(TypedDict):
    question: str
    is_relevant: Optional[bool]
    found_basic: Optional[bool]
    label: Optional[Literal["basic","advanced"]]
    matched_index: Optional[int]
    final_answer: Optional[str]
    pending_answer: Optional[str]
    exit: Optional[bool]

# ===========
# Relevancy helpers
# ===========
FORMULATION_KEYWORDS = {
    "ulv","ec","sc","sl","wg","wdg","wp","gr","cs","se","ew","od",
    "emulsion","emulsifiable","suspension","granule","capsule","suspoemulsion",
    "surfactant","adjuvant","ams","buffer","sticker","spreader","oil","mso",
    "rainfast","rainfastness","deposition","drift","zeta","particle","d90",
    "microencapsulation","controlled release","stability","compatibility",
    "tank mix","jar test","spray","nozzle","cuticle","uptake","pesticide","fertilizer",
    "formulation","formulated"
}

def normalize_question(q: str) -> str:
    q = re.sub(r"^\s*[Qq]\s*:\s*", "", q)
    return re.sub(r"\s+", " ", q).strip()

def contains_formulation_terms(q: str) -> bool:
    ql = q.lower()
    return any(term in ql for term in FORMULATION_KEYWORDS)

# ===========
# Token-based matching (unchanged)
# ===========
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_STOPWORDS = {
    "a","an","the","and","or","but","if","then","so","than","that","this","these","those",
    "of","for","to","in","on","at","by","with","from","as","is","are","was","were","be",
    "been","being","do","does","did","doing","have","has","had","having","it","its","itself",
    "you","your","yours","we","our","ours","they","their","theirs","he","she","his","her","hers",
    "what","which","who","whom","where","when","why","how","about","into","over","under","again",
    "once","only","also","can","could","should","would","may","might","must","will","shall"
}

def _normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())

def _tokens(text: str) -> List[str]:
    return [t for t in _normalize(text).split() if t and t not in _STOPWORDS]

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)

def _cosine_tf(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens: return 0.0
    def tf(tokens: List[str]) -> Dict[str,int]:
        d: Dict[str,int] = {}
        for t in tokens: d[t] = d.get(t, 0) + 1
        return d
    ta, tb = tf(a_tokens), tf(b_tokens)
    keys = set(ta) | set(tb)
    dot = sum(ta.get(k,0)*tb.get(k,0) for k in keys)
    na = math.sqrt(sum(v*v for v in ta.values()))
    nb = math.sqrt(sum(v*v for v in tb.values()))
    return (dot / (na*nb)) if na and nb else 0.0

def question_similarity(q_user: str, q_canon: str) -> float:
    tu, tc = _tokens(q_user), _tokens(q_canon)
    j = _jaccard(set(tu), set(tc))
    c = _cosine_tf(tu, tc)
    return 0.6*j + 0.4*c

def best_match(user_q: str, candidates: List[str], min_score: float = 0.4) -> tuple[Optional[int], float]:
    best_i, best_s = None, 0.0
    for i, cand in enumerate(candidates):
        s = question_similarity(user_q, cand)
        if s > best_s:
            best_i, best_s = i, s
    return (best_i, best_s) if best_i is not None and best_s >= min_score else (None, 0.0)

# ===========
# Prompts (for relevancy only)
# ===========
def relevancy_messages(q: str):
    return [
        SystemMessage(
            content=(
                "Decide if a question is about FORMULATION SCIENCE "
                "(agrochemical formulations, adjuvants, surfactants, emulsions, controlled release, carriers, "
                "wetting/spreading, pesticide/fertilizer formulation types, mixing/compatibility, safety/handling of "
                "formulated products, and application tech as it relates to formulations). "
                'Return JSON only: {"is_relevant": true|false, "reason": "..."}'
            )
        ),
        HumanMessage(content=f"Question: {q}\nReturn JSON only."),
    ]

# ===========
# Wikipedia helpers (unchanged)
# ===========
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

def _http_get_json(url: str, timeout: int = 8) -> Dict[str, Any]:
    req = Request(url, headers={"User-Agent": "LangGraph-Agent/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="ignore"))

def wiki_summary_by_title(title: str) -> Optional[Dict[str, Any]]:
    url = f"{WIKI_SUMMARY}{quote(title)}"
    try:
        data = _http_get_json(url)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("type") == "disambiguation":
        return None
    if data.get("extract"):
        return data
    return None

def wiki_opensearch(query: str, limit: int = 5) -> List[str]:
    params = {
        "action": "opensearch",
        "search": query,
        "limit": str(limit),
        "namespace": "0",
        "format": "json",
        "origin": "*",
    }
    url = f"{WIKI_API}?{urlencode(params)}"
    try:
        data = _http_get_json(url)
    except Exception:
        return []
    return data[1] if isinstance(data, list) and len(data) >= 2 else []

def wiki_search(query: str, limit: int = 5, nearmatch: bool = True) -> List[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "srqiprofile": "classic_noboostlinks",
        "format": "json",
        "utf8": "1",
        "origin": "*",
    }
    if nearmatch:
        params["srwhat"] = "nearmatch"
    url = f"{WIKI_API}?{urlencode(params)}"
    try:
        data = _http_get_json(url)
    except Exception:
        return []
    search_hits = data.get("query", {}).get("search", []) or []
    return [hit.get("title") for hit in search_hits if isinstance(hit, dict) and hit.get("title")]

_STOP_QWORDS = {
    "what","who","when","where","why","how","is","are","was","were","be","being","been",
    "does","do","did","use","used","using","for","in","of","the","a","an","and","to","on",
    "with","by","from","define","explain","difference","between"
}

def _core_topic_from_question(q: str) -> List[str]:
    q_norm = re.sub(r"[^\w\s\-–—/()]", " ", q.lower())
    q_norm = " ".join(q_norm.split())
    tokens = [t for t in q_norm.split() if t not in _STOP_QWORDS]
    cut = re.split(r"\b(in|for|of|on|with|from|to|between)\b", " ".join(tokens), maxsplit=1)[0].strip()
    core = cut if cut else " ".join(tokens)
    candidates = []
    if core:
        candidates.append(core)
    sig = [t for t in tokens if len(t) > 3][:3]
    if sig:
        candidates.append(" ".join(sig))
    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    expansions: List[str] = []
    for c in ordered:
        expansions.append(c)
        expansions.append(c.replace("-", " "))
        expansions.append(c.replace(" – ", " ").replace(" — ", " "))
    titles = []
    for e in expansions:
        t = e.strip().replace("_"," ").strip()
        if t:
            titles.append(t.title())
    out = []
    seen = set()
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def wiki_best_answer(query: str) -> Optional[str]:
    direct_titles = _core_topic_from_question(query)
    for title in direct_titles:
        summ = wiki_summary_by_title(title)
        if not summ and " " in title:
            summ = wiki_summary_by_title(title.split(" ")[0].title())
        if summ:
            extract = summ.get("extract", "").strip()
            page_url = (summ.get("content_urls", {}).get("desktop", {}).get("page")
                        or f"https://en.wikipedia.org/wiki/{quote(title)}")
            return f"{extract}\n\n— Source: Wikipedia • {summ.get('title', title)}\n{page_url}"
    os_titles = wiki_opensearch(query, limit=5)
    for title in os_titles:
        summ = wiki_summary_by_title(title)
        if summ:
            extract = summ.get("extract", "").strip()
            page_url = (summ.get("content_urls", {}).get("desktop", {}).get("page")
                        or f"https://en.wikipedia.org/wiki/{quote(title)}")
            return f"{extract}\n\n— Source: Wikipedia • {summ.get('title', title)}\n{page_url}"
    for nearmatch in (True, False):
        ft_titles = wiki_search(query, limit=5, nearmatch=nearmatch)
        for title in ft_titles:
            summ = wiki_summary_by_title(title)
            if summ:
                extract = summ.get("extract", "").strip()
                page_url = (summ.get("content_urls", {}).get("desktop", {}).get("page")
                            or f"https://en.wikipedia.org/wiki/{quote(title)}")
                return f"{extract}\n\n— Source: Wikipedia • {summ.get('title', title)}\n{page_url}"
    return None

# ===========
# Nodes
# ===========
def Relevancy(state: State) -> Dict[str, Any]:
    cleared: Dict[str, Any] = {
        "is_relevant": None,
        "found_basic": None,
        "label": None,
        "matched_index": None,
        "final_answer": None,
        "pending_answer": None,
        "exit": None,
    }
    q_raw = state["question"]
    q = normalize_question(q_raw)

    if contains_formulation_terms(q):
        cleared.update({"is_relevant": True, "exit": False})
        return cleared

    resp = LLM_RELEVANCY.invoke(relevancy_messages(q))
    try:
        data = json.loads(resp.content)
        is_rel = bool(data.get("is_relevant", False))
    except Exception:
        is_rel = False

    if not is_rel:
        cleared.update({
            "is_relevant": False,
            "exit": True,
            "final_answer": (
                "Sorry, that’s outside the scope of formulation science. "
                "Please ask about formulations (e.g., adjuvants, emulsions, mixing, safety, application)."
            ),
        })
        return cleared

    cleared.update({"is_relevant": True, "exit": False})
    return cleared

def WithinDocs(state: State) -> Dict[str, Any]:
    if not KNOWLEDGE_Q:
        return {"found_basic": False, "label": "advanced", "pending_answer": None, "matched_index": None}

    user_q = normalize_question(state["question"])
    idx, score = best_match(user_q, KNOWLEDGE_Q, min_score=0.4)

    if idx is not None:
        return {
            "found_basic": True,
            "label": "basic",
            "matched_index": idx,
            "pending_answer": KNOWLEDGE_A[idx] + f"\n\n[match_score: {score:.2f}]",
            "final_answer": None,
        }

    return {
        "found_basic": False,
        "label": "advanced",
        "matched_index": None,
        "pending_answer": None,
        "final_answer": None,
    }

def Int(state: State) -> Dict[str, Any]:
    return {}

def Ext(state: State) -> Dict[str, Any]:
    q = normalize_question(state["question"])
    try:
        answer = wiki_best_answer(q)
    except Exception:
        answer = None
    if not answer:
        answer = "I couldn’t find a reliable summary for that topic on Wikipedia."
    return {"pending_answer": answer, "final_answer": None, "matched_index": None}

def Out(state: State) -> Dict[str, Any]:
    final = state.get("final_answer") or state.get("pending_answer") or "(no answer generated)"
    return {"messages": [AIMessage(content=final)], "final_answer": final}

# ===========
# Routing
# ===========
def route_after_relevancy(output: Dict[str, Any]) -> str:
    return END if output.get("exit") else "WithinDocs"

def route_after_within(output: Dict[str, Any]) -> str:
    return "Int" if output.get("found_basic") else "Ext"

# ===========
# Graph
# ===========
builder = StateGraph(State)
builder.add_node("Relevancy", Relevancy)
builder.add_node("WithinDocs", WithinDocs)
builder.add_node("Int", Int)
builder.add_node("Ext", Ext)
builder.add_node("Out", Out)

builder.add_edge(START, "Relevancy")
builder.add_conditional_edges("Relevancy", route_after_relevancy, {END: END, "WithinDocs": "WithinDocs"})
builder.add_conditional_edges("WithinDocs", route_after_within, {"Int": "Int", "Ext": "Ext"})
builder.add_edge("Int", "Out")
builder.add_edge("Ext", "Out")
builder.add_edge("Out", END)

graph = builder.compile()
react_graph = graph

# ===========
# Minimal helper
# ===========
def run(question: str) -> str:
    state: State = {
        "question": question,
        "is_relevant": None,
        "found_basic": None,
        "label": None,
        "matched_index": None,
        "final_answer": None,
        "pending_answer": None,
        "exit": None,
    }
    result = graph.invoke(state)
    return result.get("final_answer") or ""
