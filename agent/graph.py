from langgraph.graph import StateGraph, END
from agent.state import DigestState
from loguru import logger

# ── Node placeholders ──────────────────────────────────────────────────────────
# Semua node isinya print dulu — kita isi satu per satu di hari berikutnya

def config_loader(state: DigestState) -> DigestState:
    logger.info("NODE: config_loader")
    return state

def planner(state: DigestState) -> DigestState:
    logger.info(f"NODE: planner | topic={state.get('current_topic')} | retry={state.get('retry_count', 0)}")
    return state

def fetcher(state: DigestState) -> DigestState:
    logger.info("NODE: fetcher")
    return state

def query_expander(state: DigestState) -> DigestState:
    logger.info("NODE: query_expander")
    return state

def diversity_checker(state: DigestState) -> DigestState:
    logger.info("NODE: diversity_checker")
    return state

def perspective_fetcher(state: DigestState) -> DigestState:
    logger.info("NODE: perspective_fetcher")
    return state

def summarizer(state: DigestState) -> DigestState:
    logger.info("NODE: summarizer")
    return state

def critic(state: DigestState) -> DigestState:
    logger.info("NODE: critic")
    return state

def synthesizer(state: DigestState) -> DigestState:
    logger.info("NODE: synthesizer")
    return state

def report_generator(state: DigestState) -> DigestState:
    logger.info("NODE: report_generator")
    return state

# ── Routing functions ──────────────────────────────────────────────────────────
# Ini decision points — agent decide mau ke node mana selanjutnya

def route_fetcher(state: DigestState) -> str:
    """Kalau paper kurang dari 3, expand query dulu."""
    papers = state.get("raw_papers", {}).get(state.get("current_topic", ""), [])
    if len(papers) < 3:
        logger.info(f"ROUTING: fetcher → query_expander (only {len(papers)} papers)")
        return "query_expander"
    logger.info("ROUTING: fetcher → diversity_checker")
    return "diversity_checker"

def route_diversity(state: DigestState) -> str:
    """Kalau sumber tidak beragam, cari perspektif lain dulu."""
    papers = state.get("raw_papers", {}).get(state.get("current_topic", ""), [])
    authors = set()
    for p in papers:
        authors.update(p.get("authors", []))
    if len(authors) < 5:
        logger.info("ROUTING: diversity_checker → perspective_fetcher (not diverse)")
        return "perspective_fetcher"
    logger.info("ROUTING: diversity_checker → summarizer")
    return "summarizer"

def route_critic(state: DigestState) -> str:
    """Critic decide: sufficient → synthesizer, insufficient → retry atau paksa lanjut."""
    verdict = state.get("report", {}).get("critic_verdict", "sufficient")
    retry = state.get("retry_count", 0)
    
    if verdict == "insufficient" and retry < 2:
        logger.info(f"ROUTING: critic → planner (retry {retry+1}/2)")
        return "planner"
    elif verdict == "insufficient" and retry >= 2:
        logger.info("ROUTING: critic → synthesizer (max retry reached, partial coverage)")
        return "synthesizer"
    logger.info("ROUTING: critic → synthesizer (sufficient)")
    return "synthesizer"

# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(DigestState)
    
    # Add semua node
    graph.add_node("config_loader", config_loader)
    graph.add_node("planner", planner)
    graph.add_node("fetcher", fetcher)
    graph.add_node("query_expander", query_expander)
    graph.add_node("diversity_checker", diversity_checker)
    graph.add_node("perspective_fetcher", perspective_fetcher)
    graph.add_node("summarizer", summarizer)
    graph.add_node("critic", critic)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("report_generator", report_generator)
    
    # Entry point
    graph.set_entry_point("config_loader")
    
    # Fixed edges (selalu ke node yang sama)
    graph.add_edge("config_loader", "planner")
    graph.add_edge("planner", "fetcher")
    graph.add_edge("query_expander", "diversity_checker")
    graph.add_edge("perspective_fetcher", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("synthesizer", "report_generator")
    graph.add_edge("report_generator", END)
    
    # Conditional edges (decision points)
    graph.add_conditional_edges("fetcher", route_fetcher)
    graph.add_conditional_edges("diversity_checker", route_diversity)
    graph.add_conditional_edges("critic", route_critic)
    
    return graph.compile()