from langgraph.graph import StateGraph, END
from agent.state import DigestState
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from loguru import logger
import json
from tools.arxiv_fetcher import fetch_recent_papers, expand_query, check_diversity

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# ── Nodes ──────────────────────────────────────────────────────────────────────

def config_loader(state: DigestState) -> DigestState:
    logger.info("NODE: config_loader")
    return state

def planner(state: DigestState) -> DigestState:
    topic = state.get("current_topic", "")
    retry_count = state.get("retry_count", 0)
    critique_context = state.get("critique_context", "")

    logger.info(f"NODE: planner | topic={topic} | retry={retry_count}")

    critique_section = ""
    if critique_context:
        critique_section = f"\nPrevious search was insufficient because: {critique_context}\nAdjust your search angle accordingly."

    prompt = f"""You are planning a search strategy for a weekly AI/ML research digest.
Topic: {topic}
{critique_section}

Generate a search plan. Return ONLY valid JSON, no explanation, no markdown:
{{
    "primary_query": "most specific search query for this topic",
    "sub_queries": ["broader angle query", "application-focused query"],
    "search_angle": "what specific aspect to focus on"
}}"""

    try:
        response = llm.invoke(prompt)
        plan = json.loads(response.content)
        logger.info(f"Planner generated queries: {plan}")
    except json.JSONDecodeError:
        logger.warning("Planner: LLM returned invalid JSON, using fallback")
        plan = {
            "primary_query": topic,
            "sub_queries": [topic + " survey", topic + " applications"],
            "search_angle": "general overview"
        }

    state["run_metadata"]["current_plan"] = plan
    return state

def fetcher(state: DigestState) -> DigestState:
    topic = state.get("current_topic", "")
    plan = state.get("run_metadata", {}).get("current_plan", {})
    
    logger.info(f"NODE: fetcher | topic={topic}")
    
    # Ambil semua queries dari planner
    queries = [plan.get("primary_query", topic)] + plan.get("sub_queries", [])
    
    all_papers = []
    seen_ids = set()
    
    for query in queries:
        try:
            papers = fetch_recent_papers(query, days=7, max_results=10)
            for paper in papers:
                # Deduplicate by arxiv_id
                if paper["arxiv_id"] not in seen_ids:
                    seen_ids.add(paper["arxiv_id"])
                    all_papers.append(paper)
        except Exception as e:
            logger.warning(f"Fetcher: failed for query '{query}': {e}")
            state["errors"].append(f"Fetch failed for query: {query}")
    
    logger.info(f"Fetcher: collected {len(all_papers)} unique papers for topic '{topic}'")
    
    # Simpan ke state
    if "raw_papers" not in state:
        state["raw_papers"] = {}
    state["raw_papers"][topic] = all_papers
    
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

def route_fetcher(state: DigestState) -> str:
    papers = state.get("raw_papers", {}).get(state.get("current_topic", ""), [])
    if len(papers) < 3:
        logger.info(f"ROUTING: fetcher → query_expander (only {len(papers)} papers)")
        return "query_expander"
    logger.info("ROUTING: fetcher → diversity_checker")
    return "diversity_checker"

def route_diversity(state: DigestState) -> str:
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

    graph.set_entry_point("config_loader")

    graph.add_edge("config_loader", "planner")
    graph.add_edge("planner", "fetcher")
    graph.add_edge("query_expander", "diversity_checker")
    graph.add_edge("perspective_fetcher", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("synthesizer", "report_generator")
    graph.add_edge("report_generator", END)

    graph.add_conditional_edges("fetcher", route_fetcher)
    graph.add_conditional_edges("diversity_checker", route_diversity)
    graph.add_conditional_edges("critic", route_critic)

    return graph.compile()