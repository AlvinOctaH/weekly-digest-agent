from langgraph.graph import StateGraph, END
from agent.state import DigestState
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from loguru import logger
from tools.arxiv_fetcher import fetch_recent_papers, expand_query, check_diversity
import json
import time

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

    queries = [plan.get("primary_query", topic)] + plan.get("sub_queries", [])

    all_papers = []
    seen_ids = set()

    for query in queries:
        try:
            papers = fetch_recent_papers(query, days=7, max_results=10)
            for paper in papers:
                if paper["arxiv_id"] not in seen_ids:
                    seen_ids.add(paper["arxiv_id"])
                    all_papers.append(paper)
        except Exception as e:
            logger.warning(f"Fetcher: failed for query '{query}': {e}")
            state["errors"].append(f"Fetch failed for query: {query}")

    logger.info(f"Fetcher: collected {len(all_papers)} unique papers for topic '{topic}'")

    if "raw_papers" not in state:
        state["raw_papers"] = {}
    state["raw_papers"][topic] = all_papers

    return state

def query_expander(state: DigestState) -> DigestState:
    topic = state.get("current_topic", "")
    logger.info(f"NODE: query_expander | topic={topic}")

    expanded = expand_query(topic)
    logger.info(f"Expanded query: '{expanded}'")

    try:
        papers = fetch_recent_papers(expanded, days=7, max_results=10)
        existing = state.get("raw_papers", {}).get(topic, [])
        seen_ids = {p["arxiv_id"] for p in existing}

        new_papers = [p for p in papers if p["arxiv_id"] not in seen_ids]
        state["raw_papers"][topic] = existing + new_papers
        logger.info(f"Query expander added {len(new_papers)} new papers")
    except Exception as e:
        logger.warning(f"Query expander failed: {e}")

    return state

def diversity_checker(state: DigestState) -> DigestState:
    logger.info("NODE: diversity_checker")
    return state

def perspective_fetcher(state: DigestState) -> DigestState:
    topic = state.get("current_topic", "")
    logger.info(f"NODE: perspective_fetcher | topic={topic}")

    existing = state.get("raw_papers", {}).get(topic, [])
    existing_authors = set()
    for p in existing:
        existing_authors.update(p.get("authors", []))

    # Fetch dengan query yang lebih broad
    try:
        broader_query = topic + " novel approach"
        papers = fetch_recent_papers(broader_query, days=14, max_results=10)
        seen_ids = {p["arxiv_id"] for p in existing}

        new_papers = []
        for p in papers:
            authors = set(p.get("authors", []))
            # Prioritaskan paper dari author yang belum ada
            if p["arxiv_id"] not in seen_ids and not authors.issubset(existing_authors):
                new_papers.append(p)
                seen_ids.add(p["arxiv_id"])

        state["raw_papers"][topic] = existing + new_papers
        logger.info(f"Perspective fetcher added {len(new_papers)} papers from new perspectives")
    except Exception as e:
        logger.warning(f"Perspective fetcher failed: {e}")

    return state

def evaluator(state: DigestState) -> DigestState:
    topic = state.get("current_topic", "")
    papers = state.get("raw_papers", {}).get(topic, [])

    logger.info(f"NODE: evaluator | evaluating {len(papers)} papers for topic '{topic}'")

    evaluated = []

    for paper in papers:
        prompt = f"""Evaluate this paper for an AI/ML weekly digest.
Topic context: {topic}

Title: {paper['title']}
Abstract: {paper['abstract']}

Return ONLY valid JSON, no explanation, no markdown:
{{
    "relevance_score": <1-10>,
    "novelty_score": <1-10>,
    "reason": "<one sentence why>"
}}"""

        try:
            response = llm.invoke(prompt)
            scores = json.loads(response.content)
            avg_score = (scores["relevance_score"] + scores["novelty_score"]) / 2
            paper["scores"] = scores
            paper["avg_score"] = avg_score

            logger.info(f"Paper: '{paper['title'][:50]}...' | relevance={scores['relevance_score']} novelty={scores['novelty_score']}")

            if avg_score >= 6:
                evaluated.append(paper)
            else:
                logger.info(f"Paper filtered out (score {avg_score:.1f}): {paper['title'][:50]}")

        except json.JSONDecodeError:
            logger.warning(f"Evaluator: invalid JSON for paper '{paper['title'][:40]}', using default score 5")
            paper["scores"] = {"relevance_score": 5, "novelty_score": 5, "reason": "Could not evaluate"}
            paper["avg_score"] = 5.0

        time.sleep(1)  # hindari rate limit Groq

    if "evaluated_papers" not in state:
        state["evaluated_papers"] = {}
    state["evaluated_papers"][topic] = evaluated

    logger.info(f"Evaluator: {len(evaluated)}/{len(papers)} papers passed for topic '{topic}'")
    return state

def summarizer(state: DigestState) -> DigestState:
    topic = state.get("current_topic", "")
    papers = state.get("evaluated_papers", {}).get(topic, [])

    logger.info(f"NODE: summarizer | summarizing {len(papers)} papers for topic '{topic}'")

    summaries = []

    for paper in papers:
        prompt = f"""Summarize this AI/ML research paper for a weekly digest.

Title: {paper['title']}
Abstract: {paper['abstract']}

Return ONLY valid JSON, no explanation, no markdown:
{{
    "key_contribution": "what is the main contribution in 1 sentence",
    "methodology": "how they did it in 1 sentence",
    "main_result": "what they achieved in 1 sentence",
    "why_it_matters": "why this is important for the field in 1 sentence"
}}"""

        try:
            response = llm.invoke(prompt)
            summary = json.loads(response.content)
            summary["title"] = paper["title"]
            summary["url"] = paper["url"]
            summary["authors"] = paper["authors"]
            summary["published_date"] = paper["published_date"]
            summary["avg_score"] = paper.get("avg_score", 0)
            summaries.append(summary)
            logger.info(f"Summarized: '{paper['title'][:50]}'")

        except json.JSONDecodeError:
            logger.warning(f"Summarizer: invalid JSON for '{paper['title'][:40]}', skipping")

        time.sleep(1)

    if "summaries" not in state:
        state["summaries"] = {}
    state["summaries"][topic] = summaries

    logger.info(f"Summarizer: created {len(summaries)} summaries for topic '{topic}'")
    return state

def critic(state: DigestState) -> DigestState:
    topic = state.get("current_topic", "")
    summaries = state.get("summaries", {}).get(topic, [])
    retry_count = state.get("retry_count", 0)

    logger.info(f"NODE: critic | topic={topic} | summaries={len(summaries)} | retry={retry_count}")

    if not summaries:
        logger.warning("Critic: no summaries to evaluate, forcing insufficient")
        state["report"]["critic_verdict"] = "insufficient"
        state["critique_context"] = "No papers found. Try different search terms."
        state["retry_count"] = retry_count + 1
        return state

    summaries_text = json.dumps([{
        "title": s["title"],
        "key_contribution": s["key_contribution"],
        "why_it_matters": s["why_it_matters"]
    } for s in summaries], indent=2)

    prompt = f"""You are evaluating the quality of a weekly AI/ML research digest.
Topic: {topic}
Number of papers collected: {len(summaries)}

Summaries:
{summaries_text}

Evaluate on 3 dimensions:
1. Coverage: are the main aspects of this topic covered?
2. Depth: are the summaries informative enough?
3. Diversity: are there different perspectives and approaches?

Use "sufficient" only if ALL scores >= 6.
Return ONLY valid JSON, no explanation, no markdown:
{{
    "verdict": "sufficient" or "insufficient",
    "coverage_score": <1-10>,
    "depth_score": <1-10>,
    "diversity_score": <1-10>,
    "critique_text": "specific reason if insufficient, null if sufficient",
    "specific_gaps": ["gap 1", "gap 2"]
}}"""

    try:
        response = llm.invoke(prompt)
        evaluation = json.loads(response.content)

        verdict = evaluation.get("verdict", "sufficient")
        coverage = evaluation.get("coverage_score", 0)
        depth = evaluation.get("depth_score", 0)
        diversity = evaluation.get("diversity_score", 0)

        logger.info(f"Critic verdict: {verdict} | coverage={coverage} depth={depth} diversity={diversity}")

        state["report"]["critic_verdict"] = verdict
        state["report"]["critic_scores"] = {
            "coverage": coverage,
            "depth": depth,
            "diversity": diversity
        }

        if verdict == "insufficient":
            state["critique_context"] = evaluation.get("critique_text", "")
            state["retry_count"] = retry_count + 1
            logger.info(f"Critique: {state['critique_context']}")
        else:
            state["critique_context"] = ""

    except json.JSONDecodeError:
        logger.warning("Critic: invalid JSON, defaulting to sufficient")
        state["report"]["critic_verdict"] = "sufficient"

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
    logger.info("ROUTING: diversity_checker → evaluator")
    return "evaluator"

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
    graph.add_node("evaluator", evaluator)
    graph.add_node("summarizer", summarizer)
    graph.add_node("critic", critic)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("report_generator", report_generator)

    graph.set_entry_point("config_loader")

    graph.add_edge("config_loader", "planner")
    graph.add_edge("planner", "fetcher")
    graph.add_edge("query_expander", "diversity_checker")
    graph.add_edge("perspective_fetcher", "evaluator")
    graph.add_edge("evaluator", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("synthesizer", "report_generator")
    graph.add_edge("report_generator", END)

    graph.add_conditional_edges("fetcher", route_fetcher)
    graph.add_conditional_edges("diversity_checker", route_diversity)
    graph.add_conditional_edges("critic", route_critic)

    return graph.compile()