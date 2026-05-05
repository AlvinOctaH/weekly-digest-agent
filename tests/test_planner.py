from agent.graph import build_graph
from loguru import logger

graph = build_graph()

initial_state = {
    "topics": ["retrieval augmented generation"],
    "current_topic": "retrieval augmented generation",
    "raw_papers": {
        "retrieval augmented generation": [
            {"title": "Paper 1", "authors": ["Author A", "Author B"]},
            {"title": "Paper 2", "authors": ["Author C", "Author D"]},
            {"title": "Paper 3", "authors": ["Author E", "Author F"]},
            {"title": "Paper 4", "authors": ["Author G", "Author H"]},
            {"title": "Paper 5", "authors": ["Author I", "Author J"]},
        ]
    },
    "evaluated_papers": {},
    "summaries": {},
    "critique_context": "",
    "retry_count": 0,
    "partial_coverage": False,
    "synthesis": {},
    "report": {"critic_verdict": "sufficient"},
    "errors": [],
    "run_metadata": {}
}

result = graph.invoke(initial_state)

# Cek hasil planner
plan = result["run_metadata"].get("current_plan", {})
print("\n=== PLANNER OUTPUT ===")
print(f"Primary query: {plan.get('primary_query')}")
print(f"Sub queries: {plan.get('sub_queries')}")
print(f"Search angle: {plan.get('search_angle')}")