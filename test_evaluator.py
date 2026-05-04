from agent.graph import build_graph
from loguru import logger

graph = build_graph()

initial_state = {
    "topics": ["retrieval augmented generation"],
    "current_topic": "retrieval augmented generation",
    "raw_papers": {},
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

passed = result["evaluated_papers"].get("retrieval augmented generation", [])
raw = result["raw_papers"].get("retrieval augmented generation", [])

print(f"\n=== EVALUATOR OUTPUT ===")
print(f"Raw papers: {len(raw)}")
print(f"Passed evaluation: {len(passed)}")
print(f"\nPapers that passed:")
for p in passed:
    print(f"  - {p['title'][:60]}")
    print(f"    Score: {p['avg_score']:.1f} | {p['scores']['reason']}")