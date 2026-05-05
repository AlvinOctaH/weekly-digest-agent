from agent.graph import build_graph

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

print(f"\n=== CRITIC OUTPUT ===")
print(f"Verdict  : {result['report'].get('critic_verdict')}")
scores = result['report'].get('critic_scores', {})
print(f"Coverage : {scores.get('coverage')}")
print(f"Depth    : {scores.get('depth')}")
print(f"Diversity: {scores.get('diversity')}")
print(f"Critique : {result.get('critique_context', '-')}")
print(f"Retry cnt: {result.get('retry_count')}")