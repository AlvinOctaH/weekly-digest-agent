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

summaries = result["summaries"].get("retrieval augmented generation", [])
print(f"\n=== SUMMARIZER OUTPUT ===")
print(f"Total summaries: {len(summaries)}")

for i, s in enumerate(summaries):
    print(f"\n[{i+1}] {s['title'][:60]}")
    print(f"     Contribution : {s['key_contribution']}")
    print(f"     Method       : {s['methodology']}")
    print(f"     Result       : {s['main_result']}")
    print(f"     Why it matters: {s['why_it_matters']}")