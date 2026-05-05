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

papers = result["raw_papers"].get("retrieval augmented generation", [])
print(f"\n=== FETCHER OUTPUT ===")
print(f"Total papers: {len(papers)}")
for i, p in enumerate(papers[:3]):  # print 3 pertama
    print(f"\n[{i+1}] {p['title']}")
    print(f"     Authors: {', '.join(p['authors'][:3])}")
    print(f"     Date: {p['published_date']}")