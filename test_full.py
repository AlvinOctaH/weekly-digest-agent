from agent.graph import build_graph
from loguru import logger
import json

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

logger.info("Starting full pipeline test...")
result = graph.invoke(initial_state)

print(f"\n=== FULL PIPELINE OUTPUT ===")
print(f"Verdict  : {result['report'].get('critic_verdict')}")
print(f"Retry cnt: {result.get('retry_count')}")
print(f"Week     : {result['report'].get('week')}")

for topic, data in result['report'].get('topics', {}).items():
    synth = data.get('synthesis', {})
    print(f"\nTopic: {topic}")
    print(f"  TL;DR : {synth.get('tldr')}")
    print(f"  Trend : {synth.get('weekly_trend')}")
    print(f"  Papers: {len(data.get('papers', []))}")

print(f"\nOutput files written to: output/digest_{result['report'].get('week')}/")