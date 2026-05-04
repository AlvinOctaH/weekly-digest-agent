from typing import TypedDict, Optional

class DigestState(TypedDict):
    # Input
    topics: list[str]
    current_topic: str
    
    # Paper collection
    raw_papers: dict        # {topic: [paper, ...]}
    evaluated_papers: dict  # {topic: [paper, ...]}
    summaries: dict         # {topic: [summary, ...]}
    
    # Critic & retry
    critique_context: str   # feedback dari critic untuk planner berikutnya
    retry_count: int        # berapa kali sudah retry per topic
    partial_coverage: bool  # True kalau critic tidak pernah puas setelah max retry
    
    # Output
    synthesis: dict         # {topic: {trend, breakthrough, tldr, ...}}
    report: dict            # final report structure
    
    # Meta
    errors: list[str]       # error yang terjadi tapi tidak crash
    run_metadata: dict      # timing, paper counts, dll