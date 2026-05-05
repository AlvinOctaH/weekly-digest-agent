import arxiv
from datetime import datetime, timedelta, timezone
from loguru import logger
import time

def fetch_recent_papers(query: str, days: int = 7, max_results: int = 10) -> list[dict]:
    """Fetch papers from ArXiv from the last N days."""
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results * 2,  # fetch extra, will filter by date
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in client.results(search):
        if result.published < cutoff_date:
            continue
            
        papers.append({
            "title": result.title,
            "abstract": result.summary[:500],  # truncate biar hemat token
            "authors": [a.name for a in result.authors[:5]],
            "url": result.entry_id,
            "published_date": result.published.strftime("%Y-%m-%d"),
            "arxiv_id": result.entry_id.split("/")[-1]
        })
    
    time.sleep(3)
    logger.info(f"Fetched {len(papers)} papers for query: '{query}'")
    return papers[:max_results]


def check_diversity(papers: list[dict]) -> dict:
    """Check if papers come from diverse sources."""
    
    if not papers:
        return {"unique_author_count": 0, "is_diverse": False, "total_papers": 0}
    
    all_authors = set()
    for paper in papers:
        for author in paper["authors"]:
            all_authors.add(author)
    
    is_diverse = len(all_authors) >= 5  # minimal 5 unique authors
    
    return {
        "unique_author_count": len(all_authors),
        "is_diverse": is_diverse,
        "total_papers": len(papers)
    }


def expand_query(original_query: str) -> str:
    """Expand a query to be broader when results are insufficient."""
    
    expansions = {
        "retrieval augmented generation": "information retrieval language model knowledge",
        "LLM reasoning": "large language model chain of thought reasoning inference",
        "multimodal learning": "vision language model image text multimodal",
    }
    
    return expansions.get(original_query, original_query + " machine learning deep learning")