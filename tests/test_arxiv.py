from tools.arxiv_fetcher import fetch_recent_papers, check_diversity

papers = fetch_recent_papers('retrieval augmented generation', days=7)
print(f'Papers found: {len(papers)}')

if papers:
    print(f'First paper: {papers[0]["title"]}')
    print(f'Published: {papers[0]["published_date"]}')

print(check_diversity(papers))