# Weekly AI/ML Digest Agent

An autonomous agent that monitors AI/ML research papers weekly and generates structured digest reports — automatically.

Built with LangGraph, Groq (Llama 3.3 70B), and ArXiv API.

## Demo

> 📹 Demo video — coming soon

## How It Works

The agent uses a **state machine with 3 real decision points** — not a fixed pipeline.

```
TRIGGER (manual / scheduler)
      │
      ▼
  PLANNER → generates search queries with different angles
      │
      ▼
  FETCHER → searches ArXiv with all queries, deduplicates results
      │
      ├── < 3 papers? → QUERY EXPANDER → retry with broader terms
      │
      ▼
  DIVERSITY CHECKER → checks if sources are varied enough
      │
      ├── too homogeneous? → PERSPECTIVE FETCHER → find different viewpoints
      │
      ▼
  EVALUATOR → scores each paper (relevance + novelty) via LLM
      │
      ▼
  SUMMARIZER → extracts: contribution, method, result, why it matters
      │
      ▼
  CRITIC → evaluates coverage, depth, diversity
      │
      ├── "insufficient" + retry < 2 → back to PLANNER with critique context
      ├── "insufficient" + retry = 2 → SYNTHESIZER (partial coverage flag)
      └── "sufficient" → SYNTHESIZER
              │
              ▼
          REPORT GENERATOR → JSON + Markdown output
```

**What makes this agentic (not just a pipeline):**
- The Critic decides when results are good enough — the agent does not stop at a fixed number of steps
- If coverage is insufficient, the Planner gets the critique and adjusts its search strategy
- Three different failure modes are handled gracefully: too few papers, homogeneous sources, low quality summaries

## Output

Each run produces a folder `output/digest_YYYY-WNN/` with:

```
digest_2026-W18/
├── digest.json   ← structured data, machine readable
└── digest.md     ← formatted report, human readable
```

See [`examples/`](examples/) for a sample output.

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/AlvinOctaH/weekly-digest-agent.git
cd weekly-digest-agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install langgraph langchain langchain-groq arxiv loguru pyyaml apscheduler python-dotenv
```

Get a free Groq API key at https://console.groq.com then paste into `.env`:

```
GROQ_API_KEY=your_key_here
```

Edit `config.yaml` to set your topics:

```yaml
topics:
  - "retrieval augmented generation"
  - "LLM reasoning"
  - "multimodal learning"
max_papers_per_topic: 10
min_relevance_score: 6
schedule:
  day_of_week: mon
  hour: 8
  minute: 0
```

## Usage

```bash
# Run digest now
python main.py run

# Run for specific topics
python main.py run --topics "RAG" "diffusion models"

# Re-fetch all papers (ignore seen history)
python main.py run --ignore-seen

# Start weekly scheduler (every Monday 08:00)
python main.py schedule

# Check run history
python main.py status
```

## Architecture Decisions

**Why LangGraph instead of plain LangChain?**
LangGraph gives explicit state management and conditional routing. Every decision point is visible in the graph — not buried in chain logic. It also supports checkpointing out of the box.

**Why Groq + Llama 3.3 70B?**
Free tier with generous limits (14,400 requests/day). Fast inference. No billing setup required — good for development and demo purposes.

**Why SQLite for deduplication?**
Single-machine use case, no concurrent writes, no need for a separate database server. Simple and reliable for this scope.

**Why max 2 retries in the Critic loop?**
Prevents infinite loops when papers genuinely do not exist for a topic in a given week. After 2 retries, the system produces a partial report rather than failing silently.

## Limitations

- ArXiv API rate limits can cause some queries to fail (handled gracefully — agent continues with partial results)
- LLM evaluator can be lenient — papers with tangential relevance sometimes pass the filter
- Only searches ArXiv — does not include Semantic Scholar, IEEE, or ACM yet
- Groq free tier: 14,400 requests/day, which limits how many topics can be processed per run

## What I Would Add Next

- Semantic Scholar integration for citation count and influence metrics
- Embedding-based relevance scoring (replace LLM evaluator with sentence-transformers for speed)
- Streamlit UI for non-technical users
- Email delivery of the weekly digest
