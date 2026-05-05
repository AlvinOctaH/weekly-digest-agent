import sqlite3
import os
from loguru import logger

DB_PATH = "memory/seen.db"

def _get_conn():
    """Buat koneksi ke SQLite database."""
    os.makedirs("memory", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            topic TEXT,
            seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def is_seen(arxiv_id: str) -> bool:
    """Cek apakah paper sudah pernah diproses sebelumnya."""
    try:
        conn = _get_conn()
        result = conn.execute(
            "SELECT 1 FROM seen_papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        conn.close()
        return result is not None
    except Exception as e:
        logger.warning(f"seen_papers: error checking {arxiv_id}: {e}")
        return False

def mark_seen(papers: list[dict], topic: str):
    """Tandai paper sebagai sudah diproses."""
    try:
        conn = _get_conn()
        for paper in papers:
            conn.execute(
                "INSERT OR IGNORE INTO seen_papers (arxiv_id, title, topic) VALUES (?, ?, ?)",
                (paper["arxiv_id"], paper["title"], topic)
            )
        conn.commit()
        conn.close()
        logger.info(f"Marked {len(papers)} papers as seen for topic '{topic}'")
    except Exception as e:
        logger.warning(f"seen_papers: error marking papers: {e}")

def get_seen_count() -> int:
    """Berapa total paper yang sudah pernah diproses."""
    try:
        conn = _get_conn()
        count = conn.execute("SELECT COUNT(*) FROM seen_papers").fetchone()[0]
        conn.close()
        return count
    except:
        return 0