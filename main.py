import argparse
import yaml
from loguru import logger
from agent.graph import build_graph
from utils.seen_papers import get_seen_count


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_digest(topics: list[str] = None, ignore_seen: bool = False):
    """Jalankan digest pipeline sekali."""
    config = load_config()

    if topics is None:
        topics = config.get("topics", [])

    if not topics:
        logger.error("No topics found. Check config.yaml")
        return

    if ignore_seen:
        logger.info("--ignore-seen flag active: semua paper akan di-fetch ulang")

    logger.info(f"Starting digest for {len(topics)} topics: {topics}")

    graph = build_graph()

    for topic in topics:
        logger.info(f"Processing topic: {topic}")

        initial_state = {
            "topics": topics,
            "current_topic": topic,
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

        verdict = result["report"].get("critic_verdict")
        papers = result["summaries"].get(topic, [])
        retries = result.get("retry_count", 0)

        logger.info(f"Topic '{topic}' done | papers={len(papers)} verdict={verdict} retries={retries}")

    logger.info(f"Digest complete. Total seen papers in DB: {get_seen_count()}")


def show_status():
    """Tampilkan run history dan stats."""
    from utils.seen_papers import get_seen_count
    import os

    print("\n=== DIGEST AGENT STATUS ===")
    print(f"Seen papers in database: {get_seen_count()}")

    output_dir = "output"
    if os.path.exists(output_dir):
        runs = sorted(os.listdir(output_dir), reverse=True)
        print(f"Past runs ({len(runs)} total):")
        for run in runs[:5]:  # tampilkan 5 terakhir
            run_path = os.path.join(output_dir, run)
            files = os.listdir(run_path)
            print(f"  - {run}: {', '.join(files)}")
    else:
        print("No runs yet.")


def start_scheduler():
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    config = load_config()
    schedule = config.get("schedule", {})

    day = schedule.get("day_of_week", "mon")
    hour = schedule.get("hour", 8)
    minute = schedule.get("minute", 0)
    test_interval = schedule.get("test_interval_minutes", None)

    scheduler = BlockingScheduler()

    if test_interval:
        # Mode test: jalankan setiap N menit
        scheduler.add_job(
            run_digest,
            trigger=IntervalTrigger(minutes=test_interval),
            id="weekly_digest",
            name="Weekly AI/ML Digest (TEST MODE)"
        )
        logger.info(f"TEST MODE: Scheduler running every {test_interval} minutes")
    else:
        # Mode production: jalankan setiap Senin
        scheduler.add_job(
            run_digest,
            trigger=CronTrigger(day_of_week=day, hour=hour, minute=minute),
            id="weekly_digest",
            name="Weekly AI/ML Digest"
        )
        logger.info(f"Scheduler started. Next run: every {day} at {hour:02d}:{minute:02d}")

    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Weekly AI/ML Digest Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run       Run digest now (manual trigger)
  schedule  Start scheduler (runs every Monday 08:00)
  status    Show run history and stats

Examples:
  python main.py run
  python main.py run --topics "RAG" "LLM reasoning"
  python main.py run --ignore-seen
  python main.py schedule
  python main.py status
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_parser = subparsers.add_parser("run", help="Run digest now")
    run_parser.add_argument(
        "--topics",
        nargs="+",
        help="Override topics from config.yaml"
    )
    run_parser.add_argument(
        "--ignore-seen",
        action="store_true",
        help="Ignore seen papers database, re-fetch everything"
    )

    # schedule command
    subparsers.add_parser("schedule", help="Start weekly scheduler")

    # status command
    subparsers.add_parser("status", help="Show run history and stats")

    args = parser.parse_args()

    if args.command == "run":
        run_digest(
            topics=args.topics,
            ignore_seen=args.ignore_seen
        )
    elif args.command == "schedule":
        start_scheduler()
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()