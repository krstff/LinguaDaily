#!/usr/bin/env python3
"""
Content-fetch router — dispatches to the right fetcher based on a profile's `source` field.

Usage (import):
    from src.fetch_router import fetch_article
    title, text = fetch_article(source="wikipedia", topic="Physics", config=config)

Usage (CLI):
    python3 src/fetch_router.py --config config.json --source wikipedia "Physics"
"""

import json
import sys
import os


def fetch_article(source, topic, config):
    """
    Fetch an article from the given content source.

    Parameters
    ----------
    source : str
        Content source identifier (e.g. "wikipedia", "news").
    topic : str
        Topic string to search/filter by.
    config : dict
        Full config.json contents.

    Returns
    -------
    (title, text) or (None, None) on failure.
    """
    if source == "wikipedia":
        return _fetch_wikipedia(topic, config)
    elif source == "news":
        return _fetch_news(topic, config)
    else:
        print(f"Warning: unknown source '{source}', falling back to wikipedia.")
        return _fetch_wikipedia(topic, config)


# ── Wikipedia (Kiwix) ───────────────────────────────────────────────

def _fetch_wikipedia(topic, config):
    """Fetch via the existing Kiwix-based Wikipedia fetcher (subprocess)."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    fetcher_path = os.path.join(SCRIPT_DIR, "wikipedia_fetcher.py")
    config_path = os.path.join(SCRIPT_DIR, "..", "config.json")

    cmd = [sys.executable, fetcher_path, "--config", config_path]
    if topic:
        cmd.append(topic)

    try:
        import subprocess
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"Fetcher error: {result.stderr.strip()}")
            return None, None

        data = json.loads(result.stdout)
        if data.get("error"):
            print(f"Fetcher returned error: {data['error']}")
            return None, None

        return data.get("title", "Unknown"), data.get("text", "")
    except subprocess.TimeoutExpired:
        print("Fetcher timed out after 60s")
        return None, None
    except (json.JSONDecodeError, Exception) as e:
        print(f"Fetcher exception: {e}")
        return None, None


# ── News (RSS) ──────────────────────────────────────────────────────

def _fetch_news(topic, config):
    """Fetch via RSS-based news fetcher."""
    from news_fetcher import NewsFetcher

    sources_cfg = config.get("sources", {})
    news_cfg = sources_cfg.get("news", {})

    # Article filter from profile-level or global defaults
    af = config.get("article_filter", {})
    min_words = af.get("min_words", 250)
    target_words = af.get("target_words", 400)
    max_words = af.get("max_words", 600)

    fetcher = NewsFetcher(
        feeds=news_cfg.get("feeds"),
        categories=news_cfg.get("categories", {}),
    )
    return fetcher.fetch_by_topic(
        topic,
        min_words=min_words,
        target_words=target_words,
        max_words=max_words,
    )


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Content-fetch router")
    parser.add_argument("--config", "-c", default=None, help="Path to config.json")
    parser.add_argument("--source", "-s", default="wikipedia", help="Content source (wikipedia, news)")
    parser.add_argument("topic", nargs="?", default=None, help="Topic to search for")
    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = args.config
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "config.json")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    title, text = fetch_article(args.source, args.topic, config)

    result = {
        "title": title or "Unknown",
        "text": text or "",
        "source": args.source,
        "word_count": len(text.split()) if text else 0,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
