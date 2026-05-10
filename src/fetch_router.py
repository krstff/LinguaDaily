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
import os


def fetch_article(source, topic, config, content_lang=None, article_filter=None):
    """
    Fetch an article from the given content source.

    Parameters
    ----------
    source : str
        Content source identifier (e.g. "wikipedia", "news").
    topic : str or None
        Topic string — used only for news RSS feeds; ignored for wikipedia
        which uses the /random endpoint.
    config : dict
        Full config.json contents.
    content_lang : str or None
        Language code for the desired content (used to pick the right
        Kiwix server when source is wikipedia).
    article_filter : dict or None
        Per-profile article filter overrides ({min_words, max_words}).
        Falls back to top-level config's article_filter, then built-in defaults.

    Returns
    -------
    (title, text) or (None, None) on failure.
    """
    if source == "wikipedia":
        return _fetch_wikipedia(config, content_lang=content_lang,
                                article_filter=article_filter)
    elif source == "news":
        return _fetch_news(topic, config, article_filter=article_filter)
    else:
        print(f"Warning: unknown source '{source}', falling back to wikipedia.")
        return _fetch_wikipedia(config, content_lang=content_lang,
                                article_filter=article_filter)


# ── Wikipedia (Kiwix) ───────────────────────────────────────────────

def _fetch_wikipedia(config, content_lang=None, article_filter=None):
    """Fetch a random Wikipedia article via Kiwix (direct import).

    Parameters
    ----------
    config : dict
        Full config.json contents.
    content_lang : str or None
        Language code of the desired content (e.g. "de", "en").
        If given, resolves Kiwix server from kiwix_servers[content_lang].
    article_filter : dict or None
        Per-profile article filter overrides ({min_words, max_words}).
    """
    from wikipedia_fetcher import KiwixClient, load_fetcher_config

    settings = load_fetcher_config(content_lang=content_lang)

    base_url = settings["base_url"]
    zim_name = settings["zim_name"]
    af = settings["article_filter"]

    # Profile-level overrides take precedence over config defaults
    if article_filter:
        af.update(article_filter)

    min_words = af.get("min_words", 250)
    max_words = af.get("max_words", 600)

    try:
        with KiwixClient(base_url=base_url, zim_name=zim_name) as client:
            return client.get_random_article(
                min_words=min_words,
                max_words=max_words,
            )
    except Exception as e:
        print(f"Wikipedia fetcher error: {e}")
        return None, None


# ── News (RSS) ──────────────────────────────────────────────────────

def _fetch_news(topic, config, article_filter=None):
    """Fetch via RSS-based news fetcher."""
    from news_fetcher import NewsFetcher

    sources_cfg = config.get("sources", {})
    news_cfg = sources_cfg.get("news", {})

    # Article filter: profile-level > global config > defaults
    af = article_filter or config.get("article_filter", {})
    min_words = af.get("min_words", 250)
    max_words = af.get("max_words", 600)

    fetcher = NewsFetcher(
        feeds=news_cfg.get("feeds"),
        categories=news_cfg.get("categories", {}),
    )
    return fetcher.fetch_by_topic(
        topic,
        min_words=min_words,
        max_words=max_words,
    )


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Content-fetch router")
    parser.add_argument("--config", "-c", default=None, help="Path to config.json")
    parser.add_argument("--source", "-s", default="wikipedia", help="Content source (wikipedia, news)")
    parser.add_argument("topic", nargs="?", default=None,
                        help="Topic (used only for news source; ignored for wikipedia)")
    args = parser.parse_args()

    # Load config
    from config import load_config

    config = load_config(args.config)

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
