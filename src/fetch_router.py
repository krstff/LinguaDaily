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


def fetch_article(source, topic, config,
                  learning_language=None, article_filter=None):
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
    learning_language : str or None
        Language code to fetch articles in (used to pick the right
        Kiwix server when source is wikipedia).
    article_filter : dict or None
        Per-profile article filter overrides ({min_words, max_words}).
        Falls back to top-level config's article_filter, then built-in defaults.

    Returns
    -------
    (title, text) or (None, None) on failure.
    """
    if source == "wikipedia":
        return _fetch_wikipedia(config,
                                learning_language=learning_language,
                                article_filter=article_filter)
    elif source == "news":
        return _fetch_news(topic, config,
                           learning_language=learning_language,
                           article_filter=article_filter)
    else:
        print(f"Warning: unknown source '{source}', falling back to wikipedia.")
        return _fetch_wikipedia(config,
                                learning_language=learning_language,
                                article_filter=article_filter)


# ── Wikipedia (Kiwix) ───────────────────────────────────────────────

def _fetch_wikipedia(config, learning_language=None,
                     article_filter=None):
    """Fetch a random Wikipedia article via Kiwix (direct import).

    Parameters
    ----------
    config : dict
        Full config.json contents.
    learning_language : str or None
        Language code of the desired content (e.g. "de", "en").
        If given, resolves Kiwix server from kiwix_servers[learning_language].
    article_filter : dict or None
        Per-profile article filter overrides ({min_words, max_words}).
    """
    from wikipedia_fetcher import KiwixClient, load_fetcher_config

    settings = load_fetcher_config(learning_language=learning_language)

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

def _fetch_news(topic, config, learning_language=None,
                article_filter=None):
    """Fetch via RSS-based news fetcher.

    If `topic` is None, a random topic is picked from the available feeds
    for the given language (falling back to English).
    """
    from news_fetcher import NewsFetcher

    # Article filter: profile-level > global config > defaults
    af = article_filter or config.get("article_filter", {})
    min_words = af.get("min_words", 250)
    max_words = af.get("max_words", 600)

    fetcher = NewsFetcher(config=config,
                          learning_language=learning_language)

    # Pick a random topic if none was provided
    if topic is None:
        topic = fetcher.pick_random_topic()
        if topic is None:
            print("News fetcher error: no topics available")
            return None, None

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
