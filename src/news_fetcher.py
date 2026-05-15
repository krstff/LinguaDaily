#!/usr/bin/env python3
"""
RSS-based news fetcher for language learning.

Pulls articles from curated RSS feeds, maps topics to relevant feed categories,
and returns clean readable text suitable for translation practice.

Usage:
    from src.news_fetcher import NewsFetcher
    fetcher = NewsFetcher()
    title, text = fetcher.fetch_by_topic("Technology")
"""

import random
import re
import logging

import feedparser
import requests
from bs4 import BeautifulSoup

from config import (
    DEFAULT_NATIVE_LANGUAGE,
    NEWS_FEED_CATALOGUE,
    NEWS_DEFAULT_FEEDS,
)

logger = logging.getLogger(__name__)


def load_feeds_from_config(config=None):
    """
    Load RSS feed catalogue from config.json.

    Returns a language-keyed dict: { lang_code: { topic: [urls] } }.

    Supports two formats:

    New (language-keyed):
        {
          "sources": {
            "news": {
              "feeds": {
                "en": { "Technology": ["url1"] },
                "es": { "Tecnología": ["url2"] }
              }
            }
          }
        }

    Legacy (flat — auto-wrapped under "en"):
        {
          "sources": {
            "news": {
              "feeds": { "Technology": ["url1", "url2"], ... }
            }
          }
        }
    """
    if config:
        feeds = (config.get("sources", {}) or {}).get("news", {}) or {}
        cfg_feeds = feeds.get("feeds")
        if cfg_feeds and isinstance(cfg_feeds, dict):
            # Detect format: if values are dicts of topic→urls, it's language-keyed.
            # If values are lists of URLs, it's legacy flat format.
            has_language_keys = any(
                isinstance(v, dict)
                for v in cfg_feeds.values()
            )
            if has_language_keys:
                return cfg_feeds
            else:
                # Legacy flat format — wrap under "en"
                return {"en": cfg_feeds}
    return NEWS_FEED_CATALOGUE


class NewsFetcher:
    """Fetch news articles from RSS feeds."""

    def __init__(self, feeds=None, config=None,
                 learning_language=None):
        """
        Parameters
        ----------
        feeds : dict, optional
            Override the feed catalogue. Maps topic → [feed URLs].
        config : dict, optional
            Full config.json contents. Used to load feeds from
            sources.news.feeds if `feeds` is not provided directly.
        learning_language : str, optional
            Language code (e.g. "es", "de") to resolve language-specific
            RSS feeds. Falls back to English feeds if unavailable.
        """
        if feeds is not None:
            self._feeds = feeds
        elif config:
            self._feeds = load_feeds_from_config(config)
        else:
            self._feeds = NEWS_FEED_CATALOGUE
        self._learning_language = learning_language or DEFAULT_NATIVE_LANGUAGE
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "LinguaDaily/1.0 (language-learning)"
        })

    # ── Topic → feeds ──────────────────────────────────────────────

    def _resolve_feeds(self, topic):
        """Get the list of RSS feed URLs for a given topic.

        Resolution order:
          1. Feeds for the learning_language + exact topic match
          2. Feeds for the learning_language + case-insensitive topic match
          3. Same two steps falling back to English feeds
          4. NEWS_DEFAULT_FEEDS (general BBC news)
        """
        lang = self._learning_language.lower()

        # Try target language first
        lang_feeds = self._feeds.get(lang, {})
        found = self._find_topic_in_catalogue(topic, lang_feeds)
        if found:
            return list(found)

        # Fall back to English feeds
        en_feeds = self._feeds.get("en", {})
        found = self._find_topic_in_catalogue(topic, en_feeds)
        if found:
            logger.info(
                "No '%s' feeds for topic '%s', falling back to English.",
                lang, topic,
            )
            return list(found)

        logger.warning("No feeds for topic '%s', using defaults.", topic)
        return list(NEWS_DEFAULT_FEEDS)

    @staticmethod
    def _find_topic_in_catalogue(topic, catalogue):
        """Find a topic in a single-language feed catalogue.

        Returns the list of URLs or None.
        First tries exact match, then case-insensitive.
        """
        if topic in catalogue:
            return catalogue[topic]
        topic_lower = topic.lower()
        for key, urls in catalogue.items():
            if key.lower() == topic_lower:
                return urls
        return None

    # ── RSS fetching ───────────────────────────────────────────────

    def _fetch_feed(self, url, max_items=30):
        """Parse an RSS feed and return a list of article dicts."""
        try:
            response = self._session.get(url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return []

        feed = feedparser.parse(response.text)

        articles = []
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "").strip()
            # Prefer content:encoded (full HTML), fall back to description
            body_html = ""
            for el in entry.get("content", []):
                if el.get("type") == "html":
                    body_html = el.get("value", "")
                    break
            if not body_html:
                body_html = entry.get("description", "")

            link = entry.get("link", "")
            published = entry.get("published", "")

            articles.append({
                "title": title,
                "body_html": body_html,
                "link": link,
                "published": published,
            })

        return articles

    def _fetch_all_for_topic(self, topic, max_items_per_feed=30):
        """Fetch articles from all feeds relevant to a topic."""
        urls = self._resolve_feeds(topic)
        all_articles = []
        for url in urls:
            items = self._fetch_feed(url, max_items=max_items_per_feed)
            all_articles.extend(items)
        return all_articles

    # ── Text extraction & filtering ────────────────────────────────

    @staticmethod
    def _html_to_text(html):
        """Strip HTML tags and clean up whitespace."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = text.split("\n")
        cleaned = []
        prev_blank = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                prev_blank = True
            else:
                if prev_blank:
                    cleaned.append("")
                cleaned.append(stripped)
                prev_blank = False
        return "\n".join(cleaned)

    def _filter_article(self, article, min_words=250, max_words=600):
        """Check if an article meets length requirements. Returns (title, text) or None."""
        text = self._html_to_text(article["body_html"])
        word_count = len(text.split())

        if word_count < min_words:
            return None

        # If within range, keep as-is
        if word_count <= max_words:
            return article["title"], text

        # Truncate to paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        accumulated = []
        total = 0
        for para in paragraphs:
            para_words = len(para.split())
            if total + para_words > max_words:
                break
            accumulated.append(para)
            total += para_words

        result = "\n\n".join(accumulated).strip()
        if len(result.split()) >= min_words:
            return article["title"], result

        # Fallback: hard word-level truncation (for text with no paragraph breaks)
        words = text.split()
        truncated = " ".join(words[:max_words])
        if len(truncated.split()) >= min_words:
            return article["title"], truncated
        return None

    # ── Topic selection helpers ────────────────────────────────────

    def pick_random_topic(self):
        """Pick a random topic available for the learning language.

        Tries the learning_language first, falls back to English.
        Returns a topic string or None if no topics are available.
        """
        lang = self._learning_language.lower()

        # Try target language first
        lang_feeds = self._feeds.get(lang, {})
        if lang_feeds:
            logger.info("Picking random topic from '%s' feeds", lang)
            return random.choice(list(lang_feeds.keys()))

        # Fall back to English
        en_feeds = self._feeds.get("en", {})
        if en_feeds:
            logger.info(
                "No topics for '%s', picking random topic from English feeds",
                lang,
            )
            return random.choice(list(en_feeds.keys()))

        logger.warning("No feed catalogue available — cannot pick a topic")
        return None

    # ── Public API ─────────────────────────────────────────────────

    def fetch_by_topic(self, topic, min_words=250, max_words=600):
        """
        Fetch a random news article matching the given topic.

        Returns (title, text) or (None, None).
        """
        articles = self._fetch_all_for_topic(topic)
        if not articles:
            logger.error("No articles fetched for topic '%s'.", topic)
            return None, None

        # Filter and collect eligible articles
        eligible = []
        for article in articles:
            result = self._filter_article(article, min_words=min_words, max_words=max_words)
            if result:
                eligible.append(result)

        if not eligible:
            logger.warning(
                "No articles met length requirements (%d-%d words) for topic '%s'. "
                "Got %d raw articles.",
                min_words, max_words, topic, len(articles),
            )
            return None, None

        return random.choice(eligible)

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── CLI entry point ────────────────────────────────────────────────

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="RSS news fetcher")
    parser.add_argument("topic", nargs="?", default=None, help="Topic to fetch articles for")
    parser.add_argument("--min-words", type=int, default=250)
    parser.add_argument("--max-words", type=int, default=600)
    args = parser.parse_args()

    with NewsFetcher() as fetcher:
        title, text = fetcher.fetch_by_topic(
            args.topic or "Technology",
            min_words=args.min_words,
            max_words=args.max_words,
        )

    result = {
        "title": title or "Unknown",
        "text": text or "",
        "source": "news",
        "word_count": len(text.split()) if text else 0,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
