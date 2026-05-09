import sys
import os
import re
import json
import argparse

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")


def load_config(path=None):
    """Load config from a JSON file. Defaults to config.json in project root."""
    target = path or CONFIG_PATH
    with open(target, 'r') as f:
        return json.load(f)


def clean_content(text):
    """Clean up extracted article text for display and TTS.

    Strips Wikipedia reference markers ([5], [ 1 ], etc.), removes wiki
    footer sections (Siehe auch, Literatur, Weblinks …), fixes missing
    spaces from link-adjacent words, and normalizes whitespace.
    """
    # ── Remove inline citation/reference markers like [5], [ 1 ] ──
    text = re.sub(r'\s*\[\s*\d+\s*\]\s*', '', text)

    # ── Strip Wikipedia footer sections and everything after them ──
    footer_headers = [
        'Siehe auch', 'Literatur', 'Weblinks', 'Quellen',
        'Einzelnachweise', 'Fußnoten', 'Anmerkungen',
        'Commons :',
        # English
        'See also', 'References', 'External links', 'Notes',
        'Bibliography', 'Further reading',
    ]
    for header in footer_headers:
        idx = text.find(header)
        if idx != -1:
            text = text[:idx]
            break

    # ── Fix missing spaces from adjacent wiki links ──
    # lower+Upper ("vergebenund" → "vergeben und")
    text = re.sub(r'([a-zäöüß])([A-ZÄÖÜ])', r'\1 \2', text)
    # punctuation without trailing space ("gestartet,das" → "gestartet, das")
    text = re.sub(r'([,.:;!?)\»])([A-Za-zÄÖÜäöü])', r'\1 \2', text)

    # ── Normalize whitespace ──
    # 1. Collapse 3+ newlines into double newline (paragraph boundary)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 2. Within each paragraph block, join lines broken by wiki HTML links.
    #    Only join if the preceding line ends mid-sentence (no terminal punctuation).
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    for para in paragraphs:
        lines = [l.strip() for l in para.split('\n') if l.strip()]
        if len(lines) <= 1:
            cleaned_paragraphs.append(para.strip())
            continue

        joined_lines = [lines[0]]
        for line in lines[1:]:
            prev = joined_lines[-1]
            # Join if previous line does NOT end with sentence-ending punctuation
            if not re.search(r'[.!?\"\)\»]$' , prev):
                joined_lines[-1] = prev + " " + line
            else:
                joined_lines.append(line)
        cleaned_paragraphs.append(" ".join(joined_lines))

    text = '\n\n'.join(cleaned_paragraphs)

    # 3. Collapse multiple spaces into single spaces.
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    return text


def get_profile(config, profile_name=None):
    """
    Resolve a profile from the config.

    Priority:
      1. profile_name argument
      2. config['default_profile']
      3. first profile in config['profiles']
    """
    if profile_name and profile_name in config["profiles"]:
        return profile_name, config["profiles"][profile_name]

    if profile_name:
        print(f"Warning: profile '{profile_name}' not found, falling back to default.")

    default = config.get("default_profile")
    if default and default in config["profiles"]:
        return default, config["profiles"][default]

    # Last resort: first profile
    profiles = config.get("profiles", {})
    if profiles:
        first = next(iter(profiles))
        return first, profiles[first]

    raise ValueError("No profiles defined in config.json")


def fetch_article(source="wikipedia", topic=None, config=None, content_lang=None, article_filter=None):
    """
    Fetch an article from the given content source via the router.

    Parameters
    ----------
    source : str
        Content source identifier ("wikipedia", "news").
    topic : str or None
        Topic to search/filter by.
    config : dict or None
        Full config.json contents. Loaded from disk if None.
    content_lang : str or None
        Language code for the desired content (used to pick the right
        Kiwix server when source is wikipedia).

    Returns
    -------
    (title, text) or (None, None) on failure.
    """
    if config is None:
        config = load_config()

    # Import router here to keep orchestrator lightweight
    from fetch_router import fetch_article as route_fetch

    return route_fetch(source, topic, config, content_lang=content_lang, article_filter=article_filter)


def main():
    """
    Fetch content, translate, and prepare a lesson. Originally called by an external agent; now used standalone.

    Usage:
        python3 src/orchestrator.py                      # default profile, random topic
        python3 src/orchestrator.py --profile krystof    # specific profile
        python3 src/orchestrator.py --profile anna "quantum physics"  # profile + topic
    """
    parser = argparse.ArgumentParser(description="LinguaDaily orchestrator")
    parser.add_argument("--profile", "-p", help="User profile name (default: config's default_profile)")
    parser.add_argument("--config", "-c", default=CONFIG_PATH,
        help="Path to config file (default: config.json in project root)")
    parser.add_argument("--tts-url", default=None,
        help="Override TTS base_url for local runs (e.g. http://192.168.100.60:8080/v1)")
    parser.add_argument("topic", nargs="?", default=None, help="Topic to search for")
    args = parser.parse_args()

    print("--- LinguaDaily: Task Execution Started ---")

    try:
        config = load_config(args.config)

    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    profile_name, profile = get_profile(config, args.profile)

    # Topic from CLI arg or profile config
    topic = args.topic
    if not topic and profile.get("topics"):
        import random
        topic = random.choice(profile["topics"])

    source = profile.get("source", "wikipedia")
    content_lang = profile.get("content_lang", profile.get("target_lang", "en"))

    print(f"Profile: {profile_name}")
    print(f"Source: {source}")
    print(f"Content Language: {content_lang}")
    print(f"Target Topic: {topic}")
    print(f"Languages: {profile['source_lang']} (native) -> {profile['target_lang_name']} (learning)")

    # Step 1: Fetch an article via the router
    print(f"\nFetching article from {source}...")
    article_filter = profile.get("article_filter", None)
    title, content = fetch_article(
        source=source,
        topic=topic,
        config=config,
        content_lang=content_lang,
        article_filter=article_filter,
    )

    if not content:
        print("WARNING: Could not fetch article, using fallback.")
        content = f"A Wikipedia article about {topic} could not be retrieved from the local server."

    print(f"Fetched: {title} ({len(content.split())} words)")

    # Step 1.2: Clean up formatting (strip wiki references, normalize whitespace)
    content = clean_content(content)

    # Step 1.5: Generate TTS audio of the fetched content
    wav_path = None
    use_tts = profile.get("use_tts", True)  # default True for backwards compat
    if use_tts and config.get("tts"):
        print(f"\nGenerating TTS (language: {content_lang})...")
        try:
            from tts import synthesize
            output_dir = os.path.join(PROJECT_DIR, "output", profile_name)
            if args.tts_url:
                config.setdefault("tts", {})["base_url"] = args.tts_url
            wav_path = synthesize(
                text=content,
                language_id=content_lang,
                config=config,
                output_dir=output_dir,
                voice=profile.get("tts_voice", "male"),
            )
            if wav_path:
                print(f"TTS audio: {wav_path}")
            else:
                print("WARNING: TTS generation returned no file.")
        except Exception as e:
            print(f"WARNING: TTS error (lesson will be delivered without audio): {e}")
    elif not use_tts:
        print(f"\nTTS disabled for profile '{profile_name}' — skipping audio generation.")

    # Step 2: Output structured payload for the Agent/Processor
    vocab_dir = os.path.join(PROJECT_DIR, "data", profile_name)
    vocab_path = os.path.join(vocab_dir, "vocabulary.md")

    print("\n---PAYLOAD_START---")
    payload = {
        "profile": profile_name,
        "source": source,
        "title": title,
        "content": content,
        "topic": topic,
        "content_lang": content_lang,
        "source_lang": profile["source_lang"],
        "target_lang": profile["target_lang"],
        "target_lang_name": profile["target_lang_name"],
        "vocab_path": vocab_path,
        "wav_path": wav_path,
    }
    print(json.dumps(payload, ensure_ascii=False))
    print("---PAYLOAD_END---")

    print("--- Task Execution Complete ---")


if __name__ == "__main__":
    main()
