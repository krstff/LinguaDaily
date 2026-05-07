import sys
import os
import json
import argparse

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


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


def fetch_article(source="wikipedia", topic=None, config=None, content_lang=None):
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

    return route_fetch(source, topic, config, content_lang=content_lang)


def main():
    """
    Called by the OpenClaw Agent to fetch content and prepare it for translation.

    Usage:
        python3 src/orchestrator.py                      # default profile, random topic
        python3 src/orchestrator.py --profile krystof    # specific profile
        python3 src/orchestrator.py --profile anna "quantum physics"  # profile + topic
    """
    parser = argparse.ArgumentParser(description="OpenClaw-Lingua orchestrator")
    parser.add_argument("--profile", "-p", help="User profile name (default: config's default_profile)")
    parser.add_argument("topic", nargs="?", default=None, help="Topic to search for")
    args = parser.parse_args()

    print("--- OpenClaw-Lingua: Task Execution Started ---")

    try:
        config = load_config()
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
    title, content = fetch_article(source=source, topic=topic, config=config, content_lang=content_lang)

    if not content:
        print("WARNING: Could not fetch article, using fallback.")
        content = f"A Wikipedia article about {topic} could not be retrieved from the local server."

    print(f"Fetched: {title} ({len(content.split())} words)")

    # Step 1.5: Generate TTS audio of the fetched content
    wav_path = None
    if config.get("tts"):
        print(f"\nGenerating TTS (language: {content_lang})...")
        try:
            from tts import synthesize
            output_dir = os.path.join(PROJECT_DIR, "output", profile_name)
            wav_path = synthesize(
                text=content,
                language_id=content_lang,
                config=config,
                output_dir=output_dir,
            )
            if wav_path:
                print(f"TTS audio: {wav_path}")
            else:
                print("WARNING: TTS generation returned no file.")
        except Exception as e:
            print(f"WARNING: TTS error (lesson will be delivered without audio): {e}")

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
