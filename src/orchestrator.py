import sys
import os
import json
import subprocess
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


def fetch_random_wiki_article(topic=None, config=None):
    """
    Fetch a random Wikipedia article from the local Kiwix server.
    Uses src/wikipedia_fetcher.py as a subprocess.
    The fetcher handles length filtering via the profile's article_filter settings.

    If topic is given, search for articles matching that topic.
    Returns (title, text) or (None, None) on failure.
    """
    fetcher_path = os.path.join(SCRIPT_DIR, "wikipedia_fetcher.py")
    cmd = [sys.executable, fetcher_path, "--config", CONFIG_PATH]
    if topic:
        cmd.append(topic)

    try:
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

        title = data.get("title", "Unknown")
        text = data.get("text", "")

        return title, text
    except subprocess.TimeoutExpired:
        print("Fetcher timed out after 60s")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Failed to parse fetcher output: {e}")
        print(f"Raw output: {result.stdout[:500]}")
        return None, None
    except Exception as e:
        print(f"Fetcher exception: {e}")
        return None, None


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

    print(f"Profile: {profile_name}")
    print(f"Target Topic: {topic}")
    print(f"Languages: {profile['source_lang']} -> {profile['target_lang_name']}")

    # Step 1: Fetch a real Wikipedia article
    print("\nFetching random Wikipedia article from Kiwix...")
    title, content = fetch_random_wiki_article(topic, config)

    if not content:
        print("WARNING: Could not fetch article, using fallback.")
        content = f"A Wikipedia article about {topic} could not be retrieved from the local server."

    print(f"Fetched: {title} ({len(content.split())} words)")

    # Step 2: Output structured payload for the Agent/Processor
    vocab_dir = os.path.join(PROJECT_DIR, "data", profile_name)
    vocab_path = os.path.join(vocab_dir, "vocabulary.md")

    print("\n---PAYLOAD_START---")
    payload = {
        "profile": profile_name,
        "title": title,
        "content": content,
        "topic": topic,
        "source_lang": profile["source_lang"],
        "target_lang": profile["target_lang"],
        "target_lang_name": profile["target_lang_name"],
        "vocab_path": vocab_path,
    }
    print(json.dumps(payload, ensure_ascii=False))
    print("---PAYLOAD_END---")

    print("--- Task Execution Complete ---")


if __name__ == "__main__":
    main()
