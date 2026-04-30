import sys
import os
import json
import subprocess

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")

KIWIX_URL = "http://192.168.100.52:8080"
ZIM_NAME = "wikipedia_en_all_maxi_2026-02"


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def fetch_random_wiki_article(topic=None):
    """
    Fetch a random Wikipedia article from the local Kiwix server.
    Uses src/wikipedia_fetcher.py as a subprocess.
    The fetcher handles length filtering via config.json article_filter settings.
    
    If topic is given, search for articles matching that topic.
    Returns (title, text) or (None, None) on failure.
    """
    fetcher_path = os.path.join(SCRIPT_DIR, "wikipedia_fetcher.py")
    cmd = [sys.executable, fetcher_path]
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
    """
    print("--- OpenClaw-Lingua: Task Execution Started ---")
    
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Topic from CLI arg or config
    topic = sys.argv[1] if len(sys.argv) > 1 else None
    if not topic and config.get("topics"):
        import random
        topic = random.choice(config["topics"])
    
    print(f"Target Topic: {topic}")
    print(f"Languages: {config['source_lang']} -> {config['target_lang_name']}")
    
    # Step 1: Fetch a real Wikipedia article
    print("\nFetching random Wikipedia article from Kiwix...")
    title, content = fetch_random_wiki_article(topic)
    
    if not content:
        print("WARNING: Could not fetch article, using fallback.")
        content = f"A Wikipedia article about {topic} could not be retrieved from the local server."
    
    print(f"Fetched: {title} ({len(content.split())} words)")
    
    # Step 2: Output structured payload for the Agent/Processor
    print("\n---PAYLOAD_START---")
    payload = {
        "title": title,
        "content": content,
        "topic": topic,
        "source_lang": config["source_lang"],
        "target_lang_name": config["target_lang_name"],
        "vocab_path": config.get("vocab_path", "data/vocabulary.md"),
    }
    print(json.dumps(payload, ensure_ascii=False))
    print("---PAYLOAD_END---")
    
    print("--- Task Execution Complete ---")

if __name__ == "__main__":
    main()
