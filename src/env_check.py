import os
import shutil
import subprocess
import json


def check_env():
    print("--- 🛠️ LinguaDaily: Environment Health Check ---")
    errors = []
    warnings = []

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    config_path = os.path.join(project_root, "config.json")

    # 1. Check Project Root
    if os.path.exists(project_root):
        print(f"✅ Project Root: {project_root}")
    else:
        errors.append(f"❌ Project Root not found at {project_root}")

    # 2. Check Python Availability
    try:
        version = subprocess.check_output(["python3", "--version"], stderr=subprocess.STDOUT).decode().strip()
        print(f"✅ Python: {version}")
    except Exception:
        errors.append("❌ Python3 is not accessible in the current PATH.")

    # 3. Check config.json
    if os.path.exists(config_path):
        print(f"✅ Config: {config_path}")

        # Parse profiles
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            profiles = config.get("profiles", {})
            default = config.get("default_profile", "(none)")
            print(f"✅ Profiles: {', '.join(profiles.keys())} (default: {default})")

            # Check each profile's data directory
            for name in profiles:
                data_dir = os.path.join(project_root, "data", name)
                vocab_file = os.path.join(data_dir, "vocabulary.md")
                if os.path.isdir(data_dir):
                    if os.path.exists(vocab_file):
                        print(f"  ✅ {name}: data dir + vocab file OK")
                    else:
                        warnings.append(f"⚠️  {name}: data dir exists but no vocabulary.md yet")
                else:
                    warnings.append(f"⚠️  {name}: data directory missing at {data_dir}")
        except Exception as e:
            errors.append(f"❌ Config parse error: {e}")
    else:
        errors.append(f"❌ Config file missing at {config_path}")

    # 4. Check Legacy vocab file (should be migrated)
    legacy_vocab = os.path.join(project_root, "data", "vocabulary.md")
    if os.path.exists(legacy_vocab):
        warnings.append(
            f"⚠️  Legacy vocab file found at {legacy_vocab} — "
            f"consider migrating: mkdir -p data/<profile> && mv data/vocabulary.md data/<profile>/"
        )

    # 5. Check Dependencies
    src_path = os.path.join(project_root, "src", "orchestrator.py")
    if os.path.exists(src_path):
        print(f"✅ Orchestrator: Found")
    else:
        errors.append(f"❌ Orchestrator missing at {src_path}")

    fetcher_path = os.path.join(project_root, "src", "wikipedia_fetcher.py")
    if os.path.exists(fetcher_path):
        print(f"✅ Fetcher: Found")
    else:
        errors.append(f"❌ Fetcher missing at {fetcher_path}")

    # 6. Check Kiwix connectivity (if configured)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            kiwix = config.get("kiwix", {})
            if kiwix.get("base_url"):
                import urllib.request
                try:
                    urllib.request.urlopen(
                        f"{kiwix['base_url']}/",
                        timeout=5
                    )
                    print(f"✅ Kiwix: {kiwix['base_url']} reachable")
                except Exception:
                    errors.append(f"❌ Kiwix: {kiwix['base_url']} unreachable")
        except Exception:
            pass

    # Summary
    print("\n--- Summary ---")
    if not errors and not warnings:
        print("🚀 Environment looks HEALTHY. Ready to go!")
    else:
        if errors:
            print(f"🚨 Found {len(errors)} error(s):")
            for err in errors:
                print(f"  {err}")
        if warnings:
            print(f"⚠️  Found {len(warnings)} warning(s):")
            for w in warnings:
                print(f"  {w}")


if __name__ == "__main__":
    check_env()
