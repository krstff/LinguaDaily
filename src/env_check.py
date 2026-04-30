import os
import shutil
import subprocess

def check_env():
    print("--- 🛠️ OpenClaw-Lingua: Environment Health Check ---")
    errors = []
    
    # 1. Check Project Root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
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

    # 3. Check OpenClaw CLI availability
    # We try to run 'openclaw --help' or similar to see if it's reachable
    try:
        # Using a generic check as we don't know the exact command name on your host
        # but we'll try 'openclaw' based on your previous context
        subprocess.run(["openclaw", "--help"], capture_output=True, timeout=5)
        print("✅ OpenClaw CLI: Reachable")
    except FileNotFoundError:
        errors.append("❌ OpenClaw CLI not found in PATH (Is it installed on the host?)")
    except subprocess.TimeoutExpired:
        errors.append("⚠️ OpenClaw CLI: Found but timed out (Is the service overloaded?)")
    except Exception as e:
        errors.append(f"❌ OpenClaw CLI Error: {str(e)}")

    # 4. Check Write Permissions for Data
    vocab_path = os.path.join(project_root, "data/vocabulary.md")
    if os.path.exists(vocab_path):
        if os.access(vocab_path, os.W_OK):
            print(f"✅ Vocabulary Write Access: OK ({vocab_path})")
        else:
            errors.append(f"❌ Vocabulary Write Access: DENIED ({vocab_path})")
    else:
        errors.append(f"❌ Vocabulary file missing at {vocab_path}")

    # 5. Check Dependencies (the 'src' folder)
    src_path = os.path.join(project_root, "src", "orchestrator.py")
    if os.path.exists(src_path):
        print(f"✅ Orchestrator script: Found")
    else:
        errors.append(f"❌ Orchestrator script missing at {src_path}")

    print("\n--- Summary ---")
    if not errors:
        print("🚀 Environment looks HEALTHY. Ready for deployment!")
    else:
        print(f"🚨 Environment is UNHEALTHY. Found {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        print("\nAction Required: Please fix the errors above.")

if __name__ == "__main__":
    check_env()
