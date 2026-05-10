#!/usr/bin/env python3
"""
LLM client for local llama.cpp models (OpenAI-compatible API).

Supports translation, vocabulary extraction, and interactive tutoring.
Designed for a default single-model setup with optional per-task model overrides
for future extensibility.

Usage (import):
    from src.llama_client import LlamaClient
    client = LlamaClient(config)
    translated = client.translate(text, source_lang="de", target_lang="en")
    vocab = client.extract_vocab(translated, source_text=text)
    reply = client.tutor_chat(user_id, message, history)

Config structure in config.json:
    {
      "llm": {
        "base_url": "http://localhost:8080/v1",
        "default_model": "gemma-4-26B-language",
        "api_key": "",
        "timeout": 600
      },
      "profiles": {
        "krystof": {
          "llm_model": "other-model",       // optional per-profile override
          "llm_translate_model": "...",    // optional: separate model for translation
          "llm_tutor_model": "..."         // optional: separate model for tutoring
        }
      }
    }
"""

import json
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")


# ── System prompts ───────────────────────────────────────────────────

TRANSLATE_SYSTEM_PROMPT = """You are a professional translator helping a language learner.

Translate the following article from {source_lang} to {target_lang}.
Preserve the original structure (headings, paragraphs).
Keep technical terms accurate and natural-sounding.
Do NOT add commentary, summaries, or notes — only output the translation."""

VOCAB_SYSTEM_PROMPT = """You are a language-learning tutor extracting vocabulary from a translated article.

The user is learning {source_lang}. Extract useful vocabulary words (nouns, verbs, adjectives)
from the ORIGINAL text that a learner should know.

Output ONLY a JSON array with no surrounding text. Each entry:
[
  {{"word": "original_word", "meaning": "brief definition in {target_lang}" }},
  ...
]

Only include words that appear in the original text. Keep meanings concise."""

TUTOR_SYSTEM_PROMPT = """You are a friendly language tutor helping someone learn {language_name}.

Rules:
- The user's native language is {native_lang} and they are learning {language_name}.
- Explain grammar, vocabulary, and usage clearly.
- Use examples in the target language with translations when helpful.
- Be encouraging and patient.
- Keep responses concise unless the user asks for detail."""


# ── Client ───────────────────────────────────────────────────────────

class LlamaClient:
    """Client for local llama.cpp models via OpenAI-compatible API."""

    def __init__(self, config=None, profile_name=None):
        """
        Parameters
        ----------
        config : dict or None
            Full config.json contents. Loaded from default path if None.
        profile_name : str or None
            Profile name to resolve per-profile model overrides.
        """
        if config is None:
            config = self._load_config()

        self.config = config
        self.profile_name = profile_name
        self.llm_cfg = config.get("llm", {})
        self.base_url = self.llm_cfg.get(
            "base_url", os.environ.get("LLAMA_BASE_URL", "http://localhost:8080/v1")
        )
        self.default_model = self.llm_cfg.get(
            "default_model", os.environ.get("LLAMA_MODEL", "gemma-4-26B-language")
        )
        self.api_key = self.llm_cfg.get("api_key", "") or "none"

        # Timeout for LLM requests (model swap can be slow with large models)
        self.timeout = float(self.llm_cfg.get(
            "timeout", os.environ.get("LLAMA_TIMEOUT", "600")
        ))

        # Resolve profile-level overrides
        self.profile = {}
        if profile_name and config.get("profiles"):
            self.profile = config["profiles"].get(profile_name, {})

        self._client = None

    # ── Model resolution ───────────────────────────────────────────

    def resolve_model(self, task: str = "default") -> str:
        """
        Resolve which model to use for a given task.

        Priority (highest first):
          1. Profile-level override (e.g. profile.llm_translate_model)
          2. Profile-level generic override (profile.llm_model)
          3. LLM-level task default (llm.translate_model, llm.tutor_model)
          4. Global default model

        Parameters
        ----------
        task : str
            Task identifier: "translate", "vocab", "tutor", or "default".

        Returns
        -------
        str
            Model name to use.
        """
        # Profile-level task-specific override (e.g. llm_translate_model)
        task_key = f"llm_{task}_model"
        if self.profile and task_key in self.profile:
            return self.profile[task_key]

        # Profile-level generic override
        if self.profile and "llm_model" in self.profile:
            return self.profile["llm_model"]

        # LLM-level task default (future extensibility)
        llm_task = self.llm_cfg.get(f"{task}_model")
        if llm_task:
            return llm_task

        return self.default_model

    # ── OpenAI client ──────────────────────────────────────────────

    def _get_client(self):
        """Lazy-initialize the OpenAI-compatible client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                logger.error("'openai' package not installed — LLM calls will fail.")
                return None

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    # ── Core chat completion ───────────────────────────────────────

    def _chat(self, messages: list, model: str = None, temperature: float = 0.3) -> Optional[str]:
        """
        Send a chat completion request and return the assistant's text reply.

        Parameters
        ----------
        messages : list[dict]
            Standard OpenAI-style messages: [{"role": "system", "content": "..."}, ...]
        model : str or None
            Model name override. Falls back to self.default_model.
        temperature : float
            Sampling temperature (lower = more deterministic, good for translation).

        Returns
        -------
        str or None
            Assistant's reply text, or None on failure.
        """
        client = self._get_client()
        if client is None:
            return None

        if model is None:
            model = self.default_model

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            error_msg = str(e)
            # Show a short message for connection/timeout errors instead of huge tracebacks
            if any(kw in error_msg.lower() for kw in ("connection", "refused", "timeout", "unreachable", "network")):
                logger.error("LLM unreachable (%s)", error_msg[:80])
            else:
                logger.error("LLM chat error: %s", e, exc_info=True)
            return None

    # ── Public API ─────────────────────────────────────────────────

    def translate(self, text: str, source_lang: str = "de", target_lang: str = "en") -> Optional[str]:
        """
        Translate a text from source_lang to target_lang.

        Parameters
        ----------
        text : str
            Text to translate (in source_lang).
        source_lang : str
            Source language code or name (e.g. "de", "German").
        target_lang : str
            Target language code or name (e.g. "en", "English").

        Returns
        -------
        str or None
            Translated text, or None on failure.
        """
        model = self.resolve_model("translate")
        system = TRANSLATE_SYSTEM_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]

        return self._chat(messages, model=model, temperature=0.1)

    def extract_vocab(
        self,
        original_text: str,
        translated_text: Optional[str] = None,
        source_lang: str = "de",
        target_lang: str = "en",
        max_words: int = 30,
    ) -> list:
        """
        Extract useful vocabulary words from the original text.

        Parameters
        ----------
        original_text : str
            The original article text (in source_lang).
        translated_text : str or None
            The translated version (used for context in prompts).
        source_lang : str
            Language of the original text.
        target_lang : str
            User's native language (for definitions).
        max_words : int
            Maximum number of vocabulary words to extract.

        Returns
        -------
        list[dict]
            List of {word, meaning} dicts, or empty list on failure.
        """
        model = self.resolve_model("vocab")
        system = VOCAB_SYSTEM_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
        )

        user_text = f"Original text ({source_lang}):\n{original_text}"
        if translated_text:
            user_text += f"\n\nTranslation ({target_lang}):\n{translated_text}"
        user_text += f"\n\nExtract up to {max_words} useful words."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]

        result = self._chat(messages, model=model, temperature=0.1)
        if not result:
            return []

        # Parse JSON array from response (handle markdown code fences)
        result = result.strip()
        if result.startswith("```"):
            # Strip code fence markers
            lines = result.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            result = "\n".join(lines).strip()

        try:
            vocab = json.loads(result)
            if isinstance(vocab, list):
                return vocab[:max_words]
            return []
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse vocab JSON: %s — raw: %s", e, result[:200])
            return []

    def tutor_chat(
        self,
        message: str,
        language_name: str = "German",
        native_lang: str = "English",
        history: Optional[list] = None,
        max_history: int = 10,
    ) -> Optional[str]:
        """
        Handle an interactive tutoring chat message.

        Parameters
        ----------
        message : str
            The user's message/question.
        language_name : str
            Name of the language being learned (e.g. "German").
        native_lang : str
            User's native language (e.g. "English").
        history : list[dict] or None
            Previous conversation messages for context.
        max_history : int
            Maximum number of history turns to include (keeps token usage low).

        Returns
        -------
        str or None
            Tutor's reply, or None on failure.
        """
        model = self.resolve_model("tutor")
        system = TUTOR_SYSTEM_PROMPT.format(
            language_name=language_name,
            native_lang=native_lang,
        )

        messages = [{"role": "system", "content": system}]

        # Add history (trim to max_history turns = pairs of user/assistant)
        if history:
            keep = history[-max_history * 2:]
            messages.extend(keep)

        messages.append({"role": "user", "content": message})

        return self._chat(messages, model=model, temperature=0.7)

    def health_check(self) -> bool:
        """Check if the LLM endpoint is reachable."""
        try:
            reply = self._chat(
                [{"role": "user", "content": "Reply with exactly 'OK' and nothing else."}],
                temperature=0.0,
            )
            return reply is not None and "OK" in reply
        except Exception as e:
            logger.error("LLM health check failed: %s", e)
            return False

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_config():
        """Load config from default path."""
        config_path = os.path.join(PROJECT_DIR, "config.json")
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load config for LLM client: %s", e)
            return {}


# ── CLI entry point ────────────────────────────────────────────────

def main():
    """CLI for testing LLM operations.

    Usage:
        python3 src/llama_client.py translate "Hallo Welt" --source de --target en
        python3 src/llama_client.py vocab "original text ..." --source de --target en
        python3 src/llama_client.py chat "Wie sagt man hello?" --lang German
        python3 src/llama_client.py health
    """
    import argparse

    parser = argparse.ArgumentParser(description="Llama.cpp LLM client")
    parser.add_argument("command", choices=["translate", "vocab", "chat", "health"],
                        help="Operation to perform")
    parser.add_argument("text", nargs="?", default="", help="Input text (or question for chat)")
    parser.add_argument("--source", "-s", default="de", help="Source language code")
    parser.add_argument("--target", "-t", default="en", help="Target language code")
    parser.add_argument("--lang", default="German", help="Language name (for tutor)")
    parser.add_argument("--native", default="English", help="Native language (for tutor)")
    parser.add_argument("--config", "-c", default=None, help="Path to config.json")
    parser.add_argument("--profile", "-p", default=None, help="Profile name")
    parser.add_argument("--llm-url", default=None, help="Override LLM base_url")
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = LlamaClient._load_config()

    if args.llm_url:
        config.setdefault("llm", {})["base_url"] = args.llm_url

    client = LlamaClient(config=config, profile_name=args.profile)

    if args.command == "health":
        ok = client.health_check()
        status = "✅ LLM endpoint healthy" if ok else "❌ LLM endpoint unreachable"
        print(status)
        sys.exit(0 if ok else 1)

    elif args.command == "translate":
        if not args.text:
            print("Error: provide text to translate", file=sys.stderr)
            sys.exit(1)
        result = client.translate(args.text, source_lang=args.source, target_lang=args.target)
        if result:
            print(result)
        else:
            print("Translation failed", file=sys.stderr)
            sys.exit(1)

    elif args.command == "vocab":
        if not args.text:
            print("Error: provide text to extract vocab from", file=sys.stderr)
            sys.exit(1)
        words = client.extract_vocab(args.text, source_lang=args.source, target_lang=args.target)
        if words:
            for entry in words:
                word = entry.get("word", "?")
                meaning = entry.get("meaning", "(no definition)")
                print(f"  {word} — {meaning}")
        else:
            print("No vocabulary extracted", file=sys.stderr)

    elif args.command == "chat":
        if not args.text:
            print("Error: provide a message for the tutor", file=sys.stderr)
            sys.exit(1)
        result = client.tutor_chat(
            args.text,
            language_name=args.lang,
            native_lang=args.native,
        )
        if result:
            print(result)
        else:
            print("Chat failed", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
