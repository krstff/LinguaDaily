#!/usr/bin/env python3
"""
LLM client for OpenAI-compatible models (local llama.cpp or hosted APIs).

Supports translation, vocabulary extraction, and interactive tutoring.
A single "general" model (llm.default_model) handles all chat tasks;
per-profile model overrides remain available as an escape hatch.

Usage (import):
    from src.llama_client import LlamaClient
    client = LlamaClient(config)
    translated = client.translate(text, source_lang="de", target_lang="en")
    vocab = client.extract_vocab(translated, source_text=text)
    reply = client.tutor_chat(user_id, message, history)

Config structure in config.json:
    {
      "llm": {
        "base_url": "http://llama-swap:8080/v1",  # see LLM_DEFAULT_BASE_URL
        "default_model": "see LLM_DEFAULT_MODEL",
        "api_key": "",
        "timeout": 600
      },
      "profiles": {
        "krystof": {
          "llm_model": "other-model"       // optional per-profile override
        }
      }
    }
"""

import asyncio
import json
import logging
import os
import re
import sys
from typing import Optional

from config import (
    DEFAULT_LEARNING_LANGUAGE,
    DEFAULT_NATIVE_LANGUAGE,
    LLM_DEFAULT_BASE_URL,
    LLM_DEFAULT_MODEL,
    LLM_DEFAULT_TIMEOUT,
    PROJECT_DIR,
    load_config,
)

logger = logging.getLogger(__name__)


# ── LaTeX → Unicode cleanup for tutor messages ───────────────────────
#
# LLMs sometimes emit LaTeX math notation (e.g. $\rightarrow$, $\neq$).
# Convert common patterns to plain-text / Unicode equivalents so they
# render correctly in Telegram.

_LATEX_REPLACEMENTS = [
    # arrows
    (r'\$\\rightarrow\$', '→'),
    (r'\$\\leftarrow\$', '←'),
    (r'\$\\Rightarrow\$', '⇒'),
    (r'\$\\Leftarrow\$', '⇐'),
    (r'\$\\Leftrightarrow\$', '⇔'),
    # relations
    (r'\$\\neq\$', '≠'),
    (r'\$\\leq\$', '≤'),
    (r'\$\\geq\$', '≥'),
    (r'\$\\approx\$', '≈'),
    (r'\$\\sim\$', '∼'),
    (r'\$\\in\$', '∈'),
    (r'\$\\notin\$', '∉'),
    (r'\$\\subset\$', '⊂'),
    (r'\$\\supset\$', '⊃'),
    (r'\$\\subseteq\$', '⊆'),
    (r'\$\\supseteq\$', '⊇'),
    (r'\$\\forall\$', '∀'),
    (r'\$\\exists\$', '∃'),
    # operators
    (r'\$\\times\$', '×'),
    (r'\$\\cdot\$', '·'),
    (r'\$\\pm\$', '±'),
    (r'\$\\infty\$', '∞'),
    (r'\$\\sum\$', '∑'),
    (r'\$\\prod\$', '∏'),
    # greek (lowercase)
    (r'\$\\alpha\$', 'α'),
    (r'\$\\beta\$', 'β'),
    (r'\$\\gamma\$', 'γ'),
    (r'\$\\delta\$', 'δ'),
    (r'\$\\epsilon\$', 'ε'),
    (r'\$\\theta\$', 'θ'),
    (r'\$\\lambda\$', 'λ'),
    (r'\$\\mu\$', 'μ'),
    (r'\$\\pi\$', 'π'),
    (r'\$\\sigma\$', 'σ'),
    (r'\$\\phi\$', 'φ'),
    (r'\$\\omega\$', 'ω'),
]


def _clean_latex(text: str) -> str:
    """Replace common LaTeX math notation with Unicode equivalents."""
    for pattern, replacement in _LATEX_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    # Strip remaining standalone $...$ or $$...$$ blocks
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]+\$', '', text)

    return text


# ── System prompts ───────────────────────────────────────────────────

TRANSLATE_SYSTEM_PROMPT = """You are a professional translator helping a language learner.

Translate the following article from {source_lang} to {target_lang}.
Preserve the original structure (headings, paragraphs).
Keep technical terms accurate and natural-sounding.
Do NOT add commentary, summaries, or notes — only output the translation."""

# ── Text simplification prompt ─────────────────────────────────

SIMPLIFY_SYSTEM_PROMPT = """You are a text simplifier for language learners.

Rewrite the following article in {language_name} at CEFR level {level}.
Rules:
- Keep ALL key information, facts, and names from the original.
- Use vocabulary and grammar appropriate for {level} level learners.
- Shorten sentences where possible.
- Keep the same paragraph structure and headings.
- Do NOT add commentary, summaries, or notes — only output the simplified text."""

VOCAB_SYSTEM_PROMPT = """You are a language-learning tutor extracting vocabulary from a translated article.

The user is learning {source_lang} ({source_lang_name}). Extract useful vocabulary words (nouns, verbs, adjectives,
idioms) from the ORIGINAL text that a learner should know.

Output ONLY a JSON array with no surrounding text. Each entry:
[
  {{
    "word": "original_word",
    "meaning": "brief definition in {target_lang_name}",
    "example": "a short example sentence in {source_lang_name} showing the word in context",
    "highlight_source": ["exact_word_forms_as_they_appear_in_original_text"],
    "highlight_target": ["exact_word_forms_as_they_appear_in_translation"]
  }},
  ...
]

Rules:
- "word": the base/citation form of the vocabulary word in {source_lang_name}.
- "meaning": a concise definition/translation written entirely in {target_lang_name}. Never use English if {target_lang_name} is not English.
- "example": a natural example sentence in {source_lang_name}.
- "highlight_source": list ALL exact word forms from the original text that
  should be highlighted (e.g. ["Pferd", "Pferde"] if both singular and plural
  appear). Only include words/phrases that literally appear in the original text.
- "highlight_target": list ALL exact word forms from the translation that
  should be highlighted. These are the {target_lang_name} equivalents that literally
  appear in the translated text.

Keep meanings concise. For highlight lists, be thorough — include every
inflected form (plural, past tense, etc.) that appears in the respective texts."""

# ── Intent classification prompt ─────────────────────────────────────
# Lightweight LLM call to decide whether RAG grounding is needed.
# Returns JSON: {"intent": "chitchat" | "grammar_query" | "vocab_query"}

INTENT_SYSTEM_PROMPT = """You are the front-end filter for a language tutor's knowledge base.
Classify the learner's latest message, then rewrite it as a retrieval query.

Reply with ONLY a JSON object:
{{"intent": "...", "query": "...", "terms": [...], "lemma": "..."}}

"intent" — exactly one of:
  "chitchat"      — casual conversation, greetings, opinions, non-educational
  "grammar_query" — grammar rules, conjugation, syntax, sentence structure, cases, tenses
  "vocab_query"   — word meaning, translation, vocabulary, phrases, idioms, usage

"query" — rules:
  - For grammar_query / vocab_query: write a short SELF-CONTAINED retrieval
    query (max 25 words) that states what the learner is asking about, using
    the key language-specific terms (case names, verb forms, the exact word
    in question).  Resolve pronouns and ellipsis using the conversation
    context below.  Prefer the terms as they appear in textbooks of the
    language being learned.
  - For chitchat: null

"terms" — up to 3 specific words or phrases in the language being learned
  that the question is ABOUT: the word to translate, the verb to conjugate,
  the noun to decline.  Use the exact word forms as the user wrote them.
  Empty list [] when the question is not about a specific word.

"lemma" — the base / dictionary form of the first term
  (e.g. "geht" → "gehen", "Häuser" → "Haus"), or null when there is no term.

Conversation context (may be empty):
{context}

The user is learning {language_name}. Respond in JSON only."""

# ── Tutor system prompt (base) ───────────────────────────────────────

TUTOR_SYSTEM_PROMPT = """You are a language tutor helping someone learn {language_name}.

Rules:
- The user's native language is {native_lang}. Always respond in {native_lang} when
  explaining concepts, grammar rules, or answering questions.
- Use {language_name} only for example sentences, vocabulary words, and short phrases
  that the user is studying. Provide a {native_lang} translation immediately after.
- Be direct and factual — no filler phrases like "Great question!", "That's interesting!",
  "Excellent!", "Good job!", or any other praise/encouragement padding.
- Never start responses with exclamations, praise, or commentary. Jump straight into
  the answer.
- Keep responses concise unless the user explicitly asks for detail.
- NEVER use tables (| column | format). They render poorly in chat.
  Use bullet points instead:
    • **word** — meaning/definition
    • **word2** — another definition
  For comparisons, use paired bullet lines:
    • German: *Das ist gut.*
    • English: *That is good.*"""

# ── RAG-injected block appended to tutor system prompt ───────────────

RAG_REFERENCE_BLOCK = """
You have access to reference material from textbooks, learning resources and
dictionary entries.  Use it to ground your answers.  If the reference material
contradicts your own knowledge, prefer the reference.  Do NOT mention that you
are using references unless the user asks.

Entries marked "=== word ===" are dictionary lookups (translations, parts of
speech, conjugation/declension tables).  When the question is about one of
those words, quote its translations and forms directly.

=== REFERENCE MATERIAL ===
{references}
=== END REFERENCES ==="""

# ── RAG intent label → display name (for logging) ───────────────────
_INTENT_LABELS = {"chitchat": "chat", "grammar_query": "grammar", "vocab_query": "vocab"}


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
            config = load_config(fallback={})

        self.config = config
        self.profile_name = profile_name
        self.llm_cfg = config.get("llm", {})
        self.base_url = self.llm_cfg.get(
            "base_url", os.environ.get("LLAMA_BASE_URL", LLM_DEFAULT_BASE_URL)
        )
        self.default_model = self.llm_cfg.get(
            "default_model", os.environ.get("LLAMA_MODEL", LLM_DEFAULT_MODEL)
        )
        self.api_key = self.llm_cfg.get("api_key", "") or "none"

        # Timeout for LLM requests (model swap can be slow with large models)
        self.timeout = float(self.llm_cfg.get(
            "timeout", os.environ.get("LLAMA_TIMEOUT", str(LLM_DEFAULT_TIMEOUT))
        ))

        # Resolve profile-level overrides
        self.profile = {}
        if profile_name and config.get("profiles"):
            self.profile = config["profiles"].get(profile_name, {})

    # ── Model resolution ───────────────────────────────────────────

    def resolve_model(self, task: str = "default") -> str:
        """
        Resolve which model to use for a given task.

        A single "general" model (llm.default_model) handles all chat
        tasks (translate, simplify, vocab, tutor).  Profile-level
        overrides remain as an escape hatch.

        Priority (highest first):
          1. Profile-level task override (e.g. profile.llm_translate_model)
          2. Profile-level generic override (profile.llm_model)
          3. Global default model (llm.default_model)

        Parameters
        ----------
        task : str
            Task identifier: "translate", "vocab", "tutor", or "default"
            (kept for call-site compatibility; all tasks share the general model).

        Returns
        -------
        str
            Model name to use.
        """
        # Profile-level escape-hatch overrides
        if self.profile:
            task_key = f"llm_{task}_model"
            if task_key in self.profile:
                return self.profile[task_key]
            if "llm_model" in self.profile:
                return self.profile["llm_model"]

        return self.default_model

    # ── OpenAI client (shared per-endpoint cache) ───────────────

    def _get_client(self):
        """Get the shared OpenAI-compatible client for this endpoint.

        All LlamaClient instances with the same endpoint share ONE
        underlying OpenAI client (cached per base_url+api_key+timeout in
        config) to avoid multiple connection pools fighting the server.
        """
        from config import get_openai_client
        return get_openai_client(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

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

    def translate(
        self,
        text: str,
        source_lang: str = DEFAULT_LEARNING_LANGUAGE,
        target_lang: str = DEFAULT_NATIVE_LANGUAGE,
    ) -> Optional[str]:
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

    def simplify_language(
        self,
        text: str,
        language: str = DEFAULT_LEARNING_LANGUAGE,
        level: str = "B1",
    ) -> Optional[str]:
        """
        Simplify a text to a target CEFR reading level.

        Parameters
        ----------
        text : str
            Text to simplify (in `language`).
        language : str
            Language code or name (e.g. "de", "German").
        level : str
            Target CEFR level: A1, A2, B1, B2, C1, C2.

        Returns
        -------
        str or None
            Simplified text, or None on failure.
        """
        from config import resolve_language_name

        model = self.resolve_model("simplify")
        language_name = resolve_language_name(language)
        system = SIMPLIFY_SYSTEM_PROMPT.format(
            language_name=language_name,
            level=level.upper(),
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
        source_lang: str = DEFAULT_LEARNING_LANGUAGE,
        target_lang: str = DEFAULT_NATIVE_LANGUAGE,
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
            Language of the original text (code or name, e.g. "de", "German").
        target_lang : str
            User's native language for definitions (code or name, e.g. "cs", "Czech").
        max_words : int
            Maximum number of vocabulary words to extract.

        Returns
        -------
        list[dict]
            List of {word, meaning} dicts, or empty list on failure.
        """
        from config import resolve_language_name

        model = self.resolve_model("vocab")
        source_lang_name = resolve_language_name(source_lang)
        target_lang_name = resolve_language_name(target_lang)
        system = VOCAB_SYSTEM_PROMPT.format(
            source_lang=source_lang,
            source_lang_name=source_lang_name,
            target_lang=target_lang,
            target_lang_name=target_lang_name,
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

    def _classify_intent(
        self,
        message: str,
        language_name: str = "German",
        history: Optional[list] = None,
    ) -> dict:
        """
        Lightweight intent classifier + retrieval-query rewriter.

        Returns a dict:
          intent — "chitchat" | "grammar_query" | "vocab_query"
          query  — standalone, self-contained query for the knowledge base
                   (pronouns resolved, key terms included); empty string if
                   no rewrite is available (caller falls back to raw message)
          terms  — list of specific words/phrases (max 3) the question is
                   about, for dictionary lookup
          lemma  — base form of the first term (or "")

        Recent conversation history is included so the rewrite can resolve
        anaphora ("what does it mean?" after a word was discussed).
        Falls back to a chitchat result on any error (safe default — no RAG).
        """
        empty = {"intent": "chitchat", "query": "", "terms": [], "lemma": ""}
        model = self.resolve_model("tutor")

        # Build short context from the most recent turns
        context_lines: list = []
        if history:
            for m in history[-4:]:
                role = str(m.get("role", "?"))
                content = str(m.get("content", ""))[:300]
                context_lines.append(f"{role}: {content}")
        context = "\n".join(context_lines) if context_lines else "(no prior conversation)"

        system = INTENT_SYSTEM_PROMPT.format(
            language_name=language_name, context=context,
        )

        result = self._chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            model=model,
            temperature=0.0,
        )
        if not result:
            return empty

        # Parse JSON from response (strip fences if present)
        text = result.strip()
        if text.startswith("`"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("`"))

        try:
            data = json.loads(text)
            intent = data.get("intent", "chitchat")
            if intent in _INTENT_LABELS:
                query = data.get("query")
                query = query.strip() if isinstance(query, str) else ""
                raw_terms = data.get("terms")
                terms = [t.strip() for t in raw_terms
                         if isinstance(t, str) and t.strip()][:3] \
                    if isinstance(raw_terms, list) else []
                lemma = data.get("lemma")
                lemma = lemma.strip() if isinstance(lemma, str) else ""
                return {"intent": intent, "query": query,
                        "terms": terms, "lemma": lemma}
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fuzzy fallback — check for keywords in raw output (no rewrite/terms)
        lower = text.lower()
        if "grammar" in lower:
            return {"intent": "grammar_query", "query": "", "terms": [], "lemma": ""}
        if "vocab" in lower:
            return {"intent": "vocab_query", "query": "", "terms": [], "lemma": ""}
        return empty

    def _fetch_rag_context(
        self,
        message: str,
        language_code: str = "",
        search_query: str = None,
    ) -> list[str]:
        """
        Query the RAG knowledge base and return relevant text chunks.

        ``search_query`` (from the intent rewriter) is embedded when
        provided; otherwise the raw ``message`` is used.  Returns an empty
        list on any failure (graceful degradation).
        """
        try:
            from src.rag_service import get_rag_service
            rag = get_rag_service()
            query = (search_query or "").strip() or message
            hits = rag.query_knowledge_base(
                query_vector=rag.embed_text(query),
                language=language_code,
                top_k=3,
            )
            logger.info(
                "RAG search: %d hits (lang=%s) query=%r",
                len(hits), language_code or "(any)", query[:80],
            )
            return [h["text"] for h in hits]
        except Exception as e:
            logger.warning("RAG query failed (continuing without grounding): %s", e)
            return []

    def tutor_chat(
        self,
        message: str,
        language_name: str = "German",
        native_lang: str = "English",
        history: Optional[list] = None,
        max_history: int = 10,
        lesson: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Handle an interactive tutoring chat message.

        Flow:
          1. Classify intent (chitchat / grammar_query / vocab_query) and
             extract the specific word(s) the question is about
          2. If grammar or vocab:
             a. dictionary lookup for the extracted terms (Wiktionary via
                local Kiwix ZIM or wiktionary.org — see wiktionary_client)
             b. semantic RAG query for document grounding — only when the
                dictionary lookup produced no reference (the dictionary
                entry is the authoritative source; RAG is the fallback)
          3. Inject lesson + references into system prompt
          4. Generate tutor reply

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
        lesson : dict or None
            Latest delivered lesson with keys: title, original_content,
            translated_content, vocab. Injected into the system prompt
            so the tutor can answer questions about the lesson.

        Returns
        -------
        str or None
            Tutor's reply, or None on failure.
        """
        # ── Prepare prompt (intent, references, system prompt) ────
        messages, model = self._prepare_tutor_messages(
            message=message,
            language_name=language_name,
            native_lang=native_lang,
            history=history,
            max_history=max_history,
            lesson=lesson,
        )

        # ── Generate reply ────────────────────────────────────────
        reply = self._chat(messages, model=model, temperature=0.7)
        if reply:
            reply = _clean_latex(reply)
        return reply

    def _prepare_tutor_messages(
        self,
        message: str,
        language_name: str = "German",
        native_lang: str = "English",
        history: Optional[list] = None,
        max_history: int = 10,
        lesson: Optional[dict] = None,
    ) -> tuple[list, str]:
        """Run the pre-generation steps of tutor_chat (intent classification,
        dictionary/RAG reference gathering, system prompt + history assembly).

        Returns (messages, model) ready for a chat completion call.  Kept
        synchronous so both the blocking tutor_chat() and the streaming
        tutor_chat_stream() can share it.
        """
        model = self.resolve_model("tutor")

        # ── Step 1: Classify intent + rewrite retrieval query ─────
        intent_info = self._classify_intent(message, language_name, history=history)
        intent = intent_info["intent"]
        search_query = intent_info["query"]
        terms = intent_info["terms"]
        lemma = intent_info["lemma"]
        logger.info("Tutor intent=%s | msg=%s | query=%r | terms=%s",
                    _INTENT_LABELS.get(intent, intent), message[:60],
                    search_query[:80], terms)

        # ── Step 2: Gather references for educational queries ─────
        references = []
        if intent in ("grammar_query", "vocab_query"):
            # Derive language code from name (e.g. "German" → "de")
            from config import LANGUAGE_NAMES
            lang_code = ""
            for code, name in LANGUAGE_NAMES.items():
                if name.lower() == language_name.lower():
                    lang_code = code
                    break

            # Tier 1: exact dictionary lookup (Wiktionary, Kiwix or online)
            dict_ref_found = False
            if terms:
                try:
                    from src.wiktionary_client import get_dictionary_reference
                    dict_ref = get_dictionary_reference(
                        terms, lemma, lang_code, self.config,
                    )
                    if dict_ref:
                        references.append(dict_ref)
                        dict_ref_found = True
                        logger.info("Dictionary reference injected for %s", terms)
                except Exception as e:
                    logger.warning("Wiktionary lookup failed (continuing): %s", e)

            # Tier 2: semantic RAG over indexed documents — skipped when a
            # dictionary reference was found (no need to fetch Qdrant data)
            if not dict_ref_found:
                references.extend(
                    self._fetch_rag_context(message, lang_code, search_query=search_query)
                )
            logger.info("Total references: %d", len(references))

        # ── Step 3: Build system prompt ───────────────────────────
        system = TUTOR_SYSTEM_PROMPT.format(
            language_name=language_name,
            native_lang=native_lang,
        )

        # Inject RAG references (if any)
        if references:
            ref_text = "\n---\n".join(references[:5])  # cap at 5 chunks
            system += RAG_REFERENCE_BLOCK.format(references=ref_text)
            logger.info(
                "RAG grounding injected (%d chunks, %d chars):\n%s",
                len(references), len(ref_text),
                "\n".join(f"  [{i}] {r[:100]}..." for i, r in enumerate(references[:5])),
            )

        # Inject today's lesson
        if lesson:
            original = (lesson.get("original_content") or "")[:2000]
            translated = (lesson.get("translated_content") or "")[:2000]
            vocab = lesson.get("vocab", [])

            lesson_block = f"""
You have access to the user's most recent daily lesson.  Use it when
answering questions.

=== TODAY'S LESSON ===
Title: {lesson.get('title', '?')}
Delivered: {lesson.get('delivered_at', '?')}

Original article ({language_name}):
{original}

Translation ({native_lang}):
{translated}
"""
            if vocab:
                vocab_lines = []
                for entry in vocab:
                    if isinstance(entry, dict):
                        w = entry.get("word", "")
                        m = entry.get("meaning", "")
                        vocab_lines.append(f"  {w} — {m}")
                    else:
                        vocab_lines.append(f"  {entry}")
                lesson_block += f"\nVocabulary ({len(vocab)} words):\n" + "\n".join(vocab_lines)
            lesson_block += "\n=== END LESSON ==="
            system += lesson_block

        # ── Step 4: Assemble messages ─────────────────────────────
        messages = [{"role": "system", "content": system}]

        if history:
            keep = history[-max_history * 2:]
            messages.extend(keep)

        messages.append({"role": "user", "content": message})

        return messages, model

    async def tutor_chat_stream(
        self,
        message: str,
        language_name: str = "German",
        native_lang: str = "English",
        history: Optional[list] = None,
        max_history: int = 10,
        lesson: Optional[dict] = None,
    ) -> Optional[str]:
        """Streaming variant of tutor_chat() for the Telegram tutor.

        Generates the reply with stream=True via the async OpenAI client.
        If the awaiting task is cancelled (user pressed /stop), the HTTP
        connection is closed, which makes the inference server abort the
        in-flight generation — so model compute actually stops.

        Returns
        -------
        str or None
            Tutor's reply, or None on failure.
        """
        from config import get_async_openai_client

        # Intent classification + RAG are sync LLM/IO calls — run them in a
        # worker thread so the event loop stays responsive.
        messages, model = await asyncio.to_thread(
            self._prepare_tutor_messages,
            message=message,
            language_name=language_name,
            native_lang=native_lang,
            history=history,
            max_history=max_history,
            lesson=lesson,
        )

        client = get_async_openai_client(
            base_url=self.base_url, api_key=self.api_key, timeout=self.timeout
        )
        if client is None:
            return None

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )
            parts: list[str] = []
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        parts.append(delta)
            finally:
                # Cancelling the task raises CancelledError here; closing the
                # stream drops the TCP connection so the server stops
                # generating.  On a normal exit this is a no-op cleanup.
                await stream.close()
        except asyncio.CancelledError:
            logger.info("Tutor stream cancelled for %r", message[:60])
            raise
        except Exception as e:
            logger.error("Tutor stream error: %s", e, exc_info=True)
            return None

        reply = "".join(parts).strip()
        return _clean_latex(reply) if reply else None

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
        config = load_config(args.config)
    else:
        config = load_config(fallback={})

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
