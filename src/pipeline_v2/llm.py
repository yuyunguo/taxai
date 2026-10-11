"""Deterministic, resumable Ollama client.

Every call uses temperature 0 and a fixed seed, and every prompt/response
pair is appended to a JSONL cache keyed by (model, task, method, hadm_id), so
interrupted runs resume without repeating calls and every LLM output in the
study is auditable.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ollama import Client

from .config import LLM

logger = logging.getLogger(__name__)
Key = Tuple[str, str, str, int]


class CachedLLM:
    """Ollama chat wrapper with an append-only JSONL cache."""

    def __init__(self, cache_path: Path, max_retries: int = 3, host: Optional[str] = None,
                 options: Optional[Dict[str, float]] = None) -> None:
        self.client = Client(host=host or LLM.host)
        self.options = {**LLM.options, **(options or {})}
        self.cache_path = cache_path
        self.max_retries = max_retries
        self.cache: Dict[Key, str] = {}
        if cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self.cache[(rec["model"], rec["task"], rec["method"], rec["hadm_id"])] = rec["response"]
            logger.info(f"Loaded {len(self.cache):,} cached responses from {cache_path}")

    def ask(self, model: str, task: str, method: str, hadm_id: int, prompt: str) -> str:
        key = (model, task, method, int(hadm_id))
        if key in self.cache:
            return self.cache[key]
        response: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                out = self.client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                                       options=dict(self.options))
                response = out["message"]["content"]
                break
            except Exception as e:  # network / server errors from ollama
                logger.warning(f"{key} attempt {attempt} failed: {e}")
                time.sleep(5 * attempt)
        if response is None:
            raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {key}")
        self.cache[key] = response
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"model": model, "task": task, "method": method,
                                "hadm_id": int(hadm_id), "prompt": prompt,
                                "response": response}) + "\n")
        return response


    def ask_chat(self, model: str, task: str, method: str, hadm_id: int,
                 messages: List[Dict[str, str]]) -> str:
        """Multi-turn variant of ask(); the full message history is part of the logged prompt."""
        key = (model, task, method, int(hadm_id))
        if key in self.cache:
            return self.cache[key]
        response: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                out = self.client.chat(model=model, messages=messages, options=dict(self.options))
                response = out["message"]["content"]
                break
            except Exception as e:  # network / server errors from ollama
                logger.warning(f"{key} attempt {attempt} failed: {e}")
                time.sleep(5 * attempt)
        if response is None:
            raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {key}")
        self.cache[key] = response
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"model": model, "task": task, "method": method,
                                "hadm_id": int(hadm_id), "prompt": json.dumps(messages),
                                "response": response}) + "\n")
        return response
