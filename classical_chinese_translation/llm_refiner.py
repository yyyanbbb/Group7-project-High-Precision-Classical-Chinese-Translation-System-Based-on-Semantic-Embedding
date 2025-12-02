#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local LLM refinement utilities.

This module optionally leverages a locally deployed large language model
to polish retrieval-based translations and to provide an additional
confidence signal. It never triggers a fresh download — the model must
already exist on disk (set via environment variables).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None


PROJECT_DIR = Path(__file__).resolve().parent


def _default_model_candidates() -> List[str]:
    """
    Build a prioritized list of local model directories.

    Order:
        1. LOCAL_LLM_MODEL_PATH environment variable
        2. LLM_MODEL_PATH environment variable (fallback name)
        3. ./local_models/qwen2_5
        4. ./local_models/qwen2_7b
    """
    env_primary = os.environ.get("LOCAL_LLM_MODEL_PATH")
    env_secondary = os.environ.get("LLM_MODEL_PATH")
    local_models_dir = PROJECT_DIR / "local_models"
    return [
        env_primary,
        env_secondary,
        str(local_models_dir / "qwen2_5"),
        str(local_models_dir / "qwen2_7b"),
    ]


@dataclass
class RefinementResult:
    """Structured response from the local LLM."""

    translation: str
    confidence: float
    notes: str = ""


class LocalLLMRefiner:
    """
    Optional helper that asks a locally hosted instruction model
    to refine translations for better fluency and fidelity.
    """

    PROMPT = (
        "你是一名专业的古文翻译专家。请根据提供的原文与若干候选译文，"
        "在确保忠实原意的前提下输出最优现代汉语翻译。允许对候选结果进行"
        "润色或综合，但不得臆造无依据内容。\n\n"
        "【原文】\n{original}\n\n"
        "【候选译文】\n{candidates}\n\n"
        "请严格输出 JSON，字段为 translation（字符串）、confidence（0-1 之间的小数）、notes（简短中文说明）。"
    )

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.15,
    ):
        self._candidates = model_paths or _default_model_candidates()
        self.model_path = self._resolve_model_path(self._candidates)
        self.available = bool(
            self.model_path and AutoTokenizer is not None and AutoModelForCausalLM is not None
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer = None
        self._model = None

    @staticmethod
    def _resolve_model_path(candidates: List[Optional[str]]) -> Optional[str]:
        """Return the first existing directory from candidates."""
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def _load_model(self):
        """Lazy-load tokenizer and model."""
        if not self.available:
            return
        if self._model is not None:
            return

        device_map = "auto"
        torch_dtype = torch.float16 if torch and torch.cuda.is_available() else None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self._model.eval()

    def _format_candidates(self, candidates: List[Dict[str, str]]) -> str:
        """Format candidate list for the prompt."""
        rows = []
        for idx, cand in enumerate(candidates, start=1):
            title = cand.get("title") or "未知来源"
            classical = cand.get("classical", "")
            modern = cand.get("modern", "")
            rows.append(
                f"{idx}. 《{title}》\n原句：{classical}\n译文：{modern}"
            )
        return "\n\n".join(rows)

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract JSON substring from raw LLM output."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return None

    def refine_translation(
        self,
        original: str,
        primary_candidate: Dict[str, str],
        support_candidates: List[Dict[str, str]],
    ) -> Optional[RefinementResult]:
        """
        Ask the local LLM to polish translations.

        Args:
            original: Classical Chinese sentence provided by user.
            primary_candidate: Top retrieval-based candidate.
            support_candidates: Additional candidates for context.
        """
        if not self.available:
            return None

        self._load_model()
        if self._model is None or self._tokenizer is None:
            return None

        candidates = [primary_candidate] + support_candidates
        prompt = self.PROMPT.format(
            original=original.strip(),
            candidates=self._format_candidates(candidates),
        )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if torch and torch.cuda.is_available():
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=False,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = self._tokenizer.decode(generated, skip_special_tokens=True)

        json_text = self._extract_json(raw_text) or raw_text.strip()
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return None

        translation = payload.get("translation") or primary_candidate.get("modern", "")
        confidence = float(payload.get("confidence", 0.0))
        notes = payload.get("notes", "").strip()

        confidence = max(0.0, min(1.0, confidence))

        return RefinementResult(
            translation=translation.strip(),
            confidence=confidence,
            notes=notes,
        )

    REWRITE_PROMPT = (
        "你是一名专业的古文翻译专家。现有一段古代汉语原文以及若干参考译文。"
        "请在理解原意的基础上，按照指定的风格生成新的现代汉语译文。"
        "禁止直接复制候选译文的句子，必须用全新的措辞表达。\n\n"
        "【原文】\n{original}\n\n"
        "【候选译文】\n{candidates}\n\n"
        "需要生成的风格：\n{styles}\n\n"
        "输出 JSON 数组，每个元素包含 style（风格名称）、translation（译文）和 notes（简短说明）。"
    )

    def rewrite_with_styles(
        self,
        original: str,
        candidates: List[Dict[str, str]],
        styles: List[str],
    ) -> List[Dict[str, str]]:
        """Generate rewritten translations for different styles."""
        if not self.available or not styles:
            return []

        self._load_model()
        if self._model is None or self._tokenizer is None:
            return []

        prompt = self.REWRITE_PROMPT.format(
            original=original.strip(),
            candidates=self._format_candidates(candidates),
            styles="\n".join(f"- {s}" for s in styles),
        )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if torch and torch.cuda.is_available():
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=False,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = self._tokenizer.decode(generated, skip_special_tokens=True)

        json_text = self._extract_json(raw_text) or raw_text.strip()
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return []

        rewrites: List[Dict[str, str]] = []
        if isinstance(payload, list):
            for item in payload:
                style = str(item.get("style", "")).strip()
                translation = str(item.get("translation", "")).strip()
                notes = str(item.get("notes", "")).strip()
                if style and translation:
                    rewrites.append(
                        {
                            "style": style,
                            "translation": translation,
                            "notes": notes,
                        }
                    )
        return rewrites


__all__ = ["LocalLLMRefiner", "RefinementResult"]


