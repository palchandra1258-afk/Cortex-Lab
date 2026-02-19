"""
Local LLM Interface for Cortex Lab
Interfaces with the DeepSeek-R1-1.5B model loaded in server.py.
Used by agents and retrieval pipeline for all LLM calls.
"""

import time
from typing import Optional
import torch


class LocalLLM:
    """
    Interface to the loaded DeepSeek-R1-1.5B model.
    Provides structured LLM calls for:
    - Classification (intent, emotion, memory type)
    - Query transformation (multi-query, HyDE, step-back)
    - Generation (answers, summaries, propositions)
    - Self-reflection (critique, grading)
    
    Designed to minimize token usage and latency.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._call_count = 0
        self._total_tokens = 0

    def set_model(self, model, tokenizer):
        """Set model reference (called after model loads in server.py)."""
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.3, top_p: float = 0.9,
                 stop_sequences: Optional[list] = None) -> str:
        """Generate text from the model."""
        if self.model is None or self.tokenizer is None:
            return self._fallback_generate(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        # Strip thinking tags if present (we want clean output for internal calls)
        if "<think>" in generated:
            think_end = generated.find("</think>")
            if think_end > -1:
                generated = generated[think_end + len("</think>"):].strip()

        # Apply stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in generated:
                    generated = generated[:generated.index(seq)]

        self._call_count += 1
        self._total_tokens += input_len + len(self.tokenizer.encode(generated))

        return generated

    def classify(self, prompt: str, options: list, default: str = "") -> str:
        """Quick classification with constrained output."""
        full_prompt = f"{prompt}\n\nChoose EXACTLY ONE from: {', '.join(options)}\nAnswer:"
        result = self.generate(full_prompt, max_tokens=20, temperature=0.1)
        result_lower = result.strip().lower()

        # Match to closest option
        for opt in options:
            if opt.lower() in result_lower:
                return opt
        return default or options[0]

    def extract_json(self, prompt: str, max_tokens: int = 256) -> dict:
        """Generate and parse JSON output."""
        result = self.generate(prompt + "\n\nOutput valid JSON only:", max_tokens=max_tokens, temperature=0.1)
        # Try to extract JSON from response
        import json
        try:
            # Find JSON in response
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except json.JSONDecodeError:
            pass
        return {}

    def summarize(self, text: str, max_length: int = 100) -> str:
        """Generate a concise summary."""
        prompt = f"""Summarize the following text in {max_length} words or less. 
Be concise and preserve key information.

Text: {text}

Summary:"""
        return self.generate(prompt, max_tokens=max_length * 2, temperature=0.2)

    def _fallback_generate(self, prompt: str) -> str:
        """Fallback when model is not loaded."""
        return "[Model not loaded - cannot generate response]"

    def get_stats(self) -> dict:
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "model_loaded": self.model is not None,
        }

    def reset_stats(self):
        self._call_count = 0
        self._total_tokens = 0
