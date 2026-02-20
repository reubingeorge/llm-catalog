"""Hardcoded fallback model data.

# Last verified: 2026-02-20

Used as the last-resort data source when all scraping fails.
"""

import re

from openai_models.models import ModelCapabilities, ModelPricing, OpenAIModel

# Map of model_id -> dict with known attributes
KNOWN_MODELS: dict[str, dict[str, object]] = {
    # GPT-5.2 family
    "gpt-5.2": {
        "name": "GPT-5.2",
        "family": "gpt-5.2",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.75,
            output_price_per_1m=14.00,
            cached_input_price_per_1m=0.175,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5.2-pro": {
        "name": "GPT-5.2 Pro",
        "family": "gpt-5.2",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=21.00,
            output_price_per_1m=168.00,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5.2-chat": {
        "name": "GPT-5.2 Chat",
        "family": "gpt-5.2",
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.75,
            output_price_per_1m=14.00,
            cached_input_price_per_1m=0.175,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5.2-codex": {
        "name": "GPT-5.2 Codex",
        "family": "gpt-5.2",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.75,
            output_price_per_1m=14.00,
            cached_input_price_per_1m=0.175,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    # GPT-5.1 family
    "gpt-5.1": {
        "name": "GPT-5.1",
        "family": "gpt-5.1",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.25,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=0.125,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5.1-codex": {
        "name": "GPT-5.1 Codex",
        "family": "gpt-5.1",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.25,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=0.125,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5.1-codex-mini": {
        "name": "GPT-5.1 Codex Mini",
        "family": "gpt-5.1",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=0.25,
            output_price_per_1m=2.00,
            cached_input_price_per_1m=0.025,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5.1-codex-max": {
        "name": "GPT-5.1 Codex Max",
        "family": "gpt-5.1",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.25,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=0.125,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5.1-chat": {
        "name": "GPT-5.1 Chat",
        "family": "gpt-5.1",
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.25,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=0.125,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    # GPT-5 family
    "gpt-5": {
        "name": "GPT-5",
        "family": "gpt-5",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.25,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=0.125,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5-mini": {
        "name": "GPT-5 Mini",
        "family": "gpt-5",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=0.25,
            output_price_per_1m=2.00,
            cached_input_price_per_1m=0.025,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5-nano": {
        "name": "GPT-5 Nano",
        "family": "gpt-5",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=0.05,
            output_price_per_1m=0.40,
            cached_input_price_per_1m=0.005,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5-pro": {
        "name": "GPT-5 Pro",
        "family": "gpt-5",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=15.00,
            output_price_per_1m=120.00,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5-chat": {
        "name": "GPT-5 Chat",
        "family": "gpt-5",
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.25,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=0.125,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-5-codex": {
        "name": "GPT-5 Codex",
        "family": "gpt-5",
        "provider": "openai",
        "context_window": 400_000,
        "max_output_tokens": 128_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.25,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=0.125,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    # GPT-4.1 family
    "gpt-4.1": {
        "name": "GPT-4.1",
        "family": "gpt-4.1",
        "provider": "openai",
        "context_window": 1_048_000,
        "max_output_tokens": 32_000,
        "pricing": ModelPricing(
            input_price_per_1m=2.00,
            output_price_per_1m=8.00,
            cached_input_price_per_1m=0.50,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-4.1-mini": {
        "name": "GPT-4.1 Mini",
        "family": "gpt-4.1",
        "provider": "openai",
        "context_window": 1_048_000,
        "max_output_tokens": 32_000,
        "pricing": ModelPricing(
            input_price_per_1m=0.40,
            output_price_per_1m=1.60,
            cached_input_price_per_1m=0.10,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-4.1-nano": {
        "name": "GPT-4.1 Nano",
        "family": "gpt-4.1",
        "provider": "openai",
        "context_window": 1_048_000,
        "max_output_tokens": 32_000,
        "pricing": ModelPricing(
            input_price_per_1m=0.10,
            output_price_per_1m=0.40,
            cached_input_price_per_1m=0.025,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    # o-series (reasoning models)
    "o4-mini": {
        "name": "o4-mini",
        "family": "o4",
        "provider": "openai",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.10,
            output_price_per_1m=4.40,
            cached_input_price_per_1m=0.275,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "o3": {
        "name": "o3",
        "family": "o3",
        "provider": "openai",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "pricing": ModelPricing(
            input_price_per_1m=2.00,
            output_price_per_1m=8.00,
            cached_input_price_per_1m=0.50,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "o3-mini": {
        "name": "o3-mini",
        "family": "o3",
        "provider": "openai",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "pricing": ModelPricing(
            input_price_per_1m=1.10,
            output_price_per_1m=4.40,
            cached_input_price_per_1m=0.55,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "o3-pro": {
        "name": "o3-pro",
        "family": "o3",
        "provider": "openai",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "pricing": ModelPricing(
            input_price_per_1m=20.00,
            output_price_per_1m=80.00,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "o1": {
        "name": "o1",
        "family": "o1",
        "provider": "openai",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "pricing": ModelPricing(
            input_price_per_1m=15.00,
            output_price_per_1m=60.00,
            cached_input_price_per_1m=7.50,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "o1-pro": {
        "name": "o1-pro",
        "family": "o1",
        "provider": "openai",
        "context_window": 200_000,
        "max_output_tokens": 100_000,
        "pricing": ModelPricing(
            input_price_per_1m=150.00,
            output_price_per_1m=600.00,
        ),
        "capabilities": ModelCapabilities(
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    # GPT-4o family
    "gpt-4o": {
        "name": "GPT-4o",
        "family": "gpt-4o",
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 16_000,
        "pricing": ModelPricing(
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=1.25,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "family": "gpt-4o",
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 16_000,
        "pricing": ModelPricing(
            input_price_per_1m=0.15,
            output_price_per_1m=0.60,
            cached_input_price_per_1m=0.075,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    # Open-weight
    "gpt-oss-120b": {
        "name": "GPT-OSS 120B",
        "family": "gpt-oss",
        "provider": "openai",
        "context_window": 131_000,
        "pricing": ModelPricing(
            input_price_per_1m=0.039,
            output_price_per_1m=0.19,
        ),
        "capabilities": ModelCapabilities(
            streaming=True,
        ),
    },
    "gpt-oss-20b": {
        "name": "GPT-OSS 20B",
        "family": "gpt-oss",
        "provider": "openai",
        "context_window": 131_000,
        "pricing": ModelPricing(
            input_price_per_1m=0.03,
            output_price_per_1m=0.14,
        ),
        "capabilities": ModelCapabilities(
            streaming=True,
        ),
    },
    # Legacy
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "family": "gpt-4",
        "provider": "openai",
        "context_window": 128_000,
        "max_output_tokens": 4_000,
        "pricing": ModelPricing(
            input_price_per_1m=10.00,
            output_price_per_1m=30.00,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-4": {
        "name": "GPT-4",
        "family": "gpt-4",
        "provider": "openai",
        "context_window": 8_000,
        "max_output_tokens": 8_000,
        "pricing": ModelPricing(
            input_price_per_1m=30.00,
            output_price_per_1m=60.00,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "family": "gpt-3.5",
        "provider": "openai",
        "context_window": 16_000,
        "max_output_tokens": 4_000,
        "deprecated": True,
        "pricing": ModelPricing(
            input_price_per_1m=0.50,
            output_price_per_1m=1.50,
        ),
        "capabilities": ModelCapabilities(
            function_calling=True,
            streaming=True,
            json_mode=True,
        ),
    },
    # Claude (Anthropic) models
    "claude-opus-4-6": {
        "name": "Claude Opus 4.6",
        "family": "claude-opus",
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 32_000,
        "pricing": ModelPricing(
            input_price_per_1m=15.00,
            output_price_per_1m=75.00,
            cached_input_price_per_1m=1.50,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "claude-sonnet-4-5-20250929": {
        "name": "Claude Sonnet 4.5",
        "family": "claude-sonnet",
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 16_000,
        "pricing": ModelPricing(
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            cached_input_price_per_1m=0.30,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "claude-haiku-4-5-20251001": {
        "name": "Claude Haiku 4.5",
        "family": "claude-haiku",
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 8_192,
        "pricing": ModelPricing(
            input_price_per_1m=0.80,
            output_price_per_1m=4.00,
            cached_input_price_per_1m=0.08,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "claude-sonnet-4-20250514": {
        "name": "Claude Sonnet 4",
        "family": "claude-sonnet",
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 16_000,
        "pricing": ModelPricing(
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            cached_input_price_per_1m=0.30,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "claude-3-5-haiku-20241022": {
        "name": "Claude 3.5 Haiku",
        "family": "claude-haiku",
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 8_192,
        "pricing": ModelPricing(
            input_price_per_1m=0.80,
            output_price_per_1m=4.00,
            cached_input_price_per_1m=0.08,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "claude-3-5-sonnet-20241022": {
        "name": "Claude 3.5 Sonnet",
        "family": "claude-sonnet",
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 8_192,
        "pricing": ModelPricing(
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            cached_input_price_per_1m=0.30,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "claude-3-opus-20240229": {
        "name": "Claude 3 Opus",
        "family": "claude-opus",
        "provider": "anthropic",
        "context_window": 200_000,
        "max_output_tokens": 4_096,
        "pricing": ModelPricing(
            input_price_per_1m=15.00,
            output_price_per_1m=75.00,
            cached_input_price_per_1m=1.50,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    # Gemini (Google) models
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "family": "gemini-2.5",
        "provider": "google",
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
        "pricing": ModelPricing(
            input_price_per_1m=1.25,
            output_price_per_1m=10.00,
            cached_input_price_per_1m=0.3125,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "family": "gemini-2.5",
        "provider": "google",
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
        "pricing": ModelPricing(
            input_price_per_1m=0.15,
            output_price_per_1m=0.60,
            cached_input_price_per_1m=0.0375,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            reasoning=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gemini-2.5-flash-lite": {
        "name": "Gemini 2.5 Flash-Lite",
        "family": "gemini-2.5",
        "provider": "google",
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
        "pricing": ModelPricing(
            input_price_per_1m=0.075,
            output_price_per_1m=0.30,
            cached_input_price_per_1m=0.01875,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gemini-2.0-flash": {
        "name": "Gemini 2.0 Flash",
        "family": "gemini-2.0",
        "provider": "google",
        "context_window": 1_048_576,
        "max_output_tokens": 8_192,
        "pricing": ModelPricing(
            input_price_per_1m=0.10,
            output_price_per_1m=0.40,
            cached_input_price_per_1m=0.025,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            structured_output=True,
            streaming=True,
            json_mode=True,
        ),
    },
    "gemini-2.0-flash-lite": {
        "name": "Gemini 2.0 Flash-Lite",
        "family": "gemini-2.0",
        "provider": "google",
        "context_window": 1_048_576,
        "max_output_tokens": 8_192,
        "pricing": ModelPricing(
            input_price_per_1m=0.075,
            output_price_per_1m=0.30,
            cached_input_price_per_1m=0.01875,
        ),
        "capabilities": ModelCapabilities(
            vision=True,
            function_calling=True,
            streaming=True,
            json_mode=True,
        ),
    },
}

# Patterns for inferring model family from ID
_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^gpt-5\.2"), "gpt-5.2"),
    (re.compile(r"^gpt-5\.1"), "gpt-5.1"),
    (re.compile(r"^gpt-5"), "gpt-5"),
    (re.compile(r"^gpt-4\.1"), "gpt-4.1"),
    (re.compile(r"^gpt-4o"), "gpt-4o"),
    (re.compile(r"^gpt-4-turbo"), "gpt-4"),
    (re.compile(r"^gpt-4"), "gpt-4"),
    (re.compile(r"^gpt-3\.5"), "gpt-3.5"),
    (re.compile(r"^gpt-oss"), "gpt-oss"),
    (re.compile(r"^o4"), "o4"),
    (re.compile(r"^o3"), "o3"),
    (re.compile(r"^o1"), "o1"),
    (re.compile(r"^dall-e"), "dall-e"),
    (re.compile(r"^tts"), "tts"),
    (re.compile(r"^whisper"), "whisper"),
    (re.compile(r"^text-embedding"), "text-embedding"),
    (re.compile(r"^text-moderation"), "text-moderation"),
    # Claude family patterns
    (re.compile(r"^claude-opus"), "claude-opus"),
    (re.compile(r"^claude-sonnet"), "claude-sonnet"),
    (re.compile(r"^claude-haiku"), "claude-haiku"),
    (re.compile(r"^claude-3-5-sonnet"), "claude-sonnet"),
    (re.compile(r"^claude-3-5-haiku"), "claude-haiku"),
    (re.compile(r"^claude-3-opus"), "claude-opus"),
    (re.compile(r"^claude-3-sonnet"), "claude-sonnet"),
    (re.compile(r"^claude-3-haiku"), "claude-haiku"),
    (re.compile(r"^claude"), "claude"),
    # Gemini family patterns
    (re.compile(r"^gemini-2\.5"), "gemini-2.5"),
    (re.compile(r"^gemini-2\.0"), "gemini-2.0"),
    (re.compile(r"^gemini-1\.5"), "gemini-1.5"),
    (re.compile(r"^gemini"), "gemini"),
]

# Provider inference from model ID
_PROVIDER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^claude"), "anthropic"),
    (re.compile(r"^gemini"), "google"),
]


def infer_family(model_id: str) -> str:
    """Infer the model family from a model ID using regex patterns."""
    for pattern, family in _FAMILY_PATTERNS:
        if pattern.search(model_id):
            return family
    return ""


def infer_provider(model_id: str) -> str:
    """Infer the provider from a model ID using regex patterns."""
    for pattern, provider in _PROVIDER_PATTERNS:
        if pattern.search(model_id):
            return provider
    return "openai"


def get_known_model(model_id: str) -> OpenAIModel | None:
    """Look up a model in the hardcoded capability map."""
    data = KNOWN_MODELS.get(model_id)
    if data is None:
        return None

    ctx = data.get("context_window")
    max_out = data.get("max_output_tokens")
    caps = data.get("capabilities")
    price = data.get("pricing")

    return OpenAIModel(
        id=model_id,
        name=str(data.get("name", model_id)),
        family=str(data.get("family", infer_family(model_id))),
        provider=str(data.get("provider", infer_provider(model_id))),
        description=str(data.get("description", "")),
        context_window=int(str(ctx)) if ctx is not None else None,
        max_output_tokens=int(str(max_out)) if max_out is not None else None,
        deprecated=bool(data.get("deprecated", False)),
        capabilities=(
            caps if isinstance(caps, ModelCapabilities) else ModelCapabilities()
        ),
        pricing=price if isinstance(price, ModelPricing) else ModelPricing(),
    )
