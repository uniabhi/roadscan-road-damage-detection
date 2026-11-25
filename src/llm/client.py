"""
LLM client interface for RoadScan.

Provides a unified interface for LLM API calls with support for
different providers (Bedrock, OpenAI, etc.) and mock responses for testing.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 500) -> str:
        """Generate text from the LLM."""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                      temperature: float = 0.1) -> Dict:
        """Generate JSON response from the LLM."""
        pass


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, response_template: Optional[Dict] = None):
        self.response_template = response_template or {
            "summary": "Road damage detected requiring attention.",
            "priority": "medium",
            "rationale": "Based on the detected damage types and severity.",
            "recommended_action": "Schedule maintenance inspection within 30 days.",
            "cited_sources": ["doc_1"]
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 500) -> str:
        return json.dumps(self.response_template)

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                      temperature: float = 0.1) -> Dict:
        return self.response_template.copy()


class BedrockLLMClient(LLMClient):
    """AWS Bedrock LLM client."""

    def __init__(self, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
                 region: str = "us-east-1"):
        self.model_id = model_id
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 500) -> str:
        client = self._get_client()

        messages = [{"role": "user", "content": prompt}]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system_prompt:
            body["system"] = system_prompt

        response = client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                      temperature: float = 0.1) -> Dict:
        text = self.generate(prompt, system_prompt, temperature)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw_text": text}


class OpenAILLMClient(LLMClient):
    """OpenAI LLM client."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 500) -> str:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                      temperature: float = 0.1) -> Dict:
        text = self.generate(prompt, system_prompt, temperature)
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw_text": text}


def get_llm_client(provider: str = "mock", **kwargs) -> LLMClient:
    """
    Get an LLM client instance.

    Args:
        provider: Provider name ("mock", "bedrock", "openai")
        **kwargs: Provider-specific arguments

    Returns:
        LLMClient instance
    """
    if provider == "mock":
        return MockLLMClient(**kwargs)
    elif provider == "bedrock":
        return BedrockLLMClient(**kwargs)
    elif provider == "openai":
        return OpenAILLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
