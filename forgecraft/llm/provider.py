"""LLM Provider 抽象层 — 支持多种后端

设计原则:
  - 统一接口, 后端可切换 (OpenAI / Anthropic / Ollama / 本地模型)
  - 零强制依赖: 未安装的 provider 静默降级
  - 支持 Function Calling (tool_use)
  - 成本追踪 (token 计数)

支持的后端:
  - OpenAI (gpt-4o, gpt-4o-mini)
  - Anthropic (claude-3.5-sonnet)
  - Ollama (本地 llama3, qwen2.5 等)
  - LiteLLM (统一网关, 需要额外安装)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from forgecraft.logging import get_logger

_logger = get_logger(__name__)


class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    MOCK = "mock"  # 测试用假 provider


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: Optional[List[Dict]] = None
    usage: Dict[str, int] = field(default_factory=dict)  # {"prompt_tokens": N, "completion_tokens": N}
    finish_reason: str = "stop"
    model: str = ""


@dataclass
class ToolDefinition:
    """Function Calling 工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Optional[Callable] = None  # Python 执行函数


class BaseLLMProvider:
    """LLM Provider 基类"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        raise NotImplementedError

    def supports_tools(self) -> bool:
        return False

    def get_model_name(self) -> str:
        return self.model


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider

    环境变量: OPENAI_API_KEY, OPENAI_BASE_URL (可选)
    """

    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(model, **kwargs)
        self._client = None
        self.api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL")

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = OpenAI(**client_kwargs)
        except ImportError:
            _logger.error("openai package not installed. pip install openai")
            raise
        return self._client

    def supports_tools(self) -> bool:
        return True

    def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()

        openai_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.name:
                m["name"] = msg.name
            openai_messages.append(m)

        kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            _logger.error(f"OpenAI API error: {e}")
            # 回退到 mock
            return _mock_response(f"OpenAI error: {e}")

        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
            model=response.model,
        )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider

    环境变量: ANTHROPIC_API_KEY
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(model, **kwargs)
        self._client = None
        self.api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            _logger.error("anthropic package not installed. pip install anthropic")
            raise
        return self._client

    def supports_tools(self) -> bool:
        return True

    def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        client = self._get_client()

        # Anthropic 格式: system 单独提取
        system_msg = ""
        anthropic_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg += msg.content + "\n"
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": temperature,
        }
        if system_msg.strip():
            kwargs["system"] = system_msg.strip()

        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        try:
            response = client.messages.create(**kwargs)
        except Exception as e:
            _logger.error(f"Anthropic API error: {e}")
            return _mock_response(f"Anthropic error: {e}")

        content = ""
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=response.stop_reason or "stop",
            model=response.model,
        )


class OllamaProvider(BaseLLMProvider):
    """本地 Ollama Provider

    默认地址: http://localhost:11434
    """

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url

    def supports_tools(self) -> bool:
        # Ollama 支持原生 tool calling (较新版本)
        return True

    def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        try:
            import requests
        except ImportError:
            return _mock_response("requests not installed")

        ollama_messages = []
        for msg in messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _logger.error(f"Ollama API error: {e}")
            return _mock_response(f"Ollama error: {e}")

        message = data.get("message", {})
        content = message.get("content", "")

        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = message["tool_calls"]

        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=data.get("done_reason", "stop"),
            model=data.get("model", self.model),
        )


class MockProvider(BaseLLMProvider):
    """测试用 Mock Provider — 不调用真实 API"""

    def __init__(self, model: str = "mock", responses: Optional[List[str]] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.responses = responses or ["Mock response from LLM agent."]
        self._call_count = 0

    def supports_tools(self) -> bool:
        return True

    def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        idx = self._call_count % len(self.responses)
        self._call_count += 1
        return LLMResponse(
            content=self.responses[idx],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="mock",
        )


# ── Provider 工厂 ────────────────────────────────────────────


def create_provider(
    provider_type: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> BaseLLMProvider:
    """创建 LLM Provider

    Args:
        provider_type: "openai" | "anthropic" | "ollama" | "litellm" | "mock"
        model: 模型名 (不指定则使用默认)
        **kwargs: 传递给 provider 的额外参数 (api_key, base_url 等)

    Returns:
        BaseLLMProvider 实例
    """
    provider_type = provider_type.lower()

    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3.1",
        "litellm": "gpt-4o-mini",
        "mock": "mock",
    }

    if model is None:
        model = default_models.get(provider_type, "gpt-4o-mini")

    if provider_type == "openai":
        return OpenAIProvider(model=model, **kwargs)
    elif provider_type == "anthropic":
        return AnthropicProvider(model=model, **kwargs)
    elif provider_type == "ollama":
        return OllamaProvider(model=model, **kwargs)
    elif provider_type == "mock":
        return MockProvider(model=model, **kwargs)
    else:
        _logger.warning(f"Unknown provider '{provider_type}', falling back to mock")
        return MockProvider(model="mock-fallback")


def _mock_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        model="mock-fallback",
    )
