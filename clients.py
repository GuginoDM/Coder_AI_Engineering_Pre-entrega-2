import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator

from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIError
from anthropic import AsyncAnthropic, APIConnectionError as AnthropicConnectionError, RateLimitError as AnthropicRateLimitError, APIError as AnthropicAPIError

from schemas import ChatMessage, ModelConfig, ModelResponse

class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        pass

    @abstractmethod
    async def generate_stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY no encontrada.")
        self.client = AsyncOpenAI(api_key=key)

    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        try:
            formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
            response = await self.client.chat.completions.create(
                model=config.model_name,
                messages=formatted_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            choice = response.choices[0].message
            usage = response.usage
            
            return ModelResponse(
                content=choice.content or "",
                provider="OpenAI",
                model=config.model_name,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        except RateLimitError:
            return ModelResponse(content="[Error] Límite de tasa excedido en OpenAI.", provider="OpenAI", model=config.model_name)
        except APIConnectionError:
            return ModelResponse(content="[Error] Falla de conexión con los servidores de OpenAI.", provider="OpenAI", model=config.model_name)
        except APIError as e:
            return ModelResponse(content=f"[Error API OpenAI]: {str(e)}", provider="OpenAI", model=config.model_name)

    async def generate_stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        try:
            formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
            stream = await self.client.chat.completions.create(
                model=config.model_name,
                messages=formatted_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except (RateLimitError, APIConnectionError, APIError) as e:
            yield f"\n[Error en streaming de OpenAI: {str(e)}]"


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY no encontrada.")
        self.client = AsyncAnthropic(api_key=key)

    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        try:
            system_prompt = next((m.content for m in messages if m.role == "system"), None)
            user_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

            kwargs = {
                "model": config.model_name,
                "max_tokens": config.max_tokens,
                "messages": user_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self.client.messages.create(**kwargs)
            text_content = response.content[0].text if response.content else ""
            
            return ModelResponse(
                content=text_content,
                provider="Anthropic",
                model=config.model_name,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
        except AnthropicRateLimitError:
            return ModelResponse(content="[Error] Límite de tasa excedido en Anthropic.", provider="Anthropic", model=config.model_name)
        except AnthropicConnectionError:
            return ModelResponse(content="[Error] Falla de conexión con los servidores de Anthropic.", provider="Anthropic", model=config.model_name)
        except AnthropicAPIError as e:
            return ModelResponse(content=f"[Error API Anthropic]: {str(e)}", provider="Anthropic", model=config.model_name)

    async def generate_stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        try:
            system_prompt = next((m.content for m in messages if m.role == "system"), None)
            user_messages = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

            kwargs = {
                "model": config.model_name,
                "max_tokens": config.max_tokens,
                "messages": user_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except (AnthropicRateLimitError, AnthropicConnectionError, AnthropicAPIError) as e:
            yield f"\n[Error en streaming de Anthropic: {str(e)}]"


class AsyncLLMManager:
    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.client: BaseLLMClient = OpenAIClient()
        elif self.provider == "anthropic":
            self.client = AnthropicClient()
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")

    async def generate(self, messages: list[ChatMessage], config: ModelConfig) -> ModelResponse:
        return await self.client.generate(messages, config)

    async def generate_stream(self, messages: list[ChatMessage], config: ModelConfig) -> AsyncGenerator[str, None]:
        async for chunk in self.client.generate_stream(messages, config):
            yield chunk