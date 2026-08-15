import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from gemq.openai_server import GEMQOpenAIService, ServerSettings


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, **kwargs):
        return f"User: {messages[0]['content']}\n\nAssistant:"


def _settings(model_name="deepseek-ai/DeepSeek-V2-Lite", prompt_format="auto"):
    return ServerSettings(
        model_path="/models/checkpoint",
        model_name=model_name,
        served_model_name="test-model",
        prompt_format=prompt_format,
    )


def _prompt_service(prompt_format):
    service = object.__new__(GEMQOpenAIService)
    service.prompt_format = prompt_format
    service.tokenizer = FakeTokenizer()
    return service


def test_auto_uses_raw_for_base_model_even_when_tokenizer_has_chat_template():
    assert GEMQOpenAIService._resolve_prompt_format(_settings()) == "raw"


def test_auto_uses_chat_for_instruction_model():
    settings = _settings(model_name="deepseek-ai/DeepSeek-V2-Lite-Chat")
    assert GEMQOpenAIService._resolve_prompt_format(settings) == "chat"


def test_raw_prompt_does_not_add_chat_wrappers():
    prompt, add_special_tokens = _prompt_service("raw")._build_prompt(
        [{"role": "user", "content": "Question and few-shot examples"}]
    )

    assert prompt == "Question and few-shot examples"
    assert add_special_tokens is True


def test_chat_prompt_uses_tokenizer_template():
    prompt, add_special_tokens = _prompt_service("chat")._build_prompt(
        [{"role": "user", "content": "Question"}]
    )

    assert prompt == "User: Question\n\nAssistant:"
    assert add_special_tokens is False
