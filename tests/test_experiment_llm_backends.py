"""Tests for LLM backends and code extraction."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

from experiments.llm import extract_code


class TestExtractCode:
    def test_no_fences(self):
        code = "[Mesh]\n  [gen]\n    type = GeneratedMeshGenerator\n  []\n[]"
        assert extract_code(code) == code

    def test_plain_fences(self):
        response = "Here is the code:\n```\n[Mesh]\n  [gen]\n  []\n[]\n```\nDone."
        assert extract_code(response) == "[Mesh]\n  [gen]\n  []\n[]"

    def test_language_tag_ini(self):
        response = "```ini\n[Mesh]\n[]\n```"
        assert extract_code(response) == "[Mesh]\n[]"

    def test_language_tag_hit(self):
        response = "Sure!\n```hit\n[Variables]\n  [u]\n  []\n[]\n```\nThat should work."
        assert extract_code(response) == "[Variables]\n  [u]\n  []\n[]"

    def test_multiple_fences_takes_first(self):
        response = "```\n[Mesh]\n[]\n```\nAlso:\n```\n[Vars]\n[]\n```"
        assert extract_code(response) == "[Mesh]\n[]"

    def test_whitespace_only_stripped(self):
        assert extract_code("  \n [Mesh]\n []\n  ") == "[Mesh]\n []"


from unittest.mock import MagicMock, patch


class TestAnthropicLLM:
    def test_generate_calls_api_and_extracts(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")

        mock_text_block = MagicMock()
        mock_text_block.text = "```\n[Mesh]\n  type = GeneratedMeshGenerator\n[]\n```"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_response
            from experiments.llm import AnthropicLLM

            llm = AnthropicLLM(model="claude-test")
            result = llm.generate("sys prompt", "user prompt", temperature=0)

        assert result == "[Mesh]\n  type = GeneratedMeshGenerator\n[]"
        MockClient.return_value.messages.create.assert_called_once_with(
            model="claude-test",
            system="sys prompt",
            messages=[{"role": "user", "content": "user prompt"}],
            temperature=0,
            max_tokens=4096,
        )

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("anthropic.Anthropic"):
            from experiments.llm import AnthropicLLM
            import pytest
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicLLM()


class TestGeminiLLM:
    def test_generate_calls_api_and_extracts(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-456")

        mock_response = MagicMock()
        mock_response.text = "```ini\n[Variables]\n  [T]\n  []\n[]\n```"

        with patch("google.genai.Client") as MockClient:
            MockClient.return_value.models.generate_content.return_value = mock_response
            from experiments.llm import GeminiLLM

            llm = GeminiLLM(model="gemini-test")
            result = llm.generate("sys prompt", "user prompt", temperature=0)

        assert result == "[Variables]\n  [T]\n  []\n[]"
        MockClient.return_value.models.generate_content.assert_called_once()

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("google.genai.Client"):
            from experiments.llm import GeminiLLM
            import pytest
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiLLM()


class TestOpenAICompatibleLLM:
    def test_generate_calls_api_and_extracts(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-789")

        mock_choice = MagicMock()
        mock_choice.message.content = "```\n[BCs]\n  [left]\n    type = DirichletBC\n  []\n[]\n```"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.chat.completions.create.return_value = mock_response
            from experiments.llm import OpenAICompatibleLLM

            llm = OpenAICompatibleLLM(model="gpt-4o")
            result = llm.generate("sys prompt", "user prompt", temperature=0)

        assert result == "[BCs]\n  [left]\n    type = DirichletBC\n  []\n[]"
        MockClient.return_value.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            temperature=0,
            max_tokens=4096,
        )
