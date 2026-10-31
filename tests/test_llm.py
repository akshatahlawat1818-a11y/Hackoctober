"""Tests for utils.llm calling call_gemini and call_groq."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APIConfig, API_CONFIG, OUTPUTS_DIR
from utils.llm import (
    COST_LOG_PATH,
    call_gemini,
    call_groq,
    execute_with_backoff,
    extract_status_code,
    is_retryable_error,
    parse_json_response,
    strip_code_fences,
    _sanitize_error_message,
)

MOCK_CONFIG = APIConfig(
    gemini_api_key="mock_gemini_key",
    groq_api_key="mock_groq_key",
)


class TestLLMFunctions(unittest.TestCase):
    def test_strip_code_fences_and_json_parsing(self):
        """Verify markdown code fence stripping and json response decoding."""
        fenced_json = "```json\n{\"status\": \"ok\", \"score\": 9.5}\n```"
        parsed = parse_json_response(fenced_json)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["score"], 9.5)

        raw_fence = "```\n{\"greeting\": \"hello\"}\n```"
        self.assertEqual(strip_code_fences(raw_fence), "{\"greeting\": \"hello\"}")

    @patch("utils.llm.API_CONFIG", MOCK_CONFIG)
    @patch("utils.llm.genai.Client")
    def test_call_gemini_mocked(self, mock_client_cls):
        """Verify call_gemini invocation and cost logging with mocked SDK."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = '{"message": "Hello from Gemini!", "status": "active"}'
        mock_response.usage_metadata.prompt_token_count = 12
        mock_response.usage_metadata.candidates_token_count = 15
        mock_response.usage_metadata.total_token_count = 27
        mock_client.models.generate_content.return_value = mock_response

        result = call_gemini("Test prompt for Gemini", json_mode=True)
        print("\n[Mocked Test] call_gemini result:", result)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "active")
        self.assertTrue(COST_LOG_PATH.exists())

    @patch("utils.llm.API_CONFIG", MOCK_CONFIG)
    @patch("utils.llm.Groq")
    def test_call_groq_mocked(self, mock_groq_cls):
        """Verify call_groq invocation and cost logging with mocked SDK."""
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"critique": "Solid script structure", "score": 9.2}'
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 18
        mock_response.usage.completion_tokens = 22
        mock_response.usage.total_tokens = 40
        mock_client.chat.completions.create.return_value = mock_response

        result = call_groq("Test prompt for Groq Critic", json_mode=True)
        print("[Mocked Test] call_groq result:", result)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["score"], 9.2)
        self.assertTrue(COST_LOG_PATH.exists())

    def test_retry_on_503_and_recovery(self):
        """Verify retry with backoff on 503 UNAVAILABLE up to recovery."""
        calls = [0]
        recorded_delays = []

        def failing_function():
            calls[0] += 1
            if calls[0] < 3:
                raise Exception("503 UNAVAILABLE (high demand)")
            return "recovered_payload"

        result = execute_with_backoff(
            call_fn=failing_function,
            provider="gemini",
            model="gemini-3.8-flash",
            delays=(2.0, 4.0, 8.0, 16.0, 32.0),
            sleep_fn=lambda s: recorded_delays.append(s),
        )
        self.assertEqual(result, "recovered_payload")
        self.assertEqual(calls[0], 3)
        self.assertEqual(recorded_delays, [2.0, 4.0])

    def test_all_retries_fail_raises_clear_error(self):
        """Verify exhausting all 5 retries raises a clear RuntimeError rather than returning empty."""
        calls = [0]
        recorded_delays = []

        def always_fails_429():
            calls[0] += 1
            err = Exception("429 Too Many Requests: Rate limit exceeded")
            err.status_code = 429
            raise err

        with self.assertRaises(RuntimeError) as ctx:
            execute_with_backoff(
                call_fn=always_fails_429,
                provider="groq",
                model="openai/gpt-oss-120b",
                delays=(2.0, 4.0, 8.0, 16.0, 32.0),
                sleep_fn=lambda s: recorded_delays.append(s),
            )
        err_msg = str(ctx.exception)
        self.assertIn("failed after 5 retries", err_msg)
        self.assertIn("openai/gpt-oss-120b", err_msg)
        self.assertEqual(calls[0], 6)  # Initial call + 5 retries
        self.assertEqual(recorded_delays, [2.0, 4.0, 8.0, 16.0, 32.0])

    def test_non_retryable_404_raises_immediately(self):
        """Verify 404 Not Found error raises immediately without retrying."""
        calls = [0]
        recorded_delays = []

        def fails_with_404():
            calls[0] += 1
            err = Exception("404 Not Found: Model does not exist")
            err.code = 404
            raise err

        with self.assertRaises(Exception) as ctx:
            execute_with_backoff(
                call_fn=fails_with_404,
                provider="gemini",
                model="gemini-3.8-flash",
                delays=(2.0, 4.0, 8.0, 16.0, 32.0),
                sleep_fn=lambda s: recorded_delays.append(s),
            )
        self.assertIn("404", str(ctx.exception))
        self.assertEqual(calls[0], 1)  # Only 1 call, zero retries
        self.assertEqual(recorded_delays, [])

    def test_sanitize_error_message_redacts_api_keys(self):
        """Verify API keys are redacted from error logs and exception messages."""
        msg_with_gemini = "Failed request to https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyA1234567890abcdef1234567890abcde"
        msg_with_groq = "Failed groq call with Bearer gsk_1234567890abcdef1234567890abcdef12345678"

        sanitized_gemini = _sanitize_error_message(msg_with_gemini)
        sanitized_groq = _sanitize_error_message(msg_with_groq)

        self.assertNotIn("AIzaSyA", sanitized_gemini)
        self.assertIn("[REDACTED_API_KEY]", sanitized_gemini)
        self.assertNotIn("gsk_123456", sanitized_groq)
        self.assertIn("[REDACTED_API_KEY]", sanitized_groq)


def run_one_call_to_each():
    """Make one call to call_gemini and call_groq and print results."""
    print("=" * 60)
    print("Executing one call to call_gemini() and call_groq()")
    print("=" * 60)

    has_gemini_key = bool(
        (API_CONFIG.gemini_api_key and "your_gemini" not in API_CONFIG.gemini_api_key)
        or (os.getenv("GEMINI_API_KEY") and "your_gemini" not in os.getenv("GEMINI_API_KEY", ""))
    )
    has_groq_key = bool(
        (API_CONFIG.groq_api_key and "your_groq" not in API_CONFIG.groq_api_key)
        or (os.getenv("GROQ_API_KEY") and "your_groq" not in os.getenv("GROQ_API_KEY", ""))
    )

    # 1. Call Gemini
    if has_gemini_key:
        print("\n--- 1. Calling call_gemini (Live API) ---")
        try:
            gemini_result = call_gemini(
                prompt="Respond in JSON with keys 'agent' ('Planner') and 'ready' (true).",
                system="You are an AI planner agent.",
                json_mode=True,
            )
            print("Result from call_gemini:")
            print(gemini_result)
        except Exception as e:
            print(f"Error calling live Gemini: {_sanitize_error_message(str(e))}")
    else:
        print("\n--- 1. Calling call_gemini (Simulated SDK Client) ---")
        with patch("utils.llm.API_CONFIG", MOCK_CONFIG):
            with patch("utils.llm.genai.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_resp = MagicMock()
                mock_resp.text = '{"agent": "Planner", "model": "gemini-3.8-flash", "ready": true}'
                mock_resp.usage_metadata.prompt_token_count = 14
                mock_resp.usage_metadata.candidates_token_count = 20
                mock_resp.usage_metadata.total_token_count = 34
                mock_client.models.generate_content.return_value = mock_resp

                gemini_result = call_gemini(
                    prompt="Respond in JSON with keys 'agent' and 'ready'.",
                    system="You are an AI planner agent.",
                    json_mode=True,
                )
                print("Result from call_gemini:")
                print(gemini_result)

    # 2. Call Groq
    if has_groq_key:
        print("\n--- 2. Calling call_groq (Live API) ---")
        try:
            groq_result = call_groq(
                prompt="Respond in JSON with keys 'critic' ('Critic') and 'score' (9.5).",
                system="You are a rubric critic agent.",
                json_mode=True,
            )
            print("Result from call_groq:")
            print(groq_result)
        except Exception as e:
            print(f"Error calling live Groq: {_sanitize_error_message(str(e))}")
    else:
        print("\n--- 2. Calling call_groq (Simulated SDK Client) ---")
        with patch("utils.llm.API_CONFIG", MOCK_CONFIG):
            with patch("utils.llm.Groq") as mock_groq_cls:
                mock_client = MagicMock()
                mock_groq_cls.return_value = mock_client
                mock_resp = MagicMock()
                mock_choice = MagicMock()
                mock_choice.message.content = '{"critic": "Critic", "model": "openai/gpt-oss-120b", "score": 9.5}'
                mock_resp.choices = [mock_choice]
                mock_resp.usage.prompt_tokens = 22
                mock_resp.usage.completion_tokens = 28
                mock_resp.usage.total_tokens = 50
                mock_client.chat.completions.create.return_value = mock_resp

                groq_result = call_groq(
                    prompt="Respond in JSON with keys 'critic' and 'score'.",
                    system="You are a rubric critic agent.",
                    json_mode=True,
                )
                print("Result from call_groq:")
                print(groq_result)

    print("\n" + "=" * 60)
    print("Recent rows in outputs/cost_log.csv:")
    print("=" * 60)
    if COST_LOG_PATH.exists():
        with open(COST_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f.readlines()[-5:]:
                print(line.strip())
    print("=" * 60)


if __name__ == "__main__":
    run_one_call_to_each()
