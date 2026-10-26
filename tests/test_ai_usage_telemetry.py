import json
import os
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

_TEST_ROOT = tempfile.TemporaryDirectory(prefix="ai-ch-usage-tests-")
os.environ["SESSION_SECRET"] = "test-only-session-secret-with-more-than-32-chars"
os.environ["DATA_DIR"] = str(Path(_TEST_ROOT.name) / "data")
os.environ["UPLOAD_DIR"] = str(Path(_TEST_ROOT.name) / "uploads")
os.environ["FRONTEND_DIR"] = str(Path(_TEST_ROOT.name) / "frontend")
os.environ["AI_USAGE_RETENTION_DAYS"] = "30"
os.environ["AI_STREAM_INCLUDE_USAGE"] = "1"

from backend.app import (  # noqa: E402
    _normalise_ai_usage,
    _record_ai_usage_event,
    _safe_endpoint,
    get_db,
    ModelRouter,
)


class UsageNormalisationTests(unittest.TestCase):
    def test_missing_cache_fields_remain_unknown_not_zero(self):
        usage = _normalise_ai_usage({
            "prompt_tokens": 100,
            "completion_tokens": 15,
            "total_tokens": 115,
        })
        self.assertEqual(usage["usage_status"], "available")
        self.assertIsNone(usage["cache_hit_tokens"])
        self.assertIsNone(usage["cache_miss_tokens"])
        self.assertIsNone(usage["cache_hit_rate"])
        self.assertEqual(usage["cache_status"], "missing")

    def test_explicit_zero_hit_is_preserved(self):
        usage = _normalise_ai_usage({
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "total_tokens": 23,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 20,
        })
        self.assertEqual(usage["cache_hit_tokens"], 0)
        self.assertEqual(usage["cache_miss_tokens"], 20)
        self.assertEqual(usage["cache_status"], "available")
        self.assertEqual(usage["cache_hit_rate"], 0.0)

    def test_cached_tokens_alias_is_supported(self):
        usage = _normalise_ai_usage({
            "prompt_tokens": 40,
            "completion_tokens": 4,
            "total_tokens": 44,
            "prompt_tokens_details": {"cached_tokens": 10},
        })
        self.assertEqual(usage["cache_hit_tokens"], 10)
        self.assertEqual(usage["cache_hit_rate"], 25.0)
        self.assertIsNone(usage["cache_miss_tokens"])

    def test_zero_prompt_does_not_produce_rate(self):
        usage = _normalise_ai_usage({
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        })
        self.assertEqual(usage["cache_hit_tokens"], 0)
        self.assertIsNone(usage["cache_hit_rate"])

    def test_endpoint_does_not_store_credentials_or_query(self):
        self.assertEqual(
            _safe_endpoint("https://user:secret@example.test/v1?token=secret"),
            "https://example.test",
        )

    def test_usage_event_is_persisted_without_prompt_content(self):
        with closing(get_db()) as db:
            db.execute("DELETE FROM ai_usage_events")
            db.commit()
        _record_ai_usage_event(
            request_id=42,
            task_key="main_generation",
            routing_set_id=7,
            model={
                "provider_name": "Test Provider",
                "model_name": "test-model",
                "base_url": "https://user:secret@example.test/v1?token=secret",
            },
            response_model="provider-returned-model",
            streaming=True,
            usage={
                "prompt_tokens": 12,
                "completion_tokens": 2,
                "total_tokens": 14,
                "prompt_cache_hit_tokens": 4,
                "prompt_cache_miss_tokens": 8,
            },
            attempt_status="success",
            duration_ms=123,
        )
        with closing(get_db()) as db:
            row = db.execute("SELECT * FROM ai_usage_events").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["cache_hit_tokens"], 4)
        self.assertEqual(row["cache_miss_tokens"], 8)
        self.assertAlmostEqual(row["cache_hit_rate"], 100.0 / 3.0)
        self.assertEqual(row["cache_status"], "available")
        self.assertEqual(row["endpoint"], "https://example.test")
        self.assertNotIn("secret", str(dict(row)))
        self.assertTrue({"messages", "content", "api_key", "prompt"}.isdisjoint(set(row.keys())))


class _FakeSSEStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class StreamingUsageTests(unittest.TestCase):
    def setUp(self):
        with closing(get_db()) as db:
            db.execute("DELETE FROM ai_usage_events")
            db.commit()

    @staticmethod
    def _model():
        return {
            "id": 1,
            "name": "Test model",
            "model_name": "test-stream-model",
            "temperature": 0.2,
            "max_tokens": 100,
            "timeout": 5,
            "provider_name": "Test Provider",
            "base_url": "https://api.example.test/v1",
            "api_key_env": None,
        }

    def test_stream_usage_only_chunk_is_read_without_fallback(self):
        response = _FakeSSEStream([
            b'data: {"model":"test-stream-model","choices":[{"delta":{"content":"hello"}}]}\n',
            b'data: {"model":"test-stream-model","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11,"prompt_cache_hit_tokens":6,"prompt_cache_miss_tokens":4}}\n',
            b"data: [DONE]\n",
        ])
        router = ModelRouter()
        with patch.object(router, "_models_for_task", return_value=([self._model()], 3)):
            with patch("backend.app.urllib.request.urlopen", return_value=response) as urlopen:
                events = list(router.stream_generate(
                    "main_generation",
                    [{"role": "user", "content": "test"}],
                    request_id=321,
                ))
        self.assertEqual([item["type"] for item in events], ["delta", "done"])
        self.assertTrue(response.closed)
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        with closing(get_db()) as db:
            row = db.execute("SELECT * FROM ai_usage_events WHERE request_id=321").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["attempt_status"], "success")
        self.assertEqual(row["usage_status"], "available")
        self.assertEqual(row["cache_status"], "available")
        self.assertEqual(row["cache_hit_tokens"], 6)
        self.assertEqual(row["cache_miss_tokens"], 4)
        self.assertEqual(row["response_model"], "test-stream-model")

    def test_stream_ending_before_final_usage_is_marked_incomplete(self):
        response = _FakeSSEStream([
            b'data: {"model":"test-stream-model","choices":[{"delta":{"content":"partial"}}]}\n',
        ])
        router = ModelRouter()
        with patch.object(router, "_models_for_task", return_value=([self._model()], 3)):
            with patch("backend.app.urllib.request.urlopen", return_value=response):
                events = list(router.stream_generate(
                    "main_generation",
                    [{"role": "user", "content": "test"}],
                    request_id=322,
                ))
        self.assertEqual([item["type"] for item in events], ["delta", "done"])
        with closing(get_db()) as db:
            row = db.execute("SELECT * FROM ai_usage_events WHERE request_id=322").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["attempt_status"], "success")
        self.assertEqual(row["usage_status"], "incomplete")
        self.assertEqual(row["cache_status"], "incomplete")
        self.assertIsNone(row["cache_hit_tokens"])


if __name__ == "__main__":
    unittest.main()
