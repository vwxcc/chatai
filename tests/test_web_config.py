import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import closing

_TEST_DIR = tempfile.TemporaryDirectory(prefix="ai-ch-tests-")
os.environ.setdefault("SESSION_SECRET", "test-session-secret-long-enough-for-session-middleware")
os.environ["DATA_DIR"] = os.path.join(_TEST_DIR.name, "data")
os.environ["UPLOAD_DIR"] = os.path.join(_TEST_DIR.name, "uploads")
os.environ["FRONTEND_DIR"] = os.path.abspath("frontend")
os.environ.setdefault("AI_API_BASE_URL", "https://api.example/v1")
os.environ.setdefault("AI_API_KEY", "test-key")
os.environ.setdefault("AI_MODEL", "test-model")
os.environ.setdefault("GLOBAL_AI_CONCURRENCY", "2")

from backend import app as web


class AiChWebsiteTests(unittest.TestCase):
    def test_environment_model_is_used_when_router_has_no_route(self):
        with closing(web.get_db()) as db:
            db.execute("DELETE FROM task_routes")
            db.commit()
            models, route_id = web.MODEL_ROUTER._models_for_task(db, "main_generation")
        self.assertEqual(route_id, 0)
        self.assertEqual(models[0]["base_url"], os.environ["AI_API_BASE_URL"].rstrip("/"))
        self.assertEqual(models[0]["api_key_env"], "AI_API_KEY")
        self.assertEqual(models[0]["model_name"], "test-model")

    def test_environment_model_has_priority_for_main_generation(self):
        model = {
            "id": -1, "name": "ENV model", "model_name": "env-primary",
            "temperature": 0.2, "max_tokens": 100, "timeout": 30,
            "provider_name": "Environment API", "base_url": "https://api.example/v1",
            "api_key_env": "AI_API_KEY",
        }
        with patch.dict(os.environ, {"AI_ENV_PRIORITY": "1"}):
            with patch.object(web.MODEL_ROUTER, "_environment_models", return_value=[model]):
                db = Mock()
                models, route_id = web.MODEL_ROUTER._models_for_task(db, "main_generation")
        self.assertEqual(route_id, 0)
        self.assertEqual(models[0]["model_name"], "env-primary")
        db.execute.assert_not_called()

    def test_concurrency_limit_is_positive(self):
        self.assertGreaterEqual(web.GLOBAL_AI_CONCURRENCY, 1)
        self.assertIsNotNone(web.AI_SEMAPHORE)

    def test_ram_history_does_not_insert_database_messages(self):
        user_id, chat_id = 987654321, 123456789
        web._remember_volatile_history(user_id, chat_id, [{"role": "user", "content": "question"}], "answer")
        context = web._with_volatile_history(user_id, chat_id, [{"role": "user", "content": "next"}])
        self.assertEqual(context[0]["content"], "question")
        with closing(web.get_db()) as db:
            count = db.execute("SELECT COUNT(*) AS n FROM messages WHERE chat_id=?", (chat_id,)).fetchone()["n"]
        self.assertEqual(count, 0)

    def test_telegram_payload_chunk_size(self):
        chunks = web.telegram_archive.split_text(("строка\n" * 1200), 3700)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 3700 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
