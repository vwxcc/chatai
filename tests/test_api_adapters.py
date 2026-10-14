import unittest

import main


class APIAdapterTests(unittest.TestCase):
    def test_endpoint_joins_base_and_path(self):
        self.assertEqual(
            main._endpoint("https://api.example/v1", "/chat/completions"),
            "https://api.example/v1/chat/completions",
        )

    def test_endpoint_does_not_duplicate_path(self):
        self.assertEqual(
            main._endpoint("https://api.example/v1/chat/completions", "/chat/completions"),
            "https://api.example/v1/chat/completions",
        )

    def test_absolute_endpoint_is_preserved(self):
        self.assertEqual(
            main._endpoint("https://api.example/v1", "https://other.example/generate"),
            "https://other.example/generate",
        )

    def test_custom_json_template_preserves_typed_values(self):
        template = (
            '{"model":"{{model}}","messages":"{{conversation}}",'
            '"temperature":"{{temperature}}","system":"{{system_prompt}}"}'
        )
        messages = [{"role": "user", "content": "Привет"}]
        result = main._build_template(
            template,
            {
                "model": "test-model",
                "conversation": messages,
                "temperature": 0.25,
                "system_prompt": "Инструкция",
            },
        )
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["messages"], messages)
        self.assertEqual(result["temperature"], 0.25)
        self.assertEqual(result["system"], "Инструкция")

    def test_response_path_reads_nested_arrays_and_objects(self):
        payload = {"data": {"results": [{"output": {"text": "Готово"}}]}}
        self.assertEqual(
            main._response_path(payload, "data.results.0.output.text"),
            "Готово",
        )

    def test_api_headers_support_non_bearer_auth(self):
        headers = main._api_headers(
            "secret",
            "x-api-key",
            "",
            '{"anthropic-version":"2023-06-01"}',
        )
        self.assertEqual(headers["x-api-key"], "secret")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")

    def test_ai_output_json_has_separate_speech_text(self):
        answer, speech = main.parse_ai_result(
            '{"answer_html":"<b>Привет</b>","speech_text":"Привет!"}'
        )
        self.assertIn("<b>", answer)
        self.assertEqual(speech, "Привет!")


if __name__ == "__main__":
    unittest.main()
