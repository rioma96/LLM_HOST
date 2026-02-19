import json
import unittest

import main


class ChatReviewNormalizationTests(unittest.TestCase):
    def _normalize_review(self, sample_ids, model_output):
        content, _, _ = main._normalize_mode_output(
            mode="review",
            raw_text=model_output,
            input_sample_ids=sample_ids,
            input_triples=[],
        )
        normalized = json.loads(content)
        self.assertIn("samples", normalized)
        self.assertIsInstance(normalized["samples"], list)
        return normalized

    def test_bare_object_output_is_wrapped(self):
        normalized = self._normalize_review(
            ["s1"],
            '{"sample_id":"s1","keep":["/rel/a"],"delete":false,"notes":"ok"}',
        )

        self.assertEqual(
            normalized,
            {
                "samples": [
                    {
                        "sample_id": "s1",
                        "keep": ["/rel/a"],
                        "delete": False,
                        "notes": "ok",
                    }
                ]
            },
        )

    def test_array_output_is_wrapped(self):
        normalized = self._normalize_review(
            ["s1", "s2"],
            '[{"sample_id":"s1","keep":["/rel/a"],"delete":false,"notes":"ok1"},'
            '{"sample_id":"s2","keep":[],"delete":false,"notes":"ok2"}]',
        )

        self.assertEqual([item["sample_id"] for item in normalized["samples"]], ["s1", "s2"])
        self.assertEqual(normalized["samples"][1]["notes"], "ok2")

    def test_already_correct_wrapper_is_preserved(self):
        normalized = self._normalize_review(
            ["s1"],
            '{"samples":[{"sample_id":"s1","keep":["/rel/a"],"delete":false,"notes":"ok"}]}',
        )

        self.assertEqual(len(normalized["samples"]), 1)
        self.assertEqual(normalized["samples"][0]["sample_id"], "s1")
        self.assertEqual(normalized["samples"][0]["keep"], ["/rel/a"])

    def test_fenced_json_is_repaired(self):
        normalized = self._normalize_review(
            ["s1"],
            "Some prose before\n```json\n"
            "{\"sample_id\":\"s1\",\"keep\":[\"/rel/a\"],\"delete\":false,\"notes\":\"ok\"}\n"
            "```\nSome prose after",
        )

        self.assertEqual(len(normalized["samples"]), 1)
        self.assertEqual(normalized["samples"][0]["sample_id"], "s1")
        self.assertEqual(normalized["samples"][0]["keep"], ["/rel/a"])

    def test_wrapped_review_samples(self):
        normalized = self._normalize_review(
            ["s1", "s2"],
            '{"samples":['
            '{"sample_id":"s1","keep":["/rel/a"],"delete":false,"notes":"ok"},'
            '{"sample_id":"s2","keep":["/rel/b"],"delete":false,"notes":"ok2"}]}'
        )

        self.assertEqual([item["sample_id"] for item in normalized["samples"]], ["s1", "s2"])

    def test_missing_sample_ids_are_filled_with_safe_fallback(self):
        normalized = self._normalize_review(
            ["s1", "s2"],
            '{"samples":['
            '{"sample_id":"s1","keep":["/rel/a"],"delete":false,"notes":"ok"},'
            '{"keep":["/rel/x"],"delete":true,"notes":"missing id"}]}'
        )

        self.assertEqual([item["sample_id"] for item in normalized["samples"]], ["s1", "s2"])
        self.assertEqual(normalized["samples"][0]["keep"], ["/rel/a"])
        self.assertEqual(normalized["samples"][1]["keep"], [])
        self.assertFalse(normalized["samples"][1]["delete"])
        self.assertEqual(normalized["samples"][1]["notes"], "server_normalization_fallback")

    def test_malformed_payload_review_fallback(self):
        normalized = self._normalize_review(["s1"], "not-json-at-all")
        self.assertEqual(normalized["samples"][0]["sample_id"], "s1")
        self.assertEqual(normalized["samples"][0]["notes"], "server_normalization_fallback")


class ChatGenerationNormalizationTests(unittest.TestCase):
    def _normalize_generation(self, triples, model_output):
        content, _, _ = main._normalize_mode_output(
            mode="generation",
            raw_text=model_output,
            input_sample_ids=[],
            input_triples=triples,
        )
        normalized = json.loads(content)
        self.assertIn("examples", normalized)
        self.assertIsInstance(normalized["examples"], list)
        return normalized

    def test_bare_object_generation_is_wrapped(self):
        triples = [{"head": "h1", "tail": "t1", "relation": "r1"}]
        normalized = self._normalize_generation(
            triples,
            '{"text":"example","triple":{"head":"h1","tail":"t1","relation":"r1"},"notes":"ok"}',
        )
        self.assertEqual(len(normalized["examples"]), 1)
        self.assertEqual(normalized["examples"][0]["triple"]["relation"], "r1")
        self.assertEqual(normalized["examples"][0]["labels"], ["r1"])

    def test_wrapped_generation_examples_is_preserved(self):
        triples = [{"head": "h1", "tail": "t1", "relation": "r1"}]
        normalized = self._normalize_generation(
            triples,
            '{"examples":[{"text":"x","labels":["r1"],"triple":{"head":"h1","tail":"t1","relation":"r1"},"notes":"ok"}]}',
        )
        self.assertEqual(normalized["examples"][0]["labels"], ["r1"])

    def test_array_generation_is_wrapped(self):
        triples = [
            {"head": "h1", "tail": "t1", "relation": "r1"},
            {"head": "h2", "tail": "t2", "relation": "r2"},
        ]
        normalized = self._normalize_generation(
            triples,
            '[{"text":"e1","triple":{"head":"h1","tail":"t1","relation":"r1"},"notes":"ok1"},'
            '{"text":"e2","labels":["r2"],"triple":{"head":"h2","tail":"t2","relation":"r2"},"notes":"ok2"}]',
        )
        self.assertEqual(len(normalized["examples"]), 2)
        self.assertEqual(normalized["examples"][0]["labels"], ["r1"])

    def test_markdown_wrapped_and_mixed_prose_generation(self):
        triples = [{"head": "h1", "tail": "t1", "relation": "r1"}]
        normalized = self._normalize_generation(
            triples,
            "Here the answer\n```json\n"
            "[{\"text\":\"gen\",\"triple\":{\"head\":\"h1\",\"tail\":\"t1\",\"relation\":\"r1\"},\"notes\":\"ok\"}]\n"
            "```\nthanks",
        )
        self.assertEqual(normalized["examples"][0]["triple"]["head"], "h1")
        self.assertEqual(normalized["examples"][0]["labels"], ["r1"])

    def test_malformed_payload_generation_fallback(self):
        normalized = self._normalize_generation([], "totally malformed payload")
        self.assertEqual(normalized, {"examples": []})


if __name__ == "__main__":
    unittest.main()
