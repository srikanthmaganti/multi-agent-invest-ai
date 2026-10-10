"""Tests for built-in prompt selection and MLflow prompt registration."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from invest_ai.prompt_evaluate import (
    AdvisorQualityJudgment,
    _judge_advisor_response,
    _score_advisor_case,
    _score_context,
    load_cases,
    validate_prompt_uris,
)
from invest_ai.prompt_gate import compare_metrics
from invest_ai.prompts import (
    FEW_SHOT_PROMPTS,
    PROMPTS,
    TECHNIQUE_PROMPTS,
    resolve_system_instruction,
    register_default_prompts,
    register_few_shot_prompts,
    register_technique_prompts,
)


class PromptRegistryTests(unittest.TestCase):
    def test_gate_requires_minimum_scores_and_no_regression(self):
        minimums = {"accuracy": 0.9}

        passes = compare_metrics(
            {"accuracy": 0.9},
            {"accuracy": 0.95},
            minimums,
        )
        misses_minimum = compare_metrics(
            {"accuracy": 0.85},
            {"accuracy": 0.85},
            minimums,
        )
        regresses = compare_metrics(
            {"accuracy": 1.0},
            {"accuracy": 0.95},
            minimums,
        )

        self.assertTrue(passes["passed"])
        self.assertFalse(misses_minimum["passed"])
        self.assertFalse(regresses["passed"])

    def test_evaluation_loader_reads_project_jsonl_cases(self):
        project_root = Path(__file__).resolve().parents[1]
        cases = load_cases(
            project_root / "data" / "prompt_eval" / "advisor_holdout.jsonl"
        )
        self.assertEqual(len(cases), 2)
        self.assertEqual(
            cases[0]["inputs"]["retrieved_passages"][0]["citation_id"], "S1"
        )

    def test_advisor_scorer_checks_citations_and_risky_phrases(self):
        case = {
            "case_id": "scorer-test",
            "inputs": {
                "retrieved_passages": [{"citation_id": "S1"}],
            },
            "expectations": {"must_cite": ["S1"]},
        }

        scored = _score_advisor_case(
            case,
            "I recommend you buy a fund. It is guaranteed returns [S9].",
        )

        self.assertFalse(scored["citation_coverage_pass"])
        self.assertFalse(scored["no_unknown_citations_pass"])
        self.assertFalse(scored["risky_phrase_checks_pass"])
        self.assertTrue(scored["manual_review_required"])

    def test_context_precision_and_recall_use_curated_relevance_labels(self):
        project_root = Path(__file__).resolve().parents[1]
        case = load_cases(
            project_root / "data" / "prompt_eval" / "advisor_holdout.jsonl"
        )[0]

        scored = _score_context(case)

        self.assertEqual(scored["context_precision"], 0.75)
        self.assertEqual(scored["context_recall"], 1.0)
        self.assertEqual(scored["relevant_context_ids"], ["S1", "S2", "S3"])

    def test_advisor_judge_scores_are_bounded(self):
        result = AdvisorQualityJudgment(
            groundedness=0.9,
            answer_relevancy=0.8,
            groundedness_reason="Claims are supported.",
            answer_relevancy_reason="It answers the question.",
        )
        self.assertEqual(result.groundedness, 0.9)
        with self.assertRaises(ValueError):
            AdvisorQualityJudgment(
                groundedness=1.1,
                answer_relevancy=0.8,
                groundedness_reason="Invalid score.",
                answer_relevancy_reason="Valid score.",
            )

    def test_advisor_judge_uses_structured_score_output(self):
        case = {
            "inputs": {
                "query_text": "What does the passage say?",
                "retrieved_passages": [{"citation_id": "S1", "text": "A fact."}],
            },
            "outputs": "It says a fact [S1].",
        }
        judge = Mock()
        judge.generate.return_value = (
            '{"groundedness":0.9,"answer_relevancy":0.8,'
            '"groundedness_reason":"Supported by S1.",'
            '"answer_relevancy_reason":"Answers the question."}'
        )

        result = _judge_advisor_response(judge, case, "The passage says a fact [S1].")

        self.assertEqual(result.groundedness, 0.9)
        self.assertEqual(result.answer_relevancy, 0.8)
        self.assertIs(
            judge.generate.call_args.kwargs["response_schema"],
            AdvisorQualityJudgment,
        )

    def test_prompt_uri_preflight_validates_every_version_before_evaluation(self):
        from invest_ai import prompt_evaluate

        requested = [
            "prompts:/invest-ai-advisor/1",
            "prompts:/invest-ai-advisor/2",
            "prompts:/invest-ai-advisor/7",
        ]
        with patch.object(
            prompt_evaluate,
            "load_prompt_template",
            side_effect=["baseline", ValueError("missing prompt version")],
        ) as load_template:
            with self.assertRaisesRegex(ValueError, "missing prompt version"):
                validate_prompt_uris(requested)

        self.assertEqual(
            [call.args[0] for call in load_template.call_args_list],
            requested[:2],
        )

    def test_uses_default_prompt_without_registry_uri(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_system_instruction("INVEST_AI_TEST_PROMPT_URI", "default"),
                "default",
            )

    def test_loads_configured_prompt_from_mlflow(self):
        mlflow = Mock()
        genai = Mock()
        genai.load_prompt.return_value = SimpleNamespace(template="registered")
        mlflow.genai = genai
        with (
            patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.genai": genai}),
            patch.dict(
                os.environ,
                {
                    "INVEST_AI_TEST_PROMPT_URI": "prompts:/test/1",
                    "MLFLOW_TRACKING_URI": "sqlite:///test.db",
                },
                clear=True,
            ),
        ):
            result = resolve_system_instruction(
                "INVEST_AI_TEST_PROMPT_URI", "default"
            )

        self.assertEqual(result, "registered")
        mlflow.set_tracking_uri.assert_called_once_with("sqlite:///test.db")
        genai.load_prompt.assert_called_once_with("prompts:/test/1")

    def test_rejects_registered_prompt_without_text_template(self):
        mlflow = Mock()
        genai = Mock()
        genai.load_prompt.return_value = SimpleNamespace(template="")
        mlflow.genai = genai
        with (
            patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.genai": genai}),
            patch.dict(
                os.environ,
                {"INVEST_AI_TEST_PROMPT_URI": "prompts:/test/1"},
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no non-empty text template"):
                resolve_system_instruction("INVEST_AI_TEST_PROMPT_URI", "default")

    def test_registers_builtin_prompts_as_mlflow_versions(self):
        mlflow = Mock()
        genai = Mock()
        genai.register_prompt.side_effect = [
            SimpleNamespace(version="1") for _ in PROMPTS
        ]
        mlflow.genai = genai
        with (
            patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.genai": genai}),
            patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "sqlite:///test.db"}),
            patch("builtins.print") as print_output,
        ):
            register_default_prompts()

        mlflow.set_tracking_uri.assert_called_once_with("sqlite:///test.db")
        self.assertEqual(genai.register_prompt.call_count, len(PROMPTS))
        self.assertEqual(print_output.call_count, len(PROMPTS))

    def test_few_shot_prompts_include_examples_and_safety_constraints(self):
        self.assertIn('"intent": "INVESTMENT_GUIDANCE"', FEW_SHOT_PROMPTS[
            "invest-ai-router"
        ])
        self.assertIn("These are style examples only", FEW_SHOT_PROMPTS[
            "invest-ai-advisor"
        ])
        self.assertIn("cannot recommend a specific fund", FEW_SHOT_PROMPTS[
            "invest-ai-advisor"
        ])

    def test_registers_few_shot_prompts_as_new_versions(self):
        mlflow = Mock()
        genai = Mock()
        genai.register_prompt.side_effect = [
            SimpleNamespace(version="2") for _ in FEW_SHOT_PROMPTS
        ]
        mlflow.genai = genai
        with (
            patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.genai": genai}),
            patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "sqlite:///test.db"}),
            patch("builtins.print") as print_output,
        ):
            register_few_shot_prompts()

        self.assertEqual(genai.register_prompt.call_count, len(FEW_SHOT_PROMPTS))
        for call in genai.register_prompt.call_args_list:
            self.assertEqual(
                call.kwargs["commit_message"],
                "Add few-shot examples to Invest AI prompt",
            )
        self.assertEqual(print_output.call_count, len(FEW_SHOT_PROMPTS))

    def test_technique_variants_are_distinct_and_keep_safety_rules(self):
        self.assertEqual(
            set(TECHNIQUE_PROMPTS),
            {
                "task-decomposition",
                "self-check",
                "output-constraints",
                "rag-grounding",
            },
        )
        advisor_prompts = [
            prompts["invest-ai-advisor"]
            for prompts in TECHNIQUE_PROMPTS.values()
            if "invest-ai-advisor" in prompts
        ]
        self.assertEqual(len(advisor_prompts), 4)
        for prompt in advisor_prompts:
            self.assertIn("Do not recommend a specific product", prompt)
        self.assertIn("silently verify", TECHNIQUE_PROMPTS["self-check"][
            "invest-ai-advisor"
        ])
        self.assertIn("only authority", TECHNIQUE_PROMPTS["rag-grounding"][
            "invest-ai-advisor"
        ])

    def test_registers_each_technique_as_a_new_prompt_version(self):
        mlflow = Mock()
        genai = Mock()
        genai.register_prompt.side_effect = [
            SimpleNamespace(version=str(index))
            for index in range(
                3, 3 + sum(len(prompts) for prompts in TECHNIQUE_PROMPTS.values())
            )
        ]
        mlflow.genai = genai
        with (
            patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.genai": genai}),
            patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "sqlite:///test.db"}),
            patch("builtins.print") as print_output,
        ):
            register_technique_prompts()

        expected_count = sum(len(prompts) for prompts in TECHNIQUE_PROMPTS.values())
        self.assertEqual(genai.register_prompt.call_count, expected_count)
        self.assertEqual(print_output.call_count, expected_count)
        self.assertEqual(
            [call.kwargs["commit_message"] for call in genai.register_prompt.call_args_list],
            [
                f"Prompt technique: {technique}"
                for technique, prompts in TECHNIQUE_PROMPTS.items()
                for _ in prompts
            ],
        )


if __name__ == "__main__":
    unittest.main()
