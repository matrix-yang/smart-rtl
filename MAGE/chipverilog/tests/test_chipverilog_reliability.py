import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

from mage.gen_config import DeepSeekOpenAI
from mage.rtl_editor import EditorOutputError, RTLEditor
from mage.rtl_generator import RTLGenerator
from mage.token_counter import TokenCount, TokenCounter
from mage.utils import extract_json_object


class FakeTokenCounter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.events = []
        self.requests = []

    def set_cur_tag(self, tag):
        self.tag = tag

    def count_chat(self, messages, max_output_tokens=None):
        self.requests.append((messages, max_output_tokens))
        return self.responses.pop(0), TokenCount(in_token_cnt=10, out_token_cnt=5)

    def record_event(self, event, **payload):
        self.events.append({"event": event, **payload})


class PassingReviewer:
    def review_for_edit(self):
        return {
            "is_syntax_pass": True,
            "is_sim_pass": True,
            "error_msg": "",
            "sim_mismatch_cnt": 0,
        }


class GeneratorCounter:
    def set_cur_tag(self, tag):
        self.tag = tag


class MalformedRTLGenerator(RTLGenerator):
    def generate(self, messages):
        return response("not-json", "stop")


def response(content: str, finish_reason: str) -> ChatResponse:
    return ChatResponse(
        message=ChatMessage(content=content, role=MessageRole.ASSISTANT),
        raw={"choices": [{"finish_reason": finish_reason}]},
    )


class RTLEditorReliabilityTest(unittest.TestCase):
    def test_deepseek_per_call_output_cap_overrides_generator_cap(self):
        llm = DeepSeekOpenAI(
            model="deepseek-v4-pro",
            api_key="dummy",
            api_base="https://api.deepseek.com/v1",
            max_tokens=16384,
        )
        self.assertEqual(llm._get_model_kwargs(max_tokens=4096)["max_tokens"], 4096)

    def test_truncated_json_is_retried_with_compact_patch(self):
        truncated = response('{"reasoning":"unterminated', "length")
        valid = response(
            json.dumps(
                {
                    "reasoning": "fix constant",
                    "action_input": {
                        "command": "replace_content_by_matching",
                        "args": {
                            "old_content": "assign y = 1'b0;",
                            "new_content": "assign y = 1'b1;",
                        },
                    },
                }
            ),
            "stop",
        )
        counter = FakeTokenCounter([truncated, valid])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "rtl.sv").write_text(
                "module t(output y);\nassign y = 1'b0;\nendmodule\n",
                encoding="utf-8",
            )
            editor = RTLEditor(counter, PassingReviewer())
            editor.max_trials = 1
            passed, rtl = editor.chat("spec", str(output), "failed", 100)

        self.assertTrue(passed)
        self.assertIn("assign y = 1'b1;", rtl)
        self.assertEqual([request[1] for request in counter.requests], [4096, 4096])
        self.assertEqual(counter.events[0]["event"], "editor_output_error")
        self.assertEqual(counter.events[0]["reason"], "truncated_response")
        retry_messages = counter.requests[1][0]
        self.assertIn(
            "previous editor response was rejected", retry_messages[-1].content
        )

    def test_oversized_patch_is_rejected(self):
        oversized = response(
            json.dumps(
                {
                    "reasoning": "replace too much",
                    "action_input": {
                        "command": "replace_content_by_matching",
                        "args": {"old_content": "x" * 20, "new_content": "y"},
                    },
                }
            ),
            "stop",
        )
        editor = RTLEditor(FakeTokenCounter([]), PassingReviewer())
        editor.max_patch_chars = 10
        with self.assertRaises(EditorOutputError) as raised:
            editor.parse_output(oversized)
        self.assertEqual(raised.exception.reason, "patch_too_large")

    def test_unexpected_action_argument_is_retried(self):
        invalid = response(
            json.dumps(
                {
                    "reasoning": "wrong schema",
                    "action_input": {
                        "command": "replace_content_by_matching",
                        "args": {
                            "old_content": "assign y = 1'b0;",
                            "new_content": "assign y = 1'b1;",
                            "replace_with": "assign y = 1'b1;",
                        },
                    },
                }
            ),
            "stop",
        )
        valid = response(
            json.dumps(
                {
                    "reasoning": "correct schema",
                    "action_input": {
                        "command": "replace_content_by_matching",
                        "args": {
                            "old_content": "assign y = 1'b0;",
                            "new_content": "assign y = 1'b1;",
                        },
                    },
                }
            ),
            "stop",
        )
        counter = FakeTokenCounter([invalid, valid])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "rtl.sv").write_text(
                "module t(output y);\nassign y = 1'b0;\nendmodule\n",
                encoding="utf-8",
            )
            editor = RTLEditor(counter, PassingReviewer())
            editor.max_trials = 1
            passed, rtl = editor.chat("spec", str(output), "failed", 100)

        self.assertTrue(passed)
        self.assertIn("assign y = 1'b1;", rtl)
        self.assertEqual(counter.events[0]["reason"], "invalid_args")
        self.assertIn("unexpected=['replace_with']", counter.events[0]["detail"])

    def test_llm_telemetry_records_finish_reason_and_cap(self):
        counter = object.__new__(TokenCounter)
        counter.cur_tag = "RTLEditor"
        counter.llm = SimpleNamespace(
            metadata=SimpleNamespace(model_name="mock-deepseek")
        )
        truncated = response("partial", "length")
        with tempfile.TemporaryDirectory() as temp_dir:
            telemetry = Path(temp_dir) / "telemetry.jsonl"
            previous = os.environ.get("MAGE_TELEMETRY_FILE")
            os.environ["MAGE_TELEMETRY_FILE"] = str(telemetry)
            try:
                counter._record_call(
                    [],
                    truncated,
                    TokenCount(in_token_cnt=10, out_token_cnt=4096),
                    latency=1.0,
                    async_call=False,
                    requested_max_output_tokens=4096,
                )
            finally:
                if previous is None:
                    os.environ.pop("MAGE_TELEMETRY_FILE", None)
                else:
                    os.environ["MAGE_TELEMETRY_FILE"] = previous
            event = json.loads(telemetry.read_text())
        self.assertEqual(event["finish_reason"], "length")
        self.assertTrue(event["response_truncated"])
        self.assertEqual(event["requested_max_output_tokens"], 4096)


class RTLGeneratorReliabilityTest(unittest.TestCase):
    def test_reasoning_marker_before_json_is_accepted(self):
        parsed = extract_json_object(
            '</think>\n\n{"reasoning":"ok","module":"module top; endmodule"}'
        )
        self.assertEqual(parsed["module"], "module top; endmodule")

    def test_malformed_only_response_returns_failed_empty_candidate(self):
        generator = MalformedRTLGenerator(GeneratorCounter())
        generator.max_trials = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            syntax_ok, rtl = generator.chat(
                "Generate module x.", "", "", str(Path(temp_dir) / "rtl.sv")
            )
        self.assertFalse(syntax_ok)
        self.assertEqual(rtl, "")


class ResumeReliabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script_path = Path(__file__).parents[1] / "scripts" / "run_chipverilog.py"
        spec = importlib.util.spec_from_file_location("run_chipverilog", script_path)
        cls.runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.runner)

    def test_completed_task_is_detected_and_partial_task_is_archived(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            completed = output_root / "chipverilog_done"
            completed.mkdir()
            (completed / "rtl.sv").write_text("module done; endmodule\n")
            summary = {
                "task_id": "done",
                "passed": False,
                "status": "function_fail",
                "input_tokens": 1,
                "output_tokens": 1,
                "golden_tool_calls": 1,
            }
            (completed / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(
                self.runner.load_completed_summary(completed, "done")["status"],
                "function_fail",
            )

            partial = output_root / "chipverilog_partial"
            partial.mkdir()
            (partial / "rtl.sv").write_text("partial")
            archived = self.runner.archive_existing_task_dir(output_root, "partial")
            self.assertIsNotNone(archived)
            self.assertTrue(archived.is_dir())
            self.assertFalse(partial.exists())

    def test_task_shards_are_disjoint_and_complete(self):
        tasks = list(range(64))
        shards = [self.runner.select_shard(tasks, 2, index) for index in range(2)]
        self.assertEqual(len(shards[0]), 32)
        self.assertEqual(len(shards[1]), 32)
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual(sorted(shards[0] + shards[1]), tasks)


if __name__ == "__main__":
    unittest.main()
