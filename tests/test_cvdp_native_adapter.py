from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "skills" / "ace-rtl" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from ace_cvdp_native.cli import (  # noqa: E402
    normalize_agentic_row_for_native,
    resolve_target_files,
    select_generation_processor_class,
)


class CVDPNativeProcessorRoutingTest(unittest.TestCase):
    def test_native_generation_always_uses_copilot_processor(self):
        class CopilotProcessor:
            pass

        class AgenticProcessor:
            pass

        module = SimpleNamespace(
            CopilotProcessor=CopilotProcessor,
            AgenticProcessor=AgenticProcessor,
        )
        self.assertIs(select_generation_processor_class(module), CopilotProcessor)

    def test_missing_copilot_processor_fails_closed(self):
        with self.assertRaisesRegex(AttributeError, "CopilotProcessor"):
            select_generation_processor_class(SimpleNamespace(AgenticProcessor=object))


class CVDPAgenticSchemaAdapterTest(unittest.TestCase):
    def test_cid003_multifile_targets_come_only_from_patch(self):
        row = {
            "id": "cvdp_agentic_DES_0001",
            "categories": ["cid003", "hard"],
            "prompt": "Implement DES.",
            "context": {
                "docs/spec.md": "public specification",
                "verif/tb_des.sv": "module tb_des; endmodule",
            },
            "patch": {
                "rtl/S1.sv": "REFERENCE_CONTENT_MUST_NOT_LEAK",
                "rtl/des_enc.sv": "ANOTHER_REFERENCE_SECRET",
            },
            "harness": {"docker-compose.yml": "services: {}"},
        }

        normalized = normalize_agentic_row_for_native(row)

        self.assertEqual(
            resolve_target_files(normalized),
            ["rtl/S1.sv", "rtl/des_enc.sv"],
        )
        self.assertEqual(
            normalized["output"]["context"],
            {"rtl/S1.sv": "", "rtl/des_enc.sv": ""},
        )
        self.assertEqual(normalized["input"]["context"], row["context"])
        self.assertNotIn(
            "REFERENCE_CONTENT_MUST_NOT_LEAK",
            str(normalized["input"]) + str(normalized["output"]),
        )

    def test_cid005_reuse_modules_remain_immutable_context(self):
        row = {
            "id": "cvdp_agentic_64b66b_codec_0001",
            "categories": ["cid005", "hard"],
            "prompt": "Compose the supplied encoder and decoder.",
            "context": {
                "rtl/encoder_data_64b66b.sv": "module encoder_data_64b66b; endmodule",
                "rtl/decoder_data_control_64b66b.sv": "module decoder_data_control_64b66b; endmodule",
                "docs/specification.md": "public specification",
            },
            "patch": {"rtl/top_64b66b_codec.sv": ""},
            "harness": {"docker-compose.yml": "services: {}"},
        }

        normalized = normalize_agentic_row_for_native(row)

        self.assertEqual(resolve_target_files(normalized), ["rtl/top_64b66b_codec.sv"])
        self.assertEqual(normalized["input"]["context"], row["context"])
        self.assertNotIn("rtl/encoder_data_64b66b.sv", normalized["output"]["context"])
        self.assertNotIn(
            "rtl/decoder_data_control_64b66b.sv",
            normalized["output"]["context"],
        )

    def test_context_is_never_used_as_target_fallback(self):
        with self.assertRaisesRegex(ValueError, "no declared generated targets"):
            resolve_target_files(
                {
                    "id": "broken_cid005",
                    "context": {"rtl/reusable_child.sv": "module reusable_child; endmodule"},
                }
            )

    def test_patch_and_output_target_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            resolve_target_files(
                {
                    "id": "broken_cid003",
                    "patch": {"rtl/top.sv": ""},
                    "output": {"context": {"rtl/helper.sv": ""}},
                }
            )


if __name__ == "__main__":
    unittest.main()
