import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILES = [
    Path("examples/quickstart.py"),
    Path("examples/direct_inference.py"),
    Path("examples/speaker_diarization.py"),
    Path("examples/vllm_batch.py"),
    Path("examples/streaming_sdk.py"),
]
README_EXAMPLE_LINKS = {
    "README.md": "[Runnable examples](examples/README.md)",
    "README_zh.md": "[可运行示例脚本](examples/README.md)",
    "README_ja.md": "[実行可能なサンプル](examples/README.md)",
    "README_ko.md": "[실행 가능한 예제](examples/README.md)",
}


def has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
            return True
    return False


class RunnableExamplesSmokeTest(unittest.TestCase):
    def test_documented_examples_exist(self):
        missing = [str(path) for path in EXAMPLE_FILES if not (ROOT / path).is_file()]
        self.assertEqual([], missing)

    def test_examples_parse_and_define_main(self):
        for rel_path in EXAMPLE_FILES:
            with self.subTest(example=str(rel_path)):
                source = (ROOT / rel_path).read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(rel_path))
                top_level_functions = {
                    node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                }
                self.assertIn("main", top_level_functions)
                self.assertTrue(has_main_guard(tree))

    def test_readmes_point_to_runnable_examples(self):
        for readme_path, marker in README_EXAMPLE_LINKS.items():
            with self.subTest(readme=readme_path):
                readme = (ROOT / readme_path).read_text(encoding="utf-8")
                self.assertIn(marker, readme)

    def test_readmes_do_not_show_stale_vllm_api(self):
        stale_snippets = [
            "from funasr import AutoModelVLLM",
            'AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda", dtype="bf16")',
            'model.generate(input="audio.wav", batch_size=32)',
        ]
        for readme_path in README_EXAMPLE_LINKS:
            readme = (ROOT / readme_path).read_text(encoding="utf-8")
            for snippet in stale_snippets:
                with self.subTest(readme=readme_path, snippet=snippet):
                    self.assertNotIn(snippet, readme)

    def test_localized_readmes_surface_nano_gguf_edge_path(self):
        required_links = [
            "https://www.funasr.com/llama-cpp.html",
            "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF",
            "https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF",
        ]
        for readme_path in ("README_ja.md", "README_ko.md"):
            readme = (ROOT / readme_path).read_text(encoding="utf-8")
            for link in required_links:
                with self.subTest(readme=readme_path, link=link):
                    self.assertIn(link, readme)
