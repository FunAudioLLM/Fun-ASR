import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DOCS = [
    "README.md",
    "docs/finetune.md",
    "docs/finetune_zh.md",
    "docs/vllm_guide.md",
    "docs/vllm_guide_zh.md",
    "examples/README.md",
]


def test_funasr_requirement_uses_current_release_floor():
    requirements = (ROOT / "requirements.txt").read_text()
    assert "funasr>=1.3.23" in requirements
    assert "funasr>=1.3.0" not in requirements
    assert "funasr>=1.3.19" not in requirements


def test_docs_use_quoted_current_funasr_install_commands():
    for relpath in DOCS:
        text = (ROOT / relpath).read_text()
        assert "funasr>=1.3.0" not in text
        assert "funasr>=1.3.3" not in text
        assert "funasr>=1.3.19" not in text
        assert not re.search(r"pip install funasr>=", text)

    assert '"funasr>=1.3.23"' in (ROOT / "README.md").read_text()
    assert (ROOT / "examples/README.md").read_text().count('"funasr>=1.3.23"') == 2
