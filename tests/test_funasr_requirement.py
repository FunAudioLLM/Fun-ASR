import re
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]


DOCS = [
    "README.md",
    "README_zh.md",
    "docs/finetune.md",
    "docs/finetune_zh.md",
    "docs/vllm_guide.md",
    "docs/vllm_guide_zh.md",
    "examples/README.md",
]


def test_funasr_requirement_uses_current_release_floor():
    requirements = (ROOT / "requirements.txt").read_text()
    assert "funasr>=1.3.26" in requirements
    assert "funasr>=1.3.0" not in requirements
    assert "funasr>=1.3.19" not in requirements
    assert "funasr>=1.3.23" not in requirements


def test_docs_use_quoted_current_funasr_install_commands():
    for relpath in DOCS:
        text = (ROOT / relpath).read_text()
        assert "funasr>=1.3.0" not in text
        assert "funasr>=1.3.3" not in text
        assert "funasr>=1.3.19" not in text
        assert "funasr>=1.3.23" not in text
        assert not re.search(r"pip install funasr>=", text)

    assert '"funasr>=1.3.26"' in (ROOT / "README.md").read_text()
    assert (ROOT / "examples/README.md").read_text().count('"funasr>=1.3.26"') == 2


def test_readmes_surface_funasr_1327_nano_serving_release():
    required = [
        "funasr==1.3.27",
        "AutoModel",
        "vLLM",
        "https://github.com/modelscope/FunASR/releases/tag/v1.3.27",
    ]
    guides = {
        "README.md": "https://www.funasr.com/en/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html",
        "README_zh.md": "https://www.funasr.com/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html",
        "README_ja.md": "https://www.funasr.com/en/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html",
        "README_ko.md": "https://www.funasr.com/en/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html",
    }
    for relpath, guide in guides.items():
        text = (ROOT / relpath).read_text()
        for marker in [*required, guide]:
            assert marker in text, f"{relpath} is missing {marker}"


def test_docs_relative_markdown_links_point_to_existing_files():
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for relpath in DOCS:
        doc_path = ROOT / relpath
        for target in link_pattern.findall(doc_path.read_text()):
            parsed = urlparse(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            link_path = unquote(parsed.path)
            if not link_path or link_path.startswith("#"):
                continue
            resolved = (doc_path.parent / link_path).resolve()
            assert resolved.exists(), f"{relpath} links to missing file: {target}"
