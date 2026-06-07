import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


@pytest.fixture
def openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return OpenAI(
        base_url=os.environ.get(
            "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
        ),
        api_key=api_key,
    )


@pytest.fixture
def teacher_model() -> str:
    return os.environ.get("TEACHER_MODEL", "deepseek/deepseek-v4-flash")
