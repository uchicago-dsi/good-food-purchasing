import sys
from pathlib import Path

import pytest

# TODO: move this to the package to avoid import issue — student code for now
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.pipeline import create_parser as create_parser_v2


@pytest.fixture
def v2_arg_parser():
    return create_parser_v2()
