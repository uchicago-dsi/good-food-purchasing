import pytest

# from scripts.pipeline import create_pareser
# TODO: import as package?
from scripts.pipeline_v2 import create_parser as create_parser_v2


@pytest.fixture
def v2_arg_parser():
    return create_parser_v2()
