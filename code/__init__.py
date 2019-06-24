import os

from pathlib import Path

VECTOR_SIZE = 300
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def get_path(switch):
    if switch:
        return str(Path(os.path.dirname(__file__).replace('code', switch)))
    return Path(os.path.dirname(__file__))



