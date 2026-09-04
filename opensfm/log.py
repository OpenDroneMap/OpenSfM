# pyre-strict
import logging
import os
from typing import Optional

import vmem


def setup() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s", level=logging.DEBUG, force=True
    )


def memory_available() -> Optional[int]:
    """Available memory in MB.
    """
    return vmem.virtual_memory().available / 1024 / 1024
