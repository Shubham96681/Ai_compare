"""JSON utilities: validation and safe parsing."""

import json
from typing import Any, Tuple


def is_valid_json(text: str) -> Tuple[bool, Any]:
    try:
        obj = json.loads(text)
        return True, obj
    except Exception as e:  # pragma: no cover
        return False, str(e)


