from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def get_auth_header() -> str:
    """Return the Authorization header value for Label Studio API calls."""
    scheme = os.environ.get("LABEL_STUDIO_AUTH_SCHEME", "Token")
    token = os.environ["LABEL_STUDIO_TOKEN"]
    return f"{scheme} {token}"
