from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

SESSION_FILE = Path(__file__).resolve().parent / ".ls_session"


def make_session():
    """Return a requests.Session configured for Label Studio API calls.

    Prefers session-cookie auth (from .ls_session) over the static token,
    since Label Studio may have legacy token auth disabled.
    """
    import requests

    session = requests.Session()

    if SESSION_FILE.exists():
        session.cookies.set("sessionid", SESSION_FILE.read_text().strip())
        url = os.environ["LABEL_STUDIO_URL"]
        # Fetch an HTML page so Django sets the csrftoken cookie
        session.get(f"{url}/", timeout=10)
        csrftoken = session.cookies.get("csrftoken", "")
        if csrftoken:
            session.headers["X-CSRFToken"] = csrftoken
        # Verify session is still valid
        r = session.get(f"{url}/api/projects/", timeout=10)
        if r.status_code == 200:
            return session

    # Fall back to token auth
    scheme = os.environ.get("LABEL_STUDIO_AUTH_SCHEME", "Token")
    token = os.environ["LABEL_STUDIO_TOKEN"]
    session.headers["Authorization"] = f"{scheme} {token}"
    return session


def get_auth_header() -> str:
    """Return the Authorization header value (used by scripts that build their own session)."""
    scheme = os.environ.get("LABEL_STUDIO_AUTH_SCHEME", "Token")
    token = os.environ["LABEL_STUDIO_TOKEN"]
    return f"{scheme} {token}"
