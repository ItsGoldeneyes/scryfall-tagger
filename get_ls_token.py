"""
One-time helper: log into Label Studio with username/password and save
the session cookie to .ls_session so all scripts can use it.

Run this whenever your session expires (~2 weeks by default):
  python get_ls_token.py
"""

import getpass
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

LS_URL = os.environ["LABEL_STUDIO_URL"]
SESSION_FILE = Path(__file__).parent / ".ls_session"


def login(username: str, password: str) -> requests.Session:
    session = requests.Session()

    r = session.get(f"{LS_URL}/user/login/", timeout=10)
    r.raise_for_status()
    match = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', r.text)
    if not match:
        raise RuntimeError("Could not find CSRF token on login page")
    csrf = match.group(1)

    r = session.post(
        f"{LS_URL}/user/login/",
        data={"csrfmiddlewaretoken": csrf, "email": username, "password": password},
        headers={"Referer": f"{LS_URL}/user/login/"},
        timeout=10,
        allow_redirects=True,
    )
    r.raise_for_status()

    if "/user/login/" in r.url:
        raise RuntimeError("Login failed — check your credentials")

    return session


def save_session(session: requests.Session) -> None:
    cookies = {c.name: c.value for c in session.cookies}
    if "sessionid" not in cookies:
        raise RuntimeError("No sessionid cookie after login")
    SESSION_FILE.write_text(cookies["sessionid"])
    print(f"Session saved to {SESSION_FILE}")


if __name__ == "__main__":
    print(f"Label Studio: {LS_URL}")
    username = input("Email/username: ")
    password = getpass.getpass("Password: ")

    try:
        session = login(username, password)
        save_session(session)
        r = session.get(f"{LS_URL}/api/projects/", timeout=10)
        r.raise_for_status()
        projects = r.json()
        count = len(projects.get("results", projects))
        print(f"Auth verified — {count} project(s) visible.")
        print("You can now run: python snapshot_export.py")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
