"""Application package.

Loading .env here, at package import, is the only place it happens. Python does
not read .env on its own and neither does `uv run`, so without this every
os.getenv() in the project returns None no matter what the file says -- which
looks exactly like a missing key and sends you hunting in the wrong place.

The path is resolved from __file__ rather than the working directory, because
`uv run` from a subdirectory would otherwise silently find nothing.

override=False is deliberate: a real environment variable beats the file. That is
what lets CI and Render inject secrets without a .env existing at all, and it is
why the tests can monkeypatch keys without the developer's own key leaking in.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
