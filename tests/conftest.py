import sys

# On Windows + Python 3.13, importing `trl` after `peft`/`transformers`
# triggers a segfault during pytest collection when test files
# happen to load in the wrong order. Touch them at session start
# to force the safe ordering.
try:
    import datasets  # noqa: F401
    import trl  # noqa: F401
except Exception:
    pass

# Skip the Playwright suite during normal `pytest tests` runs because the
# pytest-playwright plugin grabs the asyncio event loop in a way that
# breaks our many `@pytest.fixture` async fixtures. Run e2e explicitly:
#
#     pytest tests/e2e --browser chromium -o addopts=
#
# The condition below detects an explicit e2e target and lets pytest
# discover those tests normally.
_running_e2e = any("e2e" in arg for arg in sys.argv[1:])
if not _running_e2e:
    collect_ignore_glob = ["e2e/*"]
