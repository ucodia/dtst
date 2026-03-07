import os

DEFAULT_USER_AGENT = "dtst/0.1 (image dataset toolkit; +https://github.com/ucodia/dtst)"


def get_user_agent() -> str:
    return os.environ.get("DTST_USER_AGENT", DEFAULT_USER_AGENT)
