import os
import sys
import traceback
from dotenv import load_dotenv

ENV_FILE = os.environ.get("ENV_FILE", ".env.development")
assert (
    os.path.exists(ENV_FILE) is True
), f"[ENV_VALIDATOR_ERROR]: Environment file {ENV_FILE} not found."
load_dotenv(dotenv_path=ENV_FILE)


ENVIRONMENT = os.getenv("ENVIRONMENT")
assert ENVIRONMENT in [
    "development",
    "test",
    "staging",
    "production",
], "[ENV_VALIDATOR_ERROR]: Invalid ENVIRONMENT value."
# if ENVIRONMENT not in ('development', 'test', 'staging', 'production'):
#     print('[ENV_VALIDATOR_WARNING]: Invalid ENVIRONMENT value. Fallback to "development".')
#     ENVIRONMENT = 'development'


def get_var(env_var: str, default_var=None):
    if default_var is not None:
        return os.getenv(env_var, default_var)
    raw = os.getenv(env_var)
    assert raw is not None, f"[ENV_VALIDATOR_ERROR]: {env_var} is not defined."
    return raw


def get_bool(env_var: str, default_var=True):
    return os.getenv(env_var, str(default_var)).lower() in ("true", "1", "True")


def get_float(env_var: str, default_var=1.0):
    try:
        raw = os.getenv(env_var, default_var)
        value = float(raw)
        return value
    except Exception as err:
        print(err, traceback.format_exc())
        sys.exit(1)


def get_int(env_var: str, default_var=1):
    try:
        raw = os.getenv(env_var, default_var)
        value = int(raw)
        return value
    except Exception as err:
        print(err, traceback.format_exc())
        sys.exit(1)
