import os
from types import SimpleNamespace


def env_boolean(var, default=None):
    value = os.getenv(var, default)
    if value is None:
        return default
    if value.lower() in ("false", "0", "off", ""):
        return False
    return True


def initialize_config():
    cfg = SimpleNamespace()
    cfg.use_netcdf4_if_possible = env_boolean("EXODUSII_USE_NETCDF4", default="on")
    cfg.debug = env_boolean("EXODUSII_DEBUG", default="off")
    return cfg


config = initialize_config()
