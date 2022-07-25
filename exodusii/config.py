import os
from argparse import Namespace


def env_boolean(var, default=None):
    value = os.getenv(var, default)
    if value is None:
        return None
    if value.lower() in ("false", "0", "off", ""):
        return False
    return True


def initialize_config():
    cfg = Namespace()
    cfg.use_netcdf4_if_possible = env_boolean("SIMIO_NETCDF4", "on")
    cfg.debug = env_boolean("SIMIO_DEBUG", "off")
    return cfg


config = initialize_config()
