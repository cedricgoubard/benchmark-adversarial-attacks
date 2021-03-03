"""
config module
-------------

This module enables the user to change some main features of the project
"""

import yaml
from box import Box

CFG = None


def load_cfg(cfg_path):
    """Loads config information from the yaml file.

    :param cfg_path: path the to yaml config file
    :type cfg_path: str
    """
    global CFG  # pylint: disable=invalid-name,global-statement
    with open(cfg_path, "r") as ymlfile:
        CFG = Box(yaml.safe_load(ymlfile))
    return CFG


def get_cfg(cfg_path=None):
    """If this function has already been called with the path to the config file, returns the
    config object; otherwise, loads info from the conf file if a path is provided, or raises and
    error.

    :param cfg_path: path to the yaml config file, defaults to None
    :type cfg_path: str, optional
    :raises ValueError: if this function was never called before and no path to conf file is
    provided
    :return: the config Box object
    :rtype: box.Box
    """
    global CFG  # pylint: disable=global-statement
    if CFG is not None:
        return CFG
    if cfg_path:
        return load_cfg(cfg_path)
    raise ValueError(
        "cfg file path was never provided; you need to call get_cfg(cfg_path) at least once."
    )
