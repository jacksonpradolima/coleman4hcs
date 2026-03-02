"""
config.config - Configuration Utilities.

Provides utilities to retrieve and parse the configuration for the application.
The configuration settings are expected to be defined in a TOML file. The path
to this configuration file should be specified using the ``CONFIG_FILE``
environment variable.

Functions
---------
get_config
    Returns the application configuration as a dictionary.

Notes
-----
Ensure the ``CONFIG_FILE`` environment variable is set to the path of your
configuration file before using functions from this module.
"""
import os

import toml


def get_config() -> dict:
    """Return the configuration as a dictionary from the config TOML file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the CONFIG_FILE environment variable is not set or the file does
        not exist.
    """
    config_path = os.getenv("CONFIG_FILE")

    if config_path is None:
        raise FileNotFoundError("Environment variable CONFIG_FILE is not set.")

    with open(config_path, "r", encoding='utf-8') as file:
        config = toml.load(file)

    return config
