import toml
import os


# Return the config as dict such as defined on the config.yaml file
def get_config() -> dict:
    config_path = os.getenv("CONFIG_FILE")

    with open(config_path, "r") as file:
        config = toml.load(file)

    return config