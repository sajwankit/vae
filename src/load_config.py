import yaml

class ConfigLoader:
    """Loads and provides access to configuration settings from a YAML file."""

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Loads YAML configuration file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise Exception(f"Config file '{self.config_path}' not found!")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML: {e}")

    def get(self, key, default=None):
        """Retrieves a configuration value using dot notation, e.g., get('features.rolling_window')."""
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default  # Return default value if key not found