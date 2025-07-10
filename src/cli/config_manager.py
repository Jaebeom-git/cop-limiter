import os
import json
import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    """Recursively convert a dictionary to a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

class ConfigManager:
    def __init__(self, config_path: str = None):
        """
        Initialize ConfigManager with an optional config path.
        :param config_path: Path to the configuration file.
        """
        self.config_path = config_path
        self.config = None
        if config_path:
            # Automatically determine file type from extension
            file_format = self._infer_format(config_path)
            self.load(config_path, file_format=file_format)

    def load(self, config_path: str, file_format: str = None) -> SimpleNamespace:
        """
        Load configuration from a JSON or YAML file and convert it to SimpleNamespace.
        :param config_path: Path to the configuration file.
        :param file_format: 'json' or 'yaml'. If None, inferred from _infer_format().
        :return: SimpleNamespace containing the loaded configuration.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        if not file_format:
            file_format = self._infer_format(config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            if file_format == 'json':
                config_data = json.load(f)
            elif file_format == 'yaml':
                config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

        self.config = dict_to_namespace(config_data)
        self.config_path = config_path
        return self.config

    def init_config(self, config_data: dict):
        """
        Initialize the configuration using a dictionary.
        :param config_data: Dictionary containing configuration settings.
        :return: SimpleNamespace containing the initialized configuration.
        """
        self.config = dict_to_namespace(config_data)
        return self.config

    def save(self, config_path: str = None, file_format: str = None):
        """
        Save the current configuration to a JSON or YAML file.
        :param config_path: Path to the configuration file. If not provided, saves to the original path.
        :param file_format: 'json' or 'yaml'. If None, inferred from _infer_format().
        """
        if not self.config:
            raise ValueError("Config not loaded. Load or initialize the config first.")

        if not config_path:
            if not self.config_path:
                raise ValueError("Config path must be provided to save the file.")
            config_path = self.config_path

        if not file_format:
            file_format = self._infer_format(config_path)

        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)

        config_dict = self.to_dict()

        with open(config_path, 'w', encoding='utf-8') as f:
            if file_format == 'json':
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            elif file_format == 'yaml':
                yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

    def update(self, **kwargs):
        """
        Update the configuration with new values.
        :param kwargs: Key-value pairs to update in the configuration.
        """
        if not self.config:
            raise ValueError("Config not loaded. Load or initialize the config first.")

        for key, value in kwargs.items():
            setattr(self.config, key, value)

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.
        :return: Dictionary representation of the configuration.
        """
        if not self.config:
            raise ValueError("Config not loaded.")
        
        def namespace_to_dict(obj):
            if isinstance(obj, SimpleNamespace):
                return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, list):
                return [namespace_to_dict(item) for item in obj]
            else:
                return obj

        return namespace_to_dict(self.config)

    def print(self, format: str = 'json'):
        """
        Print the current configuration in JSON or YAML format.
        :param format: 'json' or 'yaml'.
        """
        if not self.config:
            print("Config not loaded.")
        else:
            config_dict = self.to_dict()
            if format == 'json':
                print(json.dumps(config_dict, indent=4, ensure_ascii=False))
            elif format == 'yaml':
                print(yaml.dump(config_dict, sort_keys=False, default_flow_style=False, allow_unicode=True))
            else:
                print("Unsupported format. Use 'json' or 'yaml'.")

    def _infer_format(self, path: str) -> str:
        """
        Private method to infer file format (json or yaml) from file extension.
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext in ['.yml', '.yaml']:
            return 'yaml'
        elif ext == '.json':
            return 'json'
        else:
            # Assume JSON by default or handle exception
            return 'json'