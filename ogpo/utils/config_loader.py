"""YAML-based configuration loader with inheritance and CLI override support."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class AttrDict(dict):
    """Dictionary that supports attribute-style access.

    This allows accessing config values as config.key instead of config['key'].
    Nested dicts are also converted to AttrDict for recursive attribute access.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, AttrDict):
                self[key] = AttrDict(value)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def copy(self):
        """Return a shallow copy as AttrDict."""
        return AttrDict(super().copy())

    def to_dict(self) -> dict:
        """Convert back to regular dict (recursive)."""
        result = {}
        for key, value in self.items():
            if isinstance(value, AttrDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def _try_convert_scientific_notation(value: str):
    """Try to convert a string that looks like scientific notation to float."""
    import re
    if re.match(r'^-?\d+(\.\d+)?[eE][+-]?\d+$', value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in a string ($VAR and ${VAR} syntax)."""
    return os.path.expandvars(value)


def _to_attr_dict(d: Dict) -> AttrDict:
    """Recursively convert a dict to AttrDict, normalizing value types."""
    # Keys whose list values should be tuples (hidden dimensions).
    TUPLE_KEYS = {
        'actor_hidden_dims', 'value_hidden_dims', 'hidden_dims',
        'noise_net_hidden_dims', 'edit_hidden_dims',
        'action_critic_hidden_dims', 'edit_actor_hidden_dims',
        'simba_actor_hidden_dims', 'simba_critic_hidden_dims',
    }

    result = AttrDict()
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _to_attr_dict(value)
        elif isinstance(value, list) and key in TUPLE_KEYS:
            result[key] = tuple(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float):
                result[key] = float(value)
            else:
                result[key] = int(value)
        elif isinstance(value, str):
            expanded = _expand_env_vars(value)
            result[key] = _try_convert_scientific_notation(expanded)
        else:
            result[key] = value
    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested_value(d: Dict, key_path: str, value: Any) -> None:
    """Set a nested dictionary value using dot notation (e.g. 'agent.lr')."""
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _get_nested_value(d: Dict, key_path: str, default: Any = None) -> Any:
    """Get a nested dictionary value using dot notation (e.g. 'agent.lr')."""
    keys = key_path.split('.')
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _parse_value(value_str: str) -> Any:
    """Parse a string value to int, float, bool, None, list, or string."""
    if value_str.lower() in ('none', 'null'):
        return None

    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    if (value_str.startswith('[') and value_str.endswith(']')) or \
       (value_str.startswith('(') and value_str.endswith(')')):
        inner = value_str[1:-1]
        if not inner:
            return []
        items = [_parse_value(item.strip()) for item in inner.split(',')]
        return items

    return value_str


def _get_config_dir() -> Path:
    """Get the path to the configs/algos directory."""
    this_file = Path(__file__).resolve()
    config_dir = this_file.parent.parent / 'configs' / 'algos'
    if config_dir.exists():
        return config_dir

    config_dir = Path.cwd() / 'ogpo' / 'configs' / 'algos'
    if config_dir.exists():
        return config_dir

    raise FileNotFoundError(
        f"Could not find configs/algos directory. Searched:\n"
        f"  - {this_file.parent.parent / 'configs' / 'algos'}\n"
        f"  - {Path.cwd() / 'ogpo' / 'configs' / 'algos'}"
    )


def _load_yaml_with_inheritance(yaml_path: Path) -> Dict:
    """Load a YAML file and resolve _base_ inheritance."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    if '_base_' in config:
        base_name = config.pop('_base_')
        base_path = yaml_path.parent / base_name
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        base_config = _load_yaml_with_inheritance(base_path)
        config = _deep_merge(base_config, config)

    return config


def load_config(
    algo_name: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None,
) -> Dict:
    """Load algorithm configuration with inheritance and CLI overrides."""
    if config_dir is None:
        config_dir = _get_config_dir()

    yaml_path = config_dir / f'{algo_name}.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {yaml_path}\n"
            f"Available configs: {[f.stem for f in config_dir.glob('*.yaml')]}"
        )

    config = _load_yaml_with_inheritance(yaml_path)

    if cli_overrides:
        for key_path, value in cli_overrides.items():
            _set_nested_value(config, key_path, value)

    return config


def parse_cli_args(argv: Optional[List[str]] = None) -> tuple:
    """Parse command line arguments into (algo_name, config_overrides)."""
    parser = argparse.ArgumentParser(
        description='OGPO Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--algo', '--agent', '-a',
        type=str,
        default='ogpo',
        help='Algorithm name (ogpo, qc, bptt, dsrl, expo, dsrl_plus_expo)'
    )

    args, remaining = parser.parse_known_args(argv)

    # Remaining args are config overrides in --key=value or --key value form.
    overrides = {}
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith('--'):
            key = arg[2:]
            if '=' in key:
                key, value = key.split('=', 1)
                overrides[key] = _parse_value(value)
            elif i + 1 < len(remaining) and not remaining[i + 1].startswith('--'):
                overrides[key] = _parse_value(remaining[i + 1])
                i += 1
            else:
                overrides[key] = True
        i += 1

    return args.algo, overrides


def flatten_config(config: Dict, prefix: str = '') -> Dict[str, Any]:
    """Flatten a nested config dict to dot notation keys."""
    result = {}
    for key, value in config.items():
        full_key = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten_config(value, full_key))
        else:
            result[full_key] = value
    return result


def config_to_agent_dict(config: Dict) -> AttrDict:
    """Extract the agent configuration as an AttrDict for agent.create()."""
    if hasattr(config, 'get'):
        agent_config = dict(config.get('agent', {}))
    else:
        agent_config = dict(config.agent) if hasattr(config, 'agent') else {}

    if 'horizon_length' not in agent_config:
        env_config = config.get('env', {}) if hasattr(config, 'get') else {}
        agent_config['horizon_length'] = env_config.get('horizon_length', 4)
    if 'discount' not in agent_config:
        agent_config['discount'] = agent_config.get('discount', 0.99)

    return _to_attr_dict(agent_config)


def print_config(config: Dict, indent: int = 0) -> None:
    """Pretty print configuration."""
    for key, value in sorted(config.items()):
        prefix = '  ' * indent
        if isinstance(value, dict):
            print(f'{prefix}{key}:')
            print_config(value, indent + 1)
        else:
            print(f'{prefix}{key}: {value}')
