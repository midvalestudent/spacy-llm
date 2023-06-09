import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from confection import Config
from spacy.language import Language
from spacy.util import SimpleFrozenDict, get_sourced_components, load_config
from spacy.util import load_model_from_config


def split_labels(labels: Union[str, Iterable[str]]) -> List[str]:
    """Split a comma-separated list of labels.
    If input is a list already, just strip each entry of the list

    labels (Union[str, Iterable[str]]): comma-separated string or list of labels
    RETURNS (List[str]): a split and stripped list of labels
    """
    labels = labels.split(",") if isinstance(labels, str) else labels
    return [label.strip() for label in labels]


def assemble_from_config(config: Config) -> Language:
    """Assemble a spaCy pipeline from a confection Config object.

    config (Config): Config to load spaCy pipeline from.
    RETURNS (Language): An initialized spaCy pipeline.
    """
    nlp = load_model_from_config(config, auto_fill=True)
    config = config.interpolate()
    sourced = get_sourced_components(config)
    nlp._link_components()
    with nlp.select_pipes(disable=[*sourced]):
        nlp.initialize()
    return nlp


def assemble(
    config_path: Union[str, Path], *, overrides: Dict[str, Any] = SimpleFrozenDict()
) -> Language:
    """Assemble a spaCy pipeline from a config file.

    config_path (Union[str, Path]): Path to config file.
    overrides (Dict[str, Any], optional): Dictionary of config overrides.
    RETURNS (Language): An initialized spaCy pipeline.
    """
    config_path = Path(config_path)
    config = load_config(config_path, overrides=overrides, interpolate=False)
    return assemble_from_config(config)


def consume_parameter(
    config: Dict[Any, Any],
    key: str,
    prefix: Optional[str] = None,
    default: Optional[Any] = None,
    type_: Callable[[str], Any] = str,
) -> Any:
    """Consume a parameter in the config dict.  Look for an environment
        variable with the given prefix first, then look in the config dict.
        Pop the parameter from the config dict (if it's there) even if the
        parameter is first found as an environment variable.

    config (Dict[Any, Any]): Configuration parameters.
    key (str): Parameter key.
    prefix (Optional[str]) = None: Combined with `key.upper()` to represent the
        equivalent environment variable.  None for no prefix.
    default (Optional[Any]) = None: Default value for the parameter.  None to
        require a value.
    type_ (Callable[str, Any]) = str: Converts a string to desired output.

    RETURNS (Any): The parameter value.
    RAISES (ValueError): If the parameter is not found either as an environment
        variable or as an entry in config.
    """
    env_key = key.upper() if prefix is None else (prefix + key.upper())
    env_value = os.getenv(env_key, None)
    if env_value is not None:
        config.pop(key, None)
        return type_(env_value)
    if default is None:
        try:
            return config.pop(key)
        except KeyError:
            raise ValueError(
                f"Could not find parameter {key} for Azure OpenAI REST "
                f"backend: set {key} in your config or set the environment "
                f"variable {env_key}."
            )
    return config.pop(key, default)
