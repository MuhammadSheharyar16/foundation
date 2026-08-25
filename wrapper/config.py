"""Wrapper configuration.

Configuration is read from environment variables only. Nothing is hard-coded,
nothing is committed. Required variable names are listed in the root README.
"""

import os
from dataclasses import dataclass

from .errors import ConfigurationMissingError

# Maps WrapperConfig field name -> required environment variable name.
_REQUIRED_ENV_VARS: dict[str, str] = {
    "endpoint": "MODEL_ENDPOINT",
    "chat_model_alias": "CHAT_MODEL_ALIAS",
    "embedding_model_alias": "EMBEDDING_MODEL_ALIAS",
}

# Optional: falls back to WrapperConfig.default_timeout_s when unset.
_TIMEOUT_ENV_VAR = "MODEL_TIMEOUT_S"


@dataclass(frozen=True)
class WrapperConfig:
    """Resolved configuration for the model wrapper.

    Attributes:
        endpoint: Service endpoint. Never printed in full.
        chat_model_alias: Alias for the chat deployment.
        embedding_model_alias: Alias for the embedding deployment.
        default_timeout_s: Timeout applied when a caller does not pass one.
    """

    endpoint: str
    chat_model_alias: str
    embedding_model_alias: str
    default_timeout_s: float = 30.0


def load_config() -> WrapperConfig:
    """Read configuration from environment variables.

    Required: MODEL_ENDPOINT, CHAT_MODEL_ALIAS, EMBEDDING_MODEL_ALIAS.
    Optional: MODEL_TIMEOUT_S (falls back to WrapperConfig.default_timeout_s).

    Returns:
        A populated WrapperConfig.

    Raises:
        ConfigurationMissingError: If any required variable is absent, empty,
            or if MODEL_TIMEOUT_S is set but not a valid number. The message
            names the missing/invalid variable and never its value.
    """
    values: dict[str, str] = {}
    for field_name, var_name in _REQUIRED_ENV_VARS.items():
        raw_value = os.environ.get(var_name, "").strip()
        if not raw_value:
            raise ConfigurationMissingError(
                f"Missing required configuration: {var_name}"
            )
        values[field_name] = raw_value

    kwargs: dict[str, object] = dict(values)

    raw_timeout = os.environ.get(_TIMEOUT_ENV_VAR, "").strip()
    if raw_timeout:
        try:
            kwargs["default_timeout_s"] = float(raw_timeout)
        except ValueError:
            raise ConfigurationMissingError(
                f"Invalid configuration: {_TIMEOUT_ENV_VAR} must be a number"
            ) from None

    return WrapperConfig(**kwargs)
