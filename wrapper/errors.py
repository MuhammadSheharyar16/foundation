"""Wrapper error types.

Error messages must stay safe: no token, no authorization header, no raw
sensitive content, no full endpoint with embedded credentials.
"""


class WrapperError(Exception):
    """Base class for all wrapper errors."""


class ConfigurationMissingError(WrapperError):
    """Raised when required configuration or identity is not available.

    Message should name the missing setting only, never its value.
    """


class ModelTimeoutError(WrapperError):
    """Raised when a model or embedding call exceeds its timeout budget."""


class ModelCallError(WrapperError):
    """Raised when the model call fails for any other reason.

    Underlying provider errors are wrapped so SDK exception types do not
    leak past this package.
    """
