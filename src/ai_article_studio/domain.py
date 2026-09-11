"""Dependency-free domain values and application-level failures.

The web layer deliberately continues to map ``ValueError`` to HTTP 400 and
``RuntimeError`` to HTTP 500. These classes preserve that compatibility while
giving workflow and adapter boundaries meaningful failure types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_PROVIDER_SEPARATOR = "|provider="
_AUTH_VALUE_PATTERN = re.compile(r"(?i)(\\b(?:(?:proxy-)?authorization\\s*:\\s*)?(?:bearer|basic)\\s+)([^\\s,;]+)")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\\b(?:api[_-]?key|access[_-]?token|id[_-]?token|refresh[_-]?token|token|secret|client[_-]?secret|password)\\b\\s*[:=]\\s*)([^\\s,;}&]+)"
)
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:key|api[_-]?key|access[_-]?token|id[_-]?token|refresh[_-]?token|token|secret|client[_-]?secret|password)=)([^&#\\s]+)"
)


class DomainError(Exception):
    """Base class for failures that are safe to present to an application user."""


class WorkflowValidationError(DomainError, ValueError):
    """A workflow request or persisted operation is invalid for its current state."""


class WorkflowExecutionError(DomainError, RuntimeError):
    """A valid workflow operation could not be completed."""


class WorkflowBusyError(WorkflowExecutionError):
    """A foreground mutation races an active background workflow job."""


class WorkflowCancellationRequested(WorkflowExecutionError):
    """Internal control flow used to stop an in-flight task at a safe boundary."""


class JobAlreadyRunning(WorkflowExecutionError):
    """A task is already running locally or waiting in the queue."""


class AdapterError(WorkflowExecutionError):
    """A safe, normalised failure at an external LLM or image-provider boundary."""


class AdapterConfigurationError(AdapterError):
    """An explicitly selected provider is unavailable because it is unconfigured."""


class AdapterAuthenticationError(AdapterError):
    """A provider rejected credentials with a user-actionable safe message."""


class AdapterResponseError(AdapterError):
    """A provider returned a successful but unusable response payload."""


class AdapterTransportError(AdapterError):
    """A transport or HTTP-status failure, without exposing vendor diagnostics."""


def safe_adapter_error_message(error: Exception, fallback: str) -> str:
    """Keep compatible provider diagnostics while removing credential values.

    HTTP response bodies are deliberately not inspected. Built-in adapters use
    their own fixed messages; this helper only preserves safe text supplied by
    third-party implementations at the typed image-provider port.
    """

    message = str(error).strip()
    if not message:
        return fallback
    message = _AUTH_VALUE_PATTERN.sub(r"\1[REDACTED]", message)
    message = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1[REDACTED]", message)
    return _QUERY_SECRET_PATTERN.sub(r"\1[REDACTED]", message)


@dataclass(frozen=True, slots=True)
class WorkflowOperation:
    """Immutable description of one persisted workflow action.

    ``provider`` stays a string so this domain module depends on neither
    Pydantic schemas nor adapter implementations.
    """

    action: str
    provider: str | None = None

    def encode(self) -> str:
        """Return the persisted action representation used by existing records."""

        return self.action if self.provider is None else f"{self.action}{_ACTION_PROVIDER_SEPARATOR}{self.provider}"

    @classmethod
    def parse(cls, value: str) -> WorkflowOperation:
        """Parse legacy action text without validating provider ownership here."""

        action, separator, provider = value.partition(_ACTION_PROVIDER_SEPARATOR)
        return cls(action=action, provider=provider if separator else None)
