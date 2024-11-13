import logging
import os
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

verbose_logger = logging.getLogger("liteswarm")


ANSI_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
}

LEVEL_MAP: dict[LogLevel, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class FancyFormatter(logging.Formatter):
    """A fancy formatter with colors and better visual organization."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and structure.

        Format: [TIME] LEVEL | MESSAGE
        Example: [14:23:15] INFO | Starting process...
        """
        # Get the corresponding color
        color = ANSI_COLORS.get(record.levelname, ANSI_COLORS["RESET"])

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Format the message with proper indentation for multiline
        message_lines = record.getMessage().split("\n")
        message = "\n".join(
            # First line has the normal format
            ([f"{color}{message_lines[0]}{ANSI_COLORS['RESET']}"] if message_lines else [])
            +
            # Subsequent lines are indented and dimmed
            [f"{ANSI_COLORS['DIM']}│ {line}{ANSI_COLORS['RESET']}" for line in message_lines[1:]]
        )

        # Construct the prefix with timestamp and level
        prefix = (
            f"{ANSI_COLORS['DIM']}[{timestamp}]{ANSI_COLORS['RESET']} "
            f"{color}{ANSI_COLORS['BOLD']}{record.levelname:<8}{ANSI_COLORS['RESET']}"
        )

        return f"{prefix} │ {message}"


@lru_cache(maxsize=1)
def get_log_level(default: LogLevel = "INFO") -> int:
    """Get the log level from environment or default.

    Args:
        default: Default log level if not set in environment

    Returns:
        The logging level as an int
    """
    level_name = os.getenv("LITESWARM_LOG_LEVEL", default).upper()
    if level_name in LEVEL_MAP:
        return LEVEL_MAP[level_name]

    return LEVEL_MAP[default]


@lru_cache(maxsize=1)
def get_verbose_level(default: LogLevel = "INFO") -> LogLevel | None:
    """Get the verbose printing level from environment.

    Args:
        default: Default level if not set in environment

    Returns:
        LogLevel if specific level set, or None if disabled
    """
    verbose = os.getenv("LITESWARM_VERBOSE", "").upper()

    if verbose.lower() in ("1", "true", "yes", "on"):
        return default

    if verbose in LEVEL_MAP:
        return verbose

    return None


@lru_cache(maxsize=len(LEVEL_MAP))
def should_print(level: LogLevel) -> bool:
    """Check if verbose printing is enabled for given level.

    Args:
        level: Log level to check

    Returns:
        True if verbose printing is enabled for this level
    """
    verbose_level = get_verbose_level()
    if verbose_level is None:
        return False

    return LEVEL_MAP[level] >= LEVEL_MAP[verbose_level]


def enable_logging(default_level: LogLevel = "INFO") -> None:
    """Configure logging for liteswarm.

    Args:
        default_level: Default log level if not set in environment
    """
    verbose_logger.setLevel(get_log_level(default_level))
    verbose_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(FancyFormatter())
    verbose_logger.addHandler(handler)


def log_verbose(
    message: str,
    *args: Any,
    level: LogLevel = "INFO",
    print_fn: Callable[[str], None] | None = print,
    **kwargs: Any,
) -> None:
    """Log a message with optional printing.

    Args:
        message: The message to log (supports f-strings and multiline)
        *args: Additional args passed to the message (formatting)
        level: Log level to use
        print_fn: Optional function to print the message
        **kwargs: Additional kwargs passed to logger

    Environment Variables:
        LITESWARM_LOG_LEVEL: Set logging level (default: "INFO")
        LITESWARM_VERBOSE: Enable printing for specified level and above.
                          Use level name or "1"/"true"/"yes"/"on" for INFO
    """
    log_fn = getattr(verbose_logger, level.lower())
    log_fn(message, *args, **kwargs)

    if print_fn and should_print(level):
        formatted_message = message % args
        print_fn(formatted_message)
