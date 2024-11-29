# Copyright 2024 GlyphyAI

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal, cast

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
    r"""Enhanced log formatter with color and visual organization.

    Provides a visually appealing log format with:
    - Color-coded log levels
    - Timestamp prefixes
    - Proper indentation for multiline messages
    - Visual separators between components

    Examples:
        Basic usage:
            ```python
            handler = logging.StreamHandler()
            handler.setFormatter(FancyFormatter())
            logger.addHandler(handler)

            logger.info("Starting process")
            # [14:23:15] INFO    │ Starting process

            logger.error("Error occurred:\\nDetails:\\n- Missing file")
            # [14:23:16] ERROR   │ Error occurred
            # │ Details:
            # │ - Missing file
            ```

        Custom logger:
            ```python
            logger = logging.getLogger("myapp")
            logger.setLevel(logging.DEBUG)

            handler = logging.StreamHandler()
            handler.setFormatter(FancyFormatter())
            logger.addHandler(handler)

            logger.debug("Initializing...")
            # [14:23:15] DEBUG   │ Initializing...
            ```
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and structure.

        Formats the log record with the following structure:
        [TIME] LEVEL │ MESSAGE

        Args:
            record: Log record to format.

        Returns:
            Formatted log message with colors and structure.
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
    """Get configured logging level from environment.

    Checks LITESWARM_LOG_LEVEL environment variable and falls
    back to default if not set or invalid.

    Args:
        default: Fallback log level if not configured.

    Returns:
        Numeric logging level (e.g., logging.INFO).

    Examples:
        Default behavior:
            ```python
            # No environment variable set
            level = get_log_level()  # Returns logging.INFO

            # With default override
            level = get_log_level("DEBUG")  # Returns logging.DEBUG
            ```

        Environment variable:
            ```python
            # In shell:
            # export LITESWARM_LOG_LEVEL=DEBUG

            level = get_log_level()  # Returns logging.DEBUG
            ```
    """
    level_name = os.getenv("LITESWARM_LOG_LEVEL", default).upper()
    if level_name in LEVEL_MAP:
        return LEVEL_MAP[cast(LogLevel, level_name)]

    return LEVEL_MAP[default]


@lru_cache(maxsize=1)
def get_verbose_level(default: LogLevel = "INFO") -> LogLevel | None:
    """Get configured verbose logging level.

    Checks LITESWARM_VERBOSE environment variable for:
    - Boolean flags: "1", "true", "yes", "on" (enables default level)
    - Specific level: "DEBUG", "INFO", etc.
    - Empty/other: Disables verbose logging

    Args:
        default: Level to use when enabled without specific level.

    Returns:
        Configured level or None if verbose logging disabled.

    Examples:
        Boolean flags:
            ```python
            # export LITESWARM_VERBOSE=true
            level = get_verbose_level()  # Returns "INFO"

            # export LITESWARM_VERBOSE=1
            level = get_verbose_level("DEBUG")  # Returns "DEBUG"
            ```

        Specific level:
            ```python
            # export LITESWARM_VERBOSE=WARNING
            level = get_verbose_level()  # Returns "WARNING"
            ```
    """
    verbose = os.getenv("LITESWARM_VERBOSE", "").upper()

    if verbose.lower() in ("1", "true", "yes", "on"):
        return default

    if verbose in LEVEL_MAP:
        return cast(LogLevel, verbose)

    return None


@lru_cache(maxsize=len(LEVEL_MAP))
def should_print(level: LogLevel) -> bool:
    """Check if message at given level should be printed.

    Determines if verbose printing is enabled for the specified
    level based on LITESWARM_VERBOSE configuration.

    Args:
        level: Log level to check.

    Returns:
        True if messages at this level should be printed.

    Examples:
        Level comparison:
            ```python
            # export LITESWARM_VERBOSE=WARNING

            should_print("ERROR")    # Returns True
            should_print("WARNING")  # Returns True
            should_print("INFO")     # Returns False
            ```

        Disabled verbose:
            ```python
            # LITESWARM_VERBOSE not set
            should_print("ERROR")  # Returns False
            ```
    """
    verbose_level = get_verbose_level()
    if verbose_level is None:
        return False

    return LEVEL_MAP[level] >= LEVEL_MAP[verbose_level]


def enable_logging(default_level: LogLevel = "INFO") -> None:
    """Configure liteswarm logger with fancy formatting.

    Sets up the liteswarm logger with:
    - Configured or default log level
    - Color-coded fancy formatter
    - Console output handler

    Args:
        default_level: Fallback level if not configured.

    Examples:
        Basic setup:
            ```python
            enable_logging()  # Uses INFO level
            logger.info("Ready")  # [14:23:15] INFO    │ Ready
            ```

        Custom level:
            ```python
            enable_logging("DEBUG")
            logger.debug("Starting")  # [14:23:15] DEBUG   │ Starting
            ```
    """
    verbose_logger.setLevel(get_log_level(default_level))
    verbose_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(FancyFormatter())
    verbose_logger.addHandler(handler)


@contextmanager
def disable_logging() -> Generator[None, None, None]:
    """Temporarily disable all logging output.

    Context manager that suppresses all log messages by setting
    the root logger's level above CRITICAL. Restores the original
    level when exiting the context.

    Examples:
        Suppress specific section:
            ```python
            logger.info("This will show")

            with disable_logging():
                logger.info("This won't show")
                process_data()  # No logs from this
                logger.error("Not even errors show")

            logger.info("Back to normal")
            ```

        In testing:
            ```python
            def test_noisy_function():
                with disable_logging():
                    result = noisy_function()  # Logs suppressed
                assert result == expected
            ```
    """
    old_level = logging.root.getEffectiveLevel()
    logging.root.setLevel(logging.CRITICAL + 1)
    yield
    logging.root.setLevel(old_level)


def log_verbose(
    message: str,
    *args: Any,
    level: LogLevel = "INFO",
    print_fn: Callable[[str], None] | None = print,
    **kwargs: Any,
) -> None:
    r"""Log message with optional console output.

    Logs message through liteswarm logger and optionally prints
    to console if verbose output is enabled for the specified level.

    Args:
        message: Log message (supports formatting).
        *args: Format string arguments.
        level: Log level to use.
        print_fn: Function for console output (None to disable).
        **kwargs: Additional logger arguments.

    Examples:
        Basic logging:
            ```python
            log_verbose("Processing %d items", 5)
            # [14:23:15] INFO    │ Processing 5 items
            ```

        Different levels:
            ```python
            log_verbose("Debug info", level="DEBUG")
            log_verbose("Warning!", level="WARNING")
            ```

        Custom print function:
            ```python
            def my_print(msg: str) -> None:
                sys.stderr.write(f"{msg}\\n")

            log_verbose("Error", level="ERROR", print_fn=my_print)
            ```

        Environment control:
            ```python
            # export LITESWARM_VERBOSE=WARNING

            # This only logs, no console output
            log_verbose("Starting", level="INFO")

            # This logs and prints to console
            log_verbose("Warning!", level="WARNING")
            ```
    """
    log_fn = getattr(verbose_logger, level.lower())
    log_fn(message, *args, **kwargs)

    if print_fn and should_print(level):
        formatted_message = message % args
        print_fn(formatted_message)
