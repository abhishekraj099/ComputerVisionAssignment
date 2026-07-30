"""
Logging setup shared by the application.

Purpose:
    Provide one consistent way for every module in the project to obtain a
    configured logger, so log format, destinations, and level are defined
    in exactly one place.

Responsibilities:
    Attach a console handler and a file handler (writing under
    config.LOG_DIR/config.LOG_FILE) to a named logger, with a shared
    timestamped format, so a run can be inspected after the fact.

Scope of the current phase:
    Logging infrastructure only; no phase-specific behavior. Used
    unchanged since Phase 1 by app.py, models/person_detector.py, and
    tracker/byte_tracker.py, and expected to be reused as-is by every
    future phase's module.

What this module intentionally does NOT handle:
    Log rotation/retention, remote log shipping, or structured (e.g. JSON)
    logging - plain timestamped text to console + a single growing file is
    sufficient for this project's scope.

Which future modules will consume this module's output:
    Every module in the project calls `setup_logger(__name__, ...)` once
    at import time; this will remain true for any future phase's module.
"""

import logging
import os


def setup_logger(name: str, log_dir: str, log_file: str) -> logging.Logger:
    """
    Create (or fetch) a configured logger.

    Args:
        name: Logger name, typically __name__ of the caller. Each distinct
            name gets its own logger instance and its own pair of handlers.
        log_dir: Directory the log file should live in. Created if missing.
        log_file: Log file name (created/appended to inside log_dir).

    Returns:
        A logging.Logger instance with console + file handlers attached,
        set to INFO level.

    Raises:
        Does not raise: if the log directory/file cannot be created (e.g.
        a permissions issue), the failure is caught and logged as a
        warning, and the logger falls back to console-only output rather
        than failing the caller's import.

    Side effects:
        On the first call for a given `name`, creates `log_dir` on disk if
        it does not exist and opens `log_file` for appending. Subsequent
        calls with the same `name` are no-ops (see below) and perform no
        further I/O.

    Performance considerations:
        Cheap; typically called once per module at import time, not in any
        hot path.

    Thread safety:
        Reads/writes to Python's global logger registry
        (`logging.getLogger`), which is itself thread-safe. The
        "already configured" check below (`if logger.handlers`) is not
        atomic, so calling this function for the *same* new `name`
        simultaneously from multiple threads could in theory attach
        duplicate handlers; this project only ever calls it once, at
        import time, from the main thread, so this does not occur in
        practice.
    """
    logger = logging.getLogger(name)

    # Avoid attaching duplicate handlers if setup_logger is called more than once.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # If the log directory/file can't be created, continue with console-only logging
        # rather than crashing the whole application over a logging problem.
        logger.warning("Could not create file log handler (%s). Continuing with console logging only.", exc)

    return logger
