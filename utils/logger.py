"""
Logging setup shared by the application.

Logs are written both to the console and to a rotating log file under
config.LOG_DIR, so a run can be inspected after the fact.
"""

import logging
import os


def setup_logger(name: str, log_dir: str, log_file: str) -> logging.Logger:
    """
    Create (or fetch) a configured logger.

    Args:
        name: Logger name, typically __name__ of the caller.
        log_dir: Directory the log file should live in. Created if missing.
        log_file: Log file name.

    Returns:
        A logging.Logger instance with console + file handlers attached.
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
