import sys

from loguru import logger

from RetailApp.core.config import config


def setup_logging():
    log_level = config.LOG_LEVEL.upper()

    logger.configure(extra={"request_id": "N/A"})  # Default value for request_id
    logger.remove()  # Remove default logger

    logger.add(
        sys.stderr,  # Log to standard error
        level=log_level,  # Log level
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<magenta>{extra[request_id]}</magenta> | "  # Add this line
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,  # Enable color in console
    )

    logger.add(
        "RetailApp/logs/app.log",
        rotation="10 MB",  # Rotate after 10 MB
        retention="7 days",  # Keep logs for 7 days
        compression="zip",  # Compress old logs
        encoding="utf-8",  # Use UTF-8 encoding
        level=log_level,  # Log level
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[request_id]} |{name}:{function}:{line} | {message}",
    )

    logger.add(
        "RetailApp/logs/errors.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} |  {extra[request_id]} |{name}:{function}:{line} | {message}",
        rotation="1 week",
        retention="90 days",
        compression="zip",
        encoding="utf-8",
        backtrace=True,  # show full traceback context
        diagnose=True,  # show variable values in traceback
    )
