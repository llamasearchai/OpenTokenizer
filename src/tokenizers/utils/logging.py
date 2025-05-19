import sys
from typing import Optional

from loguru import logger


# Configure default logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Create specialized loggers for different components
tokenizer_logger = logger.bind(component="tokenizer")
registry_logger = logger.bind(component="registry")
server_logger = logger.bind(component="server")
performance_logger = logger.bind(component="performance")


def set_log_level(level: str) -> None:
    """Set the log level for all loggers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )


def get_module_logger(module_name: str) -> logger:
    """Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Logger instance for the module
    """
    return logger.bind(component=module_name)