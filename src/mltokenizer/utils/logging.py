from loguru import logger
import sys

# Configure a default logger for the tokenizer package
# Users can reconfigure this if they wish by accessing logger.configure()

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",  # Default level
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

tokenizer_logger = logger.bind(name="mltokenizer")

__all__ = ["tokenizer_logger"] 