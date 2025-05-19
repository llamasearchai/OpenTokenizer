from typing import Any, Type, TypeVar, List, Optional, Dict
from pydantic import BaseModel, ValidationError, validator
from loguru import logger

T = TypeVar('T', bound=BaseModel)

def validate_model(data: Any, model_class: Type[T]) -> Optional[T]:
    """Validates data against a Pydantic model.

    Args:
        data: The data to validate (e.g., a dictionary).
        model_class: The Pydantic model class to validate against.

    Returns:
        An instance of model_class if validation is successful, None otherwise.
    """
    try:
        return model_class(**data if isinstance(data, dict) else data)
    except ValidationError as e:
        logger.error(f"Validation failed for {model_class.__name__}:\n{e}")
        return None
    except TypeError as e: # Handle cases where data is not a dict and model expects one
        logger.error(f"Type error during validation for {model_class.__name__} (data might not be a dict or compatible):\n{e}")
        return None

# Example Pydantic model for tokenizer configuration (can be more specific)
class BaseTokenizerConfig(BaseModel):
    vocab_size: int
    model_type: str
    files: Optional[Dict[str, str]] = None # e.g., {"vocab": "path/to/vocab.json"}

    @validator('vocab_size')
    def vocab_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('vocab_size must be positive')
        return v

class PathValidatorMixin:
    @validator('*', pre=True, each_item=True, check_fields=False)
    def check_path_fields_exist(cls, v, field):
        from pathlib import Path # Local import to avoid circularity if Path is used widely
        # Example: if a field is annotated as Path, check if it exists
        # This is a generic example; more specific validators are better.
        if field.outer_type_ is Path or field.type_ is Path:
            if isinstance(v, (str, Path)):
                p = Path(v)
                if not p.exists():
                    logger.warning(f"Path specified for field '{field.name}' does not exist: {p}")
                # You might return p or raise ValueError based on strictness
        return v

logger.info("Validation module loaded. Use for Pydantic model validation.") 