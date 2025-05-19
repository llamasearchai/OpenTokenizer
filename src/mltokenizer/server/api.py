from typing import Dict, List, Optional, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import Depends

from mltokenizer.core.base_tokenizer import BatchTokenizedOutput, TokenizedOutput
from mltokenizer.core.registry import TokenizerRegistry # Assuming TokenizerRegistry will be in mltokenizer.core.registry
from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.algorithms.wordpiece import WordpieceTokenizer
from mltokenizer.algorithms.unigram import UnigramTokenizer
from mltokenizer.algorithms.sentencepiece_tokenizer import SentencePieceTokenizer
from mltokenizer.algorithms.character import CharacterTokenizer
from mltokenizer.models.bert import BertTokenizer
from mltokenizer.models.gpt import GPTTokenizer


class TokenizeRequest(BaseModel):
    """Request model for tokenization."""
    
    text: str = Field(..., description="Text to tokenize")
    text_pair: Optional[str] = Field(None, description="Optional paired text")
    add_special_tokens: bool = Field(True, description="Whether to add special tokens")
    return_tokens: bool = Field(False, description="Whether to return string tokens")
    tokenizer_id: str = Field(..., description="ID of the tokenizer to use")


class TokenizeBatchRequest(BaseModel):
    """Request model for batch tokenization."""
    
    texts: List[str] = Field(..., description="Texts to tokenize")
    text_pairs: Optional[List[str]] = Field(None, description="Optional paired texts")
    add_special_tokens: bool = Field(True, description="Whether to add special tokens")
    return_tokens: bool = Field(False, description="Whether to return string tokens")
    tokenizer_id: str = Field(..., description="ID of the tokenizer to use")


class DecodeRequest(BaseModel):
    """Request model for decoding."""
    
    token_ids: List[int] = Field(..., description="Token IDs to decode")
    skip_special_tokens: bool = Field(True, description="Whether to skip special tokens")
    tokenizer_id: str = Field(..., description="ID of the tokenizer to use")


class DecodeBatchRequest(BaseModel):
    """Request model for batch decoding."""
    
    batch_token_ids: List[List[int]] = Field(..., description="Batch of token IDs to decode")
    skip_special_tokens: bool = Field(True, description="Whether to skip special tokens")
    tokenizer_id: str = Field(..., description="ID of the tokenizer to use")


class TokenizerInfo(BaseModel):
    """Model for tokenizer information."""
    
    id: str = Field(..., description="Tokenizer ID")
    type: str = Field(..., description="Tokenizer type")
    vocab_size: int = Field(..., description="Vocabulary size")
    is_trained: bool = Field(..., description="Whether the tokenizer is trained")
    special_tokens: Dict[str, str] = Field(..., description="Special tokens")
    options: Dict[str, Union[str, int, bool, None]] = Field(..., description="Tokenizer options")


class LoadTokenizerRequest(BaseModel):
    path: str = Field(..., description="Path to the saved tokenizer directory")
    tokenizer_type: str = Field(..., description="Type of the tokenizer (e.g., bpe, wordpiece)")
    tokenizer_id: Optional[str] = Field(None, description="Optional ID to register the tokenizer with")


class AddTokensRequest(BaseModel):
    tokens: List[str] = Field(..., description="List of tokens to add")


class AddSpecialTokensRequest(BaseModel):
    special_tokens_dict: Dict[str, str] = Field(..., description="Dictionary of special token types to token strings (e.g., {\"bos_token\": \"<s>\"})")


# Create the FastAPI app
app = FastAPI(
    title="ML Tokenization System API",
    description="API for interacting with the ML tokenization system",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the registry passed from CLI
# This is a simple way; for more complex apps, dependency injection (e.g. FastAPI's Depends) or app.state is better.
_shared_registry: Optional[TokenizerRegistry] = None


@app.on_event("startup")
async def startup_event():
    # This is just a placeholder if we need to initialize things with the registry at startup
    # The actual registry will be set by set_tokenizer_registry before uvicorn.run
    pass


def get_registry() -> TokenizerRegistry:
    if _shared_registry is None:
        # This case should ideally not be hit if server is started correctly via CLI
        raise RuntimeError("TokenizerRegistry not initialized. Server might not have been started via CLI.")
    return _shared_registry


@app.get("/tokenizers", response_model=List[TokenizerInfo])
async def list_tokenizers(registry: TokenizerRegistry = Depends(get_registry)):
    """List all available tokenizers."""
    tokenizers = registry.list_tokenizers()
    
    result = []
    for tokenizer_id, tokenizer in tokenizers.items():
        result.append(
            TokenizerInfo(
                id=tokenizer_id,
                type=tokenizer.tokenizer_type.value,
                vocab_size=tokenizer.vocab_size,
                is_trained=tokenizer.is_trained,
                special_tokens=tokenizer.special_tokens.get_special_tokens_dict() if tokenizer.special_tokens else {},
                options=tokenizer.options.dict() if tokenizer.options else {}
            )
        )
    
    return result


@app.get("/tokenizers/{tokenizer_id}", response_model=TokenizerInfo)
async def get_tokenizer_details(
    tokenizer_id: str, 
    registry: TokenizerRegistry = Depends(get_registry)
):
    """Get detailed information about a specific tokenizer."""
    tokenizer = registry.get_tokenizer(tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not found")
    
    return TokenizerInfo(
        id=tokenizer_id, # Or derive ID from how it was registered if different
        type=tokenizer.tokenizer_type.value,
        vocab_size=tokenizer.get_vocab_size(), # Use method for consistency
        is_trained=tokenizer.is_trained,
        special_tokens=tokenizer.special_tokens.get_special_tokens_dict() if tokenizer.special_tokens else {},
        options=tokenizer.options.dict() if tokenizer.options else {}
    )


@app.post("/tokenize", response_model=TokenizedOutput)
async def tokenize(request: TokenizeRequest, registry: TokenizerRegistry = Depends(get_registry)):
    """Tokenize a text."""
    tokenizer = registry.get_tokenizer(request.tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{request.tokenizer_id}' not found")
    
    if not tokenizer.is_trained:
        raise HTTPException(status_code=400, detail=f"Tokenizer '{request.tokenizer_id}' is not trained")
    
    result = tokenizer.encode(
        text=request.text,
        text_pair=request.text_pair,
        add_special_tokens=request.add_special_tokens,
        return_tokens=request.return_tokens
    )
    
    return result


@app.post("/tokenize_batch", response_model=BatchTokenizedOutput)
async def tokenize_batch(request: TokenizeBatchRequest, registry: TokenizerRegistry = Depends(get_registry)):
    """Tokenize a batch of texts."""
    tokenizer = registry.get_tokenizer(request.tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{request.tokenizer_id}' not found")
    
    if not tokenizer.is_trained:
        raise HTTPException(status_code=400, detail=f"Tokenizer '{request.tokenizer_id}' is not trained")
    
    result = tokenizer.encode_batch(
        texts=request.texts,
        text_pairs=request.text_pairs,
        add_special_tokens=request.add_special_tokens,
        return_tokens=request.return_tokens
    )
    
    return result


@app.post("/decode", response_model=str)
async def decode(request: DecodeRequest, registry: TokenizerRegistry = Depends(get_registry)):
    """Decode token IDs to text."""
    tokenizer = registry.get_tokenizer(request.tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{request.tokenizer_id}' not found")
    
    if not tokenizer.is_trained:
        raise HTTPException(status_code=400, detail=f"Tokenizer '{request.tokenizer_id}' is not trained")
    
    result = tokenizer.decode(
        token_ids=request.token_ids,
        skip_special_tokens=request.skip_special_tokens
    )
    
    return result


@app.post("/decode_batch", response_model=List[str])
async def decode_batch(request: DecodeBatchRequest, registry: TokenizerRegistry = Depends(get_registry)):
    """Decode batch of token IDs to texts."""
    tokenizer = registry.get_tokenizer(request.tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{request.tokenizer_id}' not found")
    
    if not tokenizer.is_trained:
        raise HTTPException(status_code=400, detail=f"Tokenizer '{request.tokenizer_id}' is not trained")
    
    result = tokenizer.decode_batch(
        batch_token_ids=request.batch_token_ids,
        skip_special_tokens=request.skip_special_tokens
    )
    
    return result


@app.get("/vocabulary/{tokenizer_id}", response_model=Dict[str, int])
async def get_vocabulary(tokenizer_id: str, limit: int = Query(100, ge=1, le=1000), registry: TokenizerRegistry = Depends(get_registry)):
    """Get the vocabulary of a tokenizer."""
    tokenizer = registry.get_tokenizer(tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not found")
    
    if not tokenizer.is_trained:
        raise HTTPException(status_code=400, detail=f"Tokenizer '{tokenizer_id}' is not trained")
    
    vocab = tokenizer.get_vocab()
    
    # Limit the size of the returned vocabulary
    if len(vocab) > limit:
        vocab = dict(list(vocab.items())[:limit])
    
    return vocab


@app.get("/metrics/{tokenizer_id}")
async def get_metrics(tokenizer_id: str, registry: TokenizerRegistry = Depends(get_registry)):
    """Get performance metrics for a tokenizer."""
    tokenizer = registry.get_tokenizer(tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not found")
    
    if not hasattr(tokenizer, "metrics_tracker"):
        raise HTTPException(status_code=400, detail=f"No metrics available for tokenizer '{tokenizer_id}'")
    
    metrics = tokenizer.metrics_tracker.get_metrics()
    return metrics.to_dict()


@app.post("/tokenizers/load", status_code=201)
async def load_tokenizer_api(
    request: LoadTokenizerRequest, 
    registry: TokenizerRegistry = Depends(get_registry)
):
    """Load a tokenizer from a specified path into the registry."""
    tokenizer_path = Path(request.path)
    if not tokenizer_path.exists() or not tokenizer_path.is_dir():
        raise HTTPException(
            status_code=404, 
            detail=f"Tokenizer path {request.path} does not exist or is not a directory"
        )

    tokenizer_instance = None
    try:
        logger.info(f"API attempting to load {request.tokenizer_type} tokenizer from {tokenizer_path}")
        if request.tokenizer_type.lower() == "bpe":
            tokenizer_instance = BPETokenizer.load(tokenizer_path)
        elif request.tokenizer_type.lower() == "wordpiece":
            tokenizer_instance = WordpieceTokenizer.load(tokenizer_path)
        elif request.tokenizer_type.lower() == "unigram":
            tokenizer_instance = UnigramTokenizer.load(tokenizer_path)
        elif request.tokenizer_type.lower() == "sentencepiece":
            tokenizer_instance = SentencePieceTokenizer.load(tokenizer_path)
        elif request.tokenizer_type.lower() == "character":
            tokenizer_instance = CharacterTokenizer.load(tokenizer_path)
        # TODO: Add other types like unigram, sentencepiece, custom models (BertTokenizer.from_pretrained for local path?)
        # elif request.tokenizer_type.lower() == "bert_local":
        #     tokenizer_instance = BertTokenizer.from_pretrained(tokenizer_path) # Needs check if path is dir
        # elif request.tokenizer_type.lower() == "gpt_local":
        #     tokenizer_instance = GPTTokenizer.from_pretrained(tokenizer_path)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported tokenizer type: {request.tokenizer_type}. Supported: bpe, wordpiece, unigram, sentencepiece, character (extend for others)"
            )
        
        tokenizer_id_to_register = request.tokenizer_id or f"{request.tokenizer_type.lower()}_{tokenizer_path.name}"
        registry.register_tokenizer(tokenizer_id_to_register, tokenizer_instance)
        logger.info(f"Tokenizer '{tokenizer_id_to_register}' loaded and registered via API from {tokenizer_path}.")
        return {
            "message": "Tokenizer loaded successfully", 
            "tokenizer_id": tokenizer_id_to_register,
            "type": tokenizer_instance.tokenizer_type.value,
            "vocab_size": tokenizer_instance.vocab_size,
            "is_trained": tokenizer_instance.is_trained
        }

    except FileNotFoundError as e:
        logger.error(f"API load_tokenizer FileNotFoundError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"API error loading tokenizer from {tokenizer_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading tokenizer: {e}")


@app.post("/tokenizers/{tokenizer_id}/add-tokens", status_code=200)
async def add_tokens_api(
    tokenizer_id: str,
    request: AddTokensRequest,
    registry: TokenizerRegistry = Depends(get_registry)
):
    """Add new regular tokens to a tokenizer's vocabulary."""
    tokenizer = registry.get_tokenizer(tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not found")
    if not tokenizer.is_trained:
         raise HTTPException(status_code=400, detail=f"Tokenizer '{tokenizer_id}' is not trained. Cannot add tokens.")
    
    try:
        added_count = tokenizer.add_tokens(request.tokens)
        logger.info(f"Added {added_count} tokens to tokenizer '{tokenizer_id}'. New vocab size: {tokenizer.get_vocab_size()}")
        return {"message": f"Successfully added {added_count} tokens.", "new_vocab_size": tokenizer.get_vocab_size()}
    except Exception as e:
        logger.error(f"Error adding tokens to '{tokenizer_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error adding tokens: {e}")


@app.post("/tokenizers/{tokenizer_id}/add-special-tokens", status_code=200)
async def add_special_tokens_api(
    tokenizer_id: str,
    request: AddSpecialTokensRequest,
    registry: TokenizerRegistry = Depends(get_registry)
):
    """Add new special tokens to a tokenizer."""
    tokenizer = registry.get_tokenizer(tokenizer_id)
    if not tokenizer:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not found")
    if not tokenizer.is_trained:
         # Some tokenizers might allow adding special tokens before full vocab training,
         # but typically vocab should exist. Let's assume it needs to be trained / have an encoder.
         raise HTTPException(status_code=400, detail=f"Tokenizer '{tokenizer_id}' is not trained. Cannot add special tokens.")

    try:
        added_count = tokenizer.add_special_tokens(request.special_tokens_dict)
        logger.info(f"Added {added_count} special tokens to tokenizer '{tokenizer_id}'. New vocab size: {tokenizer.get_vocab_size()}")
        return {"message": f"Successfully added {added_count} special tokens.", "new_vocab_size": tokenizer.get_vocab_size()}
    except Exception as e:
        logger.error(f"Error adding special tokens to '{tokenizer_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error adding special tokens: {e}")


# Function to set the registry before starting the server
# This will be called from cli.py
def set_tokenizer_registry(registry_instance: TokenizerRegistry):
    global _shared_registry
    _shared_registry = registry_instance
    logger.info(f"TokenizerRegistry instance set for API server: {id(registry_instance)}")

def start_server(host: str = "0.0.0.0", port: int = 8000, registry_instance: Optional[TokenizerRegistry] = None):
    """Start the tokenizer API server."""
    if registry_instance:
        set_tokenizer_registry(registry_instance)
    else:
        # Fallback: if no registry is passed, create a new one (though CLI should always pass it)
        logger.warning("API server starting with a new TokenizerRegistry instance. CLI-loaded tokenizers won't be available.")
        set_tokenizer_registry(TokenizerRegistry()) # Create a default one as a fallback
        
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # This direct run will use a default, empty registry
    logger.info("Starting server directly from api.py. No pre-loaded tokenizers from CLI.")
    start_server() 