# src/mltokenizer/cli.py
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mltokenizer.algorithms.bpe import BPETokenizer
from mltokenizer.algorithms.wordpiece import WordpieceTokenizer
# from mltokenizer.core.base_tokenizer import TokenizerType # Not directly used, can be removed if not needed for future extension
from mltokenizer.core.registry import TokenizerRegistry
from mltokenizer.models.bert import BertTokenizer
from mltokenizer.models.gpt import GPTTokenizer
from mltokenizer.server.api import start_server, set_tokenizer_registry # Import set_tokenizer_registry if direct manipulation needed, or pass via start_server

app = typer.Typer(help="ML Tokenization System CLI")
console = Console()
registry = TokenizerRegistry() # This is the master registry for the CLI application


@app.command()
def train_tokenizer(
    tokenizer_type: str = typer.Option(..., help="Type of tokenizer to train (bpe, wordpiece, unigram, sentencepiece)"),
    input_file: Path = typer.Option(..., help="Input file with training texts (one per line)"),
    output_dir: Path = typer.Option(..., help="Output directory to save the tokenizer"),
    vocab_size: int = typer.Option(30000, help="Size of the vocabulary"),
    min_frequency: int = typer.Option(2, help="Minimum frequency for a token to be included"),
    num_workers: int = typer.Option(-1, help="Number of worker processes (-1 for all cores)"),
):
    """Train a new tokenizer on text data."""
    console.print(Panel(f"Training {tokenizer_type} tokenizer with vocab size {vocab_size}", title="ML Tokenization System"))
    
    # Load training data
    console.print(f"Loading training data from {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        console.print(f"Loaded {len(texts)} training examples")
    except Exception as e:
        console.print(f"[red]Error loading training data: {e}[/red]")
        raise typer.Exit(code=1)
    
    # Create tokenizer
    if tokenizer_type.lower() == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
    elif tokenizer_type.lower() == "wordpiece":
        tokenizer = WordpieceTokenizer(vocab_size=vocab_size)
    # Add other types like unigram, sentencepiece if they are implemented
    # elif tokenizer_type.lower() == "unigram":
    #     from mltokenizer.algorithms.unigram import UnigramTokenizer # Example
    #     tokenizer = UnigramTokenizer(vocab_size=vocab_size)
    # elif tokenizer_type.lower() == "sentencepiece":
    #     from mltokenizer.algorithms.sentencepiece import SentencePieceTokenizer # Example
    #     tokenizer = SentencePieceTokenizer(vocab_size=vocab_size)
    else:
        console.print(f"[red]Unsupported tokenizer type: {tokenizer_type}. Supported: bpe, wordpiece" +
                      " (extend for unigram, sentencepiece if implemented).[/red]")
        raise typer.Exit(code=1)
    
    # Train the tokenizer
    console.print(f"Training tokenizer on {len(texts)} texts...")
    try:
        tokenizer.train(texts=texts, min_frequency=min_frequency, num_workers=num_workers)
        console.print(f"[green]Training complete![/green]")
    except Exception as e:
        console.print(f"[red]Error training tokenizer: {e}[/red]")
        raise typer.Exit(code=1)
    
    # Save the tokenizer
    console.print(f"Saving tokenizer to {output_dir}...")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(output_dir)
        console.print(f"[green]Tokenizer saved to {output_dir}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving tokenizer: {e}[/red]")
        raise typer.Exit(code=1)
    
    # Register the tokenizer
    tokenizer_id = f"{tokenizer_type.lower()}_{output_dir.name}"
    registry.register_tokenizer(tokenizer_id, tokenizer)
    console.print(f"Registered tokenizer as '{tokenizer_id}'")


@app.command()
def load_tokenizer(
    tokenizer_path: Path = typer.Option(..., help="Path to the tokenizer directory"),
    tokenizer_type: str = typer.Option(..., help="Type of tokenizer (bpe, wordpiece, etc.)"),
    tokenizer_id: str = typer.Option(None, help="ID to register the tokenizer with (default: auto-generated)"),
):
    """Load an existing tokenizer from disk."""
    if not tokenizer_path.exists() or not tokenizer_path.is_dir():
        console.print(f"[red]Tokenizer path {tokenizer_path} does not exist or is not a directory[/red]")
        raise typer.Exit(code=1)
    
    # Load the tokenizer
    console.print(f"Loading {tokenizer_type} tokenizer from {tokenizer_path}...")
    try:
        if tokenizer_type.lower() == "bpe":
            tokenizer = BPETokenizer.load(tokenizer_path)
        elif tokenizer_type.lower() == "wordpiece":
            tokenizer = WordpieceTokenizer.load(tokenizer_path)
        # Add other types as they are implemented
        # elif tokenizer_type.lower() == "unigram":
        #     from mltokenizer.algorithms.unigram import UnigramTokenizer # Example
        #     tokenizer = UnigramTokenizer.load(tokenizer_path)
        # elif tokenizer_type.lower() == "sentencepiece":
        #     from mltokenizer.algorithms.sentencepiece import SentencePieceTokenizer # Example
        #     tokenizer = SentencePieceTokenizer.load(tokenizer_path)
        else:
            console.print(f"[red]Unsupported tokenizer type: {tokenizer_type}. Supported: bpe, wordpiece" +
                          " (extend for unigram, sentencepiece if implemented).[/red]")
            raise typer.Exit(code=1)
        
        console.print(f"[green]Tokenizer loaded successfully![/green]")
        
        # Register the tokenizer
        if not tokenizer_id:
            tokenizer_id = f"{tokenizer_type.lower()}_{tokenizer_path.name}"
        
        registry.register_tokenizer(tokenizer_id, tokenizer)
        console.print(f"Registered tokenizer as '{tokenizer_id}'")
        
    except Exception as e:
        console.print(f"[red]Error loading tokenizer: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def load_pretrained(
    model_name: str = typer.Option(..., help="Pretrained model name (e.g., bert-base-uncased, gpt2)"),
    tokenizer_id: str = typer.Option(None, help="ID to register the tokenizer with (default: model name)"),
    local_files_only: bool = typer.Option(False, help="Use only local files (don't download)"),
):
    """Load a pretrained tokenizer from Hugging Face."""
    console.print(f"Loading pretrained tokenizer '{model_name}'...")
    
    try:
        # Determine tokenizer type from model name
        if model_name.startswith(("bert", "roberta", "albert")):
            tokenizer = BertTokenizer.from_huggingface(model_name, local_files_only=local_files_only)
        elif model_name.startswith(("gpt", "openai")):
            tokenizer = GPTTokenizer.from_huggingface(model_name, local_files_only=local_files_only)
        # Add other model types/families as needed
        # elif model_name.startswith("t5"):
        #     from mltokenizer.models.t5 import T5Tokenizer # Example
        #     tokenizer = T5Tokenizer.from_huggingface(model_name, local_files_only=local_files_only)
        else:
            console.print(f"[red]Unsupported model type for pretrained loading: {model_name}. Supported: bert-like, gpt-like."+
                           " (extend for T5, Llama etc. if model adapters are implemented).[/red]")
            raise typer.Exit(code=1)
        
        console.print(f"[green]Tokenizer loaded successfully![/green]")
        
        # Register the tokenizer
        if not tokenizer_id:
            tokenizer_id = model_name.replace("-", "_").replace("/", "_") # Make ID more robust
        
        registry.register_tokenizer(tokenizer_id, tokenizer)
        console.print(f"Registered tokenizer as '{tokenizer_id}'")
        
    except Exception as e:
        console.print(f"[red]Error loading pretrained tokenizer: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def tokenize(
    text: str = typer.Argument(..., help="Text to tokenize"),
    tokenizer_id: str = typer.Option(..., help="ID of the tokenizer to use"),
    output_format: str = typer.Option("text", help="Output format (text, json, ids, tokens)"),
    add_special_tokens: bool = typer.Option(True, help="Add special tokens"),
):
    """Tokenize a text using a registered tokenizer."""
    # Get the tokenizer
    tokenizer = registry.get_tokenizer(tokenizer_id)
    if not tokenizer:
        console.print(f"[red]Tokenizer '{tokenizer_id}' not found[/red]")
        raise typer.Exit(code=1)
    
    # Tokenize the text
    try:
        result = tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tokens=True)
        
        # Format the output
        if output_format == "json":
            # For pydantic models, .model_dump_json() is preferred if available (pydantic v2+)
            # or .json() for pydantic v1
            try:
                console.print(result.model_dump_json(indent=2))
            except AttributeError:
                 console.print(json.dumps(result.dict(), indent=2)) # Fallback for older pydantic or non-pydantic objects
        elif output_format == "ids":
            console.print(result.input_ids)
        elif output_format == "tokens":
            if not result.tokens:
                # This case should ideally not happen if return_tokens=True is respected by encode()
                console.print("[yellow]Tokens not available in result (ensure tokenizer.encode() populates them when return_tokens=True).[/yellow]")
                # We can still print input_ids as a fallback
                console.print(f"Input IDs: {result.input_ids}") 
            else:
                console.print(result.tokens)
        else:  # text
            table = Table(title=f"Tokenization Results for [bold magenta]{tokenizer_id}[/bold magenta]")
            table.add_column("Token", style="cyan", no_wrap=False)
            table.add_column("ID", style="green")
            
            if result.tokens:
                for token, token_id in zip(result.tokens, result.input_ids):
                    table.add_row(token, str(token_id))
            else:
                # Fallback if tokens are not present
                for token_id in result.input_ids:
                    table.add_row("[i]N/A[/i]", str(token_id))
            
            console.print(table)
    
    except Exception as e:
        console.print(f"[red]Error tokenizing text: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def list_tokenizers():
    """List all registered tokenizers."""
    tokenizers = registry.list_tokenizers()
    
    if not tokenizers:
        console.print("[yellow]No tokenizers registered yet.[/yellow]")
        return
    
    table = Table(title="[bold blue]Registered Tokenizers[/bold blue]")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Vocab Size", style="magenta")
    table.add_column("Trained", style="yellow")
    
    for tokenizer_id, tokenizer_instance in tokenizers.items(): # Renamed to tokenizer_instance for clarity
        table.add_row(
            tokenizer_id,
            tokenizer_instance.tokenizer_type.value,
            str(tokenizer_instance.vocab_size),
            "[green]✓[/green]" if tokenizer_instance.is_trained else "[red]✗[/red]"
        )
    
    console.print(table)


@app.command()
def benchmark(
    tokenizer_id: str = typer.Option(..., help="ID of the tokenizer to use"),
    input_file: Path = typer.Option(..., help="Input file with texts to benchmark (one per line)"),
    batch_size: int = typer.Option(32, help="Batch size for processing"),
    iterations: int = typer.Option(5, help="Number of iterations to run"),
    warm_up: int = typer.Option(1, help="Number of warm-up iterations"),
):
    """Benchmark a tokenizer's performance."""
    # Get the tokenizer
    tokenizer = registry.get_tokenizer(tokenizer_id)
    if not tokenizer:
        console.print(f"[red]Tokenizer '{tokenizer_id}' not found[/red]")
        raise typer.Exit(code=1)
    
    if not tokenizer.is_trained:
        console.print(f"[red]Tokenizer '{tokenizer_id}' is not trained. Cannot benchmark.[/red]")
        raise typer.Exit(code=1)

    # Load benchmark data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        if not texts:
            console.print(f"[yellow]No texts found in {input_file} for benchmarking.[/yellow]")
            raise typer.Exit(code=0)
        console.print(f"Loaded {len(texts)} texts for benchmarking from {input_file}")
    except Exception as e:
        console.print(f"[red]Error loading benchmark data: {e}[/red]")
        raise typer.Exit(code=1)
    
    # Prepare batches
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    console.print(f"Running benchmark with {len(batches)} batches of size up to {batch_size}...")
    
    # Run warm-up iterations
    if warm_up > 0:
        console.print(f"Warming up for {warm_up} iterations...")
        for _ in range(warm_up):
            for batch in batches:
                tokenizer.encode_batch(texts=batch, return_tokens=False) # No need for tokens in benchmark
    
    # Run benchmark iterations
    import time
    import numpy as np
    times = []
    total_tokens_processed = 0

    console.print(f"Running {iterations} benchmark iterations...")
    with console.status("[bold green]Running benchmark...[/bold green]") as status:
        for i in range(iterations):
            start_time = time.perf_counter()
            current_iter_tokens = 0
            for batch_idx, batch in enumerate(batches):
                # Update status for long benchmarks
                status.update(f"[bold green]Iteration {i+1}/{iterations}, Batch {batch_idx+1}/{len(batches)}[/bold green]")
                encoded_batch = tokenizer.encode_batch(texts=batch, return_tokens=False) # Ensure return_tokens is False for speed
                current_iter_tokens += sum(len(ids) for ids in encoded_batch.input_ids)
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            times.append(elapsed)
            if i == 0: # Count tokens only on the first proper iteration
                total_tokens_processed = current_iter_tokens
            
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # tokens_per_second based on the token count from the first iteration and average time
    tokens_per_second = (total_tokens_processed / avg_time) if avg_time > 0 else 0
    texts_per_second = (len(texts) / avg_time) if avg_time > 0 else 0
    
    # Display results
    table = Table(title=f"[bold blue]Benchmark Results for {tokenizer_id}[/bold blue]")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Texts", str(len(texts)))
    table.add_row("Total Tokens (first iter)", f"{total_tokens_processed}")
    table.add_row("Iterations", str(iterations))
    table.add_row("Batch Size (max)", str(batch_size))
    table.add_row("Avg Time per Iteration (s)", f"{avg_time:.4f}")
    table.add_row("Std Dev Time (s)", f"{std_time:.4f}")
    table.add_row("Min Time (s)", f"{min_time:.4f}")
    table.add_row("Max Time (s)", f"{max_time:.4f}")
    table.add_row("Texts per Second", f"{texts_per_second:.2f}")
    table.add_row("Tokens per Second", f"{tokens_per_second:.2f}")
    
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
):
    """Start the tokenization server."""
    console.print(Panel(f"Starting tokenization server on [bold green]{host}:{port}[/bold green]", title="[blue]ML Tokenization Server[/blue]"))
    
    try:
        # Pass the CLI's registry instance to the server
        logger.info(f"Passing CLI TokenizerRegistry (id: {id(registry)}) to server.")
        start_server(host=host, port=port, registry_instance=registry)
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def analyze_distribution(
    input_file: Path = typer.Option(..., help="Input file with texts to analyze (one per line)"),
    tokenizer_id: str = typer.Option(..., help="ID of the tokenizer to use"),
    num_samples: Optional[int] = typer.Option(None, help="Number of text samples to analyze (default: all)"),
    output_file: Optional[Path] = typer.Option(None, help="Output file for results (JSON)"),
):
    """Analyze token distribution for a corpus."""
    # Get the tokenizer
    tokenizer = registry.get_tokenizer(tokenizer_id)
    if not tokenizer:
        console.print(f"[red]Tokenizer '{tokenizer_id}' not found[/red]")
        raise typer.Exit(code=1)
    
    if not tokenizer.is_trained:
        console.print(f"[red]Tokenizer '{tokenizer_id}' is not trained. Cannot analyze distribution.[/red]")
        raise typer.Exit(code=1)

    # Load analysis data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if num_samples and num_samples > 0 and num_samples < len(texts):
            import random
            texts_to_analyze = random.sample(texts, num_samples)
            console.print(f"Analyzing {len(texts_to_analyze)} random samples from {input_file}...")
        else:
            texts_to_analyze = texts
            console.print(f"Analyzing all {len(texts_to_analyze)} texts from {input_file}...")

        if not texts_to_analyze:
            console.print(f"[yellow]No texts to analyze from {input_file}.[/yellow]")
            raise typer.Exit(code=0)

    except Exception as e:
        console.print(f"[red]Error loading analysis data: {e}[/red]")
        raise typer.Exit(code=1)
    
    # Analyze token distribution
    token_counts = defaultdict(int)
    sequence_lengths = []
    total_tokens_corpus = 0
    
    console.print(f"Analyzing token distribution with tokenizer '{tokenizer_id}'...")
    with console.status("[bold green]Analyzing...[/bold green]") as status:
        for i, text in enumerate(texts_to_analyze):
            if (i + 1) % 100 == 0 or i == len(texts_to_analyze) -1:
                status.update(f"[bold green]Analyzing... Text {i+1}/{len(texts_to_analyze)}[/bold green]")
            
            try:
                result = tokenizer.encode(text, return_tokens=True, add_special_tokens=False) # Usually analyze without special tokens
                sequence_lengths.append(len(result.input_ids))
                total_tokens_corpus += len(result.input_ids)
                
                if result.tokens:
                    for token in result.tokens:
                        token_counts[token] += 1
                else: # Fallback if tokens are not returned, count IDs (less informative)
                    for token_id in result.input_ids:
                        token_counts[str(token_id)] +=1 # Store ID as string if token string is unknown
            except Exception as e:
                console.print(f"[yellow]Skipping text due to encoding error: {text[:50]}... Error: {e}[/yellow]")
                continue

    if not sequence_lengths: # Check if any text was successfully processed
        console.print(f"[red]No texts were successfully processed. Aborting analysis.[/red]")
        raise typer.Exit(code=1)

    # Calculate statistics
    import numpy as np
    avg_length = np.mean(sequence_lengths) if sequence_lengths else 0
    median_length = np.median(sequence_lengths) if sequence_lengths else 0
    std_length = np.std(sequence_lengths) if sequence_lengths else 0
    min_len = np.min(sequence_lengths) if sequence_lengths else 0
    max_len = np.max(sequence_lengths) if sequence_lengths else 0

    # Sort tokens by frequency
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate coverage
    # total_occurrences = sum(token_counts.values()) # This is same as total_tokens_corpus if tokens were always returned
    total_occurrences = total_tokens_corpus # Use this for consistency
    cumulative_coverage_val = 0
    coverage_stats = []
    
    if total_occurrences > 0:
        for i, (token, count) in enumerate(sorted_tokens):
            cumulative_coverage_val += count
            coverage = cumulative_coverage_val / total_occurrences
            coverage_stats.append({"rank": i + 1, "token": token, "count": count, "percentage": count / total_occurrences, "cumulative_coverage": coverage})
    
    # Display results
    console.print("--- [bold blue]Token Distribution Analysis Summary[/bold blue] ---")
    summary_table = Table(title=f"Overall Statistics for {tokenizer_id}")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Texts Analyzed", str(len(texts_to_analyze)))
    summary_table.add_row("Total Tokens in Analyzed Corpus", str(total_tokens_corpus))
    summary_table.add_row("Unique Tokens Found", str(len(token_counts)))
    summary_table.add_row("Avg Sequence Length (tokens)", f"{avg_length:.2f}")
    summary_table.add_row("Median Sequence Length", f"{median_length:.2f}")
    summary_table.add_row("Std Dev Sequence Length", f"{std_length:.2f}")
    summary_table.add_row("Min Sequence Length", str(min_len))
    summary_table.add_row("Max Sequence Length", str(max_len))
    console.print(summary_table)
    
    # Display top tokens
    console.print("--- [bold blue]Top 20 Tokens by Frequency[/bold blue] ---")
    top_tokens_table = Table(title="Top 20 Tokens")
    top_tokens_table.add_column("Rank", style="dim cyan")
    top_tokens_table.add_column("Token", style="cyan")
    top_tokens_table.add_column("Count", style="magenta")
    top_tokens_table.add_column("% of Total", style="yellow")
    top_tokens_table.add_column("Cumulative %", style="green")
    
    for stat in coverage_stats[:20]:
        top_tokens_table.add_row(
            str(stat["rank"]),
            stat["token"],
            str(stat["count"]),
            f"{stat['percentage'] * 100:.2f}%".rjust(7),
            f"{stat['cumulative_coverage'] * 100:.2f}%".rjust(7)
        )
    console.print(top_tokens_table)
    
    # Display coverage milestones
    console.print("--- [bold blue]Token Coverage Milestones[/bold blue] ---")
    if total_occurrences > 0:
        for target_cov_percent in [50, 75, 90, 95, 99, 99.9]:
            target_cov = target_cov_percent / 100.0
            found_milestone = False
            for stat in coverage_stats:
                if stat["cumulative_coverage"] >= target_cov:
                    console.print(f"  [green]{target_cov_percent}%[/green] coverage with top [magenta]{stat['rank']}[/magenta] tokens (token: '{stat["token"]}')")
                    found_milestone = True
                    break
            if not found_milestone:
                 console.print(f"  [yellow]Could not reach {target_cov_percent}% coverage with available unique tokens.[/yellow]")
    else:
        console.print("[yellow]No tokens processed, cannot show coverage.[/yellow]")

    # Save results if requested
    if output_file:
        results_to_save = {
            "tokenizer_id": tokenizer_id,
            "texts_analyzed_count": len(texts_to_analyze),
            "total_tokens_corpus": total_tokens_corpus,
            "unique_tokens_count": len(token_counts),
            "avg_sequence_length": float(avg_length) if sequence_lengths else None,
            "median_sequence_length": float(median_length) if sequence_lengths else None,
            "std_dev_length": float(std_length) if sequence_lengths else None,
            "min_sequence_length": int(min_len) if sequence_lengths else None,
            "max_sequence_length": int(max_len) if sequence_lengths else None,
            "top_tokens_stats": coverage_stats, # Contains rank, token, count, percentage, cumulative_coverage
            # Consider adding sequence length histogram if useful
            # "sequence_length_histogram": np.histogram(sequence_lengths, bins=20)[0].tolist() if sequence_lengths else None
        }
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_to_save, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Full analysis results saved to {output_file}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving results to {output_file}: {e}[/red]")


if __name__ == "__main__":
    # Potentially load some default tokenizers here if desired for immediate use
    # e.g., try:
    # logger.info(f"CLI master registry ID: {id(registry)}") # For debugging
    # load_pretrained(model_name="bert-base-uncased", tokenizer_id="bert_base_uncased_default", local_files_only=True)
    # except: pass # Fail silently if not available or not desired behavior
    app() 