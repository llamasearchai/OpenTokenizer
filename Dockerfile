FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install Poetry
RUN pip install poetry==1.4.2

# Copy only dependencies to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not use a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies including dev for maturin build
# We install all dependencies here as maturin might need some dev ones.
# If specific groups are needed, --only <group> can be used.
RUN poetry install --no-interaction --no-dev

# Copy the project
COPY src ./src
COPY Cargo.toml ./Cargo.toml
# If Cargo.lock exists and is important for reproducible Rust builds, copy it too.
# COPY Cargo.lock ./Cargo.lock 
# If there are Rust sources outside of a default location maturin finds, copy them.
# e.g. COPY rust_src ./rust_src

# Build Rust extensions
# Ensure that maturin builds into a location that is then picked up.
# `maturin develop` installs to site-packages.
RUN poetry run maturin develop --release

# Build final image
FROM python:3.10-slim

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Docker image user" appuser

WORKDIR /app

# Copy installed Python packages from builder image
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy application code from builder image
# This assumes your application code is in the 'src' directory,
# and the packages within 'src' should be directly in '/app' for PYTHONPATH.
# COPY --from=builder --chown=appuser:appuser /app/src /app # Removed as package is installed

# Set environment variables
# ENV PYTHONPATH=/app # Removed as package is installed

# Ensure /app is owned by appuser, especially if WORKDIR creates it as root initially
RUN chown -R appuser:appuser /app
USER appuser

# Run the tokenization service
CMD ["python", "-m", "mltokenizer.server.api"]