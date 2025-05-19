#!/usr/bin/env python
"""
Run All Tests

This script runs all tests in the repository, including unit tests, integration tests,
performance tests, and multilingual tests. It provides a summary of test results.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_tests(test_path, label, verbose=False):
    """Run pytest on the specified path and return the results.
    
    Args:
        test_path: Path to the test directory or file
        label: Label for the test category
        verbose: Whether to show detailed output
        
    Returns:
        Tuple of (success, total tests, duration)
    """
    start_time = time.time()
    
    # Build the pytest command
    cmd = ["python", "-m", "pytest", test_path]
    if verbose:
        cmd.append("-v")
    
    # Run pytest
    print(f"\n======== Running {label} Tests ========")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Execute the tests
    process = subprocess.run(cmd, capture_output=not verbose, text=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Parse the output to get test results
    output = process.stdout if process.stdout else ""
    success = process.returncode == 0
    
    # Try to extract number of tests from output
    import re
    result_match = re.search(r'(\d+) passed, (\d+) failed', output)
    if result_match:
        passed = int(result_match.group(1))
        failed = int(result_match.group(2))
        total = passed + failed
    else:
        # If we can't parse the output, use a placeholder
        total = "?"
    
    # Print a summary
    status = "PASSED" if success else "FAILED"
    print(f"\n{label} Tests: {status} in {duration:.2f}s")
    return (success, total, duration)


def main():
    """Run all tests in the repository."""
    print("Running All Tests")
    print("================\n")
    
    # Determine the project root directory
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    os.chdir(project_root)
    
    # Define test categories
    test_categories = [
        ("tests/python/unit", "Unit"),
        ("tests/python/integration", "Integration"),
        ("tests/python/performance", "Performance"),
        ("tests/python/multilingual", "Multilingual"),
        ("tests/rust/unit", "Rust Unit")
    ]
    
    # Track results
    results = {}
    all_success = True
    total_time = 0
    
    # Run each category of tests
    for test_path, label in test_categories:
        # Skip if the directory doesn't exist
        if not Path(test_path).exists():
            print(f"\n{label} Tests: SKIPPED (directory not found)")
            results[label] = (True, 0, 0)  # Mark as success but 0 tests
            continue
        
        success, total, duration = run_tests(test_path, label, verbose=False)
        results[label] = (success, total, duration)
        total_time += duration
        
        if not success:
            all_success = False
    
    # Print summary
    print("\n======== Test Summary ========")
    for label, (success, total, duration) in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{label} Tests: {status} ({total} tests in {duration:.2f}s)")
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Overall result: {'PASSED' if all_success else 'FAILED'}")
    
    # Return appropriate exit code
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main()) 