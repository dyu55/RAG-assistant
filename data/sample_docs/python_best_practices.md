# Python Best Practices Guide

## Code Style and Structure

Python follows the PEP 8 style guide for code formatting. Key principles include using 4 spaces for indentation, keeping lines under 79 characters for code and 72 for comments, and using meaningful variable names.

### Naming Conventions

- Variables and functions use snake_case: `my_variable`, `calculate_total()`
- Classes use PascalCase: `MyClass`, `DataProcessor`
- Constants use UPPER_SNAKE_CASE: `MAX_RETRIES`, `DEFAULT_TIMEOUT`
- Private attributes use a leading underscore: `_internal_method()`

### Type Hints

Modern Python (3.10+) encourages type hints for better code documentation and IDE support:

```python
def process_data(items: list[str], threshold: float = 0.5) -> dict[str, int]:
    results: dict[str, int] = {}
    for item in items:
        if len(item) > threshold:
            results[item] = len(item)
    return results
```

## Error Handling

Python uses try/except blocks for error handling. Best practices include:

1. Catch specific exceptions rather than bare `except:` clauses
2. Use `finally` blocks for cleanup code
3. Create custom exception classes for domain-specific errors
4. Log exceptions with full traceback for debugging

```python
class DataProcessingError(Exception):
    """Raised when data processing fails."""
    pass

try:
    result = process_data(raw_input)
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
    raise DataProcessingError(f"Failed to process: {e}") from e
except IOError as e:
    logger.error(f"I/O error during processing: {e}")
    raise
finally:
    cleanup_resources()
```

## Virtual Environments

Always use virtual environments to isolate project dependencies:

- `python -m venv .venv` to create a virtual environment
- `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows) to activate
- `pip install -r requirements.txt` to install dependencies
- `pip freeze > requirements.txt` to capture current dependencies

## Testing

Python testing best practices:

- Use `pytest` as the testing framework
- Write tests in a `tests/` directory mirroring the project structure
- Aim for at least 80% code coverage
- Use fixtures for shared test setup
- Mock external dependencies with `unittest.mock` or `pytest-mock`

```python
import pytest
from myapp.processor import DataProcessor

@pytest.fixture
def processor():
    return DataProcessor(config={"timeout": 30})

def test_process_valid_input(processor):
    result = processor.process(["hello", "world"])
    assert len(result) == 2
    assert "hello" in result

def test_process_empty_input(processor):
    result = processor.process([])
    assert result == {}
```

## Logging

Use Python's built-in `logging` module instead of print statements:

- Configure logging at application startup
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Include contextual information in log messages
- Use structured logging for production systems

```python
import logging

logger = logging.getLogger(__name__)

def process_request(request_id: str, data: dict) -> dict:
    logger.info(f"Processing request {request_id}")
    try:
        result = transform(data)
        logger.debug(f"Request {request_id} transformed successfully")
        return result
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}", exc_info=True)
        raise
```

## Performance Optimization

Key performance tips for Python applications:

1. Use generators for large datasets to reduce memory usage
2. Prefer list comprehensions over loops for simple transformations
3. Use `functools.lru_cache` for expensive function calls with repeated inputs
4. Profile before optimizing with `cProfile` or `line_profiler`
5. Consider using `asyncio` for I/O-bound operations
6. Use `multiprocessing` for CPU-bound parallel tasks

## Dependency Management

Modern Python projects should use:

- `pyproject.toml` for project metadata and build configuration
- `pip-tools` or `poetry` for dependency resolution
- Pin exact versions in production, use ranges in libraries
- Regularly audit dependencies for security vulnerabilities with `pip-audit`
