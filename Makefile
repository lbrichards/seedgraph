.PHONY: install test run clean

install:
	poetry install

test:
	poetry run pytest -q

run:
	poetry run seedgraph --help

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build
