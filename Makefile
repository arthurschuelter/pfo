.DEFAULT_GOAL := run

run: 
	python3 src/main.py

test: 
	pytest tests/ -v --cov=src --cov-report=xml --cov-report=term

fix:
	black src/ tests/
	isort src/ tests/

check: 
	black src/ tests/ --check
	isort src/ tests/ --check-only
	flake8 src/ tests/
	mypy src/ tests/ --ignore-missing-imports