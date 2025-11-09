# pfo
Python tool that solves protein structure (also known as Protein Folding Optimization) using different metaheuristics.

## Features

Metaheuristics available:
- GA: Genetic Algorithm
- PSO: Particle Swarm Optimization


## Setup

### First-Time Setup

Create a virtual environment:
```bash
python3 -m venv env
```


### Running the Optimizer

1. Activate the virtual environment:
```bash
   source env/bin/activate
```

2. Install dependencies:
```bash
   pip3 install -r requirements.txt
```

3. Run the optimizer:
```bash
   python3 src/main.py
```

### Wrapping things up

When finished, deactivate the environment:
```bash
   deactivate
```

## Development


### Code Quality
Before commiting, please run

Linting verification
```bash
flake8 ./src --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 ./src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

Black for auto-formatting, isort for sorting imports and mypy to check variable typing
```bash
black --check src/ tests/
isort --check-only src/ tests/
mypy src/ --ignore-missing-imports
```

Or simply run the bash script:
```bash
./code_quality.sh
```

### Updating Dependencies

After installing new packages, update the requirements file:
```bash
pip3 freeze > requirements.txt
```

## Project Structure
```
pfo/
├── src/
│   ├── main.py
│   ├── pfo_base.py
│   └── pfo_sa.py
├── tests/
│   ├── test_fitness.py
│   ├── test_initialization.py
│   └── test_parametrized.py
├── code_quality.sh
├── requirements.txt
└── README.md
```