# AI Staffing Copilot

AI Staffing Copilot is a project scaffold for building a data-driven candidate matching and staffing assistant. It includes modules for synthetic data generation, matching algorithms, optimization, explainability, and a dashboard-ready interface.

## Setup

1. Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Open the workspace in VS Code and use the `notebooks/` folder for exploratory analysis.

## Project structure

- `data/raw/` - raw input data files
- `data/processed/` - cleaned or transformed datasets
- `src/` - Python modules for system components
- `notebooks/` - Jupyter notebooks for exploration and demos
- `tests/` - unit tests for core modules
- `docs/` - project documentation
- `config/` - configuration files and environment settings

## Files

- `src/data_generator.py` - synthetic data generation
- `src/matcher.py` - candidate-job matching logic
- `src/optimizer.py` - optimization routines
- `src/explainer.py` - explainability helpers
- `src/dashboard.py` - dashboard or UI integration
