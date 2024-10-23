# Development

Set up linting and formatting for development:
```bash
# [optional] step 1: use venv
python3 -m venv .venv
source .venv/bin/activate

# step 2: install
pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt

# step 3: pre-commit
pre-commit install
```

run pre-commit:
```bash
# pre-commit behaviour is defined in .pre-commit-config.yaml
pre-commit run --all-files
# or point to specific files
pre-commit run --files path/to/file
```
