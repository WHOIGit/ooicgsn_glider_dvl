#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ooicgsn_glider_dvl
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create the uv environment (.venv) and register its Jupyter kernel
.PHONY: create_environment
create_environment: requirements kernel

## Install the locked Python dependencies into .venv
.PHONY: requirements
requirements:
	uv sync

## Also install the OOI data-access packages (needed by notebooks 1 and 3.1)
.PHONY: ooi
ooi:
	uv sync --group ooi

## Register the .venv as the Jupyter kernel used by the notebooks
.PHONY: kernel
kernel:
	uv run python -m ipykernel install --user --name $(PROJECT_NAME) --display-name "Python ($(PROJECT_NAME))"

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -path ./.venv -prune -o -type f -name "*.py[co]" -exec rm -f {} +
	find . -path ./.venv -prune -o -type d -name "__pycache__" -exec rm -rf {} +

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format

## Serve the documentation locally
.PHONY: docs
docs:
	cd docs && uv run --group docs mkdocs serve

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
