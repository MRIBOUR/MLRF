#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlrf_cookiecutter
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ mlrf.egg-info/ venv/

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 mlrf
	isort --check --diff --profile black mlrf
	black --check --config pyproject.toml mlrf

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml mlrf

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	python -m venv venv && \
	. venv/bin/activate && \
	pip install .


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
