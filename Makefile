# Credit to sgn-cal project, most of this Makefile is adapted from there
ifeq ($(PYTHONPATH),)
	PYTHONPATH := .
else
	PYTHONPATH := .:$(PYTHONPATH)
endif
export PYTHONPATH

# Optional flags that can be overridden on command line
FLAGS ?=

.PHONY: all
all: format lint check test

.PHONY: help
help:
	@echo
	@echo 'Commands:'
	@echo
	@echo '  make format-check          run code formatter'
	@echo '  make format-fix            fix code formatting'
	@echo '  make lint-check            run linter'
	@echo '  make lint-fix              fix linter issues'
	@echo '  make check                 run static type checker'
	@echo '  make test                  run tests'
	@echo '  make all                   all of the above'
	@echo

.PHONY: test test-run
test:
	pytest -v --cov=gw_signal_tools --cov-report=term-missing $(FLAGS) tests

.PHONY: lint-check lint-fix
lint-check:
	ruff check $(FLAGS) gw_signal_tools tests

lint-fix:
	ruff check --fix $(FLAGS) gw_signal_tools tests

.PHONY: format-check format-fix
format-check:
	ruff format --diff $(FLAGS) gw_signal_tools tests

format-fix:
	ruff format $(FLAGS) gw_signal_tools tests

.PHONY: check
check:
	mypy $(FLAGS) gw_signal_tools
