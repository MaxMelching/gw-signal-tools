ifeq ($(PYTHONPATH),)
	PYTHONPATH := .
else
	PYTHONPATH := .:$(PYTHONPATH)
endif
export PYTHONPATH


.PHONY: all
all: format lint check test

.PHONY: help
help:
	@echo
	@echo 'Commands:'
	@echo
	@echo '  make format                run code formatter'
	@echo '  make lint                  run linter'
	@echo '  make check                 run static type checker'
	@echo '  make test                  run tests'
	@echo '  make all                   all of the above'
	@echo

.PHONY: test test-run
test:
	pytest -v --cov=gw_signal_tools --cov-report=term-missing tests

.PHONY: lint
lint:
	ruff check gw_signal_tools tests

.PHONY: format
format:
	ruff format gw_signal_tools tests
	ruff check --fix gw_signal_tools tests

.PHONY: check
check:
	mypy gw_signal_tools
