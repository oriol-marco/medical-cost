.PHONY: notebook docs tests

install: 
	@echo "Installing..."
	pip install -r dev-requirements.txt

activate:
	@echo "Activating virtual environment"
	source .venv/bin/activate

pull_data:
	dvc pull