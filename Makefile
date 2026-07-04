export PYTHONPATH := .
.PHONY : run lint format all clean test ml-preprocess ml-train ml-test ml-pipeline


install:
	pip install -r requirements.txt
run:
	uvicorn RetailApp.main:app --host 127.0.0.1 --port 8000 --reload
lint:
	ruff check RetailApp
format:
	isort RetailApp
	ruff format RetailApp
	black RetailApp
test:
	pytest -v ./test --tb=long --showlocals
report:
	pytest --html=report.html --self-contained-html test
	pytest --cov=RetailApp --cov-report=html test
all: format run


# ============================================================================
# MACHINE LEARNING WORKFLOW TARGETS
# ============================================================================

ml-preprocess:
	python -m PricerMlModel.src.preprocess

ml-train:
	python -m PricerMlModel.src.train

ml-test:
	python -m PricerMlModel.src.test

ml-pipeline: ml-preprocess ml-train ml-test


clean:
	@powershell -Command "Get-ChildItem -Path . -Include *.pyc, *.pyd -Recurse -File | Remove-Item -Force"
	@powershell -Command "If (Test-Path .vscode) { Remove-Item -Path .vscode -Recurse -Force }"
	@powershell -Command "If (Test-Path .ruff_cache) { Remove-Item -Path .ruff_cache -Recurse -Force }"
	@powershell -Command "If (Test-Path .pytest_cache) { Remove-Item -Path .pytest_cache -Recurse -Force }"
	@powershell -Command "If (Test-Path report.html) { Remove-Item -Path report.html -Force }"
	@powershell -Command "If (Test-Path htmlcov) { Remove-Item -Path htmlcov -Recurse -Force }"
	@powershell -Command "If (Test-Path .pytest_cache) { Remove-Item -Path .pytest_cache -Recurse -Force }"
	@powershell -Command "If (Test-Path .coverage) { Remove-Item -Path .coverage -Force }"
	@powershell -Command "Get-ChildItem -Path . -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force"
