build-docs:
  poetry run sphinx-build -M html docs/source docs/build

black:
  poetry run black dataprep
  
ci: format ci-black typeck test lint

ci-black:
  poetry run black --check --quiet dataprep

format:
  poetry run black dataprep

typeck: ci-pytype ci-mypy

test:
  poetry run pytest dataprep

testf +ARGS="dataprep":
  poetry run pytest {{ARGS}}

lint:
  poetry run pylint dataprep

ci-pytype:
  poetry run pytype dataprep

ci-mypy:
  poetry run mypy dataprep

build:
  poetry build