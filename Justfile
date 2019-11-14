build-docs:
  poetry run sphinx-build -M html docs/source docs/build

black:
  poetry run black dataprep
  
ci: ci-black ci-pytype ci-mypy ci-pytest ci-pylint

ci-black:
  poetry run black --check --quiet dataprep

ci-pytype:
  poetry run pytype dataprep

ci-pytest:
  poetry run pytest dataprep

ci-mypy:
  poetry run mypy dataprep

ci-pylint:
  poetry run pylint dataprep
