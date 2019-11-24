build-docs:
  pipenv run sphinx-build -M html docs/source docs/build

ci: ci-black ci-pytype ci-mypy ci-pytest ci-pylint

ci-black:
  pipenv run black --check --quiet dataprep

ci-pytype:
  pipenv run pytype dataprep

ci-pytest:
  pipenv run pytest dataprep

ci-mypy:
  pipenv run mypy dataprep

ci-pylint:
  pipenv run pylint dataprep