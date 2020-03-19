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

release version:
  #! /usr/bin/env bash
  if [ ! -z "$(git status --porcelain)" ]; then echo "Git tree is not clean, commit first"; exit 1; fi

  vstring="$(poetry version {{version}})"
  if [ $? -ne 0 ]; then
    echo $vstring;
    exit 1;
  fi
  
  from_version=$(echo "${vstring}" | sed -nr "s/^Bumping version from ([0-9]+\.[0-9]+\.[0-9]+) to ([0-9]+\.[0-9]+\.[0-9]+)$/\1/p")
  to_version=$(echo "${vstring}" | sed -nr "s/^Bumping version from ([0-9]+\.[0-9]+\.[0-9]+) to ([0-9]+\.[0-9]+\.[0-9]+)$/\2/p")

  echo "Releasing from ${from_version} to ${to_version}?"
  select yn in "Yes" "No"; do
      case $yn in
          Yes ) break;;
          No ) git checkout pyproject.toml; exit;;
      esac
  done

  git add pyproject.toml
  semantic-release version --{{version}}
  echo "Creating release draft"
  semantic-release changelog | sed "1iv${to_version}\n" | hub release create -F - "v${to_version}"


@ensure-git-clean:
  if [ ! -z "$(git status --porcelain)" ]; then echo "Git tree is not clean, commit first"; exit 1; fi