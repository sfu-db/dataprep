build-docs:
  poetry run sphinx-build -M html docs/source docs/build

publish-docs: build-docs
  touch docs/build/html/.nojekyll
  gh-pages --dotfiles --message "[skip ci] Updates" --dist docs/build/html
  
gen-apidocs:
  poetry run sphinx-apidoc --ext-doctest --ext-autodoc --ext-mathjax -f -o docs/source dataprep

black:
  poetry run black dataprep
  
ci: format ci-black typeck test lint

ci-black:
  poetry run black --check --quiet dataprep

format:
  poetry run black dataprep

typeck: ci-mypy

test:
  poetry run pytest dataprep

testf +ARGS="dataprep":
  poetry run pytest {{ARGS}}

lint:
  poetry run pylint dataprep

ci-mypy:
  poetry run mypy dataprep

build:
  poetry build

release version:
  #! /usr/bin/env bash

  # Sanity checks

  arr=(major minor patch)

  if [[ " ${arr[*]} " != *" {{version}} "* ]]; then
    echo "version must be one of 'major', 'minor', 'patch', got '{{version}}'";
    exit 1;
  fi

  if [ ! -z "$(git status --porcelain)" ]; then echo "Git tree is not clean, commit first"; exit 1; fi

  if [ ! -z "$(git rev-parse --verify release)" ]; then echo "delete the existing release branch before new release"; exit 1; fi

  # Pre bump the version to get the next version number
  git checkout develop

  vstring="$(poetry version {{version}})"
  if [ $? -ne 0 ]; then
    echo $vstring;
    exit 1;
  fi
  
  from_version=$(echo "${vstring}" | sed -nr "s/^Bumping version from ([0-9]+\.[0-9]+\.[0-9]+) to ([0-9]+\.[0-9]+\.[0-9]+)$/\1/p")
  to_version=$(echo "${vstring}" | sed -nr "s/^Bumping version from ([0-9]+\.[0-9]+\.[0-9]+) to ([0-9]+\.[0-9]+\.[0-9]+)$/\2/p")

  git checkout pyproject.toml # clean up

  echo "Releasing from ${from_version} to ${to_version}?"
  select yn in "Yes" "No"; do
      case $yn in
          Yes ) break;;
          No ) git checkout pyproject.toml; git checkout develop; git branch -D release; exit;;
      esac
  done

  echo ================ Release Note ================
  poetry run python scripts/release-note.py $(git rev-parse develop)
  echo ================ Release Note ================
  echo
  echo Does release note looks good?

  select yn in "Yes" "No"; do
      case $yn in
          Yes ) break;;
          No ) exit;;
      esac
  done

  # Begin of the real stuff!

  # Create new release branch
  git checkout -b "release/v${to_version}" develop

  poetry version {{version}}
  
  echo "Creating release commit"
  git add pyproject.toml
  semantic-release version --{{version}}
  
  # echo "Merge release/v${to_version} to master & develop"
  # git checkout master
  # git merge "release/v${to_version}"

  # git checkout develop
  # git merge "release/v${to_version}"

  echo "Push branch and tag to remote"
  git push origin "release/v${to_version}":master
  git push origin "release/v${to_version}":develop
  git push origin "release/v${to_version}"
  git push origin "v${to_version}"

  echo "Build artifacts"
  poetry build

  echo "Creating release draft"

  poetry run python scripts/release-note.py $(git rev-parse release/v${to_version}^) | sed "1iv${to_version}\n" | hub release create -d -a "dist/dataprep-${to_version}-py3-none-any.whl" -a "dist/dataprep-${to_version}.tar.gz" -F - "v${to_version}"


@ensure-git-clean:
  if [ ! -z "$(git status --porcelain)" ]; then echo "Git tree is not clean, commit first"; exit 1; fi

@release-note hash="":
    echo ================ Release Note ================
    poetry run python scripts/release-note.py {{hash}}
    echo ================ Release Note ================
