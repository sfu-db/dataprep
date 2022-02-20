set dotenv-load := true

#### Documentations ####
doc-clean-notebooks:
    fd ".*\.ipynb" -t f docs/source/ -x jupyter nbconvert --clear-output

build-docs:
  poetry run sphinx-build -M html docs/source docs/build

publish-docs: build-docs
  touch docs/build/html/.nojekyll
  gh-pages --dotfiles --message "[skip ci] Updates" --dist docs/build/html
  
gen-apidocs:
  poetry run sphinx-apidoc --ext-doctest --ext-autodoc --ext-mathjax -f -o docs/source dataprep

#### CI ####
ci: black pyright test pylint

black:
  poetry run black dataprep

test +ARGS="":
  poetry run pytest dataprep/tests {{ARGS}}

pylint:
  poetry run pylint dataprep

pyright:
  poetry run pyright dataprep

build:
  poetry build

@release-note hash="":
    echo ================ Release Note ================
    poetry run python scripts/release-note.py {{hash}}
    echo ================ Release Note ================

setuppy:
    python scripts/gen-setup.py
