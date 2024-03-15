"""
module for testing the function clean_headers()
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_headers

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_headers() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ISBN": [9781455582341],
            "isbn": [1455582328],
            "bookTitle": ["How Google Works"],
            "__Author": ["Eric Schmidt, Jonathan Rosenberg"],
            "Publication (year)": [2014],
            "éditeur": ["Grand Central Publishing"],
            "Number_Of_Pages": [305],
            "★ Rating": [4.06],
        }
    )
    return df


@pytest.fixture(scope="module")  # type: ignore
def df_null_headers() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "": [9781455582341],
            np.nan: ["How Google Works"],
            None: ["Eric Schmidt, Jonathan Rosenberg"],
            "N/A": [2014],
        }
    )
    return df


def test_clean_default(df_headers: pd.DataFrame) -> None:
    df_clean = clean_headers(df_headers)
    df_check = pd.DataFrame(
        {
            "isbn": [9781455582341],
            "isbn_1": [1455582328],
            "book_title": ["How Google Works"],
            "author": ["Eric Schmidt, Jonathan Rosenberg"],
            "publication_year": [2014],
            "editeur": ["Grand Central Publishing"],
            "number_of_pages": [305],
            "rating": [4.06],
        }
    )
    assert df_check.equals(df_clean)


def test_clean_case_style(df_headers: pd.DataFrame) -> None:
    df_clean_kebab = clean_headers(df_headers, case="kebab")
    df_clean_camel = clean_headers(df_headers, case="camel")
    df_clean_pascal = clean_headers(df_headers, case="pascal")
    df_clean_const = clean_headers(df_headers, case="const")
    df_clean_sentence = clean_headers(df_headers, case="sentence")
    df_clean_title = clean_headers(df_headers, case="title")
    df_clean_lower = clean_headers(df_headers, case="lower")
    df_clean_upper = clean_headers(df_headers, case="upper")
    df_check_kebab = pd.DataFrame(
        {
            "isbn": [9781455582341],
            "isbn-1": [1455582328],
            "book-title": ["How Google Works"],
            "author": ["Eric Schmidt, Jonathan Rosenberg"],
            "publication-year": [2014],
            "editeur": ["Grand Central Publishing"],
            "number-of-pages": [305],
            "rating": [4.06],
        }
    )
    df_check_camel = pd.DataFrame(
        {
            "isbn": [9781455582341],
            "isbn1": [1455582328],
            "bookTitle": ["How Google Works"],
            "author": ["Eric Schmidt, Jonathan Rosenberg"],
            "publicationYear": [2014],
            "editeur": ["Grand Central Publishing"],
            "numberOfPages": [305],
            "rating": [4.06],
        }
    )
    df_check_pascal = pd.DataFrame(
        {
            "Isbn": [9781455582341],
            "Isbn1": [1455582328],
            "BookTitle": ["How Google Works"],
            "Author": ["Eric Schmidt, Jonathan Rosenberg"],
            "PublicationYear": [2014],
            "Editeur": ["Grand Central Publishing"],
            "NumberOfPages": [305],
            "Rating": [4.06],
        }
    )
    df_check_const = pd.DataFrame(
        {
            "ISBN": [9781455582341],
            "ISBN_1": [1455582328],
            "BOOK_TITLE": ["How Google Works"],
            "AUTHOR": ["Eric Schmidt, Jonathan Rosenberg"],
            "PUBLICATION_YEAR": [2014],
            "EDITEUR": ["Grand Central Publishing"],
            "NUMBER_OF_PAGES": [305],
            "RATING": [4.06],
        }
    )
    df_check_sentence = pd.DataFrame(
        {
            "Isbn": [9781455582341],
            "Isbn 1": [1455582328],
            "Book title": ["How Google Works"],
            "Author": ["Eric Schmidt, Jonathan Rosenberg"],
            "Publication (year)": [2014],
            "Editeur": ["Grand Central Publishing"],
            "Number of pages": [305],
            "Rating": [4.06],
        }
    )
    df_check_title = pd.DataFrame(
        {
            "Isbn": [9781455582341],
            "Isbn 1": [1455582328],
            "Book Title": ["How Google Works"],
            "Author": ["Eric Schmidt, Jonathan Rosenberg"],
            "Publication (year)": [2014],
            "Editeur": ["Grand Central Publishing"],
            "Number Of Pages": [305],
            "Rating": [4.06],
        }
    )
    df_check_lower = pd.DataFrame(
        {
            "isbn": [9781455582341],
            "isbn 1": [1455582328],
            "book title": ["How Google Works"],
            "author": ["Eric Schmidt, Jonathan Rosenberg"],
            "publication (year)": [2014],
            "editeur": ["Grand Central Publishing"],
            "number of pages": [305],
            "rating": [4.06],
        }
    )
    df_check_upper = pd.DataFrame(
        {
            "ISBN": [9781455582341],
            "ISBN 1": [1455582328],
            "BOOK TITLE": ["How Google Works"],
            "AUTHOR": ["Eric Schmidt, Jonathan Rosenberg"],
            "PUBLICATION (YEAR)": [2014],
            "EDITEUR": ["Grand Central Publishing"],
            "NUMBER OF PAGES": [305],
            "RATING": [4.06],
        }
    )
    assert df_check_kebab.equals(df_clean_kebab)
    assert df_check_camel.equals(df_clean_camel)
    assert df_check_pascal.equals(df_clean_pascal)
    assert df_check_const.equals(df_clean_const)
    assert df_check_sentence.equals(df_clean_sentence)
    assert df_check_title.equals(df_clean_title)
    assert df_check_lower.equals(df_clean_lower)
    assert df_check_upper.equals(df_clean_upper)


def test_clean_replace(df_headers: pd.DataFrame) -> None:
    df_clean = clean_headers(df_headers, replace={"éditeur": "publisher", "★": "star"})
    df_check = pd.DataFrame(
        {
            "isbn": [9781455582341],
            "isbn_1": [1455582328],
            "book_title": ["How Google Works"],
            "author": ["Eric Schmidt, Jonathan Rosenberg"],
            "publication_year": [2014],
            "publisher": ["Grand Central Publishing"],
            "number_of_pages": [305],
            "star_rating": [4.06],
        }
    )
    assert df_check.equals(df_clean)


def test_clean_keep_accents(df_headers: pd.DataFrame) -> None:
    df_clean = clean_headers(df_headers, remove_accents=False)
    df_check = pd.DataFrame(
        {
            "isbn": [9781455582341],
            "isbn_1": [1455582328],
            "book_title": ["How Google Works"],
            "author": ["Eric Schmidt, Jonathan Rosenberg"],
            "publication_year": [2014],
            "éditeur": ["Grand Central Publishing"],
            "number_of_pages": [305],
            "★_rating": [4.06],
        }
    )
    assert df_check.equals(df_clean)


def test_clean_null_headers(df_null_headers: pd.DataFrame) -> None:
    df_clean = clean_headers(df_null_headers)
    df_check = pd.DataFrame(
        {
            "header": [9781455582341],
            "header_1": ["How Google Works"],
            "header_2": ["Eric Schmidt, Jonathan Rosenberg"],
            "n_a": [2014],
        }
    )
    assert df_check.equals(df_clean)
