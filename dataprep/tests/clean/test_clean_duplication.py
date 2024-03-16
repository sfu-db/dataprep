"""
module for testing the clean_duplication() function
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean.clean_duplication import UserInterface

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def clean_duplication_ui() -> UserInterface:
    df = pd.DataFrame(
        {
            "city": [
                "Québec",
                "Québec",
                "Québec",
                "Quebec",
                "Quebec",
                "quebec",
                "vancouver",
                "vancouver",
                "vancouverr",
                "Vancouver",
                "Vancouver",
                "Vancouver",
                "van",
                "Ottowa",
                "Ottowa",
                "otowa",
                "hello",
                np.nan,
            ]
        }
    )
    return UserInterface(df, "city", "df", 5)


def test_fingerprint_clusters(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._clustering_method_drop.value = "fingerprint"
    clusters = clean_duplication_ui._clusterer.get_page(0, 5)
    clusters_check = pd.Series(
        [[("Québec", 3), ("Quebec", 2), ("quebec", 1)], [("Vancouver", 3), ("vancouver", 2)]],
        name="city",
    )

    assert clusters_check.equals(clusters)


def test_ngram_clusters(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._clustering_method_drop.value = "ngram-fingerprint"
    clusters = clean_duplication_ui._clusterer.get_page(0, 5)
    clusters_check = pd.Series(
        [
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Vancouver", 3), ("vancouver", 2)],
        ],
        name="city",
    )
    # set the ngram size to 1
    clean_duplication_ui._ngram_text.value = "1"
    clusters2 = clean_duplication_ui._clusterer.get_page(0, 5)
    # check for either ordering of clusters, since they're
    # only sorted by the length of the cluster the order isn't
    # guaranteed
    clusters_check2 = pd.Series(
        [
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
            [("Ottowa", 2), ("otowa", 1)],
        ],
        name="city",
    )
    clusters_check3 = pd.Series(
        [
            [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Ottowa", 2), ("otowa", 1)],
        ],
        name="city",
    )

    assert clusters_check.equals(clusters)
    assert clusters_check2.equals(clusters2) or clusters_check3.equals(clusters2)


def test_phonetic_clusters(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._clustering_method_drop.value = "phonetic-fingerprint"
    clusters = clean_duplication_ui._clusterer.get_page(0, 5)
    # check for either ordering of clusters, since they're
    # only sorted by the length of the cluster the order isn't
    # guaranteed
    clusters_check = pd.Series(
        [
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
            [("Ottowa", 2), ("otowa", 1)],
        ],
        name="city",
    )

    clusters_check2 = pd.Series(
        [
            [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
            [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
            [("Ottowa", 2), ("otowa", 1)],
        ],
        name="city",
    )
    assert clusters_check.equals(clusters) or clusters_check2.equals(clusters)


# def test_levenshtein_clusters(clean_duplication_ui: UserInterface) -> None:
#     clean_duplication_ui._clustering_method_drop.value = "levenshtein"
#     clusters = clean_duplication_ui._clusterer.get_page(0, 5)
#     # check for either ordering of clusters, since they're
#     # only sorted by the length of the cluster the order isn't
#     # guaranteed
#     clusters_check = pd.Series(
#         [
#             [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
#             [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
#         ],
#         name="city",
#     )
#     clusters_check2 = pd.Series(
#         [
#             [("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)],
#             [("Québec", 3), ("Quebec", 2), ("quebec", 1)],
#         ],
#         name="city",
#     )
#     clean_duplication_ui._block_chars_text.value = "7"
#     clusters2 = clean_duplication_ui._clusterer.get_page(0, 5)
#     clusters_check3 = pd.Series(
#         [[("Vancouver", 3), ("vancouver", 2), ("vancouverr", 1)]],
#         name="city",
#     )
#     assert clusters_check.equals(clusters) or clusters_check2.equals(clusters)
#     assert clusters_check3.equals(clusters2)


def test_merge(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._clustering_method_drop.value = "fingerprint"
    # select the checkbox for the first cluster and
    # set the textbox contents to "hi"
    clean_duplication_ui._checks[0].value = True
    clean_duplication_ui._reprs[0].value = "hi"
    clean_duplication_ui._execute_merge({})
    # get the dataframe after merging
    df_clean = clean_duplication_ui._clusterer._df.compute()
    df_check = df_clean.copy()
    df_check["city"] = [
        "hi",
        "hi",
        "hi",
        "hi",
        "hi",
        "hi",
        "vancouver",
        "vancouver",
        "vancouverr",
        "Vancouver",
        "Vancouver",
        "Vancouver",
        "van",
        "Ottowa",
        "Ottowa",
        "otowa",
        "hello",
        "nan",
    ]

    assert df_check.equals(df_clean)


def test_select_all(clean_duplication_ui: UserInterface) -> None:
    clean_duplication_ui._sel_all.value = True
    assert all(check.value for check in clean_duplication_ui._checks)
