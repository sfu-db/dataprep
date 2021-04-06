"""
Common functions and classes for the clean_duplication function.
"""
# pylint: disable=no-name-in-module
from string import punctuation
from unicodedata import combining, category, normalize
from base64 import b64encode
from collections import defaultdict
from operator import itemgetter
from typing import List, Set, DefaultDict
from itertools import permutations

import pandas as pd
from IPython.display import Javascript, display
from metaphone import doublemetaphone
from Levenshtein import distance


class Clusterer:
    """
    Performs clustering methods on data.
    """

    clusters: pd.Series
    _df: pd.DataFrame
    _df_name: str
    _col: str
    _ngram: int
    _radius: int
    _block_size: int

    def __init__(self, df: pd.DataFrame, col_name: str, df_name: str):
        self.clusters = pd.Series()
        self._df = df
        self._df_name = df_name
        self._col = col_name
        self._ngram = 2
        self._radius = 2
        self._block_size = 6
        self._df[self._col] = self._df[self._col].astype(str)

    def cluster(self, cluster_method: str) -> None:
        """
        Create clusters using the given clustering method.
        """
        if cluster_method == "levenshtein":
            self._nearest_neighbours_cluster()
        else:
            self._key_collision_cluster(cluster_method)

    def _key_collision_cluster(self, cluster_method: str) -> None:
        """
        Create clusters using a key collision method.
        Clusters are a Pandas Series of lists (each list represents a cluster),
        each list contains tuples with the form (item, count).
        """
        key_funcs = {
            "fingerprint": self._finger_print_key,
            "ngram-fingerprint": self._ngram_finger_print_key,
            "phonetic-fingerprint": self._phonetic_fingerprint_key,
        }
        key_func = key_funcs[cluster_method]
        col = self._col
        # get the count of each item in the dataframe and remove duplicates
        counts = self._df[col].value_counts(sort=False)
        no_dups = self._df.drop_duplicates(subset=[col]).reset_index()
        # create a column "vals" containing tuples of the form (item, count)
        no_dups.loc[:, "vals"] = no_dups[col].map(lambda val: (val, counts.loc[val]))
        # create a column "key" containing keys created by the given key collision method
        no_dups.loc[:, "key"] = no_dups[col].map(key_func)
        # put items with the same key into the same list
        clusters = no_dups.groupby("key")["vals"].agg(list)
        clusters = clusters.loc[clusters.map(len) > 1]
        # sort by the size of each cluster, so that larger clusters appear first
        clusters = clusters.sort_values(key=lambda x: x.map(len), ascending=False)
        # values with greater counts appear first in the cluster
        self.clusters = clusters.map(lambda x: sorted(x, key=itemgetter(1), reverse=True))

    def _nearest_neighbours_cluster(self) -> None:
        """
        Performs nearest neighbour clustering.
        Blocking is used to speed up the process, blocks are obtained where strings in the same
        block share a substring of a given blocking size. Only strings within the same block are
        compared using the levenshtein distance function.

        Method from OpenRefine: https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth
        and simile-vicino: https://code.google.com/archive/p/simile-vicino/
        """
        col = self._col
        blocks: DefaultDict[str, Set[str]] = defaultdict(set)
        counts = self._df[col].value_counts(sort=False)
        no_dups = self._df.drop_duplicates(subset=[col]).reset_index()

        # put strings in blocks
        no_dups[col].apply(self._populate_blocks, args=(blocks, self._block_size))
        # compare strings in the same block and create clusters
        self.clusters = self._get_nearest_neighbour_clusters(blocks, counts, self._radius)

    @staticmethod
    def _populate_blocks(val: str, blocks: DefaultDict[str, Set[str]], block_size: int) -> None:
        """
        Create n gram tokens of the given string and place the string into the block
        for each n gram.
        """
        tokens = _ngram_tokens(val, block_size)
        for token in tokens:
            blocks[token].add(val)

    @staticmethod
    def _get_nearest_neighbour_clusters(
        blocks: DefaultDict[str, Set[str]], counts: pd.Series, radius: int
    ) -> pd.Series:
        """
        Compare every pair of strings in each block and add to cluster if
        their distance is less than the given radius.
        """
        cluster_map: DefaultDict[str, Set[str]] = defaultdict(set)
        for block in blocks.values():
            for center, val in permutations(block, 2):
                if val in cluster_map[center]:
                    continue

                cluster_map[center].add(center)
                dist = distance(center, val)
                if dist <= radius or radius < 0:
                    cluster_map[center].add(val)

        # remove duplicate clusters and sort so that values with greater counts
        # appear first in the cluster.
        unique_clusters = set(
            tuple(sorted(cluster, key=lambda x: counts.loc[x], reverse=True))
            for cluster in cluster_map.values()
        )

        clusters = [
            [(x, counts.loc[x]) for x in cluster] for cluster in unique_clusters if len(cluster) > 1
        ]

        return pd.Series(sorted(clusters, key=len, reverse=True))

    @staticmethod
    def _finger_print_key(val: str) -> str:
        """
        Generates a fingerprint key from a given string.

        - remove leading and trailing whitespace
        - convert to lowercase
        - remove punctuation and control characters
        - normalize extended western characters to ASCII
        - split into whitespace separated tokens
        - sort tokens and remove duplicates
        - join tokens back together

        Method taken from OpenRefine:
        https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth
        """
        val = val.strip()
        val = val.lower()
        val = val.translate(str.maketrans("", "", punctuation))
        # remove control characters
        val = "".join(ch for ch in val if category(ch)[0] != "C")
        val = normalize_non_ascii(val)
        return " ".join(sorted(set(val.split())))

    @staticmethod
    def _phonetic_fingerprint_key(val: str) -> str:
        """
        Generates n-gram fingerprint from the given string.
        Uses the double metaphone algorithm.
        """
        primary, secondary = doublemetaphone(val)
        if primary == secondary:
            secondary = ""
        return f"{primary},{secondary}"

    def _ngram_finger_print_key(self, val: str) -> str:
        """
        Generates n-gram fingerprint from the given string.

        - convert to lowercase
        - remove punctuation, whitespace and control characters
        - get string n-grams
        - sort n-grams and remove duplicates
        - join sorted n grams back together
        - normalize extended western characters to ASCII

        Method taken from OpenRefine:
        https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth
        """
        return "".join(sorted(set(_ngram_tokens(val, self._ngram))))

    def _create_replace_calls(
        self, cluster_page: pd.Series, do_merge: List[bool], new_values: List[str]
    ) -> str:
        """
        Creates a string containing the required replace function calls.

        cluster_page
            Current page of clusters being displayed by the UI
        do_merge
            Boolean values indicating whether a cluster should be merged.
            If do_merge[i] == True then merge the i-th cluster.
        """
        df_name, col = self._df_name, f"'{self._col}'"
        replace_calls = []
        for idx, cluster in enumerate(cluster_page):
            cluster_repr = f"'{new_values[idx]}'"
            if do_merge[idx]:
                cluster_vals = [f"'{cluster_val}'" for cluster_val, _ in cluster]
                code = (
                    f"{df_name}[{col}] = {df_name}[{col}].replace"
                    f"([{', '.join(cluster_vals)}], {cluster_repr})"
                )
                replace_calls.append(code)

        return "\n" + "\n".join(replace_calls)

    def live_export_code(
        self, cluster_page: pd.Series, do_merge: List[bool], new_values: List[str]
    ) -> None:
        """
        Create DataFrame replace calls to merge clusters and output
        them into the jupyter notebook.
        """
        code = self._create_replace_calls(cluster_page, do_merge, new_values)
        encoded_code = (b64encode(str.encode(code))).decode()
        code = """
            var ind = IPython.notebook.get_selected_index();
            var cell = IPython.notebook.get_cell(ind);
            var text = cell.get_text();
            cell.set_text(text.concat(atob("{0}")));
        """.format(
            encoded_code
        )
        display(Javascript(code))

    def execute_merge_code(
        self,
        cluster_page: pd.Series,
        do_merge: List[bool],
        new_values: List[str],
    ) -> None:
        """
        Merge the clusters in the DataFrame.
        """
        for idx, cluster in enumerate(cluster_page):
            cluster_repr = new_values[idx]
            if do_merge[idx]:
                self._df.loc[:, self._col] = self._df[self._col].replace(
                    [cluster_val for cluster_val, _ in cluster], cluster_repr
                )

    def final_df(self) -> None:
        """
        Displays a DataFrame with the final values in the next notebook cell.
        """
        code = "# dataframe with cleaned string values\ndf_clean"
        encoded_code = (b64encode(str.encode(code))).decode()
        code = """
                     IPython.notebook.kernel.execute("df_clean = {0}.copy()");
                     var code = IPython.notebook.insert_cell_below('code');
                     code.set_text(atob("{1}"));
                     code.execute();
                 """.format(
            self._df_name, encoded_code
        )
        display(Javascript(code))

    def set_cluster_params(self, ngram: int, radius: int, block_size: int) -> None:
        """
        Set clustering parameters.
        """
        self._ngram = ngram
        self._radius = radius
        self._block_size = block_size


def _ngram_tokens(val: str, n: int) -> List[str]:
    """
    Create n-gram tokens from the given string.
    """
    val = val.strip()
    val = val.lower()
    val = " ".join(val.split())
    val = val.translate(str.maketrans("", "", punctuation))
    # remove control characters
    val = "".join(ch for ch in val if category(ch)[0] != "C")
    val = normalize_non_ascii(val)
    n_grams = []
    for i in range(len(val) - n + 1):
        n_grams.append(val[i : i + n])
    return n_grams


def normalize_non_ascii(val: str) -> str:
    """
    Normalize extended western characters to ascii. (remove accents)
    """
    nfkd_form = normalize("NFKD", val)
    return "".join([c for c in nfkd_form if not combining(c)])
