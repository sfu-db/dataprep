"""
Common functions and classes for the clean_duplication function.
"""

# pylint: disable=no-name-in-module
from string import punctuation
from unicodedata import combining, category, normalize
from base64 import b64encode
from collections import defaultdict
from operator import itemgetter
from typing import List, Set, Union, DefaultDict
from itertools import permutations
from os import path
from tempfile import mkdtemp
import pandas as pd
import dask.dataframe as dd
import dask
from IPython.display import Javascript, display
from metaphone import doublemetaphone
from rapidfuzz.distance.Levenshtein import distance as LevenshteinDistance

from .utils import to_dask


DECODE_FUNC = """
    function b64DecodeUnicode(str) {
        // Going backwards: from bytestream, to percent-encoding, to original string.
        return decodeURIComponent(atob(str).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
    }
"""


class Clusterer:
    """
    Performs clustering methods on data.
    """

    # pylint: disable=too-many-instance-attributes

    clusters: pd.Series
    _df: dd.DataFrame
    _counts: pd.Series
    _df_name: str
    _col: str
    _ngram: int
    _radius: int
    _block_size: int

    def __init__(self, df: Union[pd.DataFrame, dd.DataFrame], col: str, df_name: str):
        self.clusters = pd.Series(dtype=object)
        self._df = to_dask(df)
        self._counts = pd.Series(dtype=object)
        self._df_name = df_name
        self._col = col
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
        Clusters are a Pandas Series of lists (each list represents a cluster).
        """
        key_funcs = {
            "fingerprint": self._finger_print_key,
            "ngram-fingerprint": self._ngram_finger_print_key,
            "phonetic-fingerprint": self._phonetic_fingerprint_key,
        }
        key_func = key_funcs[cluster_method]
        counts = self._df[self._col].value_counts(sort=False)
        # create dataframe containing unique values
        df = counts.index.to_frame(name=self._col)
        # create a column "key" containing keys created by the given key collision method
        df["key"] = df[self._col].map(key_func)
        # put items with the same key into the same list
        clusters = df.groupby("key")[self._col].apply(list, meta=(self._col, "object"))
        clusters = clusters.loc[clusters.map(len) > 1]
        clusters, self._counts = dask.compute(clusters, counts)
        # sort by the size of each cluster, so that larger clusters appear first
        self.clusters = clusters.sort_values(key=lambda x: x.map(len), ascending=False).reset_index(
            drop=True
        )

    def _nearest_neighbours_cluster(self) -> None:
        """
        Performs nearest neighbour clustering.
        Blocking is used to speed up the process, blocks are obtained where strings in the same
        block share a substring of a given blocking size. Only strings within the same block are
        compared using the levenshtein distance function.

        Method from OpenRefine: https://github.com/OpenRefine/OpenRefine/wiki/Clustering-In-Depth
        and simile-vicino: https://code.google.com/archive/p/simile-vicino/
        """
        blocks: DefaultDict[str, Set[str]] = defaultdict(set)
        counts = self._df[self._col].value_counts(sort=False)
        # create dataframe containing unique values
        df = counts.index.to_frame(name=self._col)
        # put strings in blocks
        populate_blocks = df[self._col].apply(
            self._populate_blocks, args=(blocks, self._block_size), meta=(self._col, "object")
        )
        _, self._counts = dask.compute(populate_blocks, counts)

        # compare strings in the same block and create clusters
        self.clusters = self._get_nearest_neighbour_clusters(blocks, self._radius)

    @staticmethod
    def _populate_blocks(val: str, blocks: DefaultDict[str, Set[str]], block_size: int) -> None:
        """
        Create n gram tokens of the given string and place the string into the block
        for each n gram.
        """
        tokens = _ngram_tokens(val, block_size)
        for token in tokens:
            if token not in blocks:
                blocks[token] = set()
            blocks[token].add(val)

    @staticmethod
    def _get_nearest_neighbour_clusters(
        blocks: DefaultDict[str, Set[str]], radius: int
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
                dist = LevenshteinDistance(center, val)
                if dist <= radius or radius < 0:
                    cluster_map[center].add(val)

        # remove duplicate clusters and clusters of length 1
        unique_clusters = set(
            frozenset(cluster) for cluster in cluster_map.values() if len(cluster) > 1
        )
        # convert to list of lists
        clusters = [list(cluster) for cluster in unique_clusters]
        # sort by the size of each cluster, so that larger clusters appear first
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
            if do_merge[idx]:
                # create the string that all the values in the cluster will be set to
                cluster_repr = new_values[idx].replace("'", "\\'")
                cluster_repr = f"'{cluster_repr}'"
                # create the strings to be replaced
                cluster_vals = [val.replace("'", "\\'") for val, _ in cluster]
                cluster_vals = [f"'{val}'" for val in cluster_vals if f"'{val}'" != cluster_repr]
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

        code = f"""
            {DECODE_FUNC}
            var ind = IPython.notebook.get_selected_index();
            var cell = IPython.notebook.get_cell(ind);
            var text = cell.get_text();
            cell.set_text(text.concat(b64DecodeUnicode("{encoded_code}")));
        """
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
                self._df[self._col] = self._df[self._col].replace(
                    [cluster_val for cluster_val, _ in cluster], cluster_repr
                )

    def final_df(self) -> None:
        """
        Displays a DataFrame with the final values in the next notebook cell.
        Writes the final dataframe to a pickle file then reads the file from
        inside the IPython kernel.
        """
        code = f"# dataframe with cleaned string values\n{self._df_name}_clean"
        encoded_code = (b64encode(str.encode(code))).decode()
        final_df = self._df.compute()
        # create a temporary directory for the dataframe file
        tmp_dir = mkdtemp().replace("\\", "/")
        df_file = path.join(tmp_dir, "clean_duplication_output.pkl").replace("\\", "/")
        final_df.to_pickle(df_file)
        # code to read the file and delete the temporary directory afterwards
        execute_code = (
            "import pandas as pd\n"
            "import shutil\n"
            f"{self._df_name}_clean = pd.read_pickle('{df_file}')\n"
            f"shutil.rmtree('{tmp_dir}')"
        )
        encoded_execute = (b64encode(str.encode(execute_code))).decode()
        code = f"""
                 {DECODE_FUNC}
                 IPython.notebook.kernel.execute(b64DecodeUnicode("{encoded_execute}"));
                 var code = IPython.notebook.insert_cell_below('code');
                 code.set_text(b64DecodeUnicode("{encoded_code}"));
                 code.execute();
             """
        display(Javascript(code))

    def get_page(self, start: int, end: int) -> pd.Series:
        """
        Returns a page of clusters from start to end. Adds the counts to
        each cluster entry and sorts each cluster by the counts, so that values
        with a greater count appear first in the cluster.
        """
        page = self.clusters.iloc[start:end]
        page = page.map(lambda cluster: [(val, self._counts.loc[val]) for val in cluster])
        page = page.map(lambda x: sorted(x, key=itemgetter(1), reverse=True))
        return page

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
    return [val[i : i + n] for i in range(len(val) - n + 1)]


def normalize_non_ascii(val: str) -> str:
    """
    Normalize extended western characters to ascii. (remove accents)
    """
    nfkd_form = normalize("NFKD", val)
    return "".join([c for c in nfkd_form if not combining(c)])
