from lib import Benchmark

from tempfile import TemporaryDirectory
from pandas_profiling import ProfileReport
import dask.dataframe as dd
import pandas as pd


class PandasProfilingReport(Benchmark):
    def init(self) -> None:
        pass

    def bench(self) -> None:
        if self.dpath.suffix == ".parquet":
            df = pd.read_parquet(self.dpath)
        elif self.dpath.suffix == ".csv":
            df = pd.read_csv(self.dpath, error_bad_lines=False)

        with TemporaryDirectory() as tdir:
            ProfileReport(
                df,
                correlations={
                    "phi_k": {"calculate": False},
                    "cramers": {"calculate": False},
                },
                vars={"num": {"chi_squared_threshold": 0},
                    'cat': {'characters': False}
                    },
            ).to_file(f"{tdir}/report.html")


if __name__ == "__main__":
    PandasProfilingReport().run()
