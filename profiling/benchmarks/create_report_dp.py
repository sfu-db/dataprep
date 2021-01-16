from lib import Benchmark

from dataprep.eda import create_report
from tempfile import TemporaryDirectory
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster


class DataPrepReport(Benchmark):
    def init(self) -> None:
        pass

    def bench(self) -> None:
        if self.dpath.suffix == ".parquet":
            df = dd.read_parquet(self.dpath)
        elif self.dpath.suffix == ".csv":
            df = dd.read_csv(self.dpath, error_bad_lines=False)

        with TemporaryDirectory() as tdir:
            create_report(df, 
                progress=False,
                config={
                    "kde.enable": False,
                    "qqnorm.enable": False,
                    "box.enable": False,
                    "wordcloud.enable": False,
                    "wordlen.enable": False,
                    "pie.enable": False
                }
            ).save(f"{tdir}/report.html")


if __name__ == "__main__":
    DataPrepReport().run()
