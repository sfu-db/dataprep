from lib import Benchmark


class DaskDataPrepCorr(Benchmark):
    def bench(self) -> None:
        from dataprep.eda.utils import DataArray
        from dataprep.eda.correlation.compute.nullivariate import _spearman_nxn
        import dask.dataframe as dd
        import numpy as np

        if self.dpath.suffix == ".parquet":
            df = dd.read_parquet(self.dpath)
        elif self.dpath.suffix == ".csv":
            df = dd.read_csv(self.dpath)

        df = DataArray(df.astype(np.float))
        _spearman_nxn(df.values).compute()


if __name__ == "__main__":
    DaskDataPrepCorr().run()
