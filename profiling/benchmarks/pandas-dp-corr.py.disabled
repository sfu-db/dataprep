from lib import Benchmark


class PandasDataPrepCorr(Benchmark):
    def bench(self) -> None:
        from dataprep.eda.utils import DataArray
        from dataprep.eda.correlation.compute.nullivariate import _spearman_nxn
        import pandas as pd

        if self.dpath.suffix == ".parquet":
            df = pd.read_parquet(self.dpath)
        elif self.dpath.suffix == ".csv":
            df = pd.read_csv(self.dpath)

        df = DataArray(df)
        _spearman_nxn(df.values).compute()


if __name__ == "__main__":
    PandasDataPrepCorr().run()
