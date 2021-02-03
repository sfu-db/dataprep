from lib import Benchmark


class PandasCorr(Benchmark):
    def bench(self) -> None:
        import pandas as pd

        if self.dpath.suffix == ".parquet":
            df = pd.read_parquet(self.dpath)
        elif self.dpath.suffix == ".csv":
            df = pd.read_csv(self.dpath)

        df.corr(method="spearman")


if __name__ == "__main__":
    PandasCorr().run()
