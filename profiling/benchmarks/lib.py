import sys
from json import dumps as jdumps
from pathlib import Path
from time import time

PWD = Path(__file__).resolve().parent.parent


class Benchmark:
    reader: str
    dpath: Path

    def __init__(self):
        self.reader = sys.argv[1]
        self.dpath = PWD / "data" / sys.argv[2]

    def init(self):
        pass

    def run(self) -> None:
        self.init()
        then = time()
        self.bench()
        elapsed = time() - then
        result = {"name": self.__class__.__name__, "elapsed": elapsed}
        print(jdumps(result))

    def bench(self) -> None:
        pass
