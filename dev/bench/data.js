window.BENCHMARK_DATA = {
  "lastUpdate": 1626218748324,
  "repoUrl": "https://github.com/sfu-db/dataprep",
  "entries": {
    "DataPrep.EDA Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "jlpengcs@gmail.com",
            "name": "Jinglin Peng",
            "username": "jinglinpeng"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "caac298bf1741914237f5de60726a838e1471947",
          "message": "Merge pull request #670 from sfu-db/benchmark_action\n\ntest(eda): add performance test",
          "timestamp": "2021-07-13T14:57:36-07:00",
          "tree_id": "3f7f269b138eb9b2c7c27463dcfcf92bab990484",
          "url": "https://github.com/sfu-db/dataprep/commit/caac298bf1741914237f5de60726a838e1471947"
        },
        "date": 1626213624583,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.198920389258266,
            "unit": "iter/sec",
            "range": "stddev: 0.01519471045276923",
            "extra": "mean: 5.027136754200001 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jlpengcs@gmail.com",
            "name": "Jinglin Peng",
            "username": "jinglinpeng"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d6a6af6ee2815fa03e117fe0085712d6734c5e4b",
          "message": "Merge pull request #671 from sfu-db/readme_benchmark\n\ndocs(readme): add benchmark link",
          "timestamp": "2021-07-13T16:22:30-07:00",
          "tree_id": "14bd764dd31470a0eafb1f1f5a7fa7e6c66a2ce5",
          "url": "https://github.com/sfu-db/dataprep/commit/d6a6af6ee2815fa03e117fe0085712d6734c5e4b"
        },
        "date": 1626218746886,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1666758029896957,
            "unit": "iter/sec",
            "range": "stddev: 0.23061170237253145",
            "extra": "mean: 5.999671110400004 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}