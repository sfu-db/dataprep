window.BENCHMARK_DATA = {
  "lastUpdate": 1634538611439,
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
          "id": "19077c6810bf1b684792dfbc2ebb2c690928bda9",
          "message": "Merge pull request #674 from Waterpine/song\n\ntest(eda): add test for config",
          "timestamp": "2021-07-20T17:32:09-07:00",
          "tree_id": "b1eb050b9d5305e56f83e37355d96cdc45c0b860",
          "url": "https://github.com/sfu-db/dataprep/commit/19077c6810bf1b684792dfbc2ebb2c690928bda9"
        },
        "date": 1626827704362,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.180197762286889,
            "unit": "iter/sec",
            "range": "stddev: 0.10882099934591108",
            "extra": "mean: 5.549458479999998 sec\nrounds: 5"
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
          "id": "72658db8cc238512466e02e7b2a45153ed379f12",
          "message": "Merge pull request #667 from sfu-db/bug-fix\n\nfeat(eda.diff): Add plot_diff([df1..dfn], continuous)",
          "timestamp": "2021-08-03T16:53:09-07:00",
          "tree_id": "fd66052e2b9cc1420a0ab9c9a2963f702fb5bdbc",
          "url": "https://github.com/sfu-db/dataprep/commit/72658db8cc238512466e02e7b2a45153ed379f12"
        },
        "date": 1628034978722,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.198074810508584,
            "unit": "iter/sec",
            "range": "stddev: 0.013045849110810106",
            "extra": "mean: 5.048597534600003 sec\nrounds: 5"
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
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "a868c504d90fddc61db7556b0866e495e1134c11",
          "message": "build:update varname version",
          "timestamp": "2021-09-10T19:36:31-07:00",
          "tree_id": "f2c3ff81c3e3de7a757588e23fc6db064d0ff965",
          "url": "https://github.com/sfu-db/dataprep/commit/a868c504d90fddc61db7556b0866e495e1134c11"
        },
        "date": 1631328003852,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1622125604402684,
            "unit": "iter/sec",
            "range": "stddev: 0.11301872409362092",
            "extra": "mean: 6.164750727599977 sec\nrounds: 5"
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
          "id": "bf58e30df86d176f73b5aa9a29a626b14695892f",
          "message": "Merge pull request #688 from sfu-db/scattersample\n\nfix(eda):fix scatter sample size and rate",
          "timestamp": "2021-09-18T00:06:59-07:00",
          "tree_id": "232aa18818ed19974b4270b4a566e5119825e8d6",
          "url": "https://github.com/sfu-db/dataprep/commit/bf58e30df86d176f73b5aa9a29a626b14695892f"
        },
        "date": 1631949053337,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1572721461202357,
            "unit": "iter/sec",
            "range": "stddev: 0.1333391852454711",
            "extra": "mean: 6.358404998399988 sec\nrounds: 5"
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
          "id": "13032fe3d7fb5a4dbf29f8f554122e4a5711f1ab",
          "message": "Merge pull request #691 from sahmad11/patch-1\n\ndocs(eda): scattter.sample_rate added to documentation",
          "timestamp": "2021-09-18T19:38:06-07:00",
          "tree_id": "6925a2e21290fb29d55f29a930355c23c1ceb71b",
          "url": "https://github.com/sfu-db/dataprep/commit/13032fe3d7fb5a4dbf29f8f554122e4a5711f1ab"
        },
        "date": 1632019314824,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15184162391310188,
            "unit": "iter/sec",
            "range": "stddev: 0.0861931330025079",
            "extra": "mean: 6.585809439000036 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "26f4f9e7bf4a9de3188a2cfc167900777f2548cf",
          "message": "Merge pull request #668 from NoirTree/clean_num\n\nfeat(clean): add multiple clean functions for number types",
          "timestamp": "2021-09-19T21:23:00-07:00",
          "tree_id": "2aa28902c8e49d4020084e45c81de26a5756a594",
          "url": "https://github.com/sfu-db/dataprep/commit/26f4f9e7bf4a9de3188a2cfc167900777f2548cf"
        },
        "date": 1632111997649,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1678442287550567,
            "unit": "iter/sec",
            "range": "stddev: 0.1352877254045239",
            "extra": "mean: 5.95790518040003 sec\nrounds: 5"
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
          "id": "4a0d684223dc668b1dd3cd0e93a7400d7e382f63",
          "message": "Merge pull request #621 from sfu-db/feat/clean_ml\n\nfeat(clean): add clean_ml function",
          "timestamp": "2021-09-19T21:43:06-07:00",
          "tree_id": "fa43e7f72ba5c72325c93d0e8be73210db49a0e5",
          "url": "https://github.com/sfu-db/dataprep/commit/4a0d684223dc668b1dd3cd0e93a7400d7e382f63"
        },
        "date": 1632113210731,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15471528542312268,
            "unit": "iter/sec",
            "range": "stddev: 0.027268157690841826",
            "extra": "mean: 6.463485474399977 sec\nrounds: 5"
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
          "id": "12f3c9ef9b2f70ca4562c574a8ca3d54945cd419",
          "message": "Merge pull request #622 from sfu-db/docs/clean_ml\n\ndocs(clean): add documentation for clean_ml function",
          "timestamp": "2021-09-19T22:04:24-07:00",
          "tree_id": "4809b548673a6f01ffffe0a6f55dbd355926d28b",
          "url": "https://github.com/sfu-db/dataprep/commit/12f3c9ef9b2f70ca4562c574a8ca3d54945cd419"
        },
        "date": 1632114484351,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.17354952468802165,
            "unit": "iter/sec",
            "range": "stddev: 0.0912781158708153",
            "extra": "mean: 5.7620440148 sec\nrounds: 5"
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
          "id": "1628ea850c7ab15e79fdb889ab4b145593a57be6",
          "message": "Merge pull request #669 from sfu-db/feat/clean_isbn\n\nfeat(clean): add 17 clean functions for number types",
          "timestamp": "2021-09-20T16:24:54-07:00",
          "tree_id": "7f3387b904c5c749d76b007ece9adb5ba709a56b",
          "url": "https://github.com/sfu-db/dataprep/commit/1628ea850c7ab15e79fdb889ab4b145593a57be6"
        },
        "date": 1632180505521,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.17081919511595747,
            "unit": "iter/sec",
            "range": "stddev: 0.048213157541849544",
            "extra": "mean: 5.854143027200007 sec\nrounds: 5"
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
          "id": "a8823a3122cc3c126c1cf4257220ecc9ede6c26c",
          "message": "Merge pull request #693 from devinllu/fix/save_report\n\nfix(eda): changed report save method to accept one path parameter as …",
          "timestamp": "2021-09-20T21:13:13-07:00",
          "tree_id": "abcf9111e0808eb7c36fa07362d00675594e08ce",
          "url": "https://github.com/sfu-db/dataprep/commit/a8823a3122cc3c126c1cf4257220ecc9ede6c26c"
        },
        "date": 1632197769214,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.21957163069782262,
            "unit": "iter/sec",
            "range": "stddev: 0.23070970790611833",
            "extra": "mean: 4.554322417800018 sec\nrounds: 5"
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
          "id": "9f6f5b2c033f0c9346080f8211d32ea249adb1b5",
          "message": "Merge pull request #672 from sfu-db/feat/10_clean_functions_1\n\nfeat(clean): add another 10 clean functions for number types",
          "timestamp": "2021-09-20T23:11:31-07:00",
          "tree_id": "ac0f914b256514cdae24c058cc598868020b7bdc",
          "url": "https://github.com/sfu-db/dataprep/commit/9f6f5b2c033f0c9346080f8211d32ea249adb1b5"
        },
        "date": 1632204946215,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1524718246126018,
            "unit": "iter/sec",
            "range": "stddev: 0.08905113269114114",
            "extra": "mean: 6.558588792 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4744134886017ce7381fa7ae7c201772e9bafc12",
          "message": "Merge pull request #684 from NoirTree/docs/clean_num\n\ndocs(clean): add documentation for multiple clean functions for number types",
          "timestamp": "2021-09-23T09:48:00-07:00",
          "tree_id": "f05a64a3a08b94787d48ed230584daa37452d849",
          "url": "https://github.com/sfu-db/dataprep/commit/4744134886017ce7381fa7ae7c201772e9bafc12"
        },
        "date": 1632415891359,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1718041216453461,
            "unit": "iter/sec",
            "range": "stddev: 0.06689905573945601",
            "extra": "mean: 5.820582128199999 sec\nrounds: 5"
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
          "id": "53b29b1867106c5f462d4a4c859a207d73e3c32a",
          "message": "Merge pull request #694 from sfu-db/fix/imdt_output\n\nfix(eda):remove imdt output from plot",
          "timestamp": "2021-09-28T15:29:25-07:00",
          "tree_id": "4984b420245cd55f9337630567201055fd51f880",
          "url": "https://github.com/sfu-db/dataprep/commit/53b29b1867106c5f462d4a4c859a207d73e3c32a"
        },
        "date": 1632868354927,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.18634426466269907,
            "unit": "iter/sec",
            "range": "stddev: 0.10949621123563397",
            "extra": "mean: 5.366411473999994 sec\nrounds: 5"
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
          "id": "7fa11e13fc71fb61fb04945bf4e3a03269371fcb",
          "message": "Merge pull request #696 from sfu-db/readme/colab\n\ndocs(readme):add eda colab",
          "timestamp": "2021-09-29T16:20:33-07:00",
          "tree_id": "26b4b1dc508f9a4ef1451de347ed64cc5cb0cc72",
          "url": "https://github.com/sfu-db/dataprep/commit/7fa11e13fc71fb61fb04945bf4e3a03269371fcb"
        },
        "date": 1632957870726,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15416599919379612,
            "unit": "iter/sec",
            "range": "stddev: 0.06664054546746451",
            "extra": "mean: 6.486514570200001 sec\nrounds: 5"
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
          "id": "ce25b17d401646c4f8d2ae25c3df05f254e5b99e",
          "message": "docs(eda): change eda colab position",
          "timestamp": "2021-09-29T16:36:01-07:00",
          "tree_id": "237aa6e77f7fe716e7e5356835fab4cadd287610",
          "url": "https://github.com/sfu-db/dataprep/commit/ce25b17d401646c4f8d2ae25c3df05f254e5b99e"
        },
        "date": 1632958786830,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16245631136579736,
            "unit": "iter/sec",
            "range": "stddev: 0.16965107827422618",
            "extra": "mean: 6.155501079600003 sec\nrounds: 5"
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
          "id": "92868e133ae46a3d86e3b7354a86611e3afce094",
          "message": "Merge pull request #703 from sfu-db/fix/value_table\n\nfix(eda):fix value table display",
          "timestamp": "2021-10-11T18:02:18-07:00",
          "tree_id": "b5bab581a14e531e088729d9eefdafab516b8ca0",
          "url": "https://github.com/sfu-db/dataprep/commit/92868e133ae46a3d86e3b7354a86611e3afce094"
        },
        "date": 1634000779900,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1477724662160447,
            "unit": "iter/sec",
            "range": "stddev: 0.06318230909023677",
            "extra": "mean: 6.7671605245999675 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wangxiaoying0369@gmail.com",
            "name": "Xiaoying Wang",
            "username": "wangxiaoying"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "73f9f255babb6d528b8b9e231fa7e3c5070849e0",
          "message": "docs(connector) update readme",
          "timestamp": "2021-10-17T23:26:40-07:00",
          "tree_id": "51045ede96b85d4f57907c391b38a8339a6ae43f",
          "url": "https://github.com/sfu-db/dataprep/commit/73f9f255babb6d528b8b9e231fa7e3c5070849e0"
        },
        "date": 1634538610061,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.17474320934770468,
            "unit": "iter/sec",
            "range": "stddev: 0.053936916098027025",
            "extra": "mean: 5.722683037200011 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}