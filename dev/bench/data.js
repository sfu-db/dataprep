window.BENCHMARK_DATA = {
  "lastUpdate": 1645731415001,
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
      },
      {
        "commit": {
          "author": {
            "email": "wangxiaoying0369@gmail.com",
            "name": "Xiaoying Wang",
            "username": "wangxiaoying"
          },
          "committer": {
            "email": "wangxiaoying0369@gmail.com",
            "name": "Xiaoying Wang",
            "username": "wangxiaoying"
          },
          "distinct": true,
          "id": "9beeac0a02092d12e75f068f226b674147e87c32",
          "message": "resolve conflicts",
          "timestamp": "2021-10-19T21:02:55Z",
          "tree_id": "95c8fb19172a6f250f1b37b273896e0b7454d4d9",
          "url": "https://github.com/sfu-db/dataprep/commit/9beeac0a02092d12e75f068f226b674147e87c32"
        },
        "date": 1634677617013,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1580379868502376,
            "unit": "iter/sec",
            "range": "stddev: 0.04768959562697941",
            "extra": "mean: 6.327592624598765 sec\nrounds: 5"
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
            "email": "wangxiaoying0369@gmail.com",
            "name": "Xiaoying Wang",
            "username": "wangxiaoying"
          },
          "distinct": true,
          "id": "a64e3563f4a1f95b44465bd372ddb4bd8ceb951c",
          "message": "feat(connector): integrate connectorx into connector",
          "timestamp": "2021-10-19T21:09:00Z",
          "tree_id": "95c8fb19172a6f250f1b37b273896e0b7454d4d9",
          "url": "https://github.com/sfu-db/dataprep/commit/a64e3563f4a1f95b44465bd372ddb4bd8ceb951c"
        },
        "date": 1634677975551,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.204863280161837,
            "unit": "iter/sec",
            "range": "stddev: 0.02400732527459022",
            "extra": "mean: 4.881304249400012 sec\nrounds: 5"
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
          "id": "1963c1ac5eecdd181a16f257da004cc0cfa4b911",
          "message": "Merge pull request #716 from sfu-db/fix/string\n\nfix(eda):fix string type",
          "timestamp": "2021-10-19T14:26:33-07:00",
          "tree_id": "ab7b6edaf5d189ecd4e1439a6860ff68cfb0b795",
          "url": "https://github.com/sfu-db/dataprep/commit/1963c1ac5eecdd181a16f257da004cc0cfa4b911"
        },
        "date": 1634679018899,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16048401901047046,
            "unit": "iter/sec",
            "range": "stddev: 0.10095000022809793",
            "extra": "mean: 6.231150030799995 sec\nrounds: 5"
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
          "id": "3969cc73588cef122f5a42c8b85e53df1d2fb883",
          "message": "docs/readme\n\nChange the readme part of Clean. Add the number of clean functions.",
          "timestamp": "2021-10-24T23:03:07-07:00",
          "tree_id": "aeb2e6a4d726efb3cfe4ac9195b49151df8b935d",
          "url": "https://github.com/sfu-db/dataprep/commit/3969cc73588cef122f5a42c8b85e53df1d2fb883"
        },
        "date": 1635142002469,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.18290919711389944,
            "unit": "iter/sec",
            "range": "stddev: 0.11744122707318705",
            "extra": "mean: 5.467193644600002 sec\nrounds: 5"
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
          "id": "6f7ca7ac2dc41d07e5d1c7e359f4c96885c6cebc",
          "message": "docs/readme",
          "timestamp": "2021-10-24T23:05:46-07:00",
          "tree_id": "acce9beabc15de32586fdebb7062e52a0b5a720d",
          "url": "https://github.com/sfu-db/dataprep/commit/6f7ca7ac2dc41d07e5d1c7e359f4c96885c6cebc"
        },
        "date": 1635142155844,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.18563696639131755,
            "unit": "iter/sec",
            "range": "stddev: 0.05834572810685254",
            "extra": "mean: 5.386858121199998 sec\nrounds: 5"
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
          "id": "1e9c4857989983e9d1049dbc09b807f40803423d",
          "message": "docs/readme",
          "timestamp": "2021-10-24T23:07:07-07:00",
          "tree_id": "60f2f83f3fccf03031fce35bc90d8e30b9b28325",
          "url": "https://github.com/sfu-db/dataprep/commit/1e9c4857989983e9d1049dbc09b807f40803423d"
        },
        "date": 1635142252127,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16869269050311891,
            "unit": "iter/sec",
            "range": "stddev: 0.07648509679455337",
            "extra": "mean: 5.927939124199997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "1e0dfecadf9d9a9a71ff1b1f27754e99125f48e3",
          "message": "remove yelp test",
          "timestamp": "2021-10-25T22:52:42Z",
          "tree_id": "08da199810d6a3e558d8abca147e30085a65edb9",
          "url": "https://github.com/sfu-db/dataprep/commit/1e0dfecadf9d9a9a71ff1b1f27754e99125f48e3"
        },
        "date": 1635202642926,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.19024625460585337,
            "unit": "iter/sec",
            "range": "stddev: 0.05150359901659495",
            "extra": "mean: 5.256345267199981 sec\nrounds: 5"
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
          "id": "1ec9e491db0a93332b8f8609fe19b37054a67dc4",
          "message": "Merge pull request #718 from sfu-db/connectorx\n\nDoc(Connector): add user guide and api reference for read_sql",
          "timestamp": "2021-10-25T16:33:19-07:00",
          "tree_id": "5f4cfb1faccf2d43486298d595c756f510a2e674",
          "url": "https://github.com/sfu-db/dataprep/commit/1ec9e491db0a93332b8f8609fe19b37054a67dc4"
        },
        "date": 1635205005010,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1768275068565261,
            "unit": "iter/sec",
            "range": "stddev: 0.15730813834390825",
            "extra": "mean: 5.655228746799997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "7e2a5a00df9402e7fc95d1e77f347e5acc1d25c1",
          "message": "update ci",
          "timestamp": "2021-10-26T01:21:42Z",
          "tree_id": "686abfb869bb8272393eceacfce06d68139c9112",
          "url": "https://github.com/sfu-db/dataprep/commit/7e2a5a00df9402e7fc95d1e77f347e5acc1d25c1"
        },
        "date": 1635211520724,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.17402520339979452,
            "unit": "iter/sec",
            "range": "stddev: 0.017114672484713184",
            "extra": "mean: 5.746294102599973 sec\nrounds: 5"
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
          "id": "42bf18893aee2725dfaf0222d7c0d09fc956c010",
          "message": "Merge pull request #720 from sfu-db/docs/change_clean_doc_intro\n\nChange the introduction part of clean documentation",
          "timestamp": "2021-10-25T19:14:01-07:00",
          "tree_id": "43b303605a1c59c04d8094bdde7f3d6e4aa6e36f",
          "url": "https://github.com/sfu-db/dataprep/commit/42bf18893aee2725dfaf0222d7c0d09fc956c010"
        },
        "date": 1635214656713,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16553030160412138,
            "unit": "iter/sec",
            "range": "stddev: 0.07837264203602647",
            "extra": "mean: 6.041189983400005 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "3f7b246110c0ae6ddee9d4e8d55a0ded54638e56",
          "message": "build: fix ci",
          "timestamp": "2021-10-26T02:32:02Z",
          "tree_id": "7c70d2627a5a8dd17b97a8b6556121d313a3aa33",
          "url": "https://github.com/sfu-db/dataprep/commit/3f7b246110c0ae6ddee9d4e8d55a0ded54638e56"
        },
        "date": 1635215756673,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15120582580761752,
            "unit": "iter/sec",
            "range": "stddev: 0.13307234556417016",
            "extra": "mean: 6.613501792399996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "91085c8e85bc493d844853cde5f5d2b9a219b0cd",
          "message": "build: fix ci",
          "timestamp": "2021-10-26T02:48:00Z",
          "tree_id": "eb9cf0603f592b78b1dca8e7d5548a9b3f3c07bd",
          "url": "https://github.com/sfu-db/dataprep/commit/91085c8e85bc493d844853cde5f5d2b9a219b0cd"
        },
        "date": 1635216680952,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.18671820415733806,
            "unit": "iter/sec",
            "range": "stddev: 0.16319039534237384",
            "extra": "mean: 5.3556641919999946 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "fb1b3238ee9ad79612295301c32e6be3328e163e",
          "message": "build: fix ci",
          "timestamp": "2021-10-26T03:09:46Z",
          "tree_id": "f6fab7fc23381abf6bd2113b6d0b68fbbebc9dbb",
          "url": "https://github.com/sfu-db/dataprep/commit/fb1b3238ee9ad79612295301c32e6be3328e163e"
        },
        "date": 1635218028812,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15440517408269766,
            "unit": "iter/sec",
            "range": "stddev: 0.056171522892861155",
            "extra": "mean: 6.476466905600011 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "87e77dfe372d228c698ed6a38cf1541022f12a76",
          "message": "build: fix ci",
          "timestamp": "2021-10-26T03:21:47Z",
          "tree_id": "5a11815a1e26dba5aa5e3f1ca44f2bbb7b1c8215",
          "url": "https://github.com/sfu-db/dataprep/commit/87e77dfe372d228c698ed6a38cf1541022f12a76"
        },
        "date": 1635218761808,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1535174706175434,
            "unit": "iter/sec",
            "range": "stddev: 0.07260911943227713",
            "extra": "mean: 6.513916598399999 sec\nrounds: 5"
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
          "id": "924e271c53939685f1ec089987b71ae9185090f1",
          "message": "Merge pull request #715 from Waterpine/develop\n\nfeat(eda): save imdt as json file",
          "timestamp": "2021-10-25T21:25:09-07:00",
          "tree_id": "d712d96c2a02623fa5f603b77b4a488942837ab9",
          "url": "https://github.com/sfu-db/dataprep/commit/924e271c53939685f1ec089987b71ae9185090f1"
        },
        "date": 1635222524195,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16963672476495786,
            "unit": "iter/sec",
            "range": "stddev: 0.07239214595117281",
            "extra": "mean: 5.894949937199987 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "255b6a9fb72d5483be3647820da2efc700aee808",
          "message": "build: skip codecov and codacy for forks",
          "timestamp": "2021-10-26T04:31:06Z",
          "tree_id": "54d8ec592a3b0d1835deb2838fb364c5991472c0",
          "url": "https://github.com/sfu-db/dataprep/commit/255b6a9fb72d5483be3647820da2efc700aee808"
        },
        "date": 1635222909965,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1552214022018144,
            "unit": "iter/sec",
            "range": "stddev: 0.3065213021475138",
            "extra": "mean: 6.442410555599986 sec\nrounds: 5"
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
          "id": "702ca991a8d485b98d71ebfe45de779fe6fc291d",
          "message": "Merge pull request #698 from devinllu/feat/plot_diff_density\n\nfeat(eda): added density parameter to plot_diff(df)",
          "timestamp": "2021-10-26T08:06:13-07:00",
          "tree_id": "6d31350db6bc5a957654670a897ea47c64035ddd",
          "url": "https://github.com/sfu-db/dataprep/commit/702ca991a8d485b98d71ebfe45de779fe6fc291d"
        },
        "date": 1635260989934,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16680566309010683,
            "unit": "iter/sec",
            "range": "stddev: 0.2399551173643202",
            "extra": "mean: 5.995000298399998 sec\nrounds: 5"
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
          "id": "272e980948802cc6ea94595eac6cadf5235957b9",
          "message": "Merge pull request #721 from sfu-db/fix/typo\n\nstyle(eda): fix dendrogram typo",
          "timestamp": "2021-10-26T08:20:53-07:00",
          "tree_id": "6570a0bebaf2ba5bea684d32e17b5c6fd7e1440c",
          "url": "https://github.com/sfu-db/dataprep/commit/272e980948802cc6ea94595eac6cadf5235957b9"
        },
        "date": 1635261876389,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15993185031078488,
            "unit": "iter/sec",
            "range": "stddev: 0.04387142598993089",
            "extra": "mean: 6.2526632316000015 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "jlpengcs@gmail.com",
            "name": "jinglinpeng",
            "username": "jinglinpeng"
          },
          "distinct": false,
          "id": "d76f1f5b081d58fd73277a88ddb95484d21f58a5",
          "message": "v0.4.0\n\nBump to v0.4.0",
          "timestamp": "2021-10-26T16:53:10Z",
          "tree_id": "c0cfd7cc0778d9c4a8aac60c04044dffb2064564",
          "url": "https://github.com/sfu-db/dataprep/commit/d76f1f5b081d58fd73277a88ddb95484d21f58a5"
        },
        "date": 1635267382291,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.20014093172682876,
            "unit": "iter/sec",
            "range": "stddev: 0.03745842617198819",
            "extra": "mean: 4.996479187799997 sec\nrounds: 5"
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
          "id": "29e7b82ac5cae6d7a9dd596c9e9c45544f4bfbdf",
          "message": "docs/change_readme",
          "timestamp": "2021-10-26T17:04:06-07:00",
          "tree_id": "8f53fe36bc170b0d630331d0d572f90bc18a87b2",
          "url": "https://github.com/sfu-db/dataprep/commit/29e7b82ac5cae6d7a9dd596c9e9c45544f4bfbdf"
        },
        "date": 1635293260904,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16803078263500557,
            "unit": "iter/sec",
            "range": "stddev: 0.046102617815810154",
            "extra": "mean: 5.951290497600001 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "20a1a27223fb08b6cae09ed1e600cce81ece11bd",
          "message": "release doc",
          "timestamp": "2021-10-26T17:06:23-07:00",
          "tree_id": "8fd431f3d1a27d4e20145e929ef3461447b96590",
          "url": "https://github.com/sfu-db/dataprep/commit/20a1a27223fb08b6cae09ed1e600cce81ece11bd"
        },
        "date": 1635293429193,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15679323137598564,
            "unit": "iter/sec",
            "range": "stddev: 0.03907216858240475",
            "extra": "mean: 6.377826333600007 sec\nrounds: 5"
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
          "id": "525ebf377d6e1ddfbcef66835fb227093ad37b4d",
          "message": "Merge pull request #726 from sfu-db/docs/change_clean_doc_intro\n\nupdate string of all clean documentation",
          "timestamp": "2021-10-29T20:48:45-07:00",
          "tree_id": "c04231e8befbb600e66dd74b5b080e2cbe915db4",
          "url": "https://github.com/sfu-db/dataprep/commit/525ebf377d6e1ddfbcef66835fb227093ad37b4d"
        },
        "date": 1635565939037,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.17205420725607706,
            "unit": "iter/sec",
            "range": "stddev: 0.019796424308629372",
            "extra": "mean: 5.8121217490000046 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "0b4c0ad7d2dfd9eabcfc856756e43dde33833f61",
          "message": "change all strings in clean documentation",
          "timestamp": "2021-10-29T21:35:03-07:00",
          "tree_id": "c04231e8befbb600e66dd74b5b080e2cbe915db4",
          "url": "https://github.com/sfu-db/dataprep/commit/0b4c0ad7d2dfd9eabcfc856756e43dde33833f61"
        },
        "date": 1635568798815,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15831335922862402,
            "unit": "iter/sec",
            "range": "stddev: 0.05981506420111502",
            "extra": "mean: 6.31658632519999 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "d31a31f6d48db4084492bb8703f9fce0dd81d363",
          "message": "add clean_ml documentation",
          "timestamp": "2021-10-30T12:36:25-07:00",
          "tree_id": "23cb48fb00e4ffb4270568b43889a1627ea3cc01",
          "url": "https://github.com/sfu-db/dataprep/commit/d31a31f6d48db4084492bb8703f9fce0dd81d363"
        },
        "date": 1635622827497,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1625673078725321,
            "unit": "iter/sec",
            "range": "stddev: 0.8055610803918611",
            "extra": "mean: 6.151298272000008 sec\nrounds: 5"
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
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "df81c233b098f5b80b3fc8eea6ab617237ea60cd",
          "message": "Merge pull request #732 from sfu-db/fix/cx\n\nfix(connectorx) fix and update doc on connectorx",
          "timestamp": "2021-11-01T11:38:15-07:00",
          "tree_id": "f2ee72dff44dc5dcea69d3842958cec76ec47285",
          "url": "https://github.com/sfu-db/dataprep/commit/df81c233b098f5b80b3fc8eea6ab617237ea60cd"
        },
        "date": 1635792106888,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.17963210054182274,
            "unit": "iter/sec",
            "range": "stddev: 0.09461087972831507",
            "extra": "mean: 5.566933732800033 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "ed0eac7d4e951a64c31d5c2f8df65cf9b4a4e3a0",
          "message": "fix partial clean docs",
          "timestamp": "2021-11-02T13:47:17-07:00",
          "tree_id": "9dccf7266071b3450b042d2e8a3da4d648e08f58",
          "url": "https://github.com/sfu-db/dataprep/commit/ed0eac7d4e951a64c31d5c2f8df65cf9b4a4e3a0"
        },
        "date": 1635886235550,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1971219512290125,
            "unit": "iter/sec",
            "range": "stddev: 0.03896731454125975",
            "extra": "mean: 5.073001732 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "9c889ff2198980e1057e171e4d1da4e6713ff8f2",
          "message": "fix all clean documents",
          "timestamp": "2021-11-03T17:54:39-07:00",
          "tree_id": "df36693f0a8ab0f553db1ff48997b8cb01de1613",
          "url": "https://github.com/sfu-db/dataprep/commit/9c889ff2198980e1057e171e4d1da4e6713ff8f2"
        },
        "date": 1635987500251,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16784394406939374,
            "unit": "iter/sec",
            "range": "stddev: 0.08871731886379974",
            "extra": "mean: 5.957915285799993 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "fd1057a86074d57b6a8714be7649f38d7d2439fe",
          "message": "fix all clean documents",
          "timestamp": "2021-11-03T18:03:59-07:00",
          "tree_id": "d551a1fa9f8225184e0a1151c1167dea763d9eb7",
          "url": "https://github.com/sfu-db/dataprep/commit/fd1057a86074d57b6a8714be7649f38d7d2439fe"
        },
        "date": 1635988172697,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15637032797401013,
            "unit": "iter/sec",
            "range": "stddev: 0.051226776514563754",
            "extra": "mean: 6.395075158799995 sec\nrounds: 5"
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
          "id": "57d3f3770e16e1e6998d3f66ef6b5bb744127eb5",
          "message": "Merge pull request #753 from Waterpine/develop\n\nfix(eda):fix saving imdt as json file",
          "timestamp": "2021-11-23T11:39:04-08:00",
          "tree_id": "ed9939614be18be9e8e1a92bb3e6b23b775d19a8",
          "url": "https://github.com/sfu-db/dataprep/commit/57d3f3770e16e1e6998d3f66ef6b5bb744127eb5"
        },
        "date": 1637696586545,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1580572351344921,
            "unit": "iter/sec",
            "range": "stddev: 0.06557930995648795",
            "extra": "mean: 6.32682204740006 sec\nrounds: 5"
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
          "id": "c91014fa6362c2b554132487d870fc0592bcb45b",
          "message": "Merge pull request #746 from sfu-db/fix/type\n\nfix(eda):pandas type",
          "timestamp": "2021-11-23T12:31:26-08:00",
          "tree_id": "36766f3d52474d8e900f9ae1b01ef4edd09c8de4",
          "url": "https://github.com/sfu-db/dataprep/commit/c91014fa6362c2b554132487d870fc0592bcb45b"
        },
        "date": 1637699718056,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.18085867440398032,
            "unit": "iter/sec",
            "range": "stddev: 0.05296779792059889",
            "extra": "mean: 5.529179085800001 sec\nrounds: 5"
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
          "id": "fcf47cf839ba4f2df347794b043f0facddabc0aa",
          "message": "Merge pull request #758 from sfu-db/interaction\n\nfeat(eda):add categorical interaction in create_report",
          "timestamp": "2021-11-24T01:15:18-08:00",
          "tree_id": "b5ede1611a7cee0ada4fa3561438c3ad6a4dea0a",
          "url": "https://github.com/sfu-db/dataprep/commit/fcf47cf839ba4f2df347794b043f0facddabc0aa"
        },
        "date": 1637745535708,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.20030339918217077,
            "unit": "iter/sec",
            "range": "stddev: 0.043293418855521924",
            "extra": "mean: 4.992426509400002 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "9e6f8e8464a12da0d8c2bed0e1a5d002ef188f50",
          "message": "add clean gui ci",
          "timestamp": "2021-11-24T12:20:36-08:00",
          "tree_id": "b71a1fe146e3bb60361c6d53dea8782a5f817f01",
          "url": "https://github.com/sfu-db/dataprep/commit/9e6f8e8464a12da0d8c2bed0e1a5d002ef188f50"
        },
        "date": 1637785477475,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1646051382041892,
            "unit": "iter/sec",
            "range": "stddev: 0.1566483943606885",
            "extra": "mean: 6.075144499800007 sec\nrounds: 5"
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
          "id": "e23e3a78099aeea027c028400acbd34cdfe9f9f4",
          "message": "Merge pull request #759 from sfu-db/doc/parameter\n\ndocs(eda): enrich parameters in report",
          "timestamp": "2021-11-24T18:10:53-08:00",
          "tree_id": "e5e5fe666a129d3112b6cc774510d549e2220b93",
          "url": "https://github.com/sfu-db/dataprep/commit/e23e3a78099aeea027c028400acbd34cdfe9f9f4"
        },
        "date": 1637806497718,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16983634706654624,
            "unit": "iter/sec",
            "range": "stddev: 0.11144355901094626",
            "extra": "mean: 5.888021128999992 sec\nrounds: 5"
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
          "id": "93c7d6d8e3efd2c521567ef2135ebd6c5fd76749",
          "message": "Merge pull request #752 from devinllu/feat/create_overview_variables_section\n\nFeat/create overview variables section",
          "timestamp": "2021-11-24T19:32:22-08:00",
          "tree_id": "9ee03fb1e3fe8c0548148d5f849188f8cf989bbf",
          "url": "https://github.com/sfu-db/dataprep/commit/93c7d6d8e3efd2c521567ef2135ebd6c5fd76749"
        },
        "date": 1637811335952,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.20197387127875202,
            "unit": "iter/sec",
            "range": "stddev: 0.028998817934387633",
            "extra": "mean: 4.951135479400011 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "7f4ab12a26960a5a7df7882b971880f13ace38dc",
          "message": "feat(clean): Add wiki",
          "timestamp": "2021-11-24T22:27:00-08:00",
          "tree_id": "89617ffffcc5c8b97b441e40af9489230089d8dc",
          "url": "https://github.com/sfu-db/dataprep/commit/7f4ab12a26960a5a7df7882b971880f13ace38dc"
        },
        "date": 1637821860896,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.17390831148452232,
            "unit": "iter/sec",
            "range": "stddev: 0.031510332882753754",
            "extra": "mean: 5.750156455799981 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "997b3aba82e3a226709bc79362c1f3ce3accb814",
          "message": "Add frontend dist folder",
          "timestamp": "2021-11-24T23:01:25-08:00",
          "tree_id": "ec298d63ccd494b535745076ab0d5a735b6654bf",
          "url": "https://github.com/sfu-db/dataprep/commit/997b3aba82e3a226709bc79362c1f3ce3accb814"
        },
        "date": 1637823975680,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15735705420639384,
            "unit": "iter/sec",
            "range": "stddev: 0.13065575023279716",
            "extra": "mean: 6.354974074999984 sec\nrounds: 5"
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
          "id": "3bd0277a47c29efb464580a17174882da60b6884",
          "message": "Merge pull request #763 from devinllu/refactor/diff_report\n\nrefactor(eda): removed unnecessary html code and styles",
          "timestamp": "2021-11-25T00:11:23-08:00",
          "tree_id": "4bb096e88e052fe3470ddad1d536e1f0f80eeffb",
          "url": "https://github.com/sfu-db/dataprep/commit/3bd0277a47c29efb464580a17174882da60b6884"
        },
        "date": 1637828131032,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1735616480602744,
            "unit": "iter/sec",
            "range": "stddev: 0.03639561093590472",
            "extra": "mean: 5.761641532999965 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "8216a4cd3ff375f3ce2045f80f22e9b4389ed2d9",
          "message": "Change template folder",
          "timestamp": "2021-11-25T00:23:50-08:00",
          "tree_id": "74ec04e6afa28c8eeceac51e65648e1de40a6e6e",
          "url": "https://github.com/sfu-db/dataprep/commit/8216a4cd3ff375f3ce2045f80f22e9b4389ed2d9"
        },
        "date": 1637828900355,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15410830511308496,
            "unit": "iter/sec",
            "range": "stddev: 0.05644206874736474",
            "extra": "mean: 6.4889429499999896 sec\nrounds: 5"
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
            "email": "wangxiaoying0369@gmail.com",
            "name": "Xiaoying Wang",
            "username": "wangxiaoying"
          },
          "distinct": true,
          "id": "7d872fa2b10d84b5d385044c6a740cb745b0c763",
          "message": "update justfile",
          "timestamp": "2021-11-25T18:04:12Z",
          "tree_id": "06df41a9955415d7d96a1975d22d7b75b279a89a",
          "url": "https://github.com/sfu-db/dataprep/commit/7d872fa2b10d84b5d385044c6a740cb745b0c763"
        },
        "date": 1637863693465,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1765256323728337,
            "unit": "iter/sec",
            "range": "stddev: 0.12637945937953532",
            "extra": "mean: 5.664899689399976 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "wangxiaoying0369@gmail.com",
            "name": "Xiaoying Wang",
            "username": "wangxiaoying"
          },
          "distinct": true,
          "id": "4032acb1d1f2c413d4cb000d17e8ffa611315f9f",
          "message": "v0.4.1\n\nBump to v0.4.1",
          "timestamp": "2021-11-25T18:07:07Z",
          "tree_id": "2aaece9e6d594ae7b72b2b0b60e2eb3fd5d35703",
          "url": "https://github.com/sfu-db/dataprep/commit/4032acb1d1f2c413d4cb000d17e8ffa611315f9f"
        },
        "date": 1637863894568,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.14161583027221997,
            "unit": "iter/sec",
            "range": "stddev: 0.11897119966757708",
            "extra": "mean: 7.061357463199966 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "0339b232460bf8e926f0a266db40b08a71984698",
          "message": "add workflow for release",
          "timestamp": "2021-12-14T20:02:59Z",
          "tree_id": "75f8eeaa0d64e81eb1adb6435b1ee1ca9a439ffa",
          "url": "https://github.com/sfu-db/dataprep/commit/0339b232460bf8e926f0a266db40b08a71984698"
        },
        "date": 1639512580990,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15954986587938624,
            "unit": "iter/sec",
            "range": "stddev: 0.04159729589161953",
            "extra": "mean: 6.267632971600006 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "d22701c52c5344cd6b0e92315480532eb69d3289",
          "message": "add pypi upload",
          "timestamp": "2021-12-14T23:55:55Z",
          "tree_id": "02437c90550176327b9e1705a6ec340cbc217a6d",
          "url": "https://github.com/sfu-db/dataprep/commit/d22701c52c5344cd6b0e92315480532eb69d3289"
        },
        "date": 1639526359475,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1992322942160513,
            "unit": "iter/sec",
            "range": "stddev: 0.07086473356961144",
            "extra": "mean: 5.0192666000000035 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "67500b5ab964fc2d39fbd17b1aeaa01bcd21f9e4",
          "message": "add pypi upload",
          "timestamp": "2021-12-15T00:14:34Z",
          "tree_id": "a47fedaa971c6d8aa03a4a8fc6847fe3f4a815eb",
          "url": "https://github.com/sfu-db/dataprep/commit/67500b5ab964fc2d39fbd17b1aeaa01bcd21f9e4"
        },
        "date": 1639527499464,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2011514197002141,
            "unit": "iter/sec",
            "range": "stddev: 0.009585647015040671",
            "extra": "mean: 4.97137927980001 sec\nrounds: 5"
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
          "id": "f212d174c368cb5788420bc87a2baae514f0f7f1",
          "message": "Merge pull request #772 from sfu-db/terminal\n\nfix(eda):wordcloud setting in terminal",
          "timestamp": "2021-12-20T20:34:18-08:00",
          "tree_id": "8d627b2475f95623ce34790d226562580d6a7e77",
          "url": "https://github.com/sfu-db/dataprep/commit/f212d174c368cb5788420bc87a2baae514f0f7f1"
        },
        "date": 1640061445635,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.30278453095476826,
            "unit": "iter/sec",
            "range": "stddev: 0.0427409559279298",
            "extra": "mean: 3.302678630399998 sec\nrounds: 5"
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
          "id": "a0010f61cd08db1f727e69f781c236db65f654ae",
          "message": "Merge pull request #773 from devinllu/feat/show-details-tab\n\nFeat/show details tab",
          "timestamp": "2021-12-21T17:52:15-08:00",
          "tree_id": "0407b073af9623cf57da3ed31f5c7fcf34382a99",
          "url": "https://github.com/sfu-db/dataprep/commit/a0010f61cd08db1f727e69f781c236db65f654ae"
        },
        "date": 1640138183347,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.21289793868613044,
            "unit": "iter/sec",
            "range": "stddev: 0.01901679800255346",
            "extra": "mean: 4.697086341800014 sec\nrounds: 5"
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
          "id": "6dfb9c659e8bf73f07978ae195d0372495c6f118",
          "message": "Merge pull request #774 from Bowen0729/on_yarn_doc\n\nadd the doc of run dataprep.eda on Hadoop yarn",
          "timestamp": "2021-12-24T22:12:34-08:00",
          "tree_id": "014146174665a0dc83ebf673b2d2dc6c49032454",
          "url": "https://github.com/sfu-db/dataprep/commit/6dfb9c659e8bf73f07978ae195d0372495c6f118"
        },
        "date": 1640412956629,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2966245333409923,
            "unit": "iter/sec",
            "range": "stddev: 0.058753052256889135",
            "extra": "mean: 3.3712653122000007 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bf336dec5c6587b5fbeac7f0b647ed8e5ff40d0a",
          "message": "fix 404 link for connector\n\nfixes #783",
          "timestamp": "2022-01-15T12:27:01-08:00",
          "tree_id": "15134d007fefeaf8a1d31479a3e371434611289a",
          "url": "https://github.com/sfu-db/dataprep/commit/bf336dec5c6587b5fbeac7f0b647ed8e5ff40d0a"
        },
        "date": 1642278632760,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2461884801977018,
            "unit": "iter/sec",
            "range": "stddev: 0.08392425551515681",
            "extra": "mean: 4.061928483399993 sec\nrounds: 5"
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
          "id": "e67a3a0951364b3f41c27ed1d4126038090a49ee",
          "message": "Merge pull request #789 from khoatxp/test/780-add-test-for-imdt-compute\n\ntest(eda): add tests for intermediate compute functions",
          "timestamp": "2022-01-20T21:11:12-08:00",
          "tree_id": "3a50d1b994801ead9d945cd50017fec2059c2e5e",
          "url": "https://github.com/sfu-db/dataprep/commit/e67a3a0951364b3f41c27ed1d4126038090a49ee"
        },
        "date": 1642742093817,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.24687831225969445,
            "unit": "iter/sec",
            "range": "stddev: 0.04058288381101501",
            "extra": "mean: 4.050578565800009 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bd42c70c3dd7735ed4f052128a86517fad9ad68e",
          "message": "Update README.md",
          "timestamp": "2022-01-27T21:53:46-08:00",
          "tree_id": "83b16014971eddabf3f4dfa39e855dd33b63c012",
          "url": "https://github.com/sfu-db/dataprep/commit/bd42c70c3dd7735ed4f052128a86517fad9ad68e"
        },
        "date": 1643349442741,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.29921521755542163,
            "unit": "iter/sec",
            "range": "stddev: 0.06947040680040345",
            "extra": "mean: 3.342076008599986 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3c67fa0d6e1278f8dd0ee89d8586c98d3b71c974",
          "message": "Update and rename bug_report.md to bug_report_eda.md",
          "timestamp": "2022-01-27T22:00:57-08:00",
          "tree_id": "cb0ccfcd28ac6bc69b8365c118d73d6804d4bd77",
          "url": "https://github.com/sfu-db/dataprep/commit/3c67fa0d6e1278f8dd0ee89d8586c98d3b71c974"
        },
        "date": 1643349890356,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.22995869410419206,
            "unit": "iter/sec",
            "range": "stddev: 0.08030776443776855",
            "extra": "mean: 4.348607057000026 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1ed1adfe4a4e83a8dfba877329896dd582ba6009",
          "message": "Create bug_report_connector.md",
          "timestamp": "2022-01-27T22:01:42-08:00",
          "tree_id": "54c7237e71cfac43717ac903ebc34694ed74a238",
          "url": "https://github.com/sfu-db/dataprep/commit/1ed1adfe4a4e83a8dfba877329896dd582ba6009"
        },
        "date": 1643349892110,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.29863673675329233,
            "unit": "iter/sec",
            "range": "stddev: 0.07309292116190878",
            "extra": "mean: 3.3485498498000026 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "672a1adaf9188e51c1fb4ed4d88b0a028a4f8a82",
          "message": "Create bug_report_cleaning",
          "timestamp": "2022-01-27T22:02:10-08:00",
          "tree_id": "524e11acece2f6e6d5e27734bbb2c977ee64cd29",
          "url": "https://github.com/sfu-db/dataprep/commit/672a1adaf9188e51c1fb4ed4d88b0a028a4f8a82"
        },
        "date": 1643349967142,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.22427558879857096,
            "unit": "iter/sec",
            "range": "stddev: 0.12798186852026072",
            "extra": "mean: 4.458800020799998 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6c9e7ab91c4b065df194d5164ba2b2973858615e",
          "message": "Rename bug_report_cleaning to bug_report_cleaning.md",
          "timestamp": "2022-01-27T22:03:43-08:00",
          "tree_id": "e3d06400c078ad7517b6011e3244e7671bbf1298",
          "url": "https://github.com/sfu-db/dataprep/commit/6c9e7ab91c4b065df194d5164ba2b2973858615e"
        },
        "date": 1643350007654,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.34215397845428164,
            "unit": "iter/sec",
            "range": "stddev: 0.12229847786199405",
            "extra": "mean: 2.9226607404000107 sec\nrounds: 5"
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
          "id": "5c006c26adcfbb91bec73f0324b1a32c89714676",
          "message": "Merge pull request #797 from sfu-db/new_gui\n\nfeat(clean): New version of GUI",
          "timestamp": "2022-02-01T23:11:05-08:00",
          "tree_id": "08ebde87e8aaea32f072b0d39e40bd8a970a8b85",
          "url": "https://github.com/sfu-db/dataprep/commit/5c006c26adcfbb91bec73f0324b1a32c89714676"
        },
        "date": 1643786066904,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.3077941456755818,
            "unit": "iter/sec",
            "range": "stddev: 0.05818805263716981",
            "extra": "mean: 3.248924692199995 sec\nrounds: 5"
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
          "id": "f54f3ee79d2eaece8a920185e4373c2c9ba7760e",
          "message": "Merge pull request #799 from sfu-db/remove_bottneck\n\nchore(eda): remove dependency on bottleneck lib",
          "timestamp": "2022-02-03T01:13:59-08:00",
          "tree_id": "f4832f71374b045c36307efa44b4a34392ba05bc",
          "url": "https://github.com/sfu-db/dataprep/commit/f54f3ee79d2eaece8a920185e4373c2c9ba7760e"
        },
        "date": 1643879839346,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.29594411390926756,
            "unit": "iter/sec",
            "range": "stddev: 0.0514557669141963",
            "extra": "mean: 3.379016351400003 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c0f04ec48729fbb59398c2820666433c8d006ca6",
          "message": "[skip-ci] update readme",
          "timestamp": "2022-02-03T17:39:11-08:00",
          "tree_id": "fb27956cb4023bb7946e230a6b837402f02939ba",
          "url": "https://github.com/sfu-db/dataprep/commit/c0f04ec48729fbb59398c2820666433c8d006ca6"
        },
        "date": 1643938937973,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2993361664641417,
            "unit": "iter/sec",
            "range": "stddev: 0.05134314829198782",
            "extra": "mean: 3.340725618999977 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "65895033+gremur@users.noreply.github.com",
            "name": "Grey Murav",
            "username": "gremur"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "75820abd4bc295694ab1cb8c758907433507f253",
          "message": "Update currencies.json\n\nCorrected typo in Panamanian balboa symbol and replaced symbol for Russian ruble with correct one",
          "timestamp": "2022-02-08T22:22:33-08:00",
          "tree_id": "f49c80d95c447727f1753c48a8daae785dcb71f5",
          "url": "https://github.com/sfu-db/dataprep/commit/75820abd4bc295694ab1cb8c758907433507f253"
        },
        "date": 1644388014689,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2064354196124004,
            "unit": "iter/sec",
            "range": "stddev: 0.17562988686471565",
            "extra": "mean: 4.844129955400012 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yixuy@sfu.ca",
            "name": "henryye",
            "username": "yixuy"
          },
          "committer": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "d00d2ebddcd2c535e134b22e7f78a9f4bc62f63c",
          "message": "Fix the selection bug in clean module",
          "timestamp": "2022-02-14T09:50:40-08:00",
          "tree_id": "1fc586411db3a44a4ab09c51c0d95bfdd2159668",
          "url": "https://github.com/sfu-db/dataprep/commit/d00d2ebddcd2c535e134b22e7f78a9f4bc62f63c"
        },
        "date": 1644861269462,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.253053816627072,
            "unit": "iter/sec",
            "range": "stddev: 0.12113522905882783",
            "extra": "mean: 3.9517285822000074 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "4c3b23126ffcd3b4aa16276f2c73a6ad721b5d0e",
          "message": "fix(clean): fix the bug of am, pm",
          "timestamp": "2022-02-14T11:32:01-08:00",
          "tree_id": "8ae901a18f3ef55e6a6f8c1bfee47ea14c31fa9a",
          "url": "https://github.com/sfu-db/dataprep/commit/4c3b23126ffcd3b4aa16276f2c73a6ad721b5d0e"
        },
        "date": 1644867368483,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.22673464681319427,
            "unit": "iter/sec",
            "range": "stddev: 0.06823912677261681",
            "extra": "mean: 4.410441959600007 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yixuy@sfu.ca",
            "name": "henryye",
            "username": "yixuy"
          },
          "committer": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "691257dc250682060d41ce8a33e92d1ead40e586",
          "message": "Fix when minute and second of lat and long are 60",
          "timestamp": "2022-02-14T12:18:44-08:00",
          "tree_id": "7042b04249581efc493411b0358cb0cad54f4546",
          "url": "https://github.com/sfu-db/dataprep/commit/691257dc250682060d41ce8a33e92d1ead40e586"
        },
        "date": 1644870147106,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.26137982817687866,
            "unit": "iter/sec",
            "range": "stddev: 0.07012953044713187",
            "extra": "mean: 3.8258499402000097 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yixuy@sfu.ca",
            "name": "henryye",
            "username": "yixuy"
          },
          "committer": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "5b3a9f4f4218da1e8d358451407186986f2ecedb",
          "message": "Fix the bug and make None as NaN",
          "timestamp": "2022-02-14T20:16:08-08:00",
          "tree_id": "8f30cf849908b21188483a1df0c3e6d3123c36bb",
          "url": "https://github.com/sfu-db/dataprep/commit/5b3a9f4f4218da1e8d358451407186986f2ecedb"
        },
        "date": 1644898796952,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.22636511193202674,
            "unit": "iter/sec",
            "range": "stddev: 0.148339165854006",
            "extra": "mean: 4.417641886000001 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "c192ab43259714822d0daa4ff556811dbf85c763",
          "message": "fix(clean): remove usaddress library",
          "timestamp": "2022-02-15T15:30:21-08:00",
          "tree_id": "88c404af82037913c6ce0ac48eba670e57f004e4",
          "url": "https://github.com/sfu-db/dataprep/commit/c192ab43259714822d0daa4ff556811dbf85c763"
        },
        "date": 1644968010767,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.3143412438753122,
            "unit": "iter/sec",
            "range": "stddev: 0.03996851375572618",
            "extra": "mean: 3.1812561013999927 sec\nrounds: 5"
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
          "id": "de06223cb3197c8cacbd9ece7203a437a180981b",
          "message": "Merge pull request #798 from jwa345/newt\n\ndocs(eda): add doc for getting imdt result.",
          "timestamp": "2022-02-16T17:46:41-08:00",
          "tree_id": "e67135fb00680ee1f345fa65603c1099d20b96f0",
          "url": "https://github.com/sfu-db/dataprep/commit/de06223cb3197c8cacbd9ece7203a437a180981b"
        },
        "date": 1645062604322,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.29779557301759774,
            "unit": "iter/sec",
            "range": "stddev: 0.0543909577075387",
            "extra": "mean: 3.3580082802000106 sec\nrounds: 5"
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
          "id": "ab38a36287d21f6a2b1ec208c6bb53d4572b70ea",
          "message": "Merge pull request #791 from khoatxp/feature/add-sort-by-for-variables\n\nfeat(eda): add sort by for variables section in create_report",
          "timestamp": "2022-02-16T21:02:10-08:00",
          "tree_id": "b115281e4634f581430f3126bfb5867b05d1a39a",
          "url": "https://github.com/sfu-db/dataprep/commit/ab38a36287d21f6a2b1ec208c6bb53d4572b70ea"
        },
        "date": 1645074330015,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.29534274820149226,
            "unit": "iter/sec",
            "range": "stddev: 0.06477423505339747",
            "extra": "mean: 3.3858965764000004 sec\nrounds: 5"
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
          "id": "da4dc01e283234d3c0439e5dca287ba57d009879",
          "message": "Merge pull request #817 from sfu-db/fix/layout\n\nfix(eda):fix stat layout issue",
          "timestamp": "2022-02-17T16:12:32-08:00",
          "tree_id": "935731819f66096350eb1ef7638ce947372d468f",
          "url": "https://github.com/sfu-db/dataprep/commit/da4dc01e283234d3c0439e5dca287ba57d009879"
        },
        "date": 1645143352786,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2936800933641254,
            "unit": "iter/sec",
            "range": "stddev: 0.051961856060204016",
            "extra": "mean: 3.4050656567999966 sec\nrounds: 5"
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
          "id": "0a3d037e3c543112a87acd9ba1afea18b1e6cff6",
          "message": "Merge pull request #818 from sfu-db/fix/755\n\nfix(eda): fix cat-cat error",
          "timestamp": "2022-02-17T16:58:51-08:00",
          "tree_id": "af1a13f9214098f367b558654bbd407d00542a52",
          "url": "https://github.com/sfu-db/dataprep/commit/0a3d037e3c543112a87acd9ba1afea18b1e6cff6"
        },
        "date": 1645146139079,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2758730829757548,
            "unit": "iter/sec",
            "range": "stddev: 0.08976753006101974",
            "extra": "mean: 3.6248552747999896 sec\nrounds: 5"
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
          "id": "f7b4b2fafb3f621cb2a5cef576784305d6373e2f",
          "message": "Merge pull request #816 from jwa345/bug\n\nfix(eda): index 0 is out of bounds for axis 0 with size 0 for datapre…",
          "timestamp": "2022-02-17T19:51:37-08:00",
          "tree_id": "e9cbbceedfd94ab274b20533581fd50237a9ae61",
          "url": "https://github.com/sfu-db/dataprep/commit/f7b4b2fafb3f621cb2a5cef576784305d6373e2f"
        },
        "date": 1645156482427,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.3155954679479646,
            "unit": "iter/sec",
            "range": "stddev: 0.13491785673173745",
            "extra": "mean: 3.1686133089999884 sec\nrounds: 5"
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
          "id": "1550d4bdafcf700b992c6290fe6a1bb5be775898",
          "message": "Merge pull request #807 from devinllu/fix/report_plots\n\nrefactor(eda): restyled plots, re-ordered plot titles",
          "timestamp": "2022-02-17T20:36:22-08:00",
          "tree_id": "67088e201aeb24654d40d4b74caaef6a776d08e1",
          "url": "https://github.com/sfu-db/dataprep/commit/1550d4bdafcf700b992c6290fe6a1bb5be775898"
        },
        "date": 1645159218510,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.21782962227869015,
            "unit": "iter/sec",
            "range": "stddev: 0.06477858174484921",
            "extra": "mean: 4.590743855400001 sec\nrounds: 5"
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
          "id": "ba961a1869f59fba3e6b8144a6d9c8fb0a146b7e",
          "message": "Merge pull request #810 from khoatxp/feat/802-add-pagination-in-plot\n\n802/ feat(eda.plot): Add pagination in plot",
          "timestamp": "2022-02-18T13:56:48-08:00",
          "tree_id": "3ce591aab2a17170785163633217dcc44606da89",
          "url": "https://github.com/sfu-db/dataprep/commit/ba961a1869f59fba3e6b8144a6d9c8fb0a146b7e"
        },
        "date": 1645221626992,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.24713906073601763,
            "unit": "iter/sec",
            "range": "stddev: 0.11941833895777605",
            "extra": "mean: 4.046304930599996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yixuy@sfu.ca",
            "name": "henryye",
            "username": "yixuy"
          },
          "committer": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "54d4e0a5b2356409da0f3d975b3658d9f914c974",
          "message": "Add selections in clean column and modify the UI",
          "timestamp": "2022-02-20T12:56:05-08:00",
          "tree_id": "6a02b6ba23903c88ac3832d3fda969cde155a66a",
          "url": "https://github.com/sfu-db/dataprep/commit/54d4e0a5b2356409da0f3d975b3658d9f914c974"
        },
        "date": 1645390793228,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.23641345515464715,
            "unit": "iter/sec",
            "range": "stddev: 0.06294898692125378",
            "extra": "mean: 4.229877691799993 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "c84c062e1a3427cb767caa6da6723e3b0c6d5a58",
          "message": "docs (clean): fix doc of clean_au_acn",
          "timestamp": "2022-02-20T12:57:29-08:00",
          "tree_id": "ddef2aad9fbde9c0e5252eb3195804e536303d8f",
          "url": "https://github.com/sfu-db/dataprep/commit/c84c062e1a3427cb767caa6da6723e3b0c6d5a58"
        },
        "date": 1645390855691,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.280188054751415,
            "unit": "iter/sec",
            "range": "stddev: 0.09233676122366175",
            "extra": "mean: 3.5690315237999983 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "5e2f38acc264c3b1ec945f047879833a355d6863",
          "message": "docs(clean): add doc of clean GUI",
          "timestamp": "2022-02-20T23:37:33-08:00",
          "tree_id": "9a0ba6979fadf7773d8b86a5b58f9dcf3ca2462f",
          "url": "https://github.com/sfu-db/dataprep/commit/5e2f38acc264c3b1ec945f047879833a355d6863"
        },
        "date": 1645429256949,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2899842103336491,
            "unit": "iter/sec",
            "range": "stddev: 0.05712295425771257",
            "extra": "mean: 3.448463620999996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "yixuy@sfu.ca",
            "name": "henryye",
            "username": "yixuy"
          },
          "committer": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "1176ddd7d0c2706a3ef17fd37afb03573ae26369",
          "message": "Clean up the backend",
          "timestamp": "2022-02-21T09:25:12-08:00",
          "tree_id": "7f83acd73cef1f96b0e6c84776908a2d76004547",
          "url": "https://github.com/sfu-db/dataprep/commit/1176ddd7d0c2706a3ef17fd37afb03573ae26369"
        },
        "date": 1645464507491,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.303998021350325,
            "unit": "iter/sec",
            "range": "stddev: 0.052705302196025754",
            "extra": "mean: 3.2894950945999994 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "0e072a8007554fdfdcc50485db33d43bd15747ef",
          "message": "fix(clean): delete abundant print",
          "timestamp": "2022-02-21T10:07:32-08:00",
          "tree_id": "7b629f1b8e4305311f720add47876b4981d0d011",
          "url": "https://github.com/sfu-db/dataprep/commit/0e072a8007554fdfdcc50485db33d43bd15747ef"
        },
        "date": 1645467070724,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.26613485986545904,
            "unit": "iter/sec",
            "range": "stddev: 0.08934284487791957",
            "extra": "mean: 3.7574934772000064 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "03337d80b239bf9729f0dffbf3225cc410df8266",
          "message": "build(deps-dev): bump node-notifier\n\nBumps [node-notifier](https://github.com/mikaelbr/node-notifier) from 5.4.5 to 8.0.1.\n- [Release notes](https://github.com/mikaelbr/node-notifier/releases)\n- [Changelog](https://github.com/mikaelbr/node-notifier/blob/v8.0.1/CHANGELOG.md)\n- [Commits](https://github.com/mikaelbr/node-notifier/compare/v5.4.5...v8.0.1)\n\n---\nupdated-dependencies:\n- dependency-name: node-notifier\n  dependency-type: direct:development\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2022-02-21T12:45:51-08:00",
          "tree_id": "48795ad1c8b56c4a2ec080b4ddc9df174886641d",
          "url": "https://github.com/sfu-db/dataprep/commit/03337d80b239bf9729f0dffbf3225cc410df8266"
        },
        "date": 1645476529569,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.22406782347309098,
            "unit": "iter/sec",
            "range": "stddev: 0.02277722810276499",
            "extra": "mean: 4.4629344120000045 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "8551c2b628f27a1bcb7df6181f9ab19843bf5573",
          "message": "fix some dtype warnings",
          "timestamp": "2022-02-21T12:45:14-08:00",
          "tree_id": "56aaa834b4181974b7893ffa3b0bc5c5fa7c8178",
          "url": "https://github.com/sfu-db/dataprep/commit/8551c2b628f27a1bcb7df6181f9ab19843bf5573"
        },
        "date": 1645476542467,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15507151577557687,
            "unit": "iter/sec",
            "range": "stddev: 0.05234386221044922",
            "extra": "mean: 6.448637552799982 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "14dfcd3a7b910f133d65641ac1fa68392e187dff",
          "message": "build(deps-dev): bump shelljs in /dataprep/clean/gui/clean_frontend\n\nBumps [shelljs](https://github.com/shelljs/shelljs) from 0.7.8 to 0.8.5.\n- [Release notes](https://github.com/shelljs/shelljs/releases)\n- [Changelog](https://github.com/shelljs/shelljs/blob/master/CHANGELOG.md)\n- [Commits](https://github.com/shelljs/shelljs/compare/v0.7.8...v0.8.5)\n\n---\nupdated-dependencies:\n- dependency-name: shelljs\n  dependency-type: direct:development\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2022-02-21T12:46:03-08:00",
          "tree_id": "253ab805e6e9cd4e87f75c7bcd9e1fece6fed83b",
          "url": "https://github.com/sfu-db/dataprep/commit/14dfcd3a7b910f133d65641ac1fa68392e187dff"
        },
        "date": 1645476586645,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15949942855015245,
            "unit": "iter/sec",
            "range": "stddev: 0.06733209429470548",
            "extra": "mean: 6.269614939000007 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "5c985d7992d86ae541ca331537389115936f2252",
          "message": "build(deps): bump follow-redirects in /dataprep/clean/gui/clean_frontend\n\nBumps [follow-redirects](https://github.com/follow-redirects/follow-redirects) from 1.14.5 to 1.14.8.\n- [Release notes](https://github.com/follow-redirects/follow-redirects/releases)\n- [Commits](https://github.com/follow-redirects/follow-redirects/compare/v1.14.5...v1.14.8)\n\n---\nupdated-dependencies:\n- dependency-name: follow-redirects\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2022-02-21T12:46:11-08:00",
          "tree_id": "f11a73cdeedcf9ebd791b6db99e1fc15a1f04847",
          "url": "https://github.com/sfu-db/dataprep/commit/5c985d7992d86ae541ca331537389115936f2252"
        },
        "date": 1645476588939,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.17178849606332222,
            "unit": "iter/sec",
            "range": "stddev: 0.12822760482953866",
            "extra": "mean: 5.821111558199999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "youngw@sfu.ca",
            "name": "Weiyuan Wu",
            "username": "dovahcrow"
          },
          "distinct": true,
          "id": "9016a6f10ecfc6088352a38be107841589bf4e71",
          "message": "build(deps): bump url-parse in /dataprep/clean/gui/clean_frontend\n\nBumps [url-parse](https://github.com/unshiftio/url-parse) from 1.5.3 to 1.5.7.\n- [Release notes](https://github.com/unshiftio/url-parse/releases)\n- [Commits](https://github.com/unshiftio/url-parse/compare/1.5.3...1.5.7)\n\n---\nupdated-dependencies:\n- dependency-name: url-parse\n  dependency-type: indirect\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2022-02-21T12:46:19-08:00",
          "tree_id": "3140dd1245bc884e08c3bba5e834497b5f2ce066",
          "url": "https://github.com/sfu-db/dataprep/commit/9016a6f10ecfc6088352a38be107841589bf4e71"
        },
        "date": 1645476651810,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.203419296021057,
            "unit": "iter/sec",
            "range": "stddev: 0.022911959042757",
            "extra": "mean: 4.915954482000001 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "qidanrui",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "c6bc677d85d4750cd0343b212f2d8a733c878f5d",
          "message": "update release version to 0.4.2",
          "timestamp": "2022-02-21T14:06:23-08:00",
          "tree_id": "db1a4814675ac031b2df45e734938745582a1ab3",
          "url": "https://github.com/sfu-db/dataprep/commit/c6bc677d85d4750cd0343b212f2d8a733c878f5d"
        },
        "date": 1645481407206,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.2229410288360227,
            "unit": "iter/sec",
            "range": "stddev: 0.0907752968707628",
            "extra": "mean: 4.485491096999999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f934b9eac9392d988184393057f3d21a32134377",
          "message": "Update README.md",
          "timestamp": "2022-02-23T18:36:20-08:00",
          "tree_id": "6c9876548e327acf1b1c2f5209133881fb8f68ce",
          "url": "https://github.com/sfu-db/dataprep/commit/f934b9eac9392d988184393057f3d21a32134377"
        },
        "date": 1645670357793,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.20955226380426686,
            "unit": "iter/sec",
            "range": "stddev: 0.15589569096455266",
            "extra": "mean: 4.772079202799995 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e3f11ffbadda7eb7396bb2f1f27fb1ca26953e1b",
          "message": "Update README.md\n\nAdd clean gui video",
          "timestamp": "2022-02-23T19:15:13-08:00",
          "tree_id": "204fc9afdc35fe3fd7cc5434c76ddb4049b871c9",
          "url": "https://github.com/sfu-db/dataprep/commit/e3f11ffbadda7eb7396bb2f1f27fb1ca26953e1b"
        },
        "date": 1645672716959,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1962441152305623,
            "unit": "iter/sec",
            "range": "stddev: 0.02821317676072066",
            "extra": "mean: 5.095694201199995 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "56e6f2b0cdd0962c7de3cb83d3ac8ac08c19f94c",
          "message": "Update README.md",
          "timestamp": "2022-02-23T19:21:35-08:00",
          "tree_id": "9e037c50c4b4e7de8c4e74ec38b8dc3a89a946f1",
          "url": "https://github.com/sfu-db/dataprep/commit/56e6f2b0cdd0962c7de3cb83d3ac8ac08c19f94c"
        },
        "date": 1645673081400,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.20841968993897886,
            "unit": "iter/sec",
            "range": "stddev: 0.053213031086330576",
            "extra": "mean: 4.7980111681999915 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "96c3f17bdb776fad7aac9f14f1556b3b67b6986b",
          "message": "revise the levenshtein package to python-levenshtein to fit for conda forge",
          "timestamp": "2022-02-23T19:22:36-08:00",
          "tree_id": "87db8c3be560ee10232f584556c1788577425af8",
          "url": "https://github.com/sfu-db/dataprep/commit/96c3f17bdb776fad7aac9f14f1556b3b67b6986b"
        },
        "date": 1645673200591,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.1498088480790314,
            "unit": "iter/sec",
            "range": "stddev: 0.14336656541757997",
            "extra": "mean: 6.675173147800001 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "c741937297248c047076c9ac42701b08bca419dc",
          "message": "add video cover of clean gui",
          "timestamp": "2022-02-23T19:28:36-08:00",
          "tree_id": "db87874d62ac24fa447ef8d2717b1b9f3e302388",
          "url": "https://github.com/sfu-db/dataprep/commit/c741937297248c047076c9ac42701b08bca419dc"
        },
        "date": 1645673544561,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.16197948247757415,
            "unit": "iter/sec",
            "range": "stddev: 0.10995020585594967",
            "extra": "mean: 6.1736214038000075 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "87b59da3414fbb652780474dce9af9954600b77b",
          "message": "add video cover of clean gui",
          "timestamp": "2022-02-23T19:34:49-08:00",
          "tree_id": "77332000281b9446a78e97019077f4aca349450b",
          "url": "https://github.com/sfu-db/dataprep/commit/87b59da3414fbb652780474dce9af9954600b77b"
        },
        "date": 1645673926647,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.14804016922780824,
            "unit": "iter/sec",
            "range": "stddev: 0.044311482500312256",
            "extra": "mean: 6.754923378000012 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b2f79ed95bcdb0f0f22034e0ef4acc78c0d3f3b8",
          "message": "Update README.md",
          "timestamp": "2022-02-23T19:36:53-08:00",
          "tree_id": "cfa3e2545dba9d93e022ec3e26fc37bdb875e1f4",
          "url": "https://github.com/sfu-db/dataprep/commit/b2f79ed95bcdb0f0f22034e0ef4acc78c0d3f3b8"
        },
        "date": 1645674034094,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.15921292849342994,
            "unit": "iter/sec",
            "range": "stddev: 0.04453994167473252",
            "extra": "mean: 6.280896969000014 sec\nrounds: 5"
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
            "email": "qidanrui@gmail.com",
            "name": "Danrui Qi",
            "username": "qidanrui"
          },
          "distinct": true,
          "id": "7e60e9b08fc0b7745583e77f18da77224f5673d6",
          "message": "fix the log expression of clean df on clean gui",
          "timestamp": "2022-02-24T11:33:30-08:00",
          "tree_id": "46bba0b55fd81ed35b034a7b81c67d8f2bc4091e",
          "url": "https://github.com/sfu-db/dataprep/commit/7e60e9b08fc0b7745583e77f18da77224f5673d6"
        },
        "date": 1645731410284,
        "tool": "pytest",
        "benches": [
          {
            "name": "dataprep/tests/benchmarks/eda.py::test_create_report",
            "value": 0.20111904594454078,
            "unit": "iter/sec",
            "range": "stddev: 0.051514263297475923",
            "extra": "mean: 4.972179513400005 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}