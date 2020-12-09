# type: ignore
from os import environ
import asyncio
import pytest

from ...connector import Connector
from ...connector.utils import Request
from ...utils import display_dataframe


@pytest.mark.skipif(
    environ.get("DATAPREP_CREDENTIAL_TESTS", "0") == "0",
    reason="Skip tests that requires credential",
)
def test_connector() -> None:
    token = environ["DATAPREP_DATA_CONNECTOR_YELP_TOKEN"]
    dc = Connector("yelp", _auth={"access_token": token}, _concurrency=3)
    df = asyncio.run(dc.query("businesses", term="ramen", location="vancouver"))

    assert len(df) > 0

    dc.info()

    display_dataframe(df)

    df = asyncio.run(dc.query("businesses", _count=120, term="ramen", location="vancouver"))

    assert len(df) == 120

    df = asyncio.run(dc.query("businesses", _count=1000, term="ramen", location="vancouver"))

    assert len(df) < 1000


@pytest.mark.skipif(
    environ.get("DATAPREP_CREDENTIAL_TESTS", "0") == "0",
    reason="Skip tests that requires credential",
)
def test_query_params() -> None:

    token = environ["DATAPREP_DATA_CONNECTOR_YOUTUBE_TOKEN"]

    dc = Connector("youtube", _auth={"access_token": token})
    df = asyncio.run(dc.query("videos", q="covid", part="snippet"))

    assert len(df) != 0


def test_requests() -> None:
    # GET request
    requests = Request("https://www.python.org/")
    response = requests.get()
    assert response.status == 200

    # POST request
    params = {"@number": 12524, "@type": "issue", "@action": "show"}
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    requests = Request("https://bugs.python.org/")
    response = requests.post(_data=params, _headers=headers)
    assert response.status == 302

    # PUT request
    params = {"@number": 12524, "@type": "issue", "@action": "show"}
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    requests = Request("https://bugs.python.org/")
    response = requests.put(_data=params, _headers=headers)
    assert response.status == 302
