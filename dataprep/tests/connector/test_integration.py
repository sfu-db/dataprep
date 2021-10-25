# type: ignore
from os import environ
import asyncio
import pytest

from ...connector import Connector, websites
from ...utils import display_dataframe
from ...connector.utils import Request


# @pytest.mark.skipif(
#     environ.get("DATAPREP_CREDENTIAL_TESTS", "0") == "0",
#     reason="Skip tests that requires credential",
# )
# def test_connector() -> None:
#    token = environ["DATAPREP_DATA_CONNECTOR_YELP_TOKEN"]
#    dc = Connector("yelp", _auth={"access_token": token}, _concurrency=3)
#    df = asyncio.run(dc.query("businesses", term="ramen", location="vancouver"))
#
#    assert len(df) > 0
#
#    websites()
#
#    dc.info()
#
#    display_dataframe(df)
#
#    df = asyncio.run(dc.query("businesses", _count=120, term="ramen", location="vancouver"))
#
#    assert len(df) == 120
#
#    df = asyncio.run(dc.query("businesses", _count=1000, term="ramen", location="vancouver"))
#
#    assert len(df) < 1000
#


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
    req1 = Request("https://www.python.org/")
    get_resp = req1.get()
    assert get_resp.status == 200

    # POST request
    params = {"@number": 12524, "@type": "issue", "@action": "show"}
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    req2 = Request("https://bugs.python.org/")
    post_resp = req2.post(_data=params, _headers=headers)
    assert post_resp.status == 302

    # PUT request
    params = {"@number": 12524, "@type": "issue", "@action": "show"}
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    req3 = Request("https://bugs.python.org/")
    put_resp = req3.put(_data=params, _headers=headers)
    assert put_resp.status == 302
