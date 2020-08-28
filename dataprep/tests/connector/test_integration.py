# type: ignore
from os import environ
import asyncio
import pytest

from ...connector import Connector


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

    schema = dc.show_schema("businesses")

    assert len(schema) > 0

    df = asyncio.run(
        dc.query("businesses", _count=120, term="ramen", location="vancouver")
    )

    assert len(df) == 120

    df = asyncio.run(
        dc.query("businesses", _count=1000, term="ramen", location="vancouver")
    )

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
