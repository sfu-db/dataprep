# type: ignore
from os import environ

import pytest

from ...data_connector import Connector


@pytest.mark.skipif(
    environ.get("DATAPREP_SKIP_CREDENTIAL_TESTS") == "1",
    reason="Skip tests that requires credential",
)
def test_data_connector() -> None:
    token = environ["DATAPREP_DATA_CONNECTOR_YELP_TOKEN"]
    dc = Connector("yelp", _auth={"access_token": token})
    df = dc.query("businesses", term="ramen", location="vancouver")

    assert len(df) > 0

    dc.info()

    schema = dc.show_schema("businesses")

    assert len(schema) > 0

    df = dc.query("businesses", _count=120, term="ramen", location="vancouver")

    assert len(df) == 120

    df = dc.query("businesses", _count=10000, term="ramen", location="vancouver")

    assert len(df) < 1000


@pytest.mark.skipif(
    environ.get("DATAPREP_SKIP_CREDENTIAL_TESTS") == "1",
    reason="Skip tests that requires credential",
)
def test_query_params() -> None:

    token = environ["DATAPREP_DATA_CONNECTOR_YOUTUBE_TOKEN"]

    dc = Connector("youtube", _auth={"access_token": token})
    df = dc.query("videos", q="covid", part="snippet")

    assert len(df) != 0
