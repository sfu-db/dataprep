from ...data_connector import Connector
from os import environ


def test_data_connector() -> None:
    token = environ["DATAPREP_DATA_CONNECTOR_YELP_TOKEN"]
    dc = Connector("yelp", auth_params={"access_token": token})
    df = dc.query("businesses", term="ramen", location="vancouver")

    assert len(df) > 0

    dc.info()

    schema = dc.show_schema("businesses")

    assert len(schema) > 0
