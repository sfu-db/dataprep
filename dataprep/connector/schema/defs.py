"""Strong typed schema definition."""

from base64 import b64encode
from copy import deepcopy
from enum import Enum
from time import time
from typing import Any, Dict, Optional, Union

import requests
from pydantic import Field

from .base import BaseDef


# pylint: disable=missing-class-docstring,missing-function-docstring
class PaginationDef(BaseDef):
    type: str = Field(regex=r"^(offset|seek)$")
    max_count: int
    offset_key: Optional[str]
    limit_key: str
    seek_id: Optional[str]
    seek_key: Optional[str]


class FieldDef(BaseDef):
    required: bool
    from_key: Optional[str]
    to_key: Optional[str]
    template: Optional[str]
    remove_if_empty: bool


FieldDefUnion = Union[FieldDef, bool, str]  # Put bool before str


class OAuth2AuthorizationDef(BaseDef):
    type: str = Field("OAuth2", const=True)
    grant_type: str
    token_server_url: str

    def build(
        self,
        req_data: Dict[str, Any],
        params: Dict[str, Any],
        storage: Optional[Dict[str, Any]] = None,
    ) -> None:
        if storage is None:
            raise ValueError("storage is required for OAuth2")

        if self.grant_type == "ClientCredentials":
            if "access_token" not in storage or storage.get("expires_at", 0) < time():
                # Not yet authorized
                ckey = params["client_id"]
                csecret = params["client_secret"]
                b64cred = b64encode(f"{ckey}:{csecret}".encode("ascii")).decode()
                resp: Dict[str, Any] = requests.post(
                    self.token_server_url,
                    headers={"Authorization": f"Basic {b64cred}"},
                    data={"grant_type": "client_credentials"},
                ).json()
                if resp["token_type"].lower() != "bearer":
                    raise RuntimeError("token_type is not bearer")

                access_token = resp["access_token"]
                storage["access_token"] = access_token
                if "expires_in" in resp:
                    storage["expires_at"] = (
                        time() + resp["expires_in"] - 60
                    )  # 60 seconds grace period to avoid clock lag

            req_data["headers"]["Authorization"] = f"Bearer {storage['access_token']}"

            # TODO: handle auto refresh
        elif self.grant_type == "AuthorizationCode":
            raise NotImplementedError


class QueryParamAuthorizationDef(BaseDef):
    type: str = Field("QueryParam", const=True)
    key_param: str

    def build(
        self,
        req_data: Dict[str, Any],
        params: Dict[str, Any],
        storage: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    ) -> None:
        """Populate some required fields to the request data."""

        req_data["params"][self.key_param] = params["access_token"]


class BearerAuthorizationDef(BaseDef):
    type: str = Field("Bearer", const=True)

    @staticmethod
    def build(
        req_data: Dict[str, Any],
        params: Dict[str, Any],
        storage: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    ) -> None:
        """Populate some required fields to the request data."""

        req_data["headers"]["Authorization"] = f"Bearer {params['access_token']}"


class HeaderAuthorizationDef(BaseDef):
    type: str = Field("Header", const=True)
    key_name: str
    extra: Dict[str, str] = Field(default_factory=dict)

    def build(
        self,
        req_data: Dict[str, Any],
        params: Dict[str, Any],
        storage: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
    ) -> None:
        """Populate some required fields to the request data."""

        req_data["headers"][self.key_name] = params["access_token"]
        req_data["headers"].update(self.extra)


AuthorizationDef = Union[
    OAuth2AuthorizationDef,
    QueryParamAuthorizationDef,
    BearerAuthorizationDef,
    HeaderAuthorizationDef,
]


class BodyDef(BaseDef):
    ctype: str = Field(regex=r"^(application/x-www-form-urlencoded|application/json)$")
    content: Dict[str, FieldDefUnion]


class Method(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"


class RequestDef(BaseDef):
    url: str
    method: Method
    authorization: Optional[AuthorizationDef]
    headers: Optional[Dict[str, FieldDefUnion]]
    params: Dict[str, FieldDefUnion]
    pagination: Optional[PaginationDef]
    body: Optional[BodyDef]
    cookies: Optional[Dict[str, FieldDefUnion]]


class SchemaFieldDef(BaseDef):
    target: str
    type: str
    description: Optional[str]

    def merge(self, rhs: Any) -> "SchemaFieldDef":
        if not isinstance(rhs, SchemaFieldDef):
            raise ValueError(f"Cannot merge {type(self)} with {type(rhs)}")

        if self.target != rhs.target:
            raise ValueError("Cannot merge SchemaFieldDef with different target.")

        merged_type = merge_type(self.type, rhs.type)

        cur = deepcopy(self)
        cur.type = merged_type
        cur.description = rhs.description

        return cur


TYPE_TREE = {
    "object": None,
    "string": None,
    "float": "string",
    "int": "float",
    "bool": "string",
}


def merge_type(a: str, b: str) -> str:  # pylint: disable=invalid-name
    if a == b:
        return a

    aset = {a}
    bset = {b}

    while True:
        aparent = TYPE_TREE[a]
        if aparent is not None:
            if aparent in bset:
                return aparent
            else:
                aset.add(aparent)
        bparent = TYPE_TREE[b]
        if bparent is not None:
            if bparent in aset:
                return bparent
            else:
                bset.add(bparent)

        if aparent is None and bparent is None:
            raise RuntimeError("Unreachable")


class ResponseDef(BaseDef):
    ctype: str = Field(regex=r"^(application/xml|application/json)$")
    table_path: str
    schema_: Dict[str, SchemaFieldDef] = Field(alias="schema")
    orient: str = Field(regex=r"^(records|split)$")


class ConfigDef(BaseDef):
    version: int = Field(1, const=True)
    request: RequestDef
    response: ResponseDef
