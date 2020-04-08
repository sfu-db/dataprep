"""
Defines useful types in this library.
"""
from base64 import b64encode
from enum import Enum
from time import time
from typing import Any, Dict, Optional, cast
from sys import stderr
import requests
from jinja2 import Environment, UndefinedError

from ..errors import UnreachableError


class AuthorizationType(Enum):
    """Enum class defines the supported authorization methods in this library.

    Note
    ----

    * Bearer: requires 'access_token' presented in user params
    * OAuth2: requires 'client_id' and 'client_secret' in user params for
      'ClientCredentials' grant type
    """

    Bearer = "Bearer"
    OAuth2 = "OAuth2"


class Authorization:
    """Class carries the authorization type and
    the corresponding parameter.
    """

    auth_type: AuthorizationType
    params: Dict[str, str]
    storage: Dict[str, Any]

    def __init__(self, auth_type: AuthorizationType, params: Dict[str, str]) -> None:
        self.auth_type = auth_type
        self.params = params
        self.storage = {}

    def build(self, req_data: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Populate some required fields to the request data.
        Complex logic may also happens in this function (e.g. start a server to do OAuth).
        """
        if self.auth_type == AuthorizationType.Bearer:  # pylint: disable=no-member
            req_data["headers"]["Authorization"] = f"Bearer {params['access_token']}"
        elif (
            self.auth_type == AuthorizationType.OAuth2
            and self.params["grantType"] == "ClientCredentials"
        ):
            # TODO: Move OAuth to a separate authenticator
            if (
                "access_token" not in self.storage
                or self.storage.get("expires_at", 0) < time()
            ):
                # Not yet authorized
                ckey = params["client_id"]
                csecret = params["client_secret"]
                b64cred = b64encode(f"{ckey}:{csecret}".encode("ascii")).decode()
                resp = requests.post(
                    self.params["tokenServerUrl"],
                    headers={"Authorization": f"Basic {b64cred}"},
                    data={"grant_type": "client_credentials"},
                ).json()

                assert resp["token_type"].lower() == "bearer"
                access_token = resp["access_token"]
                self.storage["access_token"] = access_token
                if "expires_in" in resp:
                    self.storage["expires_at"] = (
                        time() + resp["expires_in"] - 60
                    )  # 60 seconds grace period to avoid clock lag

            req_data["headers"][
                "Authorization"
            ] = f"Bearer {self.storage['access_token']}"

            # TODO: handle auto refresh
        elif (
            self.auth_type == AuthorizationType.OAuth2
            and self.params["grantType"] == "AuthorizationCode"
        ):
            raise NotImplementedError


class Fields:
    """A data structure that stores the fields information (e.g. headers, cookies, ...).
    This class is useful to populate concrete fields data with required variables provided.
    """

    fields: Dict[str, Any]

    def __init__(self, fields_config: Dict[str, Any]) -> None:
        self.fields = fields_config

    def populate(  # pylint: disable=too-many-branches
        self, jenv: Environment, params: Dict[str, Any]
    ) -> Dict[str, str]:
        """Populate a dict based on the fields definition and provided vars.
        """
        ret: Dict[str, str] = {}

        for key, def_ in self.fields.items():
            from_key, to_key = key, key

            if isinstance(def_, bool):
                required = def_
                value = params.get(from_key)
                if value is None and required:
                    raise KeyError(from_key)
                remove_if_empty = False
            elif isinstance(def_, str):
                # is a template
                template: Optional[str] = def_
                tmplt = jenv.from_string(cast(str, template))
                value = tmplt.render(**params)
                remove_if_empty = False
            elif isinstance(def_, dict):
                template = def_.get("template")
                remove_if_empty = def_["removeIfEmpty"]
                to_key = def_.get("toKey") or to_key
                from_key = def_.get("fromKey") or from_key

                if template is None:
                    required = def_["required"]
                    value = params.get(from_key)
                    if value is None and required:
                        raise KeyError(from_key)
                else:
                    tmplt = jenv.from_string(template)
                    try:
                        value = tmplt.render(**params)
                    except UndefinedError:
                        value = ""  # This empty string will be removed if `remove_if_empty` is True
            else:
                raise UnreachableError()

            if value is not None:
                str_value = str(value)

                if not (remove_if_empty and not str_value):
                    if to_key in ret:
                        print(f"Param {key} conflicting with {to_key}", file=stderr)
                    ret[to_key] = str_value
                    continue
        return ret


class Orient(Enum):
    """Different types of table orientations
    ref: (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html).
    Currently, DataConnector supports two different types of orientaions:

    1. Split, which is column store.
    2. Records, which is row store.

    Details can be found in the pandas page.
    """

    Split = "split"
    Records = "records"
