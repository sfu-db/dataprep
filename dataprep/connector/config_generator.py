"""
    This module implements the generation of connector configuration files
"""
import re
import json
from urllib import parse
from requests import Request, Response, Session


def create_config(example: str) -> 'ConfigGenerator':
    """
    Creates a ConfigGenerator object which has an in-memory
    representation of a configuration file

    Returns
    -------
    ConfigGenerator
        The ConfigGenerator instance.
    """
    config_gen = ConfigGenerator()
    config_gen.add_example(example)
    return config_gen


# pylint: disable=too-many-instance-attributes
class ConfigGenerator:
    """
    Class that generate configuration files according to
    input information provided by the user, for example
    an HTTP request example from a REST API.

    Example
    -------
    >>> from dataprep.connector import config_generator as cg
    >>> req_example = "GET https://openlibrary.org/api/books?bibkeys=ISBN:0385472579&format=json"
    >>> config = cg.create_config(req_example)

    """
    _request_example: str
    _url: str
    _parameters: dict
    _content_type: str
    _table_path: str
    _version: int
    _request: dict
    _response: dict
    _method: str
    _schema_cols: list
    _headers: dict
    _orient: str
    _session: Session
    _config: str

    def __init__(
            self
    ) -> None:
        self._request_example = str()
        self._url = str()
        self._parameters = dict()
        self._content_type = str()
        self._table_path = "$[*]"
        self._version = 1
        self._request = dict()
        self._response = dict()
        self._method = "GET"
        self._schema_cols = list()
        self._headers = dict()
        self._orient = "records"
        self._session = Session()
        self._config = str()

    def add_example(
            self,
            request_example: str
    ) -> 'ConfigGenerator':
        """
        Parse the request example, execute the request, create the in-memory
        representation of a configuration file and returns the corresponding
        ConfigGenerator object.

        Parameters
        ----------
        request_example
            The HTTP request example, e.g.:
            GET https://openlibrary.org/api/books?bibkeys=ISBN:0385472579&format=json

        Returns
        -------
        ConfigGenerator
            The ConfigGenerator instance created from the request example.
        """

        self._parse_example(request_example)
        self._execute_request()
        self._create_config_file_representation()
        return self

    def _parse_example(
            self,
            request_example: str
    ) -> None:
        """
        Parse the request example extracting all the relevant information to perform
        a request.

        Parameters
        ----------
        request_example
            The HTTP request example, for example:
            GET https://openlibrary.org/api/books?bibkeys=ISBN:0385472579&format=json
        """
        self._request_example = request_example
        try:
            request_full_url = re.search("(?P<url>https?://[^\s]+)",
                                         self._request_example).group("url")
        except Exception:
            raise RuntimeError(f"Malformed request example syntax: \
                               {self._request_example}") from None
        else:
            parsed_full_url = parse.urlparse(request_full_url)
            self._parameters = parse.parse_qs(parsed_full_url.query)
            if len(self._parameters) != 0:
                lst_parsed_full_url = list(parsed_full_url)
                lst_parsed_full_url[4] = str()
                self._url = parse.urlunparse(lst_parsed_full_url)
            else:
                raise RuntimeError(f"Malformed request example syntax: \
                                   {self._request_example}") from None

    def _execute_request(
            self
    ) -> None:
        """
        Execute an HTTP request taking as input all the parameters extracted from
        the request example, then, extract all the relevant information from the
        received HTTP response.
        """
        request = Request(
            method=self._method,
            url=self._url,
            headers=self._headers,
            params=self._parameters,
            json=None,
            data=None,
            cookies=dict(),
        )
        prep_request = request.prepare()
        resp: Response = self._session.send(prep_request)
        if resp.status_code == 200:
            self._content_type = resp.headers['content-type']
            try:
                self._response = resp.json()
            except ValueError:
                raise RuntimeError(f"Response body from {self._url} \
                                   does not contain a valid JSON.") from None
        else:
            raise RuntimeError(f"HTTP status received: {resp.status_code}. \
                                Expected: 200.") from None

    def _create_config_file_representation(
            self
    ) -> None:
        """
        Creates an in-memory representation (string) of a configuration file.
        """
        if len(self._response) == 0:
            self._schema_cols = list()
        else:
            self._schema_cols = list(dict(list(self._response.values())[0]).keys())
        config = {
            "version": self._version,
            "request": {
                "url": self._url,
                "method": self._method,
                "params": {p: False for p in self._parameters}
            },
            "response": {
                "ctype": "application/json",
                "tablePath": self._table_path,
                "schema": {sc: {"target": "$." + sc, "type": "string"}
                           for sc in self._schema_cols},
                "orient": self._orient
            }
        }
        self._config = json.dumps(config, indent=4)

    def save(
            self,
            filename: str
    ) -> None:
        """
        Save to disk the current in-memory representation (string) of a configuration file to a
        file specified as parameter.

        Parameters
        ----------
        filename
            Name of the file to be saved. It can include the path.
        """
        with open(filename, "w") as outfile:
            outfile.write(self._config)

    def to_string(
            self
    ) -> str:
        """
        Return the current in-memory representation (string) of a configuration file.

        Returns
        -------
        _config
            String of the in-memory representation (string) of a configuration file.
        """
        return self._config
