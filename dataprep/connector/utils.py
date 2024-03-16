"""
This module contains common utilities used by the connector
"""

from typing import Any, Dict, Optional
import http.client
import urllib.parse


class Request:
    """
    Provides a wrapper for the python http.client,
    to be used similar to the requests library.

    Parameters
    ----------
    _url: The requesting end-point URL.
    """

    def __init__(self, _url: str) -> None:
        self.url: urllib.parse.ParseResult = urllib.parse.urlparse(_url)

        qs = dict(urllib.parse.parse_qsl(self.url.query))
        self.url = self.url._replace(query=urllib.parse.urlencode(qs))
        self.hostname: str = self.url.hostname or ""
        self.headers: Dict[str, Any] = dict({"user-agent": "dataprep"})

    def get(self, _headers: Optional[Dict[str, Any]] = None) -> http.client.HTTPResponse:
        """
        GET request to the specified end-point.

        Parameters
        ----------
        _headers: Any additional headers to be passed
        """
        if _headers:
            self.headers.update(_headers)

        conn = http.client.HTTPSConnection(self.hostname)
        path = self.url.path
        if path == "":
            path = "/"
        if self.url.query:
            path += "?" + self.url.query
        conn.request(method="GET", url=path, headers=self.headers)

        response = conn.getresponse()

        return response

    def post(
        self, _headers: Optional[Dict[str, Any]] = None, _data: Optional[Dict[str, Any]] = None
    ) -> http.client.HTTPResponse:
        """
        POST request to the specified end-point.

        Parameters
        ----------
        _headers: Any additional headers to be passed
        _data: Body of the request
        """
        if _headers:
            self.headers.update(_headers)
        conn = http.client.HTTPSConnection(self.hostname)

        path = self.url.path
        if path == "":
            path = "/"
        if self.url.query:
            path += "?" + self.url.query

        if _data is not None:
            conn.request(
                method="POST",
                url=path,
                headers=self.headers,
                body=urllib.parse.urlencode(_data),
            )
        else:
            conn.request(method="POST", url=path, headers=self.headers)
        response = conn.getresponse()

        return response

    def put(
        self, _headers: Optional[Dict[str, Any]] = None, _data: Optional[Dict[str, Any]] = None
    ) -> http.client.HTTPResponse:
        """
        PUT request to the specified end-point.

        Parameters
        ----------
        _headers: Any additional headers to be passed
        _data: Body of the request
        """
        if _headers:
            self.headers.update(_headers)
        conn = http.client.HTTPSConnection(self.hostname)

        path = self.url.path
        if path == "":
            path = "/"
        if self.url.query:
            path += "?" + self.url.query

        if _data is not None:
            conn.request(
                method="PUT",
                url=path,
                headers=self.headers,
                body=urllib.parse.urlencode(_data),
            )
        else:
            conn.request(method="PUT", url=path, headers=self.headers)
        response = conn.getresponse()

        return response
