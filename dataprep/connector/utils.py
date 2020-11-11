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

    def __init__(self, _url: str,) -> None:
        self.url = urllib.parse.urlparse(_url)
        self.hostname = self.url.hostname
        self.path = self.url.path
        self.headers = {"user-agent": "node.js"}

    def get(self, _headers: Optional[Dict[str, Any]] = "") -> Dict[str, Any]:
        """
        GET request to the specified end-point.

        Parameters
        ----------
        _headers: Any additional headers to be passed
        """
        self.headers.update(_headers)
        conn = http.client.HTTPSConnection(self.hostname)
        conn.request(method="GET", url=self.path, headers=self.headers)
        response = conn.getresponse()

        return response

    def post(
        self, _headers: Optional[Dict[str, Any]] = "", _data: Optional[Dict[str, Any]] = ""
    ) -> Dict[str, Any]:
        """
        POST request to the specified end-point.

        Parameters
        ----------
        _headers: Any additional headers to be passed
        _data: Body of the request
        """
        self.headers.update(_headers)
        conn = http.client.HTTPSConnection(self.hostname)
        conn.request(
            method="POST", url=self.path, headers=self.headers, body=urllib.parse.urlencode(_data)
        )
        response = conn.getresponse()

        return response

    def put(
        self, _headers: Optional[Dict[str, Any]] = "", _data: Optional[Dict[str, Any]] = ""
    ) -> Dict[str, Any]:
        """
        PUT request to the specified end-point.

        Parameters
        ----------
        _headers: Any additional headers to be passed
        _data: Body of the request
        """
        self.headers.update(_headers)
        conn = http.client.HTTPSConnection(self.hostname)
        conn.request(
            method="PUT", url=self.path, headers=self.headers, body=urllib.parse.urlencode(_data)
        )
        response = conn.getresponse()

        return response
