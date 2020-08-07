from requests import Request, Response, Session
import re
from urllib import parse
from jinja2 import Template

CONFIG_TEMPLATE = Template("""
{
   "version": {{version}},
   "request": {
      "url": "{{url}}",
      "method": "{{method}}",
      "params": { {% for p in params %}
          "{{p}}": true, {% endfor %} 
      }                              
   },
   "response": {
      "ctype": "{{ctype}}",
      "tablePath": "{{table_path}}",
      "schema": { {% for sc in schemacols %}
          "{{sc}}":{
             "target": "$.{{sc}}",
             "type": "string",
           }, {% endfor %} 
      },                 
      "orient": "{{orient}}"                
   }    
}
"""
)


def create_config(example: str) -> 'ConfigGenerator':
    cg = ConfigGenerator()
    cg.create_configuration(example)
    return cg


class ConfigGenerator:
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

    def create_configuration(
            self,
            request_example: str
    ) -> 'ConfigGenerator':
        self._request_tokenizer(request_example)
        self._execute_request()
        self._create_config_file_representation()
        return self

    def _request_tokenizer(
            self,
            request_example: str
    ) -> None:
        self._request_example = request_example
        request_full_url = re.search("(?P<url>https?://[^\s]+)", self._request_example).group("url")
        parse_full_url = parse.urlparse(request_full_url)
        self._parameters = parse.parse_qs(parse_full_url.query)
        lst_parse_full_url = list(parse_full_url)
        lst_parse_full_url[4] = str()
        self._url = parse.urlunparse(lst_parse_full_url)

    def _execute_request(
            self
    ) -> None:
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
        resp: Response = self._session.send(
            prep_request
        )
        self._content_type = resp.headers['content-type']
        self._response = resp.json()

    def _create_config_file_representation(
            self
    ) -> None:
        lst_parameters = list(self._parameters.keys())
        self._schema_cols = list(dict(list(self._response.values())[0]).keys())
        self._config = CONFIG_TEMPLATE.render(
            nparams=len(lst_parameters), params=lst_parameters, nschemacols=len(self._schema_cols),
            schemacols=self._schema_cols, version=self._version, url=self._url, method=self._method,
            ctype=self._content_type, table_path=self._table_path, orient=self._orient
        )

    def save(
            self,
            filename
    ) -> None:
        text_file = open(filename, "w")
        n = text_file.write(self._config)
        text_file.close()

    def to_string(
            self
    ) -> str:
        return self._config
