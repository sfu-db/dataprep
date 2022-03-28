from pydoc import html
from tokenize import String
import pystache
import sys
import os
import views.pagedata as pagedata
from report import Report


class template_parser:
    databaseName: String
    htmlConfig: object
    template_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "layout"))

    def __init__(self, databaseName, htmlConfig=None) -> None:
        self.databaseName = databaseName
        self.htmlConfig = htmlConfig

    def getRootPath():
        return ""

    def getRootPathtoHome():
        return os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

    def write_data(
        self, pageData: pagedata, output_file, pageScript, pagination_configs, rootPath=""
    ):
        containerScope = {}
        page_template = open(
            os.path.realpath(os.path.join(self.template_directory, pageData.getTemplateName()))
        ).read()
        pageScope = {
            "toFileName": "true",
            "databaseName": self.databaseName,
            "paginationEnabled": "true",
            "displayNumRows": "true",
            "dataTableConfig": {},
        }

        for key, value in pagination_configs.items():
            pageScope["dataTableConfig"][key] = value
        pageScope_final = {**pageScope, **pageData.getScope()}
        html_template = pystache.render(page_template, pageScope_final)

        # output here from the writer object
        file = open(output_file, "w")
        file.write(html_template)
        file.close()
        contents = open(output_file, "r")
        fill = contents.read()

        tmpl = open(
            os.path.realpath(os.path.join(self.template_directory, "container.html"))
        ).read()
        examples = {
            "databaseName": self.databaseName,
            "content": fill,
            "pageScript": pageScript,
            "rootPath": rootPath or template_parser.getRootPath(),
            "rootPathtoHome": template_parser.getRootPathtoHome(),
        }
        html = pystache.render(tmpl, examples)
        open(output_file, "w").write(html)
        return Report(html, output_file)
