import os
import pystache
from tokenize import String
from .pagedata import PageData
from ..report import Report


class Template:
    htmlConfig: object
    template_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "layout"))

    def __init__(self, database_name, html_config=None) -> None:
        self.databaseName = database_name
        self.htmlConfig = html_config

    @staticmethod
    def get_root_path():
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def get_root_path_to_home():
        return os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

    def write_data(
        self, page_data: PageData, output_file, page_script, pagination_configs, root_path=""
    ):
        page_template = open(
            os.path.realpath(os.path.join(self.template_directory, page_data.getTemplateName()))
        ).read()

        page_scope = {
            "toFileName": "true",
            "databaseName": self.databaseName,
            "paginationEnabled": "true",
            "displayNumRows": "true",
            "dataTableConfig": {},
        }
        for key, value in pagination_configs.items():
            page_scope["dataTableConfig"][key] = value
        page_scope.update(page_data.getScope())
        html_template = pystache.render(page_template, page_scope)

        file = open(output_file, "w")
        file.write(html_template)
        file.close()
        contents = open(output_file, "r")
        fill = contents.read()

        tmpl = open(
            os.path.realpath(os.path.join(self.template_directory, "container.html"))
        ).read()
        container_scope = {
            "databaseName": self.databaseName,
            "content": fill,
            "pageScript": page_script,
            "rootPath": root_path or Template.get_root_path(),
            "rootPathtoHome": Template.get_root_path_to_home(),
        }
        html = pystache.render(tmpl, container_scope)
        open(output_file, "w").write(html)
        return Report(html, output_file)
