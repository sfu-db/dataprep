import os
import pystache
from .pagedata import PageData
from typing import Any, Dict


class Template:
    htmlConfig: object
    template_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "layout"))

    def __init__(self, database_name: str) -> None:
        self.databaseName = database_name

    @staticmethod
    def get_root_path():
        return ""

    @staticmethod
    def get_root_path_to_home():
        return os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

    def write_data(
        self,
        page_data: PageData,
        output_file: str,
        page_script: str,
        pagination_configs: Dict[str, Any],
        root_path: str = "",
    ):
        """
        Render the html pages using template files for each section of the database
        """
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

        file = open(output_file, "w", encoding="utf-8")
        file.write(html_template)
        file.close()
        contents = open(output_file, "r", encoding="utf-8")
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
        open(output_file, "w", encoding="utf-8").write(html)
        return output_file
