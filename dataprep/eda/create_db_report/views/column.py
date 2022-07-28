import json
from typing import List
from ..db_models.table import Table
from ..page_models.page_data import PageData
from ..page_models.page_template import PageTemplate
from ..template_models.table_column import TemplateTableColumn


class ColumnPage:
    def __init__(self, template_object: PageTemplate) -> None:
        self.template_object = template_object

    def page_writer(self, tables: List[Table], new_file: str):
        """
        Compile the data needed by the template for columns page
        """
        table_columns = set()
        for table in tables:
            for column in table.get_columns():
                table_columns.add(TemplateTableColumn(column, column.index, ""))

        json_columns = []
        for mc in table_columns:
            json_dict = {
                "table_name": mc.table_column.table.name,
                "table_file_name": mc.table_column.table.name,
                "table_type": mc.table_column.table.get_type(),
                "key_class": mc.get_key_class(),
                "key_title": mc.get_key_title(),
                "name": mc.get_key_icon() + mc.table_column.name,
                "type": mc.table_column.type_name,
                "length": "",
                "nullable": mc.get_nullable(),
                "auto_updated": mc.get_auto_updated(),
                "default_value": mc.get_default_value(),
                "comments": "",
            }
            json_columns.append(json.loads(json.dumps(json_dict)))

        page_data = PageData("column.html", "column.js")
        page_data.add_scope("table_data", json_columns)
        page_data.set_depth(0)

        pagination_configs = {
            "columnTable": {"paging": "true", "pageLength": 20, "lengthChange": "false"}
        }

        return self.template_object.write_data(page_data, new_file, "column.js", pagination_configs)
