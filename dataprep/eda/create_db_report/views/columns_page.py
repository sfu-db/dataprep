import json
from .pystache_columns import MustacheTableColumn
from .pagedata import PageData
from .template_pystache import Template

class HtmlColumnPage:
    def __init__(self, pystache_object:Template) -> None:
        self.pystache_object = pystache_object
    
    "compiles the data needed by the pystache template for columns page"
    
    def page_writer(self, tables, new_file):
        table_columns = set()
        for t in tables:
            for c in t.getColumns():
                table_columns.add(MustacheTableColumn(c, c.isIndexCol(), ""))

        json_columns = []
        for mc in table_columns:
            json_dict = {
                "tableName": mc.getColumn().getTable().getName(),
                "tableFileName": mc.getColumn().getTable().getName(),
                "tableType": mc.getColumn().getTable().getType(),
                "keyClass": mc.getKeyClass(),
                "keyTitle": mc.getKeyTitle(),
                "name": mc.getKeyIcon() + mc.getColumn().getName(),
                "type": mc.getColumn().getTypeName(),
                "length": "",
                "nullable": mc.getNullable(),
                "autoUpdated": mc.getAutoUpdated(),
                "defaultValue": mc.getDefaultValue(),
                "comments": "",
            }
            json_columns.append(json.loads(json.dumps(json_dict)))

        page_data = PageData("column.html", "column.js")
        page_data.addScope("tableData", json_columns)
        page_data.setDepth(0)

        pagination_configs = {
            "columnTable": {"paging": "true", "pageLength": 20, "lengthChange": "false"}
        }

        return self.pystache_object.write_data(page_data, new_file, "column.js", pagination_configs)
