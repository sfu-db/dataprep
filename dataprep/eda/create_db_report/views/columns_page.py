import json
import views.pystache_columns as PC
import views.pagedata as PD


class HtmlColumnPage:
    def __init__(self, pystache_object) -> None:
        self.pystache_object = pystache_object

    def pageWriter(self, tables, new_file):
        table_columns = set()
        for t in tables:
            for c in t.getColumns():
                table_columns.add(PC.MustacheTableColumn(c, c.isIndexCol(), ""))

        json_columns = []
        for mc in table_columns:
            json_dict = {}
            json_dict["tableName"] = mc.getColumn().getTable().getName()
            json_dict["tableFileName"] = mc.getColumn().getTable().getName()
            json_dict["tableType"] = mc.getColumn().getTable().getType()
            json_dict["keyClass"] = mc.getKeyClass()
            json_dict["keyTitle"] = mc.getKeyTitle()
            json_dict["name"] = mc.getKeyIcon() + mc.getColumn().getName()
            json_dict["type"] = mc.getColumn().getTypeName()
            json_dict["length"] = ""
            json_dict["nullable"] = mc.getNullable()
            json_dict["autoUpdated"] = mc.getAutoUpdated()
            json_dict["defaultValue"] = mc.getDefaultValue()
            json_dict["comments"] = ""
            json_columns.append(json.loads(json.dumps(json_dict)))

        pagedata = PD.pageData("column.html", "column.js")
        pagedata.addScope("tableData", json_columns)
        pagedata.setDepth(0)

        pagination_configs = {
            "columnTable": {"paging": "true", "pageLength": 20, "lengthChange": "false"}
        }

        return self.pystache_object.write_data(pagedata, new_file, "column.js", pagination_configs)
