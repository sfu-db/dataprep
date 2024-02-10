import re
import os
import platform
import pydot
import json
import shutil
from typing import Dict, Any
from .db_models.database import Database

GRAPHVIZ_PATH = os.environ.get("GRAPHVIZ_PATH", "C:/Program Files/Graphviz/bin")
if platform.system() == "Windows" and os.path.exists(GRAPHVIZ_PATH):
    os.add_dll_directory(GRAPHVIZ_PATH)
try:
    from eralchemy2 import render_er

    _WITH_GV = True
except ImportError:
    _WITH_GV = False


class DiagramFactory:
    def __init__(self, output_dir: str):
        self.cwd = os.getcwd()
        self.diagram_dir = output_dir + "/diagrams"
        self.dirs = {
            "table": self.diagram_dir + "/tables",
            "summary": self.diagram_dir + "/summary",
            "orphan": self.diagram_dir + "/orphans",
        }
        self.import_err = ImportError(
            "ERAlchemy is not installed."
            " Please run pip install ERAlchemy"
            "\nThis package also requires sub-dependency pygraphviz."
            " Please refer to https://pygraphviz.github.io/documentation/stable/install.html to install pygraphviz."
            f"\nFor Windows users, make sure that Graphviz is installed under {GRAPHVIZ_PATH}"
            "\nIf Graphviz was installed in a different directory, set path environment variable GRAPHVIZ_PATH to that directory."
        )
        self.create_dirs()

    def create_dirs(self):
        if not os.path.exists(self.diagram_dir):
            os.mkdir(self.diagram_dir)
        for path in self.dirs:
            if os.path.exists(self.dirs[path]) and os.path.isdir(self.dirs[path]):
                shutil.rmtree(self.dirs[path])
            os.mkdir(self.dirs[path])

    def generate_summary_diagram(self, database_object: Database, database_url: str):
        os.chdir(self.dirs["summary"])
        if _WITH_GV:
            render_er(database_url, "relationships.dot")
        else:
            raise self.import_err
        json_tables = self.generate_diagram_tables(database_object.get_tables_dict())
        file = str(
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), "layout/diagrams/summary/relationships.dot")
            )
        )
        json_relationships = self.generate_diagram_relationships(file)
        os.chdir(self.cwd)
        return json_tables, json_relationships

    def generate_table_diagrams(self, database_object: Database, database_url: str):
        # Generate diagram for each table
        database_tables = database_object.get_tables_dict()
        table_names = set(database_tables.keys())
        orphan_table_names = list()
        result_tables = {}
        for table in table_names:
            related_table_names = {table}
            related_table_names.update(database_tables[table].get_referenced_by_tables())
            table_foreign_keys = database_tables[table].get_foreign_keys_dict()
            for foreign_key in table_foreign_keys:
                related_table_names.add(
                    table_foreign_keys[foreign_key].get_parent_table().get_name()
                )
            related_table_names = list(related_table_names)
            if len(related_table_names) == 1:
                orphan_table_names.append(table)
            os.chdir(self.dirs["table"])
            if _WITH_GV:
                render_er(
                    database_url, f"{table}.dot", include_tables=" ".join(related_table_names)
                )
            else:
                raise self.import_err
            os.chdir(self.cwd)
            first_degree_tables = {
                key: value for key, value in database_tables.items() if key in related_table_names
            }
            json_tables = self.generate_diagram_tables(first_degree_tables)
            file = str(
                os.path.realpath(
                    os.path.join(os.path.dirname(__file__), f"layout/diagrams/tables/{table}.dot")
                )
            )
            json_relationships = self.generate_diagram_relationships(file)
            result_tables[table] = {
                "json_tables": json_tables,
                "json_relationships": json_relationships,
            }

        # Generate diagram for orphan tables
        os.chdir(self.dirs["orphan"])
        if _WITH_GV:
            render_er(database_url, "orphans.dot", include_tables=" ".join(orphan_table_names))
        else:
            raise self.import_err
        os.chdir(self.cwd)
        orphan_tables = {
            key: value for key, value in database_tables.items() if key in orphan_table_names
        }
        json_tables = self.generate_diagram_tables(orphan_tables)
        file = str(
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), "layout/diagrams/orphans/orphans.dot")
            )
        )
        json_relationships = self.generate_diagram_relationships(file)
        orphan_result_tables = {
            "json_tables": json_tables,
            "json_relationships": json_relationships,
        }
        return orphan_result_tables, result_tables

    def generate_diagram_tables(self, tables: Dict[str, Any]):
        table_names = set(tables.keys())
        json_tables = []
        for table in table_names:
            current_table_description = {"key": table}
            table_items = []
            related_table_names = {table}
            related_table_names.update(tables[table].get_referenced_by_tables())
            table_columns = tables[table].get_columns()
            for column in table_columns:
                current_column = {
                    "name": column.get_name(),
                    "type": column.type_name,
                    "default_value": column.default_value,
                    "nullable": column.not_null,
                }
                if column.is_primary():
                    current_column["iskey"] = True
                    current_column["figure"] = "Decision"
                    current_column["color"] = "red"
                elif column.is_foreign_key():
                    current_column["iskey"] = True
                    current_column["figure"] = "Decision"
                    current_column["color"] = "purple"
                    current_column["ref"] = ",".join(
                        [f"{x.name} in {x.table.name}" for x in column.parents]
                    )
                else:
                    current_column["iskey"] = False
                    current_column["figure"] = "Circle"
                    current_column["color"] = "green"
                table_items.append(current_column)
            current_table_description["items"] = self.sort_by_priority(table_items)
            json_tables.append(current_table_description)
        return json_tables

    def generate_diagram_relationships(self, dot_file: str):
        json_relationships = []
        graph = pydot.graph_from_dot_file(dot_file)
        rex = re.compile(r"<<FONT>(.*?)</FONT>>", re.S | re.M)
        edge_list = graph[0].get_edge_list()
        for e in edge_list:
            current_edge = {}
            node_name = str(e).split()
            labels = json.dumps(e.get_attributes())
            edge_attr = json.loads(labels)
            # { from: "Products", to: "Suppliers", text: "0..N", toText: "1" }
            current_edge["from"] = node_name[0].replace('"', "")
            current_edge["to"] = node_name[2].replace('"', "")
            match = rex.match(edge_attr["taillabel"])
            if match:
                current_edge["text"] = match.groups()[0].strip()
            match = rex.match(edge_attr["headlabel"])
            if match:
                current_edge["toText"] = match.groups()[0].strip()
            json_relationships.append(current_edge)
        return json_relationships

    def sort_by_priority(self, values):
        priority = ["red", "purple", "green"]
        priority_dict = dict(zip(priority, range(len(priority))))
        for value in values:
            value["priority"] = priority_dict[value["color"]]
        return sorted(values, key=lambda x: x["priority"])
