import os
import json
from .model.db_metadata import DbMeta
from .model.init_database import Database
from .model.init_table import Table
from .model.init_view import View
from .model.init_tablecolumn import TableColumn
from .model.init_tableindex import TableIndex
from .model.add_foreignkey import ForeignKeyConstraint
from .views.main_page import HtmlMainIndexPage
from .views.columns_page import HtmlColumnPage
from .views.template_pystache import Template
from .views.constraints_page import HtmlConstraintsPage
from .views.table_page import HtmlTablePage
from .header.sql_metadata import plot_mysql_db, plot_postgres_db, plot_sqlite_db
from typing import Any, Dict
from sqlalchemy.engine.base import Engine


def parse_database(engine_name: str, database_name: str, json_overview_dict: Dict[str, Any]):
    """
    Initialize database metadata and return database object

     Parameters
    ----------
    engine_name
        Name of database engine object
    database_name
        Name of database/schema
    json_overview_dict
        A dictionary of database metadata
    """
    metadata = DbMeta(
        engine_name,
        json_overview_dict["num_of_views"],
        json_overview_dict["num_of_schemas"],
        json_overview_dict["num_of_fk"],
        json_overview_dict["num_of_uk"],
        json_overview_dict["num_of_pk"],
        json_overview_dict["num_of_tables"],
        json_overview_dict["product_version"],
    )
    current_database = Database(database_name, json_overview_dict["schema_names"], metadata)
    return metadata, current_database


def parse_tables(
    json_table_dict: Dict[str, Any],
    json_overview_dict: Dict[str, Any],
    json_view_dict: Dict[str, Any],
    current_database: Database,
):
    """
    Initialize database tables and views

     Parameters
    ----------
    json_table_dict
        A dictionary of tables data
    json_overview_dict
        A dictionary of database metadata
    json_view_dict
        A dictionary of views data
    current_database
        Database object
    """
    for table_name in json_table_dict.keys():
        table_obj = Table(
            current_database, json_overview_dict["table_schema"][table_name], table_name
        )
        table_obj.num_row(json_table_dict[table_name]["num_of_rows"])
        table_obj.num_columns(json_table_dict[table_name]["num_of_cols"])
        current_database.addTable(table_name, table_obj)

    for view_name in json_view_dict.keys():
        view_obj = View(
            current_database,
            json_overview_dict["view_schema"][view_name],
            view_name,
            json_view_dict[view_name]["definition"],
        )
        view_obj.num_columns(json_view_dict[view_name]["num_of_cols"])
        current_database.addView(view_name, view_obj)
        current_database.addTable(view_name, view_obj)

    existing_tables = current_database.getTablesMap()
    for t in json_table_dict.keys():
        current_columns = json_table_dict[t]
        current_table = existing_tables[t]
        for c in current_columns.keys():
            if (
                c == "constraints"
                or c == "num_of_parents"
                or c == "num_of_children"
                or c == "num_of_rows"
                or c == "num_of_cols"
            ):
                continue
            elif c == "indices":
                for current_index in current_columns["indices"]:
                    create_index = TableIndex(
                        current_index, current_columns["indices"][current_index]["Index_type"]
                    )
                    if current_index == "PRIMARY" or "pkey" in current_index:
                        create_index.setPrimary()
                    col_string = current_columns["indices"][current_index]["Column_name"]
                    if col_string:
                        all_columns = col_string.upper().split(",")
                        for col in all_columns:
                            column = current_table.getColumn(col.strip())
                            column.setIndex()
                            create_index.addColumn(col_string, column)
                    current_table.setIndex(current_index, create_index)
            else:
                column = current_columns[c]
                create_table_column = TableColumn(
                    current_table,
                    c,
                    column["type"],
                    str(column["attnotnull"]).upper() == "TRUE",
                    column["default"],
                    str(column["auto_increment"]).upper() == "TRUE"
                    if "auto_increment" in column
                    else False,
                    column["description"] if "description" in column else "",
                )
                current_table.addColumn(c.upper(), create_table_column)

    existing_views = current_database.getViewsMap()
    for t in json_view_dict.keys():
        collect_columns = {}
        current_columns = json_view_dict[t]
        current_view = existing_views[t]

        for c in current_columns.keys():
            if c == "num_of_cols" or c == "definition":
                continue
            column = current_columns[c]
            create_view_column = TableColumn(
                current_view,
                c,
                column["type"],
                column["attnotnull"] == "True",
                column["default"],
                str(column["auto_increment"]).upper() == "TRUE"
                if "auto_increment" in column
                else False,
                column["description"] if "description" in column else "",
            )
            collect_columns[c.upper()] = create_view_column
        current_view.setColumns(collect_columns)


def parse_constraints(current_database: Database, json_table_dict: Dict[str, Any]):
    """
    Initialize primary and foreign key constraints

     Parameters
    ----------

    current_database
        Database object
    json_table_dict
        A dictionary of tables data
    """
    existing_tables = current_database.getTablesMap()
    for t in json_table_dict.keys():
        columns = json_table_dict[t]
        current_table = existing_tables[t]
        constraints = columns["constraints"]
        for current_constraint in constraints.keys():
            if constraints[current_constraint]["constraint_type"].upper() == "PRIMARY KEY":
                column_id = str(constraints[current_constraint]["col_name"]).upper().split(",")
                for i in column_id:
                    current_table.setPrimaryColumn(current_table.getColumn(i.strip()))
            elif constraints[current_constraint]["constraint_type"].upper() == "FOREIGN KEY":
                column_id = str(constraints[current_constraint]["col_name"]).upper().strip()
                current_column = current_table.getColumn(column_id)
                parent_table = constraints[current_constraint]["ref_table"]
                parent_col = constraints[current_constraint]["ref_col"]
                parent_column = existing_tables[parent_table].getColumn(parent_col.upper())
                delete_constraint = constraints[current_constraint]["delete_rule"]
                new_fk = ForeignKeyConstraint(
                    current_table, current_constraint, delete_constraint, 0
                )
                new_fk.addChildColumn(current_column)
                new_fk.addParentColumn(parent_column)
                current_column.addParent(parent_column, new_fk)
                parent_column.addChild(current_column, new_fk)
                current_table.addForeignKey(new_fk)


plot_db = {
    "mysql": plot_mysql_db,
    "postgresql": plot_postgres_db,
    "sqlite": plot_sqlite_db,
}


def generate_db_report(sql_engine: Engine, analyze: bool = False):
    """
    Write database analysis to template files

     Parameters
    ----------

    sql_engine
        SQL Alchemy Engine object returned from create_engine() with an url passed
    analyze
        Whether to execute ANALYZE to write database statistics to the database
    """
    overview_dict, table_dict, view_dict = (
        plot_db[sql_engine.name](sql_engine, analyze)
        if analyze
        else plot_db[sql_engine.name](sql_engine)
    )
    json_overview_dict = json.loads(json.dumps(overview_dict))
    json_table_dict = json.loads(json.dumps(table_dict))
    json_view_dict = json.loads(json.dumps(view_dict))
    database_name = os.path.splitext(os.path.basename(sql_engine.url.database))[0]

    # setup database model
    metadata, current_database = parse_database(sql_engine.name, database_name, json_overview_dict)

    # create tables and columns, add to current database
    parse_tables(json_table_dict, json_overview_dict, json_view_dict, current_database)

    # define all constraints (primary keys and foreign keys) for the database tables
    parse_constraints(current_database, json_table_dict)

    template_compiler = Template(database_name)

    f = str(os.path.realpath(os.path.join(os.path.dirname(__file__), "layout", "columns.html")))
    html_columns_page = HtmlColumnPage(template_compiler)
    html_columns_page.page_writer(current_database.getTables(), f)

    f = str(os.path.realpath(os.path.join(os.path.dirname(__file__), "layout", "constraints.html")))
    html_constraints_page = HtmlConstraintsPage(template_compiler)
    constraints = ForeignKeyConstraint.getAllForeignKeyConstraints(current_database.getTables())
    html_constraints_page.page_writer(constraints, current_database.getTables(), f)

    table_files = ["table.html", "table.js"]
    for table in current_database.getTables():
        table_file_name = table.getName() + ".html"
        table_files.append(table_file_name)
        f = str(
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), "layout/tables", table_file_name)
            )
        )
        html_columns_page = HtmlTablePage(template_compiler)
        html_columns_page.page_writer(table, f)

    delete_table_files = [
        f
        for f in os.listdir(
            os.path.realpath(os.path.join(os.path.dirname(__file__), "layout/tables"))
        )
        if f not in table_files
    ]
    for file_name in delete_table_files:
        os.remove(
            os.path.realpath(os.path.join(os.path.dirname(__file__), "layout/tables", file_name))
        )

    f = str(os.path.realpath(os.path.join(os.path.dirname(__file__), "layout", "index.html")))
    html_main_index_page = HtmlMainIndexPage(template_compiler, "", metadata)

    report_output_file = html_main_index_page.page_writer(
        current_database, current_database.getTables(), f
    )

    return report_output_file
