import os
import json
from typing import Any, Dict
from sqlalchemy.engine.base import Engine
from .db_models.db_meta import DbMeta
from .db_models.database import Database
from .db_models.table import Table
from .db_models.view import View
from .db_models.table_column import TableColumn
from .db_models.table_index import TableIndex
from .db_models.constraint import ForeignKeyConstraint
from .page_models.page_template import PageTemplate
from .views.main import MainPage
from .views.column import ColumnPage
from .views.constraint import ConstraintPage
from .views.table import TablePage
from .views.relationship import RelationshipPage
from .views.orphan import OrphanPage
from .diagram_factory import DiagramFactory
from .header.sql_metadata import plot_mysql_db, plot_postgres_db, plot_sqlite_db


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
        json_overview_dict["connection_url"],
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
        table = Table(current_database, json_overview_dict["table_schema"][table_name], table_name)
        table.num_of_rows = json_table_dict[table_name]["num_of_rows"]
        table.num_of_cols = json_table_dict[table_name]["num_of_cols"]
        current_database.add_table(table_name, table)

    for view_name in json_view_dict.keys():
        view = View(
            current_database,
            json_overview_dict["view_schema"][view_name],
            view_name,
            json_view_dict[view_name]["definition"],
        )
        view.num_of_cols = json_view_dict[view_name]["num_of_cols"]
        current_database.add_view(view_name, view)
        current_database.add_table(view_name, view)

    existing_tables = current_database.tables
    for table in json_table_dict.keys():
        current_columns = json_table_dict[table]
        current_table = existing_tables[table]
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
                        create_index.set_primary()
                    col_string = current_columns["indices"][current_index]["Column_name"]
                    if col_string:
                        all_columns = col_string.upper().split(",")
                        for col in all_columns:
                            column = current_table.get_column(col.strip())
                            column.set_index()
                            create_index.add_column(col_string, column)
                    current_table.set_index(current_index, create_index)
            else:
                column = current_columns[c]
                create_table_column = TableColumn(
                    current_table,
                    c,
                    column["type"],
                    str(column["attnotnull"]).upper() == "TRUE",
                    column["default"],
                    (
                        str(column["auto_increment"]).upper() == "TRUE"
                        if "auto_increment" in column
                        else False
                    ),
                    column["description"] if "description" in column else "",
                )
                current_table.add_column(c.upper(), create_table_column)

    existing_views = current_database.views
    for view in json_view_dict.keys():
        collect_columns = {}
        current_columns = json_view_dict[view]
        current_view = existing_views[view]

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
                (
                    str(column["auto_increment"]).upper() == "TRUE"
                    if "auto_increment" in column
                    else False
                ),
                column["description"] if "description" in column else "",
            )
            collect_columns[c.upper()] = create_view_column
        current_view.set_columns(collect_columns)


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
    existing_tables = current_database.tables
    for table in json_table_dict.keys():
        columns = json_table_dict[table]
        current_table = existing_tables[table]
        constraints = columns["constraints"]
        for current_constraint in constraints.keys():
            if constraints[current_constraint]["constraint_type"].upper() == "PRIMARY KEY":
                column_id = str(constraints[current_constraint]["col_name"]).upper().split(",")
                for i in column_id:
                    current_table.add_primary_key(current_table.get_column(i.strip()))
            elif constraints[current_constraint]["constraint_type"].upper() == "FOREIGN KEY":
                column_id = str(constraints[current_constraint]["col_name"]).upper().strip()
                current_column = current_table.get_column(column_id)
                parent_table = existing_tables[constraints[current_constraint]["ref_table"]]
                parent_column_name = constraints[current_constraint]["ref_col"]
                parent_column = parent_table.get_column(parent_column_name.upper())
                delete_constraint = constraints[current_constraint]["delete_rule"]
                new_fk = ForeignKeyConstraint(
                    current_table, current_constraint, delete_constraint, "0"
                )
                new_fk.add_child_column(current_column)
                new_fk.add_parent_column(parent_column)
                current_column.add_parent(parent_column, new_fk)
                parent_column.add_child(current_column, new_fk)
                parent_table.add_referenced_by_table(current_table)
                current_table.add_foreign_key(new_fk)


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
    overview_dict["connection_url"] = str(overview_dict["connection_url"])
    json_overview_dict = json.loads(json.dumps(overview_dict))
    json_table_dict = json.loads(json.dumps(table_dict))
    json_view_dict = json.loads(json.dumps(view_dict))
    database_name = os.path.splitext(os.path.basename(sql_engine.url.database))[0]

    # setup database db_models
    metadata, current_database = parse_database(sql_engine.name, database_name, json_overview_dict)

    # create tables and columns, add to current database
    parse_tables(json_table_dict, json_overview_dict, json_view_dict, current_database)

    # define all constraints (primary keys and foreign keys) for the database tables
    parse_constraints(current_database, json_table_dict)

    template_compiler = PageTemplate(database_name)

    file = str(os.path.realpath(os.path.join(os.path.dirname(__file__), "layout")))
    diagram_generator = DiagramFactory(file)

    file = str(os.path.realpath(os.path.join(os.path.dirname(__file__), "layout", "columns.html")))
    column_page = ColumnPage(template_compiler)
    column_page.page_writer(current_database.get_tables(), file)

    file = str(
        os.path.realpath(os.path.join(os.path.dirname(__file__), "layout", "constraints.html"))
    )
    constraint_page = ConstraintPage(template_compiler)
    constraints = ForeignKeyConstraint.get_all_foreign_key_constraints(
        current_database.get_tables()
    )
    constraint_page.page_writer(constraints, current_database.get_tables(), file)

    orphan_result_tables, result_tables = diagram_generator.generate_table_diagrams(
        current_database, str(sql_engine.url)
    )

    orphan_page = OrphanPage(template_compiler)
    file = str(os.path.realpath(os.path.join(os.path.dirname(__file__), "layout", "orphans.html")))
    orphan_page.page_writer(
        orphan_result_tables["json_tables"], orphan_result_tables["json_relationships"], file
    )

    table_files = ["table.html", "table.js"]
    for table in current_database.get_tables():
        table_file_name = table.name + ".html"
        table_files.append(table_file_name)
        file = str(
            os.path.realpath(
                os.path.join(os.path.dirname(__file__), "layout/tables", table_file_name)
            )
        )
        table_page = TablePage(template_compiler)
        table_page.page_writer(
            table,
            result_tables[table.name]["json_tables"],
            result_tables[table.name]["json_relationships"],
            file,
        )

    delete_table_files = [
        file
        for file in os.listdir(
            os.path.realpath(os.path.join(os.path.dirname(__file__), "layout/tables"))
        )
        if file not in table_files
    ]
    for file_name in delete_table_files:
        os.remove(
            os.path.realpath(os.path.join(os.path.dirname(__file__), "layout/tables", file_name))
        )

    file = str(os.path.realpath(os.path.join(os.path.dirname(__file__), "layout", "index.html")))
    main_page = MainPage(template_compiler, "", metadata)

    report_output_file = main_page.page_writer(
        current_database, current_database.get_tables(), file
    )

    json_tables, json_relationships = diagram_generator.generate_summary_diagram(
        current_database, str(sql_engine.url)
    )
    relationship_page = RelationshipPage(template_compiler)
    file = str(
        os.path.realpath(os.path.join(os.path.dirname(__file__), "layout", "relationships.html"))
    )
    relationship_page.page_writer(json_tables, json_relationships, file)

    return database_name, report_output_file
