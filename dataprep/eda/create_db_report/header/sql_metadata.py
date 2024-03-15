import re
import os
import pandas as pd
from collections import OrderedDict
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine.base import Engine
from sqlalchemy import text


def plot_mysql_db(sql_engine: Engine):
    """
    Query MySQL database and returns dictionaries for database tables, views, and metadata for database.
    sql_engine
        SQL Alchemy Engine object returned from create_engine() with an url passed
    """
    db_name = sql_engine.url.database
    # Table level SQL, schema name, table name, row count
    version_sql = pd.read_sql("""SELECT version();""", sql_engine)
    table_sql = pd.read_sql(
        """SELECT table_schema AS schemaname, table_name AS table_name, table_rows AS row_count FROM INFORMATION_SCHEMA.tables
    WHERE table_schema not in ('mysql','information_schema','performance_schema','sys', 'Z_README_TO_RECOVER') AND TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = '%s' ORDER BY 1,2;"""
        % (db_name),
        sql_engine,
    )
    view_sql = pd.read_sql(
        """SELECT table_schema AS schemaname, table_name AS view_name, view_definition AS definition FROM INFORMATION_SCHEMA.VIEWS WHERE TABLE_SCHEMA != 'sys' AND TABLE_SCHEMA = '%s' ORDER BY 1,2;"""
        % (db_name),
        sql_engine,
    )
    pk_fk = pd.read_sql(
        """SELECT k.CONSTRAINT_NAME AS constraint_name, t.CONSTRAINT_TYPE AS constraint_type, k.TABLE_NAME AS table_name, k.COLUMN_NAME AS col_name,
    CASE WHEN concat_ws('.', k.REFERENCED_TABLE_SCHEMA, k.REFERENCED_TABLE_NAME) = '' THEN NULL
    ELSE k.REFERENCED_TABLE_NAME END AS ref_table, k.REFERENCED_COLUMN_NAME AS ref_col, r.UPDATE_RULE AS update_rule, r.DELETE_RULE AS delete_rule FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
    JOIN information_schema.table_constraints t on k.CONSTRAINT_CATALOG = t. CONSTRAINT_CATALOG AND k.constraint_schema = t.constraint_schema AND k.constraint_name = t.constraint_name AND k.TABLE_NAME = t.TABLE_NAME
    LEFT JOIN information_schema.referential_constraints r on k.CONSTRAINT_CATALOG = r.CONSTRAINT_CATALOG AND k.constraint_schema = r.constraint_schema AND k.constraint_name = r.constraint_name
    WHERE k.CONSTRAINT_SCHEMA not in ('mysql', 'performance_schema', 'sys') AND k.CONSTRAINT_SCHEMA = '%s' ORDER BY 3;"""
        % (db_name),
        sql_engine,
    )

    # List of schemas and tables
    schema_list = list(table_sql["schemaname"])
    table_list = list(table_sql["table_name"])
    view_list = list(view_sql["view_name"])

    overview_dict = {}
    # Show the stats for schemas, tables and PK/FK
    overview_dict["num_of_schemas"] = len(set(schema_list))
    overview_dict["schema_names"] = list(set(schema_list))
    overview_dict["num_of_tables"] = len(table_list)
    overview_dict["table_names"] = table_list
    overview_dict["num_of_views"] = len(view_list)
    overview_dict["view_names"] = view_list
    overview_dict["connection_url"] = str(sql_engine.url)
    overview_dict["tables_no_index"] = list(
        table_sql[
            ~table_sql["table_name"].isin(
                set(pk_fk[pk_fk["constraint_type"] == "PRIMARY KEY"]["table_name"])
            )
        ]["table_name"]
    )
    overview_dict["num_of_pk"] = len(
        set(pk_fk[pk_fk["constraint_type"] == "PRIMARY KEY"]["table_name"])
    )
    overview_dict["num_of_fk"] = len(pk_fk[pk_fk["constraint_type"] == "FOREIGN KEY"])
    overview_dict["num_of_uk"] = len(pk_fk[pk_fk["constraint_type"] == "UNIQUE"])
    overview_dict["table_schema"] = dict(zip(table_sql["table_name"], table_sql["schemaname"]))
    overview_dict["view_schema"] = dict(zip(view_sql["view_name"], view_sql["schemaname"]))
    overview_dict["product_version"] = version_sql["version()"].values[0]
    # Stats for column level stats
    all_cols = pd.read_sql(
        """SELECT table_name AS table_name, COLUMN_name AS col_name, COLUMN_TYPE AS type,
    CASE WHEN IS_NULLABLE  = 'YES' THEN 'False'
    ELSE 'True' END AS attnotnull , COLUMN_DEFAULT AS `default`, column_comment AS description,
    CASE WHEN EXTRA like '%s' THEN 'True'
    ELSE 'False' END AS `auto_increment` FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema not in
    ('mysql','information_schema','performance_schema','sys', 'Z_README_TO_RECOVER')
    AND table_schema = '%s' ORDER BY 1, 2;"""
        % ("%%auto_increment%%", db_name),
        sql_engine,
    )

    # Split into intermediate result dictionary form - table
    table_dict = {}
    for i in table_list:
        indices = {}
        index = pd.read_sql("SHOW INDEX FROM " + str(i) + " FROM " + db_name + ";", sql_engine)
        for idx, row in index.iterrows():
            if row.loc["Key_name"] in indices:
                indices[row.loc["Key_name"]]["Column_name"] += "," + row.loc["Column_name"]
                # indices[row.loc['Key_name']]['Index_type']+="/"+row.loc['Index_type']
            else:
                new_index = {}
                new_index["Column_name"] = row.loc["Column_name"]
                new_index["Index_type"] = row.loc["Index_type"]
                indices[row.loc["Key_name"]] = new_index
        temp = OrderedDict()
        temp_cols = (
            all_cols[all_cols["table_name"] == i]
            .drop(columns=["table_name"])
            .to_dict(orient="records")
        )
        for j in temp_cols:
            temp[j["col_name"]] = {}
            element = j.pop("col_name")
            temp[element] = j
            temp[element]["children"] = list(
                pk_fk[(pk_fk["ref_table"] == i) & (pk_fk["ref_col"] == element)]["table_name"]
            )
            temp[element]["parents"] = list(
                pk_fk[
                    (pk_fk["table_name"] == i)
                    & (pk_fk["col_name"] == element)
                    & (pk_fk["constraint_type"] == "FOREIGN KEY")
                ]["ref_table"]
            )
        temp["num_of_parents"] = int(
            len(pk_fk[(pk_fk["table_name"] == i) & (pk_fk["constraint_type"] == "FOREIGN KEY")])
        )
        temp["num_of_children"] = int(len(pk_fk[(pk_fk["ref_table"] == i)]))
        temp["num_of_rows"] = int(table_sql[table_sql["table_name"] == i]["row_count"].values[0])
        temp["num_of_cols"] = int(len(all_cols[all_cols["table_name"] == i]))
        temp["constraints"] = {}
        temp_pk_fk = (
            pk_fk[pk_fk["table_name"] == i]
            .drop(columns=["table_name"])
            .groupby("constraint_name")
            .agg(
                {
                    "constraint_name": "first",
                    "constraint_type": "first",
                    "col_name": ", ".join,
                    "ref_table": "first",
                    "ref_col": "first",
                    "update_rule": "first",
                    "delete_rule": "first",
                }
            )
            .to_dict(orient="records")
        )
        for j in temp_pk_fk:
            temp["constraints"][j["constraint_name"]] = {}
            element = j.pop("constraint_name")
            temp["constraints"][element] = j
        temp["indices"] = indices
        table_dict[i] = temp
    # Split into intermediate result dictionary form - view
    view_dict = {}
    for i in view_list:
        temp = {}
        temp_cols = (
            all_cols[all_cols["table_name"] == i]
            .drop(columns=["table_name"])
            .to_dict(orient="records")
        )
        for j in temp_cols:
            temp[j["col_name"]] = {}
            element = j.pop("col_name")
            temp[element] = j
        temp["num_of_cols"] = len(all_cols[all_cols["table_name"] == i])
        # temp['num_of_rows'] = int(view_sql[view_sql['view_name'] == i]['row_count'].values[0])
        temp["definition"] = view_sql[view_sql["view_name"] == i]["definition"].values[0]
        view_dict[i] = temp
    return overview_dict, table_dict, view_dict


def plot_postgres_db(postgres_engine: Engine):
    """
    Query PostgresDB database and returns dictionaries for database tables, views, and metadata for database.
    sql_engine
        SQL Alchemy Engine object returned from create_engine() with an url passed
    """
    # Table level SQL, schema name, table name, row count
    table_sql = pd.read_sql(
        """SELECT s.schemaname, tablename AS table_name, hasindexes, n_live_tup AS row_count
      FROM pg_stat_user_tables s
      JOIN pg_tables t ON t.tablename = s.relname AND t.schemaname = s.schemaname ORDER BY 1,2;""",
        postgres_engine,
    )
    version_sql = pd.read_sql("""SELECT version();""", postgres_engine)
    # View level SQL
    view_sql = pd.read_sql(
        """SELECT schemaname, v.viewname AS view_name, definition FROM pg_class c
JOIN pg_views v on v.viewname = c.relname AND c.relnamespace = v.schemaname::regnamespace::oid
WHERE v.schemaname != 'pg_catalog' AND v.schemaname != 'information_schema' AND relkind = 'v' ORDER BY 1,2""",
        postgres_engine,
    )
    # PK/FK constraints
    pk_fk = pd.read_sql(
        """SELECT conname as constraint_name,
        CASE
            WHEN contype = 'p' THEN 'primary key'
            WHEN contype = 'f' THEN 'foreign key'
            WHEN contype = 'u' THEN 'unique key'
        END AS constraint_type
          , conrelid::regclass AS "table_name"
          , CASE WHEN pg_get_constraintdef(c.oid) LIKE 'FOREIGN KEY %%' THEN substring(pg_get_constraintdef(c.oid), 14, position(')' in pg_get_constraintdef(c.oid))-14) WHEN pg_get_constraintdef(c.oid) LIKE 'PRIMARY KEY %%' THEN substring(pg_get_constraintdef(c.oid), 14, position(')' in pg_get_constraintdef(c.oid))-14) END AS "col_name"
          , CASE WHEN pg_get_constraintdef(c.oid) LIKE 'FOREIGN KEY %%' THEN substring(pg_get_constraintdef(c.oid), position(' REFERENCES ' in pg_get_constraintdef(c.oid))+12, position('(' in substring(pg_get_constraintdef(c.oid), 14))-position(' REFERENCES ' in pg_get_constraintdef(c.oid))+1) END AS "ref_table"
          , CASE WHEN pg_get_constraintdef(c.oid) LIKE 'FOREIGN KEY %%' THEN substring(pg_get_constraintdef(c.oid), position('(' in substring(pg_get_constraintdef(c.oid), 14))+14, position(')' in substring(pg_get_constraintdef(c.oid), position('(' in substring(pg_get_constraintdef(c.oid), 14))+14))-1) END AS "ref_col"
          , pg_get_constraintdef(c.oid) as constraint_def,
          CASE
            WHEN confupdtype = 'a' THEN 'NO ACTION'
            WHEN confupdtype = 'r' THEN 'RESTRICT'
            WHEN confupdtype = 'c' THEN 'CASCADE'
            WHEN confupdtype = 'n' THEN 'SET NULL'
            WHEN confupdtype = 'd' THEN 'SET DEFAULT'
        END AS update_rule,
        CASE
            WHEN confdeltype = 'a' THEN 'NO ACTION'
            WHEN confdeltype = 'r' THEN 'RESTRICT'
            WHEN confdeltype = 'c' THEN 'CASCADE'
            WHEN confdeltype = 'n' THEN 'SET NULL'
            WHEN confdeltype = 'd' THEN 'SET DEFAULT'
        END AS delete_rule
    FROM   pg_constraint c
    JOIN   pg_namespace n ON n.oid = c.connamespace
    WHERE  contype IN ('f', 'p', 'u')
    ORDER  BY conrelid::regclass::text, contype DESC;""",
        postgres_engine,
    )
    # List of schemas and tables
    schema_list = list(table_sql["schemaname"])
    schema_str = ",".join(set(schema_list))
    table_list = list(table_sql["table_name"])
    view_list = list(view_sql["view_name"])
    overview_dict = {}
    # Show the stats for schemas, tables and PK/FK
    overview_dict["num_of_schemas"] = len(set(schema_list))
    overview_dict["schema_names"] = list(set(schema_list))
    overview_dict["table_schema"] = dict(zip(table_sql["table_name"], table_sql["schemaname"]))
    overview_dict["num_of_tables"] = len(table_list)
    overview_dict["table_names"] = table_list
    overview_dict["num_of_views"] = len(view_list)
    overview_dict["view_names"] = view_list
    overview_dict["connection_url"] = postgres_engine.url
    overview_dict["tables_no_index"] = list(
        table_sql[table_sql["hasindexes"] == False]["table_name"]
    )
    overview_dict["num_of_pk"] = len(pk_fk[pk_fk["constraint_type"] == "primary key"])
    overview_dict["num_of_fk"] = len(pk_fk[pk_fk["constraint_type"] == "foreign key"])
    overview_dict["num_of_uk"] = len(pk_fk[pk_fk["constraint_type"] == "unique key"])
    overview_dict["view_schema"] = dict(zip(view_sql["view_name"], view_sql["schemaname"]))
    overview_dict["product_version"] = re.findall("[\d\.]+\d+", version_sql.values[0][0])[0]

    # Stats for column level stats
    all_cols = pd.read_sql(
        """select attrelid::regclass AS table_name, f.attname AS col_name,
        pg_catalog.format_type(f.atttypid,f.atttypmod) AS type, attnotnull,
        CASE
            WHEN f.atthasdef = 't' THEN pg_get_expr(d.adbin, d.adrelid)
        END AS default, description,
        CASE
            WHEN pg_get_expr(d.adbin, d.adrelid) LIKE 'nextval%%' THEN True
            ELSE False
        END AS auto_increment, null_frac * c.reltuples AS num_null, null_frac AS perc_of_null,
        CASE WHEN s.n_distinct < 0
            THEN -s.n_distinct * c.reltuples
            ELSE s.n_distinct
       END AS num_of_distinct,
       CASE WHEN s.n_distinct < 0
            THEN round((-s.n_distinct * 100)::numeric, 2)
            ELSE round((s.n_distinct / c.reltuples * 100)::numeric, 2)
       END AS perc_of_distinct
        FROM pg_attribute f
        JOIN pg_class c ON c.oid = f.attrelid
        --JOIN pg_type t ON t.oid = f.atttypid
        LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = f.attnum
        LEFT JOIN pg_description de on de.objoid = c.oid
        LEFT JOIN pg_stats s on s.schemaname::regnamespace::oid = c.relnamespace AND s.tablename = c.relname AND s.attname = f.attname
        WHERE (c.relkind = 'v'::char or c.relkind = 'r'::char)
        AND f.attnum > 0
        AND attisdropped is False
        AND n.nspname in ('{}');""".format(
            schema_str
        ),
        postgres_engine,
    )
    # Split into intermediate result dictionary form - table
    table_dict = {}
    for i in table_list:
        indices = {}
        index = pd.read_sql(
            "SELECT * FROM pg_indexes WHERE tablename= " + "'" + str(i) + "'" + ";",
            postgres_engine,
        )
        for idx, row in index.iterrows():
            current_index = row.loc["indexname"]
            indices[current_index] = {}
            index_type, col_name = (row.loc["indexdef"].split("USING ", 1)[1]).split(" ", 1)
            col_name = col_name.replace("(", "")
            col_name = col_name.replace(")", "")
            indices[current_index]["Column_name"] = col_name
            indices[current_index]["Index_type"] = index_type
        temp = {}
        temp_cols = (
            all_cols[all_cols["table_name"] == i]
            .drop(columns=["table_name"])
            .to_dict(orient="records")
        )
        for j in temp_cols:
            temp[j["col_name"]] = {}
            element = j.pop("col_name")
            temp[element] = j
            temp[element]["children"] = list(
                pk_fk[(pk_fk["ref_table"] == i) & (pk_fk["ref_col"] == element)]["table_name"]
            )
            temp[element]["parents"] = list(
                pk_fk[
                    (pk_fk["table_name"] == i)
                    & (pk_fk["col_name"] == element)
                    & (pk_fk["constraint_type"] == "foreign key")
                ]["ref_table"]
            )
        temp["num_of_parents"] = int(
            len(pk_fk[(pk_fk["table_name"] == i) & (pk_fk["constraint_type"] == "foreign key")])
        )
        temp["num_of_children"] = int(len(pk_fk[(pk_fk["ref_table"] == i)]))
        temp["num_of_rows"] = int(table_sql[table_sql["table_name"] == i]["row_count"].values[0])
        temp["num_of_cols"] = int(len(all_cols[all_cols["table_name"] == i]))
        temp["constraints"] = {}
        temp_pk_fk = (
            pk_fk[pk_fk["table_name"] == i].drop(columns=["table_name"]).to_dict(orient="records")
        )
        for j in temp_pk_fk:
            temp["constraints"][j["constraint_name"]] = {}
            element = j.pop("constraint_name")
            temp["constraints"][element] = j
        temp["indices"] = indices
        table_dict[i] = temp
    # Split into intermediate result dictionary form - view
    view_dict = {}
    for i in view_list:
        temp = {}
        temp_cols = (
            all_cols[all_cols["table_name"] == i]
            .drop(columns=["table_name"])
            .to_dict(orient="records")
        )
        for j in temp_cols:
            temp[j["col_name"]] = {}
            element = j.pop("col_name")
            temp[element] = j
        temp["num_of_cols"] = len(all_cols[all_cols["table_name"] == i])
        temp["definition"] = view_sql[view_sql["view_name"] == i]["definition"].values[0]
        view_dict[i] = temp

    return overview_dict, table_dict, view_dict


def plot_sqlite_db(sqliteConnection: Engine, analyze: bool = False):
    """
    Query SQLite database and returns dictionaries for database tables, views, and metadata for database.
    sql_engine
        SQL Alchemy Engine object returned from create_engine() with an url passed
    analyze
        Whether to execute ANALYZE to write database statistics to the database
    """
    db_name = os.path.splitext(os.path.basename(sqliteConnection.url.database))[0]
    schema_name = []
    schema_name.append(db_name)
    if analyze:
        sqliteConnection.execute("ANALYZE")
    try:
        with sqliteConnection.begin() as conn:
            query = text("""select sqlite_version();""")
            version_sql = pd.read_sql(query, conn)
            index = pd.read_sql(text("SELECT * FROM sqlite_master WHERE type = 'index'"), conn)
            # Get all table names
            table_sql = pd.read_sql(
                text(
                    """select type, tbl_name as table_name, sql from sqlite_master where type = 'table' AND tbl_name not like 'sqlite_%';"""
                ),
                conn,
            )
            # Get row count for each table
            table_row_sql = pd.read_sql(
                text(
                    """select DISTINCT tbl_name AS table_name, CASE WHEN stat is null then 0 else cast(stat as INT) END row_count
            from sqlite_master m
            LEFT JOIN sqlite_stat1 stat on   m.tbl_name = stat.tbl
            where m.type='table'
            and m.tbl_name not like 'sqlite_%'
            order by 1"""
                ),
                conn,
            )
            # Get all the columns and their stats
            all_cols = pd.read_sql(
                text(
                    """SELECT tbl_name as table_name, p.name as col_name, p.type as type,
            CASE WHEN `notnull` = 0 THEN 'False'
            ELSE 'True' END AS attnotnull, dflt_value as `default`, pk, sql
            FROM
            sqlite_master AS m
            JOIN
            pragma_table_info(m.name) AS p
            WHERE tbl_name not like 'sqlite_%'
            ORDER BY
            m.name,
            p.cid"""
                ),
                conn,
            )
            # Get all view names
            view_sql = pd.read_sql(
                text(
                    """select type, tbl_name as view_name, sql AS definition from sqlite_master where type = 'view' AND tbl_name not like 'sqlite_%';"""
                ),
                conn,
            )
            # Get all fk stats
            fk_sql = pd.read_sql(
                text(
                    """SELECT 'foreign key' AS constraint_type, tbl_name as table_name, `from` AS col_name,
                `table` AS ref_table, `to` AS ref_col, sql AS constraint_def, on_update AS "update_rule", on_delete AS "delete_rule"
            FROM
            sqlite_master AS m
            JOIN
            pragma_foreign_key_list(m.name) AS p WHERE m.type = 'table'"""
                ),
                conn,
            )
            # Get all pk stats
            pk_sql = pd.read_sql(
                text(
                    """SELECT DISTINCT 'primary key' AS constraint_type, tbl_name as table_name
            ,group_concat(p.name) OVER (
            PARTITION BY tbl_name) AS col_name, sql AS constraint_def
            FROM
            sqlite_master AS m
            JOIN
            pragma_table_info(m.name) AS p
            WHERE tbl_name not like 'sqlite_%' AND pk != 0
            ORDER BY
            m.name,
            p.cid"""
                ),
                conn,
            )
            # Get all uk stats
            uk_sql = pd.read_sql(
                text(
                    """SELECT DISTINCT 'unique key' AS constraint_type, tbl_name as table_name, p.name as col_name, sql AS constraint_def
            FROM
            sqlite_master AS m
            JOIN
            pragma_index_list(m.name) AS p WHERE m.type = 'table' AND `unique` = 1 AND origin not in ('pk', 'fk')"""
                ),
                conn,
            )
            # Align the columns for pk and fk and concat them
            pk_sql["ref_table"], pk_sql["ref_col"], uk_sql["ref_table"], uk_sql["ref_col"] = (
                None,
                None,
                None,
                None,
            )
    except OperationalError:
        raise Exception(
            "Cannot read statistics from the database. Please run 'analyze' in the database to collect the statistics first, or set analyze=True to allow us do this (note that 'analyze' usually collects the statistics and stores the result in the database)"
        )

    pk_sql = pk_sql[
        ["constraint_type", "table_name", "col_name", "ref_table", "ref_col", "constraint_def"]
    ]
    uk_sql = uk_sql[
        ["constraint_type", "table_name", "col_name", "ref_table", "ref_col", "constraint_def"]
    ]
    pk_fk = pd.concat([pk_sql, fk_sql, uk_sql]).reset_index(drop=True)
    table_list = list(table_sql["table_name"])
    view_list = list(view_sql["view_name"])
    overview_dict = {}
    overview_dict["table_schema"] = dict([(x, "sakila") for x in table_list])
    overview_dict["num_of_schemas"] = 1
    overview_dict["schema_names"] = schema_name
    overview_dict["num_of_tables"] = int(len(table_list))
    overview_dict["table_names"] = table_list
    overview_dict["num_of_views"] = int(len(view_list))
    overview_dict["view_names"] = view_list
    overview_dict["connection_url"] = sqliteConnection.url
    overview_dict["view_schema"] = dict([(x, "sakila") for x in view_list])
    overview_dict["tables_no_index"] = list(
        table_sql[~table_sql["table_name"].isin(set(pk_sql["table_name"]))]["table_name"]
    )
    overview_dict["num_of_pk"] = int(len(pk_sql))
    overview_dict["num_of_fk"] = int(len(fk_sql))
    overview_dict["num_of_uk"] = int(len(uk_sql))
    overview_dict["product_version"] = version_sql.values[0][0]
    # Split into intermediate result dictionary form - table
    table_dict = {}
    for i in table_list:
        indices = {}
        table_indexes = index.loc[index["tbl_name"] == str(i)]
        for idx, row in table_indexes.iterrows():
            current_index = row.loc["name"]
            indices[current_index] = {}
            index_type = row.loc["type"]
            if row.loc["sql"]:
                col_name = (row.loc["sql"].split("(", 1)[1]).split(" ", 1)[0].strip()[:-1]
            else:
                col_name = None
            new_index = {}
            indices[current_index]["Column_name"] = col_name
            indices[current_index]["Index_type"] = index_type
        temp = OrderedDict()
        temp_cols = (
            all_cols[all_cols["table_name"] == i]
            .drop(columns=["table_name", "pk", "sql"])
            .to_dict(orient="records")
        )
        for j in temp_cols:
            temp[j["col_name"]] = {}
            element = j.pop("col_name")
            temp[element] = j
            temp[element]["children"] = list(
                pk_fk[(pk_fk["ref_table"] == i) & (pk_fk["ref_col"] == element)]["table_name"]
            )
            temp[element]["parents"] = list(
                pk_fk[
                    (pk_fk["table_name"] == i)
                    & (pk_fk["col_name"] == element)
                    & (pk_fk["constraint_type"] == "foreign key")
                ]["ref_table"]
            )
        temp["num_of_parents"] = len(
            pk_fk[(pk_fk["table_name"] == i) & (pk_fk["constraint_type"] == "foreign key")]
        )
        temp["num_of_children"] = len(pk_fk[(pk_fk["ref_table"] == i)])
        temp["num_of_rows"] = int(
            table_row_sql[table_row_sql["table_name"] == i]["row_count"].values[0]
        )
        temp["num_of_cols"] = len(all_cols[all_cols["table_name"] == i])
        temp["constraints"] = {}
        temp_pk_fk = (
            pk_fk[pk_fk["table_name"] == i].drop(columns=["table_name"]).to_dict(orient="records")
        )
        fk_counter, uk_counter = 1, 1
        for j in temp_pk_fk:
            if j["constraint_type"] == "primary key":
                element = i + "_pkey"
                temp["constraints"][element] = {}
            elif j["constraint_type"] == "foreign key":
                element = i + "_fkey" + str(fk_counter)
                temp["constraints"][element] = {}
                fk_counter += 1
            elif j["constraint_type"] == "unique key":
                element = i + "_ukey" + str(uk_counter)
                temp["constraints"][element] = {}
                uk_counter += 1
            temp["constraints"][element] = j
        temp["indices"] = indices
        table_dict[i] = temp
    # Split into intermediate result dictionary form - view
    view_dict = {}
    for i in view_list:
        temp = {}
        temp_cols = (
            all_cols[all_cols["table_name"] == i]
            .drop(columns=["table_name", "pk", "sql"])
            .to_dict(orient="records")
        )
        for j in temp_cols:
            temp[j["col_name"]] = {}
            element = j.pop("col_name")
            temp[element] = j
        temp["num_of_cols"] = len(temp_cols)
        temp["definition"] = view_sql[view_sql["view_name"] == i]["definition"].values[0]
        view_dict[i] = temp
    return overview_dict, table_dict, view_dict
