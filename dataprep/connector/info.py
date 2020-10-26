from .implicit_database import ImplicitDatabase
from .config_manager import config_directory, ensure_config
from pathlib import Path
from jinja2 import Template

INFO_TEMPLATE = Template(
    """{% for tb in tbs.keys() %}
Table {{dbname}}.{{tb}}

Parameters
----------
{% if tbs[tb].required_params %}{{", ".join(tbs[tb].required_params)}} required {% endif %}
{% if tbs[tb].optional_params %}{{", ".join(tbs[tb].optional_params)}} optional {% endif %}

Examples
--------
>>> dc = Connector({{", ".join([dbname] + ["concurrency=2"])}})
>>> dc.query({{", ".join(["\\\"{}\\\"".format(tb)] + tbs[tb].joined_query_fields)}})
>>> dc.show_schema("{{tb}}")
{% endfor %}
"""
)

def info(website_name, update = False) -> None:
    """Show the basic information and provide guidance for users
    to issue queries."""
     
    path = initialize_path(website_name, update)
    impdb = ImplicitDatabase(path)
     
    # get info
    tbs: Dict[str, Any] = {}
    for cur_table in impdb.tables:
        table_config_content: ConfigDef = impdb.tables[cur_table].config

        auth_params = []
        params_required = []
        params_optional = []
        example_query_fields = []
        count = False
        required_param_count = 1
        auth = table_config_content.request.authorization 
         
        if auth == None:
            pass
        elif auth.type == 'OAuth2':
            auth_params.insert(0, 'client_secret')
            auth_params.insert(0, 'client_id')
        else:
            auth_params.append('access_token')   
         
        for k, val in table_config_content.request.params.items():
            if isinstance(val, bool) and val:
                params_required.append(k)
                example_query_fields.append(f"""{k}="word{required_param_count}\"""")
                required_param_count += 1
            elif isinstance(val, bool):
                params_optional.append(k)
                 
        if table_config_content.request.pagination != None:
            count = True
             
        tbs[cur_table] = {}
        tbs[cur_table]["auth_params"] = auth_params
        tbs[cur_table]["required_params"] = params_required
        tbs[cur_table]["optional_params"] = params_optional
        tbs[cur_table]["joined_query_fields"] = example_query_fields
        tbs[cur_table]["count"] = count

    # show table info
    print(INFO_TEMPLATE.render(ntables=len(impdb.tables.keys()), dbname=impdb.name, tbs=tbs))

def initialize_path(config_path, update):
    if (
        config_path.startswith(".")
        or config_path.startswith("/")
        or config_path.startswith("~")
    ):
        path = Path(config_path).resolve()
    else:
        # From Github!
        ensure_config(config_path, update)
        path = config_directory() / config_path
    return path