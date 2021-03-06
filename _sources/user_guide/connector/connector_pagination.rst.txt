.. _connector_pagination:

==================================================
Pagination in Connector
==================================================

.. toctree::
   :maxdepth: 2

Overview
========

The **pagination feature** in Connector allows you to retrieve more results from Web APIs that support pagination.

Connector supports two mainstream pagination schemes:

- **Offset-based**
- **Cursor-based**

Additionally, Connector's **auto-pagination feature** enables you to implement pagination without getting into unnecessary detail about a specific pagination scheme.

Let's review how Connector supports pagination in the sections below.

Auto-pagination
================

Connector automatically handles pagination for you and follows the Web API's concurrency policy. You can directly specify the number of records to be collected without detailing a specific pagination scheme. Let's review an example:

`DBLP <https://dblp.org/>`_ is a Computer Science bibliography website that exposes several Web APIs, including the `publication search <https://dblp.org/faq/13501473.html>`_ Web API. DBLP restricts to **30** the maximum number of search results to return for each request. Therefore, in order to retrieve more information from this Web API using Connector's auto-pagination feature, you can execute the ``query`` function with the parameter ``_count`` as follows::

    import asyncio
    from dataprep.connector import connect
    dblp_connector = connect("dblp")
    df = await dblp_connector.query("publication", q="SIGMOD", _count=500)

.. image:: ../../_static/images/connector/connector_auto_pagination_on.png
	:align: center
   	:width: 1017
   	:height: 222

Employing the ``_count`` parameter, you define the maximum number of results to retrieve, in this case: **500**. Thus, your query is not limited to obtain a maximum of 30 results per invocation. The remaining parameters of the query function define the name of the endpoint (``publication``) and the search criteria (``q="SIGMOD"``).

In contrast, when auto-pagination is not used, only **30** records are retrieved (DBLP's search results limit)::

    import asyncio
    from dataprep.connector import connect
    dblp_connector = connect("dblp")
    df = await dblp_connector.query("publication", q="SIGMOD")

.. image:: ../../_static/images/connector/connector_auto_pagination_off.png
	:align: center
   	:width: 995
   	:height: 215


Offset-based pagination
=======================

**Offset-based pagination** scheme has two variants: **Offset & Limit** and **Page & Perpage**, as follows:


Offset & Limit
******************

**Offset** and **limit** parameters allow you to specify the number of rows to skip before selecting the actual rows. For example when parameters ``offset = 0`` and ``limit = 20`` are used, the first 20 items are fetched. Then, by sending ``offset = 20`` and ``limit = 20``, the next 20 items are fetched, and so on.

Continuing with the DBLP example, below you can find how to use offset-based pagination in Connector::

    import asyncio
    from dataprep.connector import connect
    dblp_connector = connect("dblp")
    df = await dblp_connector.query("publication", q="SIGMOD", f="0", h="10")

In this case, the DBLP endpoint specification defines that the name of the offset parameter is ``f`` and the name of the limit parameter is ``h``. For that reason, parameters with names ``f`` and ``h`` are used in the ``query`` function. These parameter names are also defined in the `DBLP's configuration file <https://github.com/sfu-db/DataConnectorConfigs/blob/develop/dblp/publication.json>`_ for the publication endpoint.::

    "pagination": {
        "type": "offset",
        "offsetKey": "f",
        "limitKey": "h",
        "maxCount": 1000
    },

After the execution of this query, 10 results are retrieved:

.. image:: ../../_static/images/connector/connector_pagination_offset_offset_limit.png
	:align: center
   	:width: 1049
   	:height: 309


Page & Perpage
******************

Instead of specifying offset and limit values, a **page number ("Page")** and the **amount of each page ("Perpage")** are used as parameters within the request.
In the following example, you can see how this pagination method works using as an example the `MapQuest Web API <https://developer.mapquest.com/>`_ - "place" endpoint::

    import asyncio
    from dataprep.connector import connect
    mapquest_connector = connect("mapquest", _auth={"access_token":"<Your MapQuest access token>"})
    df = await mapquest_connector.query("place", q="Vancouver, BC", sort="relevance", page="1", pageSize="10")

In this case, the specification of the MapQuest - "place" endpoint defines that the name of the "Page" parameter is ``page`` and the name of the "Perpage" parameter is ``pageSize``. For that reason, parameters with names ``page`` and ``pageSize`` are used into the ``query`` function. These parameter names are also defined into the `MapQuest's configuration file <https://github.com/sfu-db/DataConnectorConfigs/blob/develop/mapquest/place.json>`_ for the place endpoint.::

    "pagination": {
        "type": "page",
        "pageKey": "page",
        "limitKey": "pageSize",
        "maxCount": 50
    }

After the execution of this query, 10 results are retrieved:

.. image:: ../../_static/images/connector/connector_pagination_offset_page_perpage.png
	:align: center
   	:width: 1067
   	:height: 457


Cursor-based pagination
=======================

**Cursor-based pagination** uses a **“cursor”** within each response to fetch the next block of data. Connector supports this type of pagination through the **auto-pagination** feature (see details above). That is, if a Web API implements cursor-based pagination, you can use the auto-pagination feature to retrieve data without requiring to write code for handling cursor-based pagination explicitly.

Connector supports the two main variations of cursor-based pagination, as follows: **Header Cursor** and **Item Cursor**. Under the "Header Cursor" option, a token for the next “page” will be included in each response's metadata. On the other hand, for the "Item Cursor" option, there will be multiple items within each response, where each item represents a valid data record. At the end of the item set of each response, a cursor of the next page will be included, which can be passed as a parameter for the subsequent request.

However, remember that you just need to use Connector's auto-pagination feature to fetch data from a Web API that implements cursor-based pagination.
