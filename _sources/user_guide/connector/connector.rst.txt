.. _`userguide/connector`:

Connector
=========

.. toctree::
    :maxdepth: 1

    connector_authorization
    connector_concurrency
    connector_pagination

To see detailed examples of how to use Connector, check out our `Jupyter Notebooks repository <https://github.com/sfu-db/dataprep/tree/develop/examples>`_.

Overview
--------

Connector is a component in DataPrep that aims to simplify data collection from Web APIs by providing a standard set of operations. See Connector's `full documentation <https://sfu-db.github.io/dataprep/>`_.

Connector wraps-up complex API calls into a set of easy-to-use Python functions. By using Connector, you can skip the complex API configuration process and rapidly query different Web APIs in few steps, enabling you to execute the analysis workflow you are familiar with in a direct way.

Connector offers essential features to facilitate the process of collecting data, for example:

- **Concurrency:** Collect data from websites, in parallel, in a fast way!
- **Pagination:** Retrieve more rows of a particular query without getting into unnecessary detail about pagination schemes!
- **Authorization:** Access more Web APIs quickly! even the ones that implement authorization!

In the following sections, this guide will walk you through Connector's main features in a hands-on way, using as a case study the `Yelp API  <https://www.yelp.com/developers>`_.

Fetching data in a nutshell
---------------------------

With Connector, you can collect data from one of the top recommendations' sites online: **Yelp**. Let's see how:

First, import the ``connect`` function from the DataPrep package into your Python source code: ::

    from dataprep.connector import connect

Then, use the ``connect`` function with the ``"yelp"`` string and your Yelp access token, both specified as parameters. This action allows you to create a Connector to the Yelp Web API::

    yelp_connector = connect("yelp", _auth={"access_token":"<Your Yelp access token>"})

Click `here <https://www.yelp.com/developers/documentation/v3/authentication>`_ to request a Yelp access token. You can use it by replacing the ``<Your Yelp access token>`` string.

In this step, by using the ``info()`` function, you can get more information about Yelp's endpoints currently supported by Connector::

    yelp_connector.info()


The ``info`` function gives information and guidelines on using Connector over a particular Web API. The output of the ``info`` function is composed of four sections, as follows:

| 	a. Table - The table(s)/endpoint being accessed.
| 	b. Parameters - Identifies which parameters can be used to call the method.
| 	c. Examples - Shows how you can call the methods in the Connector class.
| 	d. Schema - Names and data types of attributes in the response.

.. image:: ../../_static/images/connector/info.png
	:align: center
   	:width: 720
   	:height: 320


As you can see in the image above, in this example, there is only one endpoint available for Yelp: ``businesses``. However, if you want to connect to a different Yelp endpoint, you can build a new configuration file. See: `Configuration Files
<https://github.com/sfu-db/DataConnectorConfigs>`_.


Now you can explore the "businesses" endpoint schema according to its configuration file definition. For this purpose, you can use the ``show_schema()`` function::

    yelp_connector.show_schema("business")

.. image:: ../../_static/images/connector/connector_yelp_show_schema.png
	:align: center
   	:width: 221
   	:height: 302

Finally, use the ``query`` function on this Connector object with the parameter ``"businesses"`` which indicates you want to query the ``"businesses"`` endpoint by providing the term ``"sushi"`` and location ``"Vancouver"``::

    df = await yelp_connector.query("businesses", term="sushi", location="Vancouver")

To see more details about the technical specification of the Yelp business search endpoint, click `here <https://www.yelp.ca/developers/documentation/v3/business_search>`_.

Note the highlighted ``await`` keyword at the beginning of the ``query`` instruction. Connector uses the `Asyncio <https://docs.python.org/3/library/asyncio.html>`_ feature from The Python Standard Library to achieve parallelism (see "Concurrency" section below). Hence, you must import the Asyncio library as well.
The final block of code is as follows::

    import asyncio
    from dataprep.connector import connect
    yelp_connector = connect("yelp", _auth={"access_token":"<Your Yelp access token>"})
    df = await yelp_connector.query("businesses", term="sushi", location="Vancouver")

You can store these results in a `Pandas Dataframe <https://pandas.pydata.org/pandas-docs/stable/reference/frame.html>`_ (named ``df`` in the example). Let’s review the results obtained:

.. image:: ../../_static/images/connector/connector_yelp_query.png
	:align: center
   	:width: 850
   	:height: 582


Authorization
-------------
Connector supports the most used authorization methods in Web APIs:

- API Key
- OAuth 2.0 "Client Credentials" and "Authorization Code" grants.

Yelp API uses the ``API Key`` authorization method for the ``businesses`` endpoint. As you noted in the last section, you merely have to pass the credentials (in the Yelp case, the API key) as a ``connect`` function parameter.
Specifically, you have to define the value of the ``_auth`` parameter by using the ``access_token`` variable::

    yelp_connector = connect("yelp", _auth={"access_token":"<Your Yelp access token>"})

Pagination
----------
Connector implements pagination more smartly! This feature allows you to get more results from the same query on a Web API endpoint. In that sense, with Connector's auto-pagination feature, you don't need to code how to handle result pages because Connector does that for you!

Continuing with the Yelp example, this Web API, by default, only returns 20 records per request (see results image above). However, with Connector, you can specify the desired number of records you want to obtain. To achieve this, you must use the ``_count`` argument.

Thus, let's add the ``_count`` argument to the ``query`` function and give it ``1000`` as a value::

    import asyncio
    from dataprep.connector import connect
    yelp_connector = connect("yelp", _auth={"access_token":"<Your Yelp access token>"})
    df = await yelp_connector.query("businesses", term="sushi", location="Vancouver", _count=1000)


By this, you get specifically up to 1000 records (in this case: 682), instead of 20.

.. image:: ../../_static/images/connector/connector_yelp_query_2.png
	:align: center
   	:width: 889
   	:height: 372


Nevertheless, the side effect of incrementing the ``_count`` argument is that Dataprep needs to issue more requests to the Web API, and this can cause that the whole query could execute slower than expected.
To alleviate this problem, Connector implements **concurrency**! Let's see the details below.

Concurrency
-----------
Another great feature of Connector is concurrency. Through this feature, you can retrieve data, in parallel, in less time!

Next, you can see a code example where the ``_concurrency`` parameter is used on the ``connect`` function. This parameter allows you to define the number of requests per second to be sent out to the Web API. In that sense, when the ``_concurrency`` parameter is used jointly with the pagination feature (``_count`` parameter), Connector accelerates the data request process, and, therefore, the total request time is improved::

    import asyncio
    from dataprep.connector import connect
    yelp_connector = connect("yelp", _auth={"access_token":"<Your Yelp access token>"}, _concurrency=10)
    df = await yelp_connector.query("businesses", term="sushi", location="Vancouver", _count=1000)


Configuration Files
-------------------
A configuration file defines the information needed to fetch data from a website, e.g., the request URL, the API authorization type, the required parameters from the user (API key, search keyword, etc.), and the returned data's schema.

All the information is reusable.
To write a configuration file for your own needs or to modify an existing one, please visit our `Configuration Files
<https://github.com/sfu-db/DataConnectorConfigs>`_ repository.
