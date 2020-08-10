.. _dataconnector:

=========
Connector
=========

.. toctree::
    :maxdepth: 1

    DC_DBLP_tut
    DC_Spotify_tut
    DC_Yelp_tut

Overview
==========
Connector is a component in the dataprep library that aims to simplify the data access by providing a standard API set. 
The goal is to help the users skip the complex API configuration.
We illustrate how to use connector library with Yelp.


Initializing a connector class for a website
=============================================
The first step is to initialize a Connector class with the configuration file location and access token specified (`How to get access token?
<https://www.yelp.com/developers/documentation/v3/authentication>`_).
Available configuration files can be manually downloaded here: `Configuration Files
<https://github.com/sfu-db/DataConnectorConfigs>`_ or automatically downloaded at usage.
To initialize a connector::

    from dataprep.connector import Connector
    dc = Connector("./DataConnectorConfigs/yelp", auth_params={"access_token":access_token})


Getting the guidline of the connector with `Connector.info`
=================================================================
| Connector's info method gives information and guideline of using the connector. In the example below, the response shows three things. 
| 	a. There is one table in Yelp, i.e. Yelp.businesses.
| 	b. To query this table, the term and location parameters are required and the longitute and latitude parameters are optional (see Connector.query() section).
| 	c. The examples of calling the methods in the Connector class.

::

    dc.info

.. image:: ../../_static/images/connector/info.png
	:align: center
   	:width: 496
   	:height: 215



Understand web data with `Connector.show_schema()`
============================================================
show_schema(table name) returns the schema of the webdata to be returned in a dataframe.
There are two columns in the response.
The first column is the column name and the second is the datatype.

::

    dc.show_schema('businesses')


.. image:: ../../_static/images/connector/show_schema.png
   :align: center
   :width: 208
   :height: 458 


Getting web data with `Connector.query()`
=================================================
the `query()` method downloads the website data.
The parameters should meet the requriement in `Connector.info`
Usually the raw data is returned in JSON or xml format.
connector re-format the data in pandas dataframe for the convenience of downstream operations.

::

    df = dc.query('businesses', term="korean", location="seattle")
    df

.. image:: ../../_static/images/connector/query.png
   :align: center
   :width: 870
   :height: 491


Advanced: writing your own connector configuration file
==============================================================
A configuration file defines the infomation neccessary to fetch data from a website, e.g. the request url; the API authorization type; the parameters needed from the uses(API key, search keyword, etc.); the returned data's schema. 
All the information are reusable.
To write a configuration file for the your own needs or modify an existing one, please refer to `Configuration Files
<https://github.com/sfu-db/DataConnectorConfigs>`_.
