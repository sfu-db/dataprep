
==================================================
Tutorial - Data Connector for DBLP 
==================================================

.. toctree::
   :maxdepth: 2

Overview
========

data_connector is a component in the dataprep library that aims to simplify the data access by providing a standard API set. 
The goal is to help the users skip the complex API configuration. In this tutorial, we demonstrate how to use data_connector library with DBLP.

Preprocessing
================
If you haven't installed dataprep, run command pip install dataprep or execute the following cell.

::

    !pip install dataprep
    

Download and store the configuration files in dataprep 
================================================================
The configuration files are used to construct the parameters and initial setup for the API. The available configuration files can be manually downloaded here: `Configuration Files
<https://github.com/sfu-db/DataConnectorConfigs>`_ or automatically downloaded at usage. 

Store the configuration file in the dataprep folder. 

.. image:: _static/images/tutorial/Config_destination.png
	:align: center
   	:width: 586
   	:height: 132


    
Initialize data_connector
=============================
To initialize, run the following code. 

* Unlike Yelp and Spotify, tokens and client information are not needed.

::

    from dataprep.data_connector import Connector
    dc = Connector("./DataConnectorConfigs/DBLP")
    
Functionalities
===================
Data connector has several functions you can perform to gain insight on the data downloaded from DBLP.

Connector.info
------------------
| The info method gives information and guidelines on using the connector. There are 3 sections in the response and they are table, parameters and examples.
| 	a. Table - The table(s) being accessed.
| 	b. Parameters - Identifies which parameters can be used to call the method. For DBLP, there is no required **parameter**. 
| 	c. Examples - Shows how you can call the methods in the Connector class.

::

    dc.info()
    
.. image:: _static/images/tutorial/dc_dblp_info.png
	:align: center
   	:width: 260
   	:height: 175

Connector.show_schema
--------------------------
The show_schema method returns the website data in a Dataframe format. There are two columns in the response. The first column is the name and the second is the datatype.
As an example, lets see what is in the publication table.

::

    dc.show_schema("publication")

.. image:: _static/images/tutorial/dc_dblp_show_schema.png
	:align: center
   	:width: 267
   	:height: 303
    
Connector.query
------------------
The query method downloads the website data. The parameters must meet the requirements as indicated in connector.info for the operation to run.

When the data is received from the server, it will either be in a JSON or XML format. The data_connector reformats the data in pandas Dataframe for the convenience of downstream operations.

As an example, let's try to get the data from the "publication" table, providing the query search for "lee".

::

    dc.query("publication", q="lee")
    
.. image:: _static/images/tutorial/dc_dblp_query.png
	:align: center
   	:width: 1000
   	:height: 500
    
From query results, you can see how easy it is to download the publication data from DBLP into a pandas Dataframe.
Now that you have an understanding of how data connector operates, you can easily accomplish the task with two lines of code.

::

    dc = Connector(...)
    dc.query(...)
    
That's all for now.
===================
Please visit the other tutorials that are available if you are interested in setting up a different data connector. 
If you are interested in writing your own configuration file or modify an existing one, refer to the `Configuration Files
<https://github.com/sfu-db/DataConnectorConfigs>`_.