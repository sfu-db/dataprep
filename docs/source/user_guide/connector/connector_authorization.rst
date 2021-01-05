
==================================================
Authorization schemes supported by Connector
==================================================

.. toctree::
   :maxdepth: 2

Overview
========

Connector supports the most used authorization methods in Web APIs:

- API Key
- OAuth 2.0 "Client Credentials" and "Authorization Code" grants.

Let's review them in detail:

API Key
========

Depending on the Web API specification, you must send your API Key as a bearer token, a parameter in the URL (query parameter), or a parameter in the request header. Let's review these cases in detail:

Bearer Token
-------------

If the Web API supports the Bearer Token authorization type, follow the next steps:

***********************************
- Adjusting your configuration file
***********************************

You have to specify in your configuration file that the desired authentication method to be used is "Bearer token".
The following property should be included below the "method" property in your configuration file::

    "authorization": {
        "type": "Bearer"
    },

For example, you can review our `Yelp configuration file for the "businesses" endpoint <https://github.com/sfu-db/DataConnectorConfigs/blob/develop/yelp/businesses.json>`_, which implements the "Bearer token" authorization method.

**********************
- Invoking the Web API
**********************

Use the ``connect`` function with the ``"yelp"`` string and your Yelp access token, both specified as parameters. This action allows you to create a Connector to the Yelp Web API.
Next, through the ``query`` function, you can retrieve data from this endpoint. The parameter ``"businesses"`` indicates you want to query the Yelp ``"businesses"`` endpoint with the search term ``"sushi"`` and location ``"Vancouver"``::


    import asyncio
    from dataprep.connector import connect
    yelp_connector = connect("yelp", _auth={"access_token":"<Your Yelp access token>"})
    df = await yelp_connector.query("businesses", term="sushi", location="Vancouver")






Query parameter
------------------
You have to specify in your configuration file that you will send your API Key as a parameter in the request URL (``"type": "QueryParam"``).
The following property should be included below the ``method`` property in your configuration file::

    "authorization": {
        "type": "QueryParam",
        "keyParam": "<API Key parameter name>"
    },

You must replace the ``<API Key parameter name>``  string with the parameter's exact name to be used to send the API Key to the remote endpoint. This parameter name is defined by the Web API. Review the Web API documentation to identify the exact name.

For example, the `Finnhub API - IPO Calendar endpoint <https://github.com/sfu-db/DataConnectorConfigs/blob/develop/finnhub/ipo_calender.json>`_, names this parameter as ``token``::

    "authorization": {
        "type": "QueryParam",
        "keyParam": "token"
    },

To query information from this Finnhub endpoint, you can use both ``connect`` and ``query`` functions::

    import asyncio
    from dataprep.connector import connect
    finnhub_connector = connect("finnhub", _auth={"access_token":"<Your Finnhub API Key>"})
    df = await finnhub_connector.query("ipo_calendar",from_="2020-01-01", to="2020-04-30")

Internally, Connector will take your API Key and send it as the ``token`` parameter's value in the request URL.

Request header parameter
-------------------------
You have to specify in your configuration file that you will send your API Key as a parameter in the request header (``"type": "Header"``).
The following property should be included below the ``method`` property in your configuration file::

    "authorization": {
        "type": "Header",
        "keyName": "<API Key parameter name>"
    },

You must replace the ``<API Key parameter name>`` string with the parameter's exact name to be used to send the API Key to the remote endpoint. This parameter name is defined by the Web API. Review the Web API documentation to identify the exact name.

For example, the `Twitch.tv API - channels endpoint <https://github.com/sfu-db/DataConnectorConfigs/blob/develop/twitch/channels.json>`_,  names this parameter as ``Client-ID``::

    "authorization": {
        "type": "Header",
        "keyName": "Client-ID",
        ...
    },

To query information from this Twitch endpoint, you can use both ``connect`` and ``query`` functions::

    import asyncio
    from dataprep.connector import connect
    twitch_connector = connect("twitch", _auth={"access_token":"<Your Twitch API Key>"})
    df = await twitch_connector.query("channels",q="a_seagull")

Internally, Connector will take your API Key and send it as the ``Client-ID`` parameter's value in the request header.

OAuth 2.0 "Client Credentials" and "Authorization Code" grants
==================================================================

Connector supports the authorization scheme OAuth 2.0 - "Client Credentials" and "Authorization Code" grants. Let's review the details:

Client Credentials grant
--------------------------
In your configuration file, you have to specify that you'll use the OAuth 2.0 authorization type - Client Credentials grant.
The following property should be included below the ``method`` property in your configuration file::

    "authorization": {
        "type": "OAuth2",
        "grantType": "ClientCredentials",
        "tokenServerUrl": "<OAuth 2.0 token URL>"
    },

You must replace the ``<OAuth 2.0 token URL"`` string with the OAuth 2.0 token URL defined by the Web API. Review the Web API documentation to identify the exact URL.

For example, see the `Twitter API - Tweets endpoint <https://github.com/sfu-db/DataConnectorConfigs/blob/develop/twitter/tweets.json>`_, configuration below::

    "authorization": {
        "type": "OAuth2",
        "grantType": "ClientCredentials",
        "tokenServerUrl": "https://api.twitter.com/oauth2/token"
    },

Before executing any query, you must get your Web API ``Client ID`` and ``Client Secret`` information. Once obtained, you can pass these values as parameters in the ``connect`` function and then execute the ``query`` method to retrieve the desired data::

    import asyncio
    from dataprep.connector import connect
    twitter_connector = connect("twitter", _auth={"client_id":twitter_client_id, "client_secret":twitter_client_secret})
    df = await twitter_connector.query("tweets", q="data science")

Internally, Connector will take both ``Client ID`` and ``Client Secret`` values and execute the OAuth 2.0 - Client Credentials grant process on your behalf.


Authorization Code grant
--------------------------
In your configuration file, you have to specify that you'll use the OAuth 2.0 authorization type - Authorization Code grant.
The following property should be included below the ``method`` property in your configuration file::

    "authorization": {
        "type": "OAuth2",
        "grantType": "AuthorizationCode",
        "tokenServerUrl": "<OAuth 2.0 token URL>"
    },

You must replace the ``<OAuth 2.0 token URL"`` string with the OAuth 2.0 token URL defined by the Web API. Review the Web API documentation to identify the exact URL.

For example, see the Twitter API - Tweets endpoint - configuration below::

    "authorization": {
        "type": "OAuth2",
        "grantType": "AuthorizationCode",
        "tokenServerUrl": "https://api.twitter.com/oauth2/token"
    },

Before executing any query, you must get your Web API ``Client ID`` and ``Client Secret`` information. Once obtained, you can pass these values as parameters in the ``connect`` function and then execute the ``query`` method to retrieve the desired data::

    import asyncio
    from dataprep.connector import connect
    twitter_connector = connect("twitter", _auth={"client_id":twitter_client_id, "client_secret":twitter_client_secret})
    df = await twitter_connector.query("tweets", q="data science")

Internally, Connector will take both ``Client ID`` and ``Client Secret`` values and execute the OAuth 2.0 - Authorization Code grant process on your behalf.