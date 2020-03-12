#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:29:51 2020

@author: xieyi
"""

from dataprep.data_connector import Connector


apis = ["yelp","dblp","twitter","spotify"]

api_id = 3

api = apis[api_id]


if api == "yelp":
    dc = Connector("./DataConnectorConfigs/yelp", auth_params={"access_token":"MBdBbgig-VEs1KjjtDq5DndLQFhKp0sjCj2FXUEuhPgEjniUm4CyCDYE-nO9TSKK8a3_jlIdsRHAJc9Fv7WXNeL8VITkZoDkBnJ7TPAWvKxey_Pcuyy6cWlGm3IzXnYx"})
    df = dc.query("businesses", term="ramen", location="vancouver")
    
    print(df)


if api == "dblp":
    dc = Connector("./DataConnectorConfigs/dblp")
    df = dc.query("publication",q = "data_mining")
    print(df["title"])

if api == "twitter":
    client_id = 'j5bcIQapZUhRCWqfeRlz6jDIO'
    client_secret = 'Bc9KLr9csRuANgu0ImoSMR0mh7VH6l19GNwF6VJvzMOV0AXxym'
    
    dc = Connector("./DataConnectorConfigs/twitter", auth_params={"client_id": client_id, "client_secret": client_secret})
    
    df = dc.query("tweets", q="machine learning",returned_number = 248)
    
    print(df["text"])

if api == "spotify":
    client_id = '9f2b33cbc59c4fb3aac9aabdbd5b8d00'
    client_secret = 'ce1e42a3b39f4410897598bb70a21828'
    
    dc = Connector("./DataConnectorConfigs/spotify", auth_params={"client_id": client_id, "client_secret": client_secret})
    
    df = dc.query("track", q="summer")
    
    print(df)
    
    