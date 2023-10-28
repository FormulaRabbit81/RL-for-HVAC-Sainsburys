# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:40:51 2023

Input data required:
    - External temps
    - Electricity prices
    
@author: Max
"""
import ddbwrapper as dbw

def get_filename(start_date: str, end_date: str) -> str:
    filename_start_date = f"{start_date[6:].split('/')[0]}{start_date[6:].split('/')[1]}{start_date[6:].split('/')[2]}"
    filename_end_date = f"{end_date[6:].split('/')[0]}{end_date[6:].split('/')[1]}{end_date[6:].split('/')[2]}"
    
    return f"measured_external_temps_{filename_start_date}-{filename_end_date}.csv"


# specify start/end dates
str_start = "00:00 01/05/2022"
str_end = "00:00 01/04/2023"

# calculate unix time from timestamp
unix_start = dbw.timestamp2unix(str_start)
unix_end = dbw.timestamp2unix(str_end)

# initialise DB connection
dynamo_table = dbw.dynamoTable("bmsTrial")

# query data
external_temps = dynamo_table.queryWeatherOrCarbon(
    'weather',
    unix_start,
    unix_end,
    attributes=["temperature"],
    forecast_horizon=0
    )['temperature_0']

# save data
filename = get_filename(str_start, str_end)
external_temps.to_csv(f"./input_data/{filename}")
