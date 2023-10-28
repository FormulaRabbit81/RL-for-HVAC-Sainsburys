# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:06:08 2021

@author: maxbi
"""
import pandas as pd
import datetime
import pytz
import os

def unix2datetime(unix: int) -> datetime.datetime:
    """ Returns aware datetime """
    dt = datetime.datetime.fromtimestamp(unix)
    timezone = pytz.timezone("Europe/London")
    return timezone.localize(dt)

def unix2strtime(unix: int) -> str:
    dt = datetime.datetime.fromtimestamp(unix)
    timezone = pytz.timezone("Europe/London")
    dt_aware = timezone.localize(dt)
    return dt_aware.strftime("%H:%M %d/%m/%Y")

def get_opening_times(dt_start: datetime.datetime, dt_end: datetime.datetime) -> (list, list):
    # datetimes must be aware
    openingTimes = []
    closingTimes = []

    currentTime = dt_start
    while currentTime < dt_end:
        if currentTime.weekday() != 6: # mon-sat inclusive
            openHour = 7
            closeHour = 23
        else: # sunday
            openHour = 10
            closeHour = 16

        openingTimes.append(datetime.datetime(currentTime.year, currentTime.month, currentTime.day, openHour, 0))
        closingTimes.append(datetime.datetime(currentTime.year, currentTime.month, currentTime.day, closeHour, 0))
        currentTime += datetime.timedelta(days=1)

    return openingTimes, closingTimes

def plot_opening_times(axs, unix_start: int, unix_end: int):
    openingTimes, closingTimes = get_opening_times(unix2datetime(unix_start), unix2datetime(unix_end))

    for ax in axs:
        for entry in range(len(openingTimes)):
            ax.axvspan(openingTimes[entry], closingTimes[entry], alpha=0.35, color="blue")       
    return axs

def get_electricity_prices(unix_start: int, unix_end: int, timestep: int, elec_price_mode: str) -> list:

    all_prices = pd.read_excel("./core/data/elec_prices/normal_and_volitile_elec_prices.xlsx", index_col=0, usecols=[2,3,4]) # prices and timestamps are correct to here

    if elec_price_mode == "sains":
        prices = all_prices["sains price [p/kWh]"]
    else:
        prices = all_prices["n2ex price [p/kWh]"]

    prices = prices.iloc[(all_prices.index >= unix_start) & (all_prices.index < unix_end)]
    prices = prices.reset_index()

    if timestep == 3600: # hourly mpc timesteps (the default)
        prices = prices[0: :2] # prices is half hourly so skip every odd value to get hourly values

    # return prices.iloc[:]
    return list(prices.iloc[:,1])

def format_carbon_data(raw_carbon_factors: pd.DataFrame) -> pd.DataFrame:

    raw_carbon_factors.drop(labels=["unixTimestamp"], inplace=True, axis=1)
    carbon_factors = raw_carbon_factors.resample("1H").mean() # resampling rows to be hourly instead of hh

    # below loop resamples columns to be hourly forecasts
    hourly_carbon_factors = pd.DataFrame()
    for x in range(int(len(carbon_factors.columns)/2)):
        hourly_carbon_factors[x] = (carbon_factors.iloc[:,x*2] + carbon_factors.iloc[:,(x*2)+1])/2

    hourly_carbon_factors.fillna(method="bfill", inplace=True)

    return hourly_carbon_factors

def format_forecast_data(external_temps: pd.DataFrame, carbon_factors: pd.DataFrame, elec_prices: list, N_steps: int) -> list[pd.DataFrame]:

    forecast_data_list = []    
    for x in range(len(external_temps)):
        forecasts = pd.DataFrame()
        forecasts["external_temp"] = external_temps.iloc[x, 1:] # external_temp contains unix_timestamp column in position 0 so this is not included
        forecasts.reset_index(drop=True, inplace=True)
        forecasts["carbon_factor"] = carbon_factors.iloc[x, :] # carbon_factors does not have the column so it does nto need ot be removed
        forecasts.reset_index(drop=True, inplace=True)
        forecasts["elec_price"] = elec_prices[x:x+N_steps]

        forecast_data_list.append(forecasts)

    return forecast_data_list

def get_min_space_temps(dt_start: datetime.datetime, dt_end: datetime.datetime, open_times: list, close_times: list, timestep: int) -> list:
    """" List of minimum space temps at the desired timestep. Datetimes must be aware in London timezone. """
    current_time = dt_start
    min_space_temps = []
    while current_time < dt_end:
        if current_time.weekday() != 6: # mon-sat
            if current_time.hour < open_times[0] or current_time.hour >= close_times[0]: # store closed
                min_space_temps.append(16)
            else: # store open
                min_space_temps.append(19)
        else: # sunday
            if current_time.hour < open_times[1] or current_time.hour >= close_times[1]:
                min_space_temps.append(16)
            else:
                min_space_temps.append(19)
        current_time += datetime.timedelta(seconds=timestep)
    return min_space_temps

def get_min_space_temps_forecast(min_space_temps: list, unix_start: int, unix_end: int, mpc_steps: int, mpc_timestep: int) -> list[list]:
    """" Produce list of lists for 24 forecasts of min space temp. Used for MPC model """
    return [min_space_temps[x : x+(mpc_steps+1)] for x in range(int((unix_end-unix_start)/mpc_timestep))]

def get_unique_xlsx_path(dir_str: str, base_filename: str) -> str:
    # Generate unique xlsx file name for simulation results
    directory = os.fsencode(dir_str)
    run_num = len([x for x in os.listdir(directory) if x.startswith(base_filename.encode('UTF-8'))]) # count number of existing run saves
    return base_filename + f"{run_num}.xlsx"

def save_xlsx(dfs: list[pd.DataFrame], sheet_names: list[str], xlsx_path: str) -> None:
    """ Save xlsx file with multiple sheets"""
    writer = pd.ExcelWriter(xlsx_path)
    for df, sheet in zip(dfs, sheet_names):
        df.to_excel(writer, sheet_name=sheet)
    writer.save()

def get_external_temps(Table, unix_start: int, unix_end: int, timestep: int) -> list:
    """ Get Dark Sky external temp readings at timestep intervals """
    hourly_external_temps = Table.queryWeatherOrCarbon("weather", unix_start, unix_end+3600, attributes=["apparentTemperature"], forecast_horizon=0)
    external_temps = hourly_external_temps.resample(str(int(timestep/60)) + "min", axis=0).asfreq() # reindex to x min intervals with NaN values
    external_temps.interpolate(method="linear", limit_direction="forward", inplace=True)

    return list(external_temps.loc[:,"apparentTemperature_0"])[:-1]
    
def format_mpc_results(unix_start: int, unix_end: int, mpc_config, sim_timestep,
                       all_mpc_space_temp_forecasts: pd.DataFrame, space_temp_history: list, q_hvac_mpc: list, q_hvac_pi: list,
                       mpc_space_temp_sp_history: list, min_space_temp_history: list, external_temp_history: list, elec_price_history: list, 
                       carbon_factor_history: list) -> list[pd.DataFrame]:

    summary_data = [unix2strtime(unix_start), unix2strtime(unix_end), unix_start, unix_end, mpc_config.N, mpc_config.timestep, sim_timestep, mpc_config.elec_price_mode, mpc_config.optimisation_mode,
                    mpc_config.weights[1], mpc_config.weights[2], mpc_config.weights[3]]
    summary_ind = ["start","end","unix_start","unix_end","mpc_steps","mpc_timestep [s]","simulation_timestep [s]","elec_price_mode","optimisation_mode",
                   "economic weighting", "carbon weighting", "comfort weighting"]
    summary_df = pd.DataFrame(data = summary_data, index = summary_ind)

    unix_timestamps = [unix_start+(x*sim_timestep) for x in range(len(space_temp_history))]
    timestamps = [unix2strtime(x) for x in unix_timestamps]      

    mpc_results_df = pd.DataFrame(list(zip(timestamps, unix_timestamps, space_temp_history, q_hvac_mpc, q_hvac_pi, mpc_space_temp_sp_history, min_space_temp_history, external_temp_history, elec_price_history, carbon_factor_history)),
                                  columns = ["Timestamp","unix_timestamp","simulated_space_temp", "q_hvac_mpc [W]","q_hvac_pi [W]","mpc_space_temp_sp","min_allowable_space_temp","external_temp","elec_price [p/kWh]", "carbon_factor [gCO2/kWh]"])    

    return [summary_df, mpc_results_df, all_mpc_space_temp_forecasts]

def format_baseline_results(unix_start: int, unix_end: int, q_hvac: list, space_temps: list, 
                            space_temp_setpoints: list, simulation_env_timestep: int, oss_duration: int) -> list[pd.DataFrame]:

    summary_data = [unix2strtime(unix_start), unix2strtime(unix_end), unix_start, unix_end, simulation_env_timestep, oss_duration]
    summary_ind = ["start","end","unix_start","unix_end","simulation_timestep [s]","OSS duration [s]"]
    summary_df = pd.DataFrame(data = summary_data, index = summary_ind)

    unix_timestamps = [unix_start+(x*simulation_env_timestep) for x in range(len(q_hvac))]
    timestamps = [unix2strtime(x) for x in unix_timestamps]    

    baseline_results_df = pd.DataFrame(list(zip(timestamps, unix_timestamps, q_hvac, space_temps, space_temp_setpoints)), 
                                       columns = ["Timestamps","unix_timestamp","q_hvac_pi [W]","simulated_space_temp","space_temp_setpoints"])

    return [summary_df, baseline_results_df]
