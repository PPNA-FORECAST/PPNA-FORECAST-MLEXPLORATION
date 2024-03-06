""" 
This Script was created with the purpose of performing data normalization on the information provided by the photodetection department of the Buenos Aires University.

The idea is have a processed data with the following features: 
date, longitude, latitude, ppna, ppt, temp
"""

### import modules ###
import pandas as pd
from datetime import datetime, timedelta


"""
Ingestion and Transformation of ppna database 
"""

def ppna_database_ingest(file_name):
    
    ppna_df = pd.read_csv(f'../data/raw/{file_name}', header=None)

    #Add id_ppna
    ppna_df['id_ppna'] = range(0, len(ppna_df)) 
    ppna_df['id_date'] = range(0, len(ppna_df)) 

    #Delete first row (headers)
    ppna_df=ppna_df[1:]

    #Melt each column of coordenate into rows 
    ppna_df = pd.melt(ppna_df, id_vars=['id_ppna', 'id_date'], var_name='id_cord', value_name='ppna')

    return ppna_df


"""
Ingestion and Transformation of precipitation database 
"""

def ppt_database_ingest(file_name):

    ppt_df = pd.read_csv(f'../data/raw/{file_name}', header=None)

    #Add id_ppna
    ppt_df['id_ppt'] = range(0, len(ppt_df)) 
    ppt_df['id_date'] = range(0, len(ppt_df)) 

    #Delete first row (headers)
    ppt_df=ppt_df[1:]

    #Melt each column of coordenate into rows 
    ppt_df = pd.melt(ppt_df, id_vars=['id_ppt', 'id_date'], var_name='id_cord', value_name='ppt')

    return ppt_df

"""
Ingestion and Transformation of temperature database 
"""

def temp_database_ingest(file_name):

    temp_df = pd.read_csv(f'../data/raw/{file_name}', header=None)

    #Add id_ppna
    temp_df['id_temp'] = range(0, len(temp_df)) 
    temp_df['id_date'] = range(0, len(temp_df)) 

    #Delete first row (headers)
    temp_df=temp_df[1:]

    #Melt each column of coordenate into rows 
    temp_df = pd.melt(temp_df, id_vars=['id_temp', 'id_date'], var_name='id_cord', value_name='temp')

    return temp_df

"""
Ingestion and Transformation of Cordenates database 
"""

def cord_database_ingest(file_name):

    cord_df = pd.read_csv(f'../data/raw/{file_name}')

    #Only kept latitude and longitude 
    cord_df = cord_df[['latitude','longitude']]
    
    #Add id_cord
    cord_df['id_cord'] = range(0, len(cord_df)) 

    return cord_df

"""
Ingestion and Transformation of date database 
"""

def date_database_ingest(file_name):

    date_df = pd.read_csv(f'../data/raw/{file_name}', names=['date','ppna'])

    #add id_date
    date_df['id_date'] = range(0, len(date_df)) 

    #only kept the date
    date_df = date_df.loc[1:, ['date', 'id_date']]

    #Convert the format {year}.{day number} to dd-mm-yyyy
    date_convertion = lambda fecha_str: datetime(int(fecha_str[1:5]), 1, 1) + timedelta(days=int(fecha_str[6:]) - 1)
    date_df['date'] = date_df['date'].apply(date_convertion)

    return date_df 

"""
Merge temperature, precipitation, ppna, date and cordenates to have a consolidate database
"""

def merge_databases(ppna_df, temp_df, ppt_df,cord_df,date_df): 

    ppna_df = ppna_df.merge(date_df, on='id_date', how='inner')
    ppna_df = ppna_df.merge(cord_df, on='id_cord', how='inner')

    ppna_df = temp_df.merge(ppna_df, on=['id_date', 'id_cord'], how='inner')
    ppna_df = ppt_df.merge(ppna_df, on=['id_date', 'id_cord'], how='inner')
    ppna_df = ppna_df[['ppna', 'temp','ppt', 'date', 'latitude', 'longitude']]

    ppna_df['ppna'] = pd.to_numeric(ppna_df['ppna'], errors='coerce')
    ppna_df['temp'] = pd.to_numeric(ppna_df['temp'], errors='coerce')
    ppna_df['ppt'] = pd.to_numeric(ppna_df['ppt'], errors='coerce')

    return ppna_df