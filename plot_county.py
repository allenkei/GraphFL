from pickle import TRUE
import torch
import numpy as np
from collections import defaultdict
import scipy.stats as st
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd



def cal_map(args, labels, annotate=True, save=True):
  # Load California county shapefile
  #shapefile_path = 'data/CA_data/tl_2023_us_county.shp' 
  #gdf = gpd.read_file(shapefile_path)
  #gdf = gdf[gdf['STATEFP'] == '06'].copy() #  Filter for California (STATEFP == '06')

  gdf = gpd.read_file('data/CA_data/California_Counties.shp')

  #print(gdf.columns); print(gdf.head())

  FIPS_name = [
    'Alameda',
    'Alpine',
    'Amador',
    'Butte',
    'Calaveras',
    'Colusa',
    'Contra Costa',
    'Del Norte',
    'El Dorado',
    'Fresno',
    'Glenn',
    'Humboldt',
    'Imperial',
    'Inyo',
    'Kern',
    'Kings',
    'Lake',
    'Lassen',
    'Los Angeles',
    'Madera',
    'Marin',
    'Mariposa',
    'Mendocino',
    'Merced',
    'Modoc',
    'Mono',
    'Monterey',
    'Napa',
    'Nevada',
    'Orange',
    'Placer',
    'Plumas',
    'Riverside',
    'Sacramento',
    'San Benito',
    'San Bernardino',
    'San Diego',
    'San Francisco',
    'San Joaquin',
    'San Luis Obispo',
    'San Mateo',
    'Santa Barbara',
    'Santa Clara',
    'Santa Cruz',
    'Shasta',
    'Sierra',
    'Siskiyou',
    'Solano',
    'Sonoma',
    'Stanislaus',
    'Sutter',
    'Tehama',
    'Trinity',
    'Tulare',
    'Tuolumne',
    'Ventura',
    'Yolo',
    'Yuba'
  ]

  FIPS_name = [name + ' County' for name in FIPS_name]

  df = pd.DataFrame({'County': FIPS_name, 'Cluster': labels})
  df['County'] = df['County'].str.lower().str.strip()
  df['Cluster'] = pd.Categorical(df['Cluster'])

  gdf['NAME'] = gdf['NAME'].str.lower().str.strip(); #print(gdf['NAME'])
  gdf = gdf.merge(df, left_on='NAME', right_on='County')

  fig, ax = plt.subplots(figsize=(12, 12))
  gdf.plot(column='Cluster', cmap='tab20', legend=True, ax=ax, edgecolor='black')

  if annotate:
      for _, row in gdf.iterrows():
          pt = row.geometry.representative_point()
          ax.annotate(
              text=row['County'].replace(' county', '').title(),
              xy=(pt.x, pt.y),
              ha='center',
              fontsize=6,
              color='black'
          )

  plt.title('California Counties by Clusters')
  plt.axis('off')
  
  if save == True:
    plt.savefig(args.output_dir +'/Counties_map', format='pdf', bbox_inches='tight')
  plt.show()



























