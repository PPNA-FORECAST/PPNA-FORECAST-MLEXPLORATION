"""
TO DO: 
    -Armar documentación 
    -Convertirlo en una clase que cree una fig 
"""


import data
import pandas as pd
import folium
from matplotlib import pyplot as plt
from folium import plugins


def show_correlation(df,fig, var1, var2): 

    fig.scatter(df[f"{var1}"], df[f"{var2}"], c='blue', alpha=0.7, s=5)
    r= df[f"{var1}"].corr(df[f"{var2}"])
    r=round(r,3)
    fig.set_title(f"Correlation between {var1} and {var2} (R = {r})")
    fig.set_xlabel(f'{var1}'.upper())
    fig.set_ylabel(f'{var2}'.upper())
    fig.grid(True)

def show_ppna_fixed_position (df, fig, latitude, longitude):

    fix_position_df = df[(df['longitude'] == longitude) & (df['latitude'] == latitude)].copy()
    fix_position_df['date'] = pd.to_datetime(fix_position_df['date'])
    fig.plot(fix_position_df['date'], fix_position_df['ppna'], c='blue')
    fig.set_title("PPNA over time in fixed location")
    fig.set_xlabel('Date')
    fig.set_ylabel('PPNA')
    fig.grid(True)

def show_ppna_year_comparision_fixed_position(df, fig, year1, year2, latitude, longitude): 
        
    fix_position_df = df[(df['longitude'] == longitude) & (df['latitude'] == latitude)].copy()
    fix_position_df['date'] = pd.to_datetime(fix_position_df['date'])

    for year in fix_position_df['date'].dt.year.unique():
        if year == year1 or year == year2:
            year_data = fix_position_df[fix_position_df['date'].dt.year == year]
            dias_meses = [fecha.strftime('%d-%m') for fecha in year_data['date']]
            fig.plot(dias_meses, year_data['ppna'], label=f'Year {year}')

    # Adjusting the x-axis labels for better readability
    fig.tick_params(axis='x', rotation=45)
    fig.set_title("PPNA Comparison over the years")
    fig.set_xlabel('Date')
    fig.set_ylabel('PPNA')
    fig.legend()
    fig.grid(True)

def show_heat_map(df):

    # Crear un mapa centrado en la ubicación media de tus datos
    mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)

    # Creo un df con fecha, latitude, longitud y valor de la ppna 
    ppna_heatmaptime_df = df[['date','latitude','longitude','ppna']]
    ppna_heatmaptime_df['date'] = ppna_heatmaptime_df['date'].sort_values(ascending=True)
    data = []
    for _, d in ppna_heatmaptime_df.groupby('date'):
        data.append([[row['latitude'], row['longitude'], row['ppna']] for _, row in d.iterrows()])
    data

    #creo el indice de tiempo para graficar en el mapa 
    time_index = list(ppna_heatmaptime_df['date'].astype('str').unique())


    plugins.HeatMapWithTime(data,
                    index=time_index,
                    auto_play=True,
                    radius=10,
                    use_local_extrema=True
                ).add_to(mapa)


    # Agrego labels
    """
    Lo dejo comentado por que por performance tarda mucho, despues vemos si lo necesitamos como solucionarlo 
    percentage_to_label = 5  # Porcentaje de puntos para los que se agregarán etiquetas
    total_points = sum(len(points) for points in data)
    points_to_label = [point for points in data for point in points if hash(tuple(point)) % 100 < percentage_to_label]

    for lat, lon, value in points_to_label:
        label = value
        folium.Marker(location=[lat, lon], popup=label).add_to(mapa)
    """

    display(mapa)

def plot_training_history(fig, history, metric):

    validation_metric = f"val_{metric}"

    fig.plot(history.history[metric], label=f"Training {metric}")
    fig.plot(history.history[validation_metric], label=f"Validation {metric}")
    fig.set_xlabel("Epoch")
    fig.set_ylabel(f"{metric}")
    fig.legend()

    merged_mae_lists = history.history[f"{metric}"][0:] + history.history[f"val_{metric}"][0:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    fig.set_ylim([bottom_of_y_axis, top_of_y_axis])

def plot_result_in_fixed_position(ax, longitude, latitude, index, model, test_df, input_steps,output_steps): 
        
    test_sequence_fixed_position_df = data.DataManager(test_df[(test_df['longitude'] == longitude) & (test_df['latitude'] == latitude)])
    test_sequence_fixed_position, test_labels_fixed_position = test_sequence_fixed_position_df.sequence_data_preparation(input_steps, output_steps)
    predict = model.predict(test_sequence_fixed_position, verbose=0)
    ax.plot(test_sequence_fixed_position[index,:, -1], label='Input data')
    ax.plot(range(input_steps , output_steps  + input_steps), predict[index], label="Model predictions")
    ax.plot(range(input_steps , output_steps  + input_steps), test_labels_fixed_position[index], label="Data labels")
    ax.set_xlabel("fortnight")  
    ax.set_ylabel("Normalized PPNA") 
    ax.set_title(f"Prediction for location ['{longitude}','{latitude}']")
    ax.legend()