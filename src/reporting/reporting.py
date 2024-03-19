"""
TO DO: 
    -Convertirlo en una clase que cree una fig 
"""


import data
import pandas as pd
import folium
from matplotlib import pyplot as plt
from folium import plugins


#PRE: The df must be previously loaded with columns labeled var1 and var2. 
#POST: Prints every datapoint in a var1 vs var2 grid plot and shows the correlation R between both variables
def show_correlation(df,fig, var1, var2): 

    fig.scatter(df[f"{var1}"], df[f"{var2}"], c='blue', alpha=0.7, s=5)
    r= df[f"{var1}"].corr(df[f"{var2}"])
    r=round(r,3)
    fig.set_title(f"Correlation between {var1} and {var2} (R = {r})")
    fig.set_xlabel(f'{var1}'.upper())
    fig.set_ylabel(f'{var2}'.upper())
    fig.grid(True)


#PRE: The df must be previously loaded. latitude and longitude must be among the lat and long values of the df
#POST: Prints every ppna value recorded for the latitud, longitude location in a date vs ppna plot.
def show_ppna_fixed_position (df, fig, latitude, longitude):

    fix_position_df = df[(df['longitude'] == longitude) & (df['latitude'] == latitude)].copy()
    fix_position_df['date'] = pd.to_datetime(fix_position_df['date'])
    fig.plot(fix_position_df['date'], fix_position_df['ppna'], c='blue')
    fig.set_title("PPNA over time in fixed location")
    fig.set_xlabel('Date')
    fig.set_ylabel('PPNA')
    fig.grid(True)


#PRE: The df must be previously loaded. latitude and longitude must be among the lat and long values of the df. There must be ppna data recorded at years
    # year1 and year2.
#POST: Prints every ppna value recorded in year1 and year2 as two separated functions in a sigle date vs ppna grid plot.
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


#PRE: The df must be previously loaded with lat, long, date and ppna values.
#POST: Prints map of earth with every ppna value recorded in it's specified position. The ppna values are graded in a thermal-like color scale
    #  and the user can choose the date of the data being shown.
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


#PRE: history must have the training information of a NN model. metric must be one of the stored metrics in history
#POST: plots a graph of the metric evolution with each epoch for the training dataset as well as the validation dataset
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


#PRE: latitude and longitude must be among the recorded values of the test_df, which must be previously loaded.
#POST: prints in a fortnight vs ppna graph the ppna values of the latitude, longitude position. It first shows the data used for training
    #  and then prints the ppna values of the test_df dataframe and the values predicted by the model for those dates in particular.
def plot_result_in_fixed_position(ax, longitude, latitude, index, model, test_df, input_steps,output_steps, mean, std): 
        
    test_sequence_fixed_position_df = data.DataManager(test_df[(test_df['longitude'] == longitude) & (test_df['latitude'] == latitude)])
    test_sequence_fixed_position, test_labels_fixed_position = test_sequence_fixed_position_df.sequence_data_preparation(input_steps, output_steps)
    predict = model.predict(test_sequence_fixed_position, verbose=0)

    labels = data.DataManager(test_labels_fixed_position[index])
    labels.denormalize_data(mean,std)
    inputs = data.DataManager(test_sequence_fixed_position[index,:, -1])
    inputs.denormalize_data(mean,std)
    predicts = data.DataManager(predict[index])
    predicts.denormalize_data(mean,std)

    ax.plot(inputs, label='Input data')
    ax.plot(range(input_steps , output_steps  + input_steps), predicts, label="Model predictions")
    ax.plot(range(input_steps , output_steps  + input_steps), labels, label="Data labels")
    ax.set_xlabel("fortnight")  
    ax.set_ylabel("PPNA") 
    ax.set_title(f"Prediction for location ['{longitude}','{latitude}']")
    ax.legend()