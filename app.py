pip install flask==2.1.0
import dash
from dash import Dash
from dash import dcc
from dash import html
from dash import callback
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import os
import joblib
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Fraud Detection"

server = app.server
app.config.suppress_callback_exceptions = True


# Load data from csv
def load_data():
    # Load data
    df = pd.read_csv('data/datosv2_clean.csv')
    return df

# Cargar datos
data = load_data()

def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            #html.H5("Proyecto 1"),
            html.H3("Pronóstico de pérdidas de fluidos en energía eléctrica"),
            html.Div(
                id="intro",
                children="Esta herramienta contiene información sobre pérdidas no técnicas de energía en América Latina. Adicionalmente, permite realizar pronósticos de posible fraude."
            ),
        ],
    )

df = data
    
consumo_promedio = df['N479'].mean()
consumo_promedio = round(consumo_promedio, 2)
consumo_minimo = df['N479'].min()
consumo_minimo = round(consumo_minimo, 2)
consumo_maximo = df['N479'].max()
consumo_maximo = round(consumo_maximo, 2)

def control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[  
            html.Div([
                html.Div([
                    html.Div([
                        html.P("Seleccionar Región:")
                    ]),
                    html.Div(
                        id="componente-region",
                        children=[
                            html.Div(
                                id="componente-regiones",
                                children=[
                                    dcc.Dropdown(
                                        id="dropdown-regiones",
                                        options=[{'label':i, 'value':i} for i in df['ID_REGION'].unique()],
                                        value = 10
                                    )
                                ],
                                style=dict(width='80%')
                            ),

                        ],
                        style=dict(display='flex')),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '25%'}),
                
                html.Div([
                    html.Div([
                        html.P("Consumo Promedio:")
                    ]),

                    html.Div([
                        dcc.Input(id='CONSUMO_PROMEDIO', type='number', value=consumo_promedio,
                                 readOnly='readOnly'),
                    ]),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '25%'}),
                
                html.Div([
                    html.Div([
                        html.P("Consumo Mínimo:")
                    ]),

                    html.Div([
                        dcc.Input(id='CONSUMO_MINIMO', type='number', value=consumo_minimo,
                                 readOnly='readOnly'),
                    ]),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '25%'}),
                
                html.Div([
                    html.Div([
                        html.P("Consumo Máximo:")
                    ]),

                    html.Div([
                        dcc.Input(id='CONSUMO_MAXIMO', type='number', value=consumo_maximo,
                                 readOnly='readOnly'),
                    ]),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '25%'}),
            ]),   
        ]
    )

def prediction_card():
    """
    :return: A Div containing prediction inputs.
    """
    return html.Div(
        id="prediction-card",
        children=[

            html.Div([
                html.Div([
                    html.Label('ID Municipio'),
                    html.Div([
                        dcc.Dropdown(id='ID_MUNICIPALITY', options=[{'label':i, 'value':i} for i in df['ID_MUNICIPALITY'].unique()], value=''),
                    ], style=dict(width='80%')),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),
                
                html.Div([
                    html.Label('ID Región'),
                    html.Div([
                        dcc.Dropdown(id='ID_REGION', options=[{'label':i, 'value':i} for i in df['ID_REGION'].unique()], value=''),
                    ], style=dict(width='80%')),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),
                
            ]),

            html.Div([
                html.Div([
                    html.Label('Ciclo de Lectura'),
                    dcc.Input(id='ID_UC_READING_CYCLE', type='number', value=''),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),
                
                html.Div([
                    html.Label('Avg. Última Medición'),
                    dcc.Input(id='N284', type='number', value=''),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),               
            ]),    

            html.Div([
                html.Div([
                    html.Label('Plan Sistema Medición'),
                    dcc.Input(id='UC_COL_ID_05', type='number', value=''),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),
                
                html.Div([
                    html.Label('Avg. Consumo 3M'),
                    dcc.Input(id='N256', type='number', value=''),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),
            ]),  

            html.Div([
                html.Div([
                    html.Label('Avg. Consumo M6-4'),
                    dcc.Input(id='N322', type='number', value=''),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),

                html.Div([
                    html.Label('Avg. Consumo 35M'),
                    dcc.Input(id='N588', type='number', value=''),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),
            ]),  

            html.Div([
                html.Div([
                    html.Label('Avg. Consumo 12M'),
                    dcc.Input(id='N319', type='number', value=''),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),
                
                html.Div([
                    html.Label('Suma Consumo 3M'),
                    dcc.Input(id='N479', type='number', value=''),
                ], style={'marginBottom': 10, 'display': 'inline-block', 'width': '50%'}),
            ]), 

            html.Div([
                html.Button('Predecir', id='submit-val', n_clicks=0, style={'color': 'white'}),
            ], style={'marginTop': 20}),

            html.Div(id='output-prediction')            
     
        ]
    )

app.layout = html.Div(
    id="app-container",
    children=[
        
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), prediction_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[control_card()]
            + [
                # Histograma
                html.Div(
                    id="model_graph",
                    children=[
                        html.Hr(),
                        dcc.Graph(
                            id="graphs",
                        )
                    ],
                ),
            
            ],
        ),
    ],
)


# Cargar el modelo
rf = joblib.load(os.getcwd() + '\Modelo_energy.pkl')

@app.callback(Output(component_id="graphs", component_property="figure"),
              Output(component_id="CONSUMO_PROMEDIO", component_property="value"),
              Output(component_id="CONSUMO_MINIMO", component_property="value"),
              Output(component_id="CONSUMO_MAXIMO", component_property="value"),
              Input(component_id="dropdown-regiones", component_property="value"))
def update_graph(region):
    df_region = df.loc[df['ID_REGION'] == region]
    consumo_promedio = df_region['N479'].mean()
    consumo_promedio = round(consumo_promedio, 2)
    consumo_minimo = df_region['N479'].min()
    consumo_minimo = round(consumo_minimo, 2)
    consumo_maximo = df_region['N479'].max()
    consumo_maximo = round(consumo_maximo, 2)
    count = df_region.shape[0]
    df_region['RESULTADO_NAME'] = df_region['RESULTADO']
    df_region['RESULTADO_NAME'] = df_region['RESULTADO_NAME'].replace({10: 'No Fraude', 7: 'Fraude'})
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"},{"type": "pie"}]],subplot_titles=(
        "Número de Revisiones Ejecutadas en el Último Año", "Distribución de Fraudes"))
    fig.add_trace(row=1, col=1, trace=go.Histogram(x=df_region['N294'])) # replace with your own data source
    fig.add_trace(row=1, col=2, trace=go.Pie(values=df_region['RESULTADO'],labels=df_region['RESULTADO_NAME']))
    fig.update_layout(title_text=f'Región {region}, Casos Acumulados: {count}')
    return fig, consumo_promedio, consumo_minimo, consumo_maximo

@app.callback(Output(component_id='output-prediction', component_property='children'),
             Input(component_id='submit-val', component_property='n_clicks'),
             State('ID_UC_READING_CYCLE', 'value'),
             State('ID_MUNICIPALITY', 'value'),
             State('ID_REGION', 'value'),
             State('UC_COL_ID_05', 'value'),
             State('N284', 'value'),
             State('N256', 'value'),
             State('N322', 'value'),
             State('N588', 'value'),
             State('N479', 'value'),
             State('N319', 'value') )
def update_prediction(n_clicks,ID_UC_READING_CYCLE, ID_MUNICIPALITY, ID_REGION, UC_COL_ID_05,
                       N284, N256, N322, N588, N479, N319):
    # Realizar la predicción solo cuando se hace clic en el botón
    if n_clicks > 0:
        # Crear un DataFrame con los valores ingresados por el usuario
        data = {
        'ID_UC_READING_CYCLE': [ID_UC_READING_CYCLE],
        'ID_MUNICIPALITY': [ID_MUNICIPALITY],
        'ID_REGION': [ID_REGION],
        'UC_COL_ID_05': [UC_COL_ID_05],
        'N284': [N284],
        'N256': [N256],
        'N322': [N322],
        'N588': [N588],
        'N479': [N479],
        'N319': [N319]
        }
        df = pd.DataFrame(data)        
        # Realizar la predicción
        try:
            prediction = round(rf.predict_proba(df)[0][1]*100,2)
            return "La probabilidad de fraude es de: "+str(prediction)+"%"
        except Exception as e:
            return "Verifique que todos los campos están completos. Error en la predicción: "+str(e)    


# Run the server
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=True)