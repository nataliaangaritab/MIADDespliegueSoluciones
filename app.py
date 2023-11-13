import dash
from dash import Dash
from dash import dcc
from dash import html
from dash import callback
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objs as go
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
data

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

def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[

            html.P("Seleccionar Región:"),
            
            # Dropdown Regiones
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
                                # style=dict(width='50%', display="inline-block")
                            )
                        ],
                        style=dict(width='20%')
                    ),
                    
                ],
                style=dict(display='flex')
            ),
            
            html.Br(),
            
            html.P("Seleccionar Usuario:"),

            # Dropdown Usuarios
            html.Div(
                id="componente-usuario",
                children=[
                    html.Div(
                        id="componente-usuarios",
                        children=[
                            dcc.Dropdown(
                                id="dropdown-usuarios",
                                #options=[{'label':i, 'value':i} for i in df['ID_REGION'].unique()],
                                value = 1
                                # style=dict(width='50%', display="inline-block")
                            )
                        ],
                        style=dict(width='20%')
                    ),
                    
                ],
            )
     
        ]
    )

app.layout = html.Div(
    id="app-container",
    children=[
        
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
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
            children=[
                
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


@app.callback(Output(component_id="graphs", component_property="figure"),
              Input(component_id="dropdown-regiones", component_property="value"),
              Input(component_id="dropdown-usuarios", component_property="value"))
def update_graph(region, usuario):
    df_region = df.loc[df['ID_REGION'] == region]
    count = df_region.shape[0]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"},{"type": "pie"}]],subplot_titles=(
        "Número de Revisiones Ejecutadas en el Último Año", "Distribución de Fraudes"))
    fig.add_trace(row=1, col=1, trace=go.Histogram(x=df_region['N294'])) # replace with your own data source
    fig.add_trace(row=1, col=2, trace=go.Pie(values=df_region['RESULTADO'],labels=df_region['RESULTADO']))
    fig.update_layout(title_text=f'Región {region}, Casos Acumulados: {count}')
    return fig


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)