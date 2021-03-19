import numpy as np
import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go

import dash  # (version 1.12.0) pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_table as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


data = pd.read_csv(r'https://api.covid19india.org/csv/latest/state_wise_daily.csv')
for i in data.columns[3:]:
  m = data[i].median()
  data[i] = data[i].apply(lambda X:m if X<0 else X)
data.reset_index(inplace=True)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = 'COVID-19: Prediction And Analysis Of Spreading Rate'



# ---------------------------------------c---------------------------------------
# App layout
app.layout = html.Div([

    html.H1("COVID-19 Analysis And Prediction Of Spreading Rate", style={'text-align': 'center',"color":'#78ABEE'}),
    dcc.Dropdown(id="slct_state",
                 options=[
                        {"label": "Andaman and Nicobar Islands", "value": "AN"},
                        {"label": "Andhra Pradesh","value" :"AP"},
                        {"label": "Arunachal Pradesh","value" :"AR"},
                        {"label": "Assam","value" :"AS"},
                        {"label": "Bihar","value" :"BR"},
                        {"label": "Chandigarh","value" :"CH"},
                        {"label": "Chattisgarh","value" :"CT"},
                        {"label": "Dadra and Nagar Haveli","value" :"DN"},
                        {"label": "Daman and Diu","value" :"DD"},
                        {"label": "Delhi","value" :"DL"},
                        {"label": "Goa","value" :"GA"},
                        {"label": "Gujarat","value" :"GJ"},
                        {"label": "Haryana","value" :"HR"},
                        {"label": "Himachal Pradesh","value" :"HP"},
                        {"label": "Jammu and Kashmir","value" :"JK"},
                        {"label": "Jharkhand","value" :"JH"},
                        {"label": "Karnataka","value" :"KA"},
                        {"label": "Kerala","value" :"KL"},
                        {"label": "Lakshadweep Islands","value" :"LD"},
                        {"label": "Madhya Pradesh","value" :"MP"},
                        {"label": "Maharashtra","value" :"MH"},
                        {"label": "Manipur","value" :"MN"},
                        {"label": "Meghalaya","value" :"ML"},
                        {"label": "Mizoram","value" :"MZ"},
                        {"label": "Nagaland","value" :"NL"},
                        {"label": "Odisha","value" :"OR"},
                        {"label": "Pondicherry","value" :"PY"},
                        {"label": "Punjab","value" :"PB"},
                        {"label": "Rajasthan","value" :"RJ"},
                        {"label": "Sikkim","value" :"SK"},
                        {"label": "Tamil Nadu","value" :"TN"},
                        {"label": "Telangana","value" :"TS"},
                        {"label": "Tripura","value" :"TR"},
                        {"label": "Uttar Pradesh","value" :"UP"},
                        {"label": "Uttarakhand","value" :"UT"},
                        {"label": "West Bengal","value" :"WB"}],

                 multi=False,
                 value="MH",
                 style={'width': "40%",'margin-left':'30%'}
                 ),

    html.Div(id='output_container', children=[],style={"color":"white","font-size":"30px","text-align":"center"}),
    html.Br(),
    html.H3("Prediction Of Spreading Rate for Next 7 days", style={'text-align': 'center',"color":'#78ABEE'}),
    html.Br(),
    html.Div(id="pred_container",children=[],),
    html.Br(),

    html.Div([
        html.Div([
            dcc.Graph(id='conf_map', figure={},
                        config={'displayModeBar': False})],className='four columns',style={"margin-left":"1%","padding":"0px"}),

        html.Div([
        dcc.Graph(id='rec_map',figure={},
                    config={'displayModeBar': False})],className='four columns',style={"margin":"0px","padding":"0px"}),

        html.Div([
        dcc.Graph(id='dec_map',figure={},
                    config={'displayModeBar': False})],className='four columns',style={"margin":"0px","padding":"0px"}),
    ],className="row")
],style={"background":"#2C2C2C"})


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='conf_map', component_property='figure'),
     Output(component_id='rec_map', component_property='figure'),
     Output(component_id='dec_map', component_property='figure')],
    [Input(component_id='slct_state', component_property='value')]
)
def update_graph(option_slctd):

    container = "The State chosen by user is : {}".format(option_slctd)

    dff = data.copy()
    Rec_dff = dff[dff["Status"] == "Recovered"][["Date",option_slctd]]
    Dec_dff = dff[dff["Status"] == "Deceased"][["Date",option_slctd]]
    Conf_dff = dff[dff["Status"] == "Confirmed"][["Date",option_slctd]]

    # print(Rec_dff)

    # Plotly Express
    fig1 = px.line(
        data_frame=Conf_dff,
        x=Conf_dff['Date'],
        y=Conf_dff[option_slctd],
        title="Confirmed Cases in {}".format(option_slctd),
        width=500,
        template="plotly_dark",
    )

    fig2 = px.line(
        data_frame=Rec_dff,
        x=Rec_dff['Date'],
        y=Rec_dff[option_slctd],
        title="Recovered Cases in {}".format(option_slctd),
        width=500,
        template="plotly_dark",
    )

    fig3 = px.line(
        data_frame=Dec_dff,
        x=Dec_dff['Date'],
        y=Dec_dff[option_slctd],
        title="Deceased Cases in {}".format(option_slctd),
        width=500,
        template="plotly_dark",
    )

    return container, fig1,fig2,fig3

# for showing the predection of next 7 day on dashboard
@app.callback(
    Output(component_id='pred_container', component_property='children'),
    Input(component_id='slct_state', component_property='value'),
)
def show_prediction(option_selected):
    data = pd.read_csv(r'https://api.covid19india.org/csv/latest/state_wise_daily.csv')
    for i in data.columns[3:]:
      m = data[i].median()
      data[i] = data[i].apply(lambda X:m if X<0 else X)
    data.reset_index(inplace=True)

    dff = data.copy()
    Confirmed = dff[dff["Status"] == "Confirmed"][["Date_YMD",option_selected]]

    Confirmed.set_index('Date_YMD',inplace=True)
    Confirmed.index = pd.to_datetime(Confirmed.index,format='%Y-%m-%d')

    x = len(Confirmed)-5
    train=Confirmed.iloc[:x]
    test = Confirmed.iloc[x:]

    scaler = MinMaxScaler()
    scaler.fit(train) #find max value


    scaled_train = scaler.transform(train)#and divide every point by max value
    scaled_test = scaler.transform(test)

    test_prediction = []

    name = "models/confirmed_{0}_2021-03-18.h5".format(option_selected)
    model = load_model(name)
    ##last n points from training set
    first_eval_batch = scaled_train[-5:]
    current_batch = first_eval_batch.reshape(1,5,1)

    for i in range(len(test)+7):
        current_pred = model.predict(current_batch)[0]
        test_prediction.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    true_prediction = scaler.inverse_transform(test_prediction)
    time_series_array = test.index

    for k in range(0,7):
        time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

    df_forecast = pd.DataFrame(columns=["Confirmed Predicted"],index=time_series_array)
    df_forecast.loc[:,"Confirmed Predicted"] = true_prediction[:,0].astype('int32')

    df_forecast.reset_index(inplace=True)
    df_forecast.rename(columns={'Date_YMD':'Date'},inplace=True)
    data = df_forecast.iloc[-7:].to_dict('records')
    columns =  [{"name": j, "id": j,} for j in (df_forecast.columns)]

    return dt.DataTable(data=data, columns=columns,
                        style_header={'backgroundColor': 'rgb(30, 30, 30)','textAlign': 'center'},
                        style_cell={
                            'textAlign': 'center',
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'color': 'white'},
                        tooltip_data=[
                              {
                                  column: {'value': str(value), 'type': 'markdown'}
                                  for column, value in row.items()
                              } for row in data
                          ],
                          tooltip_duration=None)

if __name__ == '__main__':
    app.run_server(debug=True)
