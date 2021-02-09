import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from dataset512 import CXRDatasetr512
from plotter import get_plot
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



csv_path = '/Users/semenkiselev/Documents/dev/job/kaggle/train.csv'
root = '/Users/semenkiselev/Documents/dev/job/kaggle/archive/'
d = CXRDatasetr512(root, csv_path)

df = d.df
colors = ['#f58231', "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
          "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
          "#5A0007", ]
rad2color = dict(zip(df.rad_id.unique(), colors))


app.layout = html.Div([
    html.H6("Enter image_id or index in dataset"),
    html.Div(["Input: ",
              dcc.Input(id='my-input', value='0', type='text')]),
    html.Div(id='my-output'),
    dcc.Graph(id='image-plot'),
])


@app.callback(
    [Output('image-plot', 'figure'), Output('my-output', 'children')],
    Input('my-input', 'value'))
def update_figure(str_idx):

    try:
        if len(str_idx) <= 5:
            entry = d[int(str_idx)]
        else:
            entry = d.get_by_name(str_idx)

        img = entry['image']
        rad_ids = entry['rad_id']
        class_names = entry['class_name']
        class_id = entry['class_id']
        bboxes = entry['bboxes']
        fig = get_plot(img, bboxes, class_names, rad_ids, rad2color)
        return fig, ''
    except ValueError:
        return go.Figure(), html.B(html.P('Invalid name ', style={'color': 'red'}))



if __name__ == '__main__':
    app.run_server(debug=True)