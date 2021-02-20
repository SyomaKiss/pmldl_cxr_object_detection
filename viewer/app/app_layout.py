import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

INVALID_VALUE_MESSAGE = html.B(html.P("Invalid name or index", style={"color": "red"}))

INVALID_NAME_MESSAGE = html.B(html.P("Invalid name, 0th image is loaded", style={"color": "red"}))

PRETTY_YELLOW = "#fcba03"
PRETTY_WHITE = "#ffffff"
PRETTY_GREEN = "#32a852"

def get_layout(modifiable=True, working_file_name=''):
    return html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H6("Enter index or image_id in dataset"),
                        html.Div(
                            ["Idx: ", dcc.Input(id="idx-input", value=0, type="number")]
                        ),
                        html.Div(
                            [
                                "Image_id: ",
                                dcc.Input(id="name-input", value="", type="text"),
                            ]
                        ),
                        html.Div(id="my-output"),
                        html.Div(id="my-warning-2"),
                        html.Div(children="Width of the image viewer"),
                        dcc.Slider(
                            id="plot_scale_slider",
                            min=0,
                            max=2000,
                            step=200,
                            value=1000,
                            marks={
                                i: {"label": "{}".format(i)}
                                for i in [0, *range(0, 2000, 200), 2000]
                            },
                            included=False,
                        ),
                        dcc.Checklist(
                            id="boxes-flag",
                            options=[
                                {"label": "Show boxes", "value": "on"},
                            ],
                            value=["on"],
                        ),
                        dcc.Checklist(
                            id="highres_flag",
                            options=[
                                {"label": "Highres", "value": "on"},
                            ],
                            value=[],
                        ),
                        html.H3(children="Editor"),

                        html.Div(children="Image_id:"),
                        html.Div(id="image-id-div"),
                        html.Div(
                            id="editor-view",
                            style={
                                "height": "450px",
                                "maxHeight": "650px",
                                "overflow": "scroll",
                            },
                        ),
                        html.Button("Save & Next", id="save-next-button",
                                    disabled=not modifiable),
                    ],
                    className="four columns",
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="graph1",
                            figure=go.Figure(),
                        ),
                    ],
                    className="six columns",
                ),
            ],
            className="row",
        ),
        html.Footer(
            id="footer-table",
            style={"overflow": "scroll"},
        ),
        html.Div('Open file is: ' + str(working_file_name))
    ]
)


def make_footer_table(ids, flags, current_idx):

    header_row = html.Tr(
        [html.Th("Image index")]
        + [
            html.Th(
                html.Button(
                    i,
                    id={"type": "footer-table-idx-button", "index": str(i)},
                    n_clicks=0,
                ),
                style={"background-color": PRETTY_GREEN if flag else PRETTY_WHITE},
            )
            for i, flag in zip(ids, flags)
        ]
    )

    data_row = html.Tr(
        [html.Td("Current")]
        + [
            html.Td(
                flag,
                style={
                    "background-color": PRETTY_YELLOW
                    if i == current_idx
                    else PRETTY_WHITE
                },
            )
            for i, flag in zip(ids, flags)
        ]
    )
    return html.Table([header_row, data_row])


def make_dropdown(
    id,
    options=(1, 2, 3),
    default=1,
    width="180px",
    disabled=False,
    background_color=PRETTY_WHITE,
    clearable=False,
):
    return dcc.Dropdown(
        id=id,
        options=[{"label": opt, "value": opt} for opt in options],
        value=default,
        style={"width": width, "background-color": background_color},
        disabled=disabled,
        clearable=clearable,
    )


def get_editor_layout(df, new_class_options, modifiable=True):
    trs = []
    df_columns = ["image_id", "class_name", "rad_id", "new_class", "removed"]
    for row in df[df_columns].itertuples():
        tr = html.Tr(
            [
                html.Td(row.Index),
                html.Td(row.rad_id),
                html.Td(row.class_name),
                html.Td(
                    make_dropdown(
                        {"type": "new-class-dropdown", "index": row.Index},
                        options=new_class_options,
                        default=row.new_class,
                        disabled=not modifiable,
                        background_color=PRETTY_WHITE
                        if row.new_class == row.class_name
                        else PRETTY_YELLOW,
                    )
                ),
                html.Td(
                    dcc.Checklist(
                        id={"type": "remove-box-checkbox", "index": row.Index},
                        options=[
                            {"label": "", "value": "on", "disabled": not modifiable}
                        ],
                        value=["on"] if bool(row.removed) else [],
                    )
                ),
            ]
        )
        trs.append(tr)

    children = html.Table(
        [
            html.Tr(
                [
                    html.Th("Box_id"),
                    html.Th("Rad_id"),
                    html.Th("Class"),
                    html.Th("New class"),
                    html.Th("Removed"),
                ]
            )
        ]
        + trs,
    )
    return children
