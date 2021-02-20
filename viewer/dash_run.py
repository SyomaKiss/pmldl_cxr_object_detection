import dash
import dash_html_components as html
import argparse
from loguru import logger


import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State, ALL

from app.app_layout import (
    get_layout,
    get_editor_layout,
    INVALID_VALUE_MESSAGE,
    make_footer_table,
    INVALID_NAME_MESSAGE,
)
from app.plotter import get_plot, hide_boxes
from app.helpers import (
    get_from_dataset,
    what_exactly_triggered,
    expand_df_if_needed,
    create_log_dir,
)
from app.dataset import CXRDatasetrFull
from app.dataset512 import CXRDatasetr512

from app import BBOX_REMOVED_COL_NAME, NEW_CLASS_COL_NAME, DONE_COL_NAME


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

parser = argparse.ArgumentParser()
parser.add_argument("--port", default=8050)
parser.add_argument("--host", default="10.100.10.53")
parser.add_argument(
    "--readonly",
    action="store_const",
    const=True,
    default=False,
    help="if set, user can not modify dataframe",
)
parser.add_argument("--input_path", help="path to initial .csv file")
parser.add_argument(
    "--log_dir_path", help="path to directory with runs", default="viewer_runs"
)
parser.add_argument(
    "--lowres_dataset_root",
    help="path to directory with downscaled dataset",
    default="/home/semyon/data/vinbigdata/archive/train",
)
parser.add_argument(
    "--full_dataset_root",
    help="path to directory with full resolution dataset",
    default="/home/semyon/data/VinBigData/train/",
)

args = parser.parse_args()

# read df, copy it into log dir
csv_path = args.input_path

df = pd.read_csv(csv_path)
df = expand_df_if_needed(df)

csv_path = create_log_dir(args.log_dir_path, csv_path, args.readonly)
dump_csv_path = csv_path

df.to_csv(dump_csv_path, index=False)
logger.add(f"{dump_csv_path}.log", format="{time} {level} {message}", level="INFO")

root = args.lowres_dataset_root
d512 = CXRDatasetr512(root, df=df)

root = args.full_dataset_root
d = CXRDatasetrFull(root, df=df)


app.layout = get_layout(modifiable=not args.readonly, working_file_name=dump_csv_path)


@app.callback(
    Output("graph1", "figure"),
    Output("my-output", "children"),
    Input("idx-input", "value"),
    Input("plot_scale_slider", "value"),
    Input("boxes-flag", "value"),
    Input("highres_flag", "value"),
    Input({"type": "remove-box-checkbox", "index": ALL}, "value"),
    Input({"type": "new-class-dropdown", "index": ALL}, "value"),
)
def update_figure(
    idx_input,
    plot_width,
    boxes_flag,
    highres_flag,
    remove_box_checkbox_values,
    new_class_dropdown_values,
):
    if idx_input is None or idx_input >= len(d) or idx_input < 0:
        return go.Figure(), INVALID_VALUE_MESSAGE

    global df
    cur_dataset = d if bool(highres_flag) else d512
    ctx = dash.callback_context

    idx = idx_input

    if what_exactly_triggered(ctx, "remove-box-checkbox"):
        box_id = eval(ctx.triggered[0]["prop_id"].split(".value")[0])["index"]
        to_be_removed = bool(ctx.triggered[0]["value"])
        # print("# {}, remove? {}".format(box_id, to_be_removed))
        df.loc[box_id, BBOX_REMOVED_COL_NAME] = int(to_be_removed)
        df.to_csv(dump_csv_path, index=False)
        logger.info(
            str(dict(box_id=box_id, action="set remove flag", value=to_be_removed))
        )

    if what_exactly_triggered(ctx, "new-class-dropdown"):
        box_id = eval(ctx.triggered[0]["prop_id"].split(".value")[0])["index"]
        new_class = str(ctx.triggered[0]["value"])
        # print("# {}, change to-> {}".format(box_id, new_class))
        df.loc[box_id, NEW_CLASS_COL_NAME] = str(new_class)
        df.to_csv(dump_csv_path, index=False)
        logger.info(str(dict(box_id=box_id, action="change class", value=new_class)))

    # get sample from dataset OR report error
    try:
        (
            img,
            rad_ids,
            class_names,
            class_id,
            bboxes,
            box_ids,
            new_class_names,
        ) = get_from_dataset(cur_dataset, idx)
        fig = get_plot(img, bboxes, new_class_names, rad_ids, box_ids, width=plot_width)
    except ValueError:
        return go.Figure(), INVALID_VALUE_MESSAGE

    # hide boxes from image
    if what_exactly_triggered(ctx, "boxes-flag"):
        fig = hide_boxes(fig, bool(boxes_flag))
    return fig, ""


@app.callback(
    Output("editor-view", "children"),
    Output("image-id-div", "children"),
    Input("idx-input", "value"),
    State("highres_flag", "value"),
)
def update_editor(idx_input, highres_flag):
    if idx_input is None or idx_input >= len(d) or idx_input < 0:
        return INVALID_VALUE_MESSAGE, ""
    # read updates from csv
    global df
    df = pd.read_csv(csv_path)
    d.df = df
    d512.df = df
    # choose dataset
    cur_dataset = d if bool(highres_flag) else d512
    ctx = dash.callback_context

    # get image_id
    name = d.get_name_by_idx(idx_input)

    tmp = df[df.image_id == name].copy()
    # get sample from dataset OR report error
    try:
        get_from_dataset(cur_dataset, idx_input)
    except ValueError:
        return INVALID_VALUE_MESSAGE

    # editor_content = get_editor_layour(rad_ids, class_names, class_id, box_ids, sorted(df.class_name.unique()))
    editor_content = get_editor_layout(
        tmp, sorted(df.class_name.unique()), modifiable=not args.readonly
    )
    return editor_content, name


@app.callback(
    Output("idx-input", "value"),
    Output("my-warning-2", "children"),
    Input("save-next-button", "n_clicks"),
    Input({"type": "footer-table-idx-button", "index": ALL}, "n_clicks"),
    Input("name-input", "value"),
    State("idx-input", "value"),
)
def save_and_next(n_clicks, n_clicks_values, name_input, current_idx_value):
    ctx = dash.callback_context

    if what_exactly_triggered(ctx, "name-input"):
        try:
            return d.get_idx_by_name(name_input), " "
        except ValueError:
            return 0, INVALID_NAME_MESSAGE

    if what_exactly_triggered(ctx, "footer-table-idx-button"):
        next_idx = eval(ctx.triggered[0]["prop_id"].split(".n_clicks")[0])["index"]
        return int(next_idx), ""

    if what_exactly_triggered(ctx, "save-next-button"):
        global df
        name = d.image_ids[current_idx_value]
        # set saved flag in df
        df.loc[df.image_id == name, DONE_COL_NAME] = int(True)
        df.to_csv(dump_csv_path, index=False)
        next_idx = (current_idx_value + 1) % len(d)
        logger.info(
            str(
                dict(
                    image_id=name,
                    action="save and next",
                    move_from=current_idx_value,
                    move_to=next_idx,
                )
            )
        )
        return next_idx, ""

    return current_idx_value, ""


@app.callback(
    Output("footer-table", "children"),
    Input("save-next-button", "n_clicks"),
    Input("idx-input", "value"),
)
def footer_table(n_clicks, idx_input):
    if idx_input is None or idx_input >= len(d) or idx_input < 0:
        return [INVALID_VALUE_MESSAGE]

    global df
    image_id2flag = df[["image_id", "done"]].groupby("image_id").any().to_dict()["done"]

    ids, flags = zip(*[(i, image_id2flag[name]) for i, name in enumerate(d.image_ids)])

    ids, flags = np.array(ids), np.array(flags)
    # report progress
    done_n = sum(flags)
    total_n = len(ids)
    progress_message = html.H4("{}/{} are reviewer.".format(done_n, total_n))

    # show only part of dataset in footer
    from_idx = max(0, idx_input - 15)
    to_idx = min(len(ids) - 1, idx_input + 150)
    ids = ids[from_idx:to_idx]
    flags = flags[from_idx:to_idx]

    return [progress_message, make_footer_table(ids, flags, idx_input)]


if __name__ == "__main__":
    app.run_server(debug=False, host=args.host, port=args.port)
