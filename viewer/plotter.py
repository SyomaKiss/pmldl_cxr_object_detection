import plotly.express as px
import plotly.graph_objects as go



def get_contrasted_color(key):
    colors = [
        "#f58231",
        "#FFFF00",
        "#1CE6FF",
        "#FF34FF",
        "#FF4A46",
        "#008941",
        "#006FA6",
        "#A30059",
        "#FFDBE5",
        "#7A4900",
        "#0000A6",
        "#63FFAC",
        "#B79762",
        "#004D43",
        "#8FB0FF",
        "#997D87",
        "#5A0007",
    ]  # set of most contrasted colors
    rad_ids = ['R11',
               'R7',
               'R10',
               'R9',
               'R17',
               'R3',
               'R8',
               'R6',
               'R5',
               'R4',
               'R2',
               'R16',
               'R1',
               'R15',
               'R13',
               'R12',
               'R14']
    rad2color = dict(zip(rad_ids, colors))
    color = colors[0]
    try:
        color = rad2color[key]
        return color
    except KeyError:
        return color


def get_plot(img, bboxes, class_names, rad_ids, box_ids, width=600):
    fig = px.imshow(img,  binary_string=True)
    # Add image
    img_height, img_width = img.shape[:2]
    fig.update_layout(
        width=width,
        height=width * img_height/img_width)

    for box, rad_id, class_name, box_id in zip(bboxes, rad_ids, class_names, box_ids):
        if 'No finding' in class_name:
            fig.add_trace(go.Scatter(x=[300],y=[300],
                                     name = f'{rad_id}, {class_name}',
                                     opacity=0,
                                    ))
        else:
            x0, y0, x1, y1 = box
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                uid=int(box_id),
                hovertemplate= f'<b>{box_id}</b><br> {rad_id}<br>{class_name}',
                text=[class_name],
                name = f'{rad_id}, {class_name}',
                line=dict(width=2, color=get_contrasted_color(rad_id)),
                fillcolor=get_contrasted_color(rad_id),
                mode="lines+text",

                opacity=1,
            ))

    return fig

def hide_boxes(fig, flag):
    value = True if flag else "legendonly"
    def foo(x):
        x.visible = value

    fig = fig.for_each_trace(foo, selector=dict(type="scatter"))
    return fig