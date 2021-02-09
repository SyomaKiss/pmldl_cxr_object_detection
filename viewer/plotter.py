import plotly.express as px
import plotly.graph_objects as go


def get_plot(img, bboxes, class_names, rad_ids, rad2color):
    fig = px.imshow(img)
    # Add image
    img_height, img_width, ch = img.shape
    fig.update_layout(
        width=img_width*2,
        height=img_height*2,)

    for box, rad_id, class_name in zip(bboxes, rad_ids, class_names):
        if 'No finding' in class_name:
            fig.add_trace(go.Scatter(x=[100],y=[100],
                                     name = f'{rad_id}, {class_name}',
                                     opacity=0,
                                    ))
        else:
            x0, y0, x1, y1 = box
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                hovertemplate= f'{rad_id}<br>{class_name}',
                text=[class_name],
                name = f'{rad_id}, {class_name}',
                line=dict(width=2, color=rad2color[rad_id]),
                fillcolor=rad2color[rad_id],
                mode="lines+text",

                opacity=1,
            ))

    return fig