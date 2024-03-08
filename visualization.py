import os
from typing import List, Union

import numpy as np
import pandas as pd
import plotly.express as px


def save_interactive_scatter_plot(
        save_fpath: str,
        embeddings2d: np.ndarray,
        meta_df: pd.DataFrame,
        plot_label_type: str,
        hover_label_types: List[str]
):
    meta_df = meta_df.copy()

    x = embeddings2d[:, 0]
    y = embeddings2d[:, 1]

    custom_data = [meta_df[label_type] for label_type in hover_label_types]
    trace = ["ColX: %{x}", "ColY: %{y}"] + [label_type + ": %{customdata[" + str(i) + "]}" for i, label_type in enumerate(hover_label_types)]

    fig = px.scatter(
        x=x,
        y=y,
        color=meta_df[plot_label_type],
        color_discrete_map={'DMSO': 'black'},
        color_discrete_sequence=px.colors.qualitative.Light24,
        title=f"{plot_label_type}.html",
        custom_data=custom_data,
    )

    fig.update_xaxes(range=[x.min() - 0.5, x.max() + 0.5])
    fig.update_yaxes(range=[y.min() - 0.5, y.max() + 0.5])

    fig.update_traces(
        hovertemplate="<br>".join(trace)
    )
    fig.write_html(save_fpath, include_plotlyjs="cdn")


