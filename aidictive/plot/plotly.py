
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.io as pio

from aidictive.plot import Plot
from aidictive.utils import is_jupyter_notebook


DEFAULT_LAYOUT = {
    "hovermode": "closest"
}


if is_jupyter_notebook():
    pass#init_notebook_mode(connected=True)


class PlotlyPlot(Plot):

    def __init__(self, obj, layout=None):
        lay = DEFAULT_LAYOUT.copy()
        if layout is not None:
            lay.update(layout)
        self.data = []
        self.layout = lay
        self.append(obj)

    @classmethod
    def copy(cls, obj):
        return cls(obj.data, obj.layout)

    def update_layout(self, *args, **kwargs):
        lay = kwargs
        if len(args) > 0:
            assert type(args[0]) == dict
            lay.update(args[0])
        self.layout.update(lay)
        return self

    def append(self, obj):
        out = self
        if type(obj) == list:
            self.data += obj
        elif type(obj) == PlotlyPlot:
            lay = self.layout.copy()
            lay.update(obj.layout)
            dat = self.data + obj.data
            out = PlotlyPlot(dat, lay)
        else:
            self.data.append(obj)
        return out

    def __call__(self, *args, **kwargs):
        c = PlotlyPlot.copy(self)
        return c.update_layout(*args, **kwargs)

    def __add__(self, data):
        return self.append(data)

    def __radd__(self, data):
        return self.append(data)

    def get_dict(self):
        d = {
            "data": self.data,
            "layout": self.layout,
        }
        return d

    def get_html(self, include_plotlyjs=True):
        info = self.get_dict()
        html = plot(info,
                    include_plotlyjs=include_plotlyjs,
                    output_type="div")
        return html

    def _repr_html_(self):
        pio.show(self.get_dict())

    def save(self, filename="plot.html", auto_open=True,
             include_plotlyjs=True, **kwargs):
        return plot(self.data, filename=filename,
                    include_plotlyjs=include_plotlyjs,
                    auto_open=auto_open, **kwargs)


def cmap2colorscale(cmap_name):
    import matplotlib.pyplot as plt

    cm = plt.get_cmap(cmap_name).colors
    n = len(cm) - 1
    colorscale = [(i / n, f"rgb{tuple(c)}") for i, c in enumerate(cm)]
    return colorscale

def cmap2colorway(cmap_name):
    import matplotlib.pyplot as plt

    cm = plt.get_cmap(cmap_name).colors
    colorway = ["#%02x%02x%02x" % i for i in cm]
    return colorway

def prepare_plot(plot, data, layout=None):
    if plot is None:
        plot = PlotlyPlot(data, layout=layout)
    else:
        plot += data
        if layout is not None:
            plot.update_layout(layout)
    return plot

def prepare_params(kwargs, default_dict):
    default_dict.update(kwargs)
    return default_dict

def scatter(x, y, plot=None, **kwargs):
    # Set default parameters.
    kwargs = prepare_params(kwargs, {
        "mode": "markers",
        "hoverinfo": "x+y"
    })
    # Create scatter plot.
    trace = go.Scatter(x=x, y=y, **kwargs)
    # Add to current/create plot and return.
    return prepare_plot(plot, trace)

def line(x, y, plot=None, **kwargs):
    # Set default parameters.
    kwargs = prepare_params(kwargs, {
        "mode": "lines",
        "hoverinfo": "x+y"
    })
    # Create scatter plot.
    trace = go.Scatter(x=x, y=y, **kwargs)
    # Add to current/create plot and return.
    return prepare_plot(plot, trace)

def bar(x, y, plot=None, x_cmap=None, y_cmap=None, color=None, **kwargs):
    # Set default parameters.
    kwargs = prepare_params(kwargs, {
        "hoverinfo": "x+y",
        "marker": {}
    })
    if x_cmap is not None:
        cmap = cmap2colorscale(x_cmap) if type(x_cmap) == str else x_cmap
        n = len(cmap)
        if color is None:
            color = [cmap[i % n][1] for i in range(len(x))]
        else:
            color = [cmap[i % n][1] for i in color]
        kwargs["marker"].update(dict(
            color=color
        ))
    else:  # x_cmap is None
        kwargs["marker"].update(dict(
            color=color
        ))
    if y_cmap is not None:
        cmap = cmap2colorscale(y_cmap) if type(y_cmap) == str else y_cmap
        kwargs["marker"].update(dict(
            color=y,
            colorscale=cmap
        ))
    # Create scatter plot.
    trace = go.Bar(x=x, y=y, **kwargs)
    # Add to current/create plot and return.
    return prepare_plot(plot, trace)
