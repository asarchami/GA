

import plotly.plotly as py
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='asarchami', api_key='gzblhndqc9')
data = [
    go.Heatmap(
        z=[[1, 20, 30],
           [20, 1, 60],
           [30, 60, 1]]
    )
]
py.iplot(data, filename='basic-heatmap')
