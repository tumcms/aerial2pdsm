import plotly.graph_objects as go
import numpy as np

# data = [
#     ("Zustandsmonitoring", 5, 6),
#     ("Modellierung", 8, 1.5),
#     ("Bauabweichung", 8, 4),
#     ("Prozessmonitoring", 3, 8),
#     ("Objektverortung", 1, 9),
#     ("Big Data", 3, 0.5),
#     ("Geländemodell", 5, 1),
#     ("Schadensanalyse", 9, 0.5),
#     ("Automatisierung", 8, 8),
#     ("Vorplanung", 1.5, 2)
# ]
data = [
    ("State Monitoring", 4.5, 5),
    ("Modeling", 7.5, 1.5),
    ("Deviation in Construction", 8, 4),
    ("Process Monitoring", 3, 8),
    ("Object Tracking", 1, 9),
    ("Big Data", 3, 0.5),
    ("Digital Surface Model", 5, 1),
    ("Damage Analysis", 9, 0.5),
    ("Automation", 7.5, 8),
    ("Preliminary Planning", 1.5, 2)
]


xe = [x[1] for x in data]
ye = [y[2] for y in data]
te = [t[0] for t in data]

fig = go.Figure(data=go.Scatter(
    y=ye,
    x=xe,
    text=te,
    mode='markers+text',
    textposition='bottom center',
    textfont=dict(
        # family="sans serif",
        family="Courier New, monospace",
        size=25,
        # color="LightSeaGreen"
    ),
    marker=dict(
        size=25,
        color=np.random.randn(100),  # set color equal to a variable
        colorscale='Portland',  # one of plotly colorscales
        showscale=False,
        line=dict(
            color='Gray',
            width=2
        )
    )
),
    layout=go.Layout(
        titlefont_size=32,
        font=dict(
            family="Courier New, monospace",
            size=25,
            color="#7f7f7f"
        ),
        title=f"<b>Applications in AEC</b>",
        yaxis={"title": "<b>% Temporal Resolution</b>", "range": [0, 10], },
        xaxis={"title": "<b>% Spatial Resolution</b>", "range": [0, 10]}
    )
)

fig.show()
