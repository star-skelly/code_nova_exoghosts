import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from noise_generator import centered_phase_fold, detrend

# ----------------------------
# Matplotlib-style formatting
# ----------------------------
mpl.rcParams.update({
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "lines.linewidth": 1.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Verdana"],
    "font.size": 10,
    "axes.labelsize": 15,
    "errorbar.capsize": 0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "text.usetex": True
})

# ----------------------------
# Load TESS heatmap and lightcurve data
# ----------------------------
dash_data = np.load("tess_dashboard_data.npz")
period_grid = dash_data["period_grid"]
dur_grid = dash_data["dur_grid"]
scores_white = dash_data["scores_white"]
scores_est = dash_data["scores_est"]
cadence = dash_data["cadence"]
lc = dash_data["lc"]
alpha = float(dash_data["alpha"]) if "alpha" in dash_data else 0.01
true_period = float(dash_data["true_period"]) if "true_period" in dash_data else None
true_dur = float(dash_data["true_dur"]) if "true_dur" in dash_data else None

# Axes scaling to match matplotlib script
period_scale_days = 30 * 24
dur_scale_hours = 30
period_days = period_grid / period_scale_days
dur_hours = dur_grid / dur_scale_hours

# Default selections: argmax in each heatmap
imax_w = np.unravel_index(np.argmax(scores_white), scores_white.shape)
imax_e = np.unravel_index(np.argmax(scores_est), scores_est.shape)

def _nearest(array, value):
    idx = int(np.argmin(np.abs(array - value)))
    return idx

def make_phase_folded(cad, flux, period_samples, dur_samples, alpha_val, title):
    ph, mask = centered_phase_fold(cad, period_samples, dur_samples, alpha_val)
    y_dt = detrend(cad, flux, deg=2, mask=mask)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ph[~mask], y=y_dt[~mask], mode='markers', marker=dict(size=3, color='#4c78a8'), name='out-of-transit'))
    fig.add_trace(go.Scatter(x=ph[mask], y=y_dt[mask], mode='markers', marker=dict(size=4, color='#e45756'), name='in-transit'))
    fig.update_layout(
        title=title,
        xaxis_title='Phase (centered)',
        yaxis_title='Flux (detrended)',
        template='plotly_dark',
        font=dict(family='Verdana', size=12, color='white'),
        height=360,
        margin=dict(l=60, r=30, t=60, b=60),
        title_font=dict(size=16, color='white', family='Verdana'),
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#111111'
    )
    fig.update_xaxes(showline=True, linewidth=1.0, linecolor='rgba(255,255,255,0.2)', mirror=True, ticks="inside",
                     showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False, range=[-0.5, 0.5])
    fig.update_yaxes(showline=True, linewidth=1.0, linecolor='rgba(255,255,255,0.2)', mirror=True, ticks="inside",
                     showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
    return fig

# ----------------------------
# Function to make lightcurve figures
# ----------------------------
def make_lightcurve(time, flux, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=flux,
        mode='lines',
        line=dict(color=color, width=2),
        name='Flux'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Time (days)',
        yaxis_title='Normalized Flux',
        template='plotly_dark',
        font=dict(family='Verdana', size=12, color='white'),
        margin=dict(l=60, r=30, t=60, b=60),
        title_font=dict(size=16, color='white', family='Verdana'),
        height=360,
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#111111'
    )
    fig.update_xaxes(
        showline=True, linewidth=1.0, linecolor='rgba(255,255,255,0.2)', mirror=True, ticks="inside",
        showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False
    )
    fig.update_yaxes(
        showline=True, linewidth=1.0, linecolor='rgba(255,255,255,0.2)', mirror=True, ticks="inside",
        showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False
    )
    return fig

# ----------------------------
# Initial phase-folded lightcurves from default selections
# ----------------------------
P_w0 = float(period_grid[imax_w[0]])
D_w0 = float(dur_grid[imax_w[1]])
P_e0 = float(period_grid[imax_e[0]])
D_e0 = float(dur_grid[imax_e[1]])
fig_lc1 = make_phase_folded(cadence, lc, P_w0, D_w0, alpha, "Phase-folded (baseline white-noise)")
fig_lc2 = make_phase_folded(cadence, lc, P_e0, D_e0, alpha, "Phase-folded (adaptive learned stellar model)")

def make_heatmap(z, title_label):
    fig = go.Figure(data=go.Heatmap(
        z=z.T, x=period_days, y=dur_hours,
        colorscale=[
            [0.0, "#0b0b3b"],
            [0.25, "#3a0ca3"],
            [0.5, "#7209b7"],
            [0.75, "#b5179e"],
            [1.0, "#ff5f1f"]
        ],
        showscale=False,
        hovertemplate='Period=%{x:.2f} d<br>Duration=%{y:.2f} h<br>Detection score=%{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title='Period [days]',
        yaxis_title='Duration [hours]',
        template='plotly_dark',
        font=dict(family='Verdana', size=12, color='white'),
        margin=dict(l=60, r=40, t=10, b=60),
        height=520,
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#111111',
    )
    fig.update_xaxes(showline=True, linewidth=1.0, linecolor='rgba(255,255,255,0.12)', mirror=True, ticks="inside",
                     showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False, automargin=True)
    fig.update_yaxes(showline=True, linewidth=1.0, linecolor='rgba(255,255,255,0.12)', mirror=True, ticks="inside",
                     showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False, automargin=True)
    # overlay true parameter cross
    if true_period is not None and true_dur is not None:
        x0 = true_period / period_scale_days
        y0 = true_dur / dur_scale_hours
        fig.add_shape(type='line', x0=x0, x1=x0, y0=dur_hours.min(), y1=dur_hours.max(), line=dict(color='red', width=1))
        fig.add_shape(type='line', x0=period_days.min(), x1=period_days.max(), y0=y0, y1=y0, line=dict(color='red', width=1))
        fig.add_trace(go.Scatter(x=[x0], y=[y0], mode='markers', marker=dict(symbol='x', size=10, color='red'), showlegend=False))
    return fig

heatmap_white = make_heatmap(scores_white, "baseline white-noise")
heatmap_est = make_heatmap(scores_est, "adaptive learned stellar model")

# ----------------------------
# Dash app layout
# ----------------------------
app = dash.Dash(__name__)
app.title = "ExoGhost Viewer"

app.layout = html.Div([
    html.H1("EXOGHOST LIGHTCURVE VIEWER", style={
        "textAlign": "center",
        "fontFamily": "Verdana, sans-serif",
        "fontWeight": "normal",
        "color": "white",
        "marginBottom": "8px",
        "marginTop": "8px"
    }),

    # Observation details section
    html.Div(id="observation-details", children=[
        html.H3("Observation Details", style={
            "marginTop": "6px", "marginBottom": "6px",
            "color": "#d8dffb", "fontFamily": "Verdana, sans-serif"
        }),
        html.P("Fill this area with target name, exposure, pipeline notes, etc.", style={
            "color": "#bdbdbd", "fontFamily": "Verdana, sans-serif"
        })
    ], style={
        "backgroundColor": "#121219",
        "borderRadius": "10px",
        "padding": "14px",
        "width": "78%",
        "margin": "12px auto",
        "textAlign": "center",
        "boxShadow": "0 0 18px rgba(80,200,255,0.05)"
    }),

    # centered helper text
    html.Div(html.H4("Click a region on the heatmap to view its light curves", style={
        "textAlign": "center", "color": "#cfcfcf", "fontStyle": "italic",
        "marginTop": "8px", "fontFamily": "Verdana, sans-serif"
    })),

    # Two heatmaps side-by-side
    html.Div([
        html.Div([
            html.H3("baseline white-noise", style={"textAlign": "center", "color": "#d8dffb", "fontFamily": "Verdana, sans-serif"}),
            dcc.Graph(id='heatmap_white', figure=heatmap_white, config={"responsive": True})
        ], style={"width": "48%"}),
        html.Div([
            html.H3("adaptive learned stellar model", style={"textAlign": "center", "color": "#d8dffb", "fontFamily": "Verdana, sans-serif"}),
            dcc.Graph(id='heatmap_est', figure=heatmap_est, config={"responsive": True})
        ], style={"width": "48%"}),
    ], style={"display": "flex", "justifyContent": "space-between", "width": "90%", "margin": "12px auto"}),

    # clicked-data text area
    html.Div(id="click-data", style={
        "textAlign": "center", "marginTop": "16px", "color": "#d1d1d1", "fontFamily": "Verdana, sans-serif"
    }),

    # two lightcurves side-by-side
    html.Div([
        html.Div(dcc.Graph(id='lightcurve1', figure=fig_lc1), style={"width": "48%"}),
        html.Div(dcc.Graph(id='lightcurve2', figure=fig_lc2), style={"width": "48%"}),
    ], style={"display": "flex", "justifyContent": "space-between", "width": "90%", "margin": "18px auto"})
], style={"backgroundColor": "#0d0d0d", "paddingBottom": "40px"})

@app.callback(
    [Output('lightcurve1', 'figure'),
     Output('lightcurve2', 'figure'),
     Output('click-data', 'children')],
    [Input('heatmap_white', 'clickData'),
     Input('heatmap_est', 'clickData')]
)
def update_lightcurves(cd_white, cd_est):
    # White selection
    if cd_white and 'points' in cd_white and cd_white['points']:
        x_days = float(cd_white['points'][0]['x'])
        y_hours = float(cd_white['points'][0]['y'])
        iw = _nearest(period_days, x_days)
        jw = _nearest(dur_hours, y_hours)
    else:
        iw, jw = imax_w
    P_w = float(period_grid[iw])
    D_w = float(dur_grid[jw])

    # Est selection
    if cd_est and 'points' in cd_est and cd_est['points']:
        x_days = float(cd_est['points'][0]['x'])
        y_hours = float(cd_est['points'][0]['y'])
        ie = _nearest(period_days, x_days)
        je = _nearest(dur_hours, y_hours)
    else:
        ie, je = imax_e
    P_e = float(period_grid[ie])
    D_e = float(dur_grid[je])

    fig1 = make_phase_folded(cadence, lc, P_w, D_w, alpha, "Phase-folded (baseline white-noise)")
    fig2 = make_phase_folded(cadence, lc, P_e, D_e, alpha, "Phase-folded (adaptive learned stellar model)")

    click_text = html.I(
        f"White: period={period_days[iw]:.2f} d, dur={dur_hours[jw]:.2f} h | "
        f"Adaptive: period={period_days[ie]:.2f} d, dur={dur_hours[je]:.2f} h",
        style={"color": "#d1d1d1", "fontFamily": "Verdana, sans-serif"}
    )

    return fig1, fig2, click_text

# ----------------------------
# Run app
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
