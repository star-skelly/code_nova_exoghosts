import dash
from dash import dcc, html, Input, Output
import dash_daq as daq
import plotly.graph_objs as go
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
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
# Load TESS heatmap and lightcurve data (robust path + checks)
# ----------------------------
_here = os.path.dirname(__file__)
_npz_path = os.path.join(_here, "tess_dashboard_data.npz")
dash_data = np.load(_npz_path)
period_grid = dash_data["period_grid"]
dur_grid = dash_data["dur_grid"]
scores_white = np.asarray(dash_data["scores_white"], dtype=float)
scores_est = np.asarray(dash_data["scores_est"], dtype=float)
cadence = dash_data["cadence"]
lc = dash_data["lc"]
alpha = float(dash_data["alpha"]) if "alpha" in dash_data else 0.01
true_period = float(dash_data["true_period"]) if "true_period" in dash_data else None
true_dur = float(dash_data["true_dur"]) if "true_dur" in dash_data else None
tid = '232616284'
phase_bins = dash_data["phase_bins"] if "phase_bins" in dash_data else None
pf_y = dash_data["pf_y"] if "pf_y" in dash_data else None
# Axes scaling to match matplotlib script
period_scale_days = 30 * 24
dur_scale_hours = 30
period_days = period_grid / period_scale_days
dur_hours = dur_grid / dur_scale_hours

# Ensure shapes: z must be (len(dur), len(period)) for Plotly
def _prepare_z(z):
    z = np.asarray(z, dtype=float)
    if not np.any(np.isfinite(z)):
        print("[dashboard] Warning: detection score array contains no finite values.")
    # Replace NaNs for rendering (keeps hover useful)
    z = np.nan_to_num(z, nan=np.nanmin(z) if np.isfinite(np.nanmin(z)) else 0.0)
    if z.shape == (len(period_grid), len(dur_grid)):
        z = z.T
    return z
_z_white = _prepare_z(scores_white)
_z_est = _prepare_z(scores_est)

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
    # If precomputed binned series available, plot as line for speed/clarity
    if phase_bins is not None and pf_y is not None:
        # nearest indices for current selection
        ip = _nearest(period_grid, period_samples)
        idu = _nearest(dur_grid, dur_samples)
        yb = pf_y[ip, idu]
        if np.any(np.isfinite(yb)):
            fig.add_trace(go.Scatter(x=phase_bins, y=yb, mode='lines', line=dict(color='#4c78a8', width=2), name='binned flux'))
    # Scatter overlay (thin) for context
    fig.add_trace(go.Scatter(x=ph[~mask], y=y_dt[~mask], mode='markers', marker=dict(size=2, color='rgba(76,120,168,0.35)'), name='out-of-transit'))
    fig.add_trace(go.Scatter(x=ph[mask], y=y_dt[mask], mode='markers', marker=dict(size=3, color='rgba(228,87,86,0.65)'), name='in-transit'))
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

def make_time_series(cad, flux, mask, title):
    time_days = cad / float(30 * 24)
    fig = go.Figure()
    # Unfolded view as scatter
    fig.add_trace(go.Scatter(x=time_days, y=flux, mode='markers',
                             marker=dict(size=2, color='rgba(76,120,168,0.6)'),
                             name='light curve'))
    if mask is not None and np.any(mask):
        fig.add_trace(go.Scatter(x=time_days[mask], y=flux[mask], mode='markers',
                                 marker=dict(size=4, color='rgba(228,87,86,0.9)'),
                                 name='in-transit'))
    fig.update_layout(
        title=title,
        xaxis_title='Time [days]',
        yaxis_title='Flux',
        template='plotly_dark',
        font=dict(family='Verdana', size=12, color='white'),
        height=360,
        margin=dict(l=60, r=30, t=60, b=60),
        title_font=dict(size=16, color='white', family='Verdana'),
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#111111'
    )
    fig.update_xaxes(showline=True, linewidth=1.0, linecolor='rgba(255,255,255,0.2)', mirror=True, ticks="inside",
                     showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False)
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
        z=z, x=period_days, y=dur_hours,
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
    # Title mapping without super-title
    tl = str(title_label).lower()
    fig.update_layout(
        xaxis_title='Period [days]',
        yaxis_title='Duration [hours]',
        template='plotly_dark',
        font=dict(family='Verdana', size=12, color='white'),
        margin=dict(l=60, r=40, t=10, b=40),
        height=420,
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

heatmap_white = make_heatmap(_z_white, "Baseline: white-noise detector score")
heatmap_est = make_heatmap(_z_est, "Adaptive stellar model detector score")

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
        html.P(id="obs-text", style={
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
            dcc.Graph(id='heatmap_white', figure=heatmap_white, config={"responsive": True}, style={"height": "420px"})
        ], style={"width": "48%"}),
        html.Div([
            html.H3("adaptive learned stellar model", style={"textAlign": "center", "color": "#d8dffb", "fontFamily": "Verdana, sans-serif"}),
            dcc.Graph(id='heatmap_est', figure=heatmap_est, config={"responsive": True}, style={"height": "420px"})
        ], style={"width": "48%"}),
    ], style={"display": "flex", "justifyContent": "space-between", "width": "90%", "margin": "12px auto 20px auto", "position": "relative"}),

    # clicked-data + note
    html.Div([
        html.Div(id="click-data"),
        html.Div(html.I("Note: red cross marks the true simulated transit parameters."), style={"marginTop": "6px"})
    ], style={
        "textAlign": "center", "marginTop": "16px", "color": "#d1d1d1", "fontFamily": "Verdana, sans-serif"
    }),

    # view mode toggle (compact pro switch)
    html.Div([
        html.Div([
            html.Span("View:", style={"marginRight": "10px", "fontWeight": "600", "color": "#d1d1d1"}),
            html.Span("Phase-folded", style={"marginRight": "10px", "color": "#d1d1d1"}),
            daq.BooleanSwitch(id='view-toggle', on=False, color="#4c78a8"),
            html.Span("Unfolded", style={"marginLeft": "10px", "color": "#d1d1d1"}),
        ], style={"display": "inline-flex", "alignItems": "center", "gap": "8px"}),
        html.Div(id='view-mode-help', style={"marginTop": "6px", "color": "#d1d1d1"})
    ], style={
        "textAlign": "center", "marginTop": "10px", "padding": "8px 12px",
        "borderRadius": "10px", "backgroundColor": "#121219", "display": "inline-block"
    }),

    # two lightcurves side-by-side
    html.Div([
        html.Div(dcc.Graph(id='lightcurve1', figure=fig_lc1), style={"width": "48%"}),
        html.Div(dcc.Graph(id='lightcurve2', figure=fig_lc2), style={"width": "48%"}),
    ], style={"display": "flex", "justifyContent": "space-between", "width": "90%", "margin": "24px auto", "position": "relative"})
], style={"backgroundColor": "#0d0d0d", "paddingBottom": "40px"})

@app.callback(
    [Output('lightcurve1', 'figure'),
     Output('lightcurve2', 'figure'),
     Output('click-data', 'children'),
     Output('obs-text', 'children'),
     Output('heatmap_white', 'figure'),
     Output('heatmap_est', 'figure'),
     Output('view-mode-help', 'children')],
    [Input('heatmap_white', 'clickData'),
     Input('heatmap_est', 'clickData'),
     Input('view-toggle', 'on')]
)
def update_lightcurves(cd_white, cd_est, view_toggle_on):
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

    view_label = 'Unfolded' if bool(view_toggle_on) else 'Phase-folded'
    if bool(view_toggle_on):
        # compute masks for highlighting
        _, m_w = centered_phase_fold(cadence, P_w, D_w, alpha)
        _, m_e = centered_phase_fold(cadence, P_e, D_e, alpha)
        fig1 = make_time_series(cadence, lc, m_w, "Unfolded (baseline white-noise)")
        fig2 = make_time_series(cadence, lc, m_e, "Unfolded (adaptive learned stellar model)")
    else:
        fig1 = make_phase_folded(cadence, lc, P_w, D_w, alpha, "Phase-folded (baseline white-noise)")
        fig2 = make_phase_folded(cadence, lc, P_e, D_e, alpha, "Phase-folded (adaptive learned stellar model)")

    click_text = html.I(
        f"White: period={period_days[iw]:.2f} d, dur={dur_hours[jw]:.2f} h | "
        f"Adaptive: period={period_days[ie]:.2f} d, dur={dur_hours[je]:.2f} h | "
        f"View: {view_label}",
        style={"color": "#d1d1d1", "fontFamily": "Verdana, sans-serif"}
    )
    # Observation details
    obs_lines = [
        f"TID: {tid}" if tid is not None else "TID: N/A",
        f"True period: {true_period/period_scale_days:.2f} d" if true_period is not None else "True period: N/A",
        f"True duration: {true_dur/dur_scale_hours:.2f} h" if true_dur is not None else "True duration: N/A",
        f"Cadence: 2 minutes; total baseline ≈ 90 days (3 sectors)"
    ]
    # Render without bullets: line-separated spans
    obs_elems = []
    for s in obs_lines:
        obs_elems.append(html.Span(s))
        obs_elems.append(html.Br())
    obs_text = html.Div(obs_elems)

    # Highlight selected cell on each heatmap
    def _add_cross(fig, i, j):
        x = period_days[i]
        y = dur_hours[j]
        fig.add_shape(type='rect', x0=x-(period_days[1]-period_days[0])/2, x1=x+(period_days[1]-period_days[0])/2,
                      y0=y-(dur_hours[1]-dur_hours[0])/2, y1=y+(dur_hours[1]-dur_hours[0])/2,
                      line=dict(color='#00e5ff', width=2))
        return fig

    hw = make_heatmap(_z_white, "baseline white-noise")
    he = make_heatmap(_z_est, "adaptive learned stellar model")
    hw = _add_cross(hw, iw, jw)
    he = _add_cross(he, ie, je)

    help_text = f"Currently showing: {view_label} view (use the switch to toggle)."

    return fig1, fig2, click_text, obs_text, hw, he, help_text

# ----------------------------
# Run app
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
