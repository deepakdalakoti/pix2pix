import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config["suppress_callback_exceptions"] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
