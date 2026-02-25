import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import json
import os

# --- DATA LOADING ---
def load_episode_data(episode_num):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if episode_num == "Full":
        filename = "starwars-full-interactions-allCharacters.json"
    else:
        filename = f"starwars-episode-{episode_num}-interactions-allCharacters.json"
    
    file_path = os.path.join(base_dir, filename)
    # Using utf-8 encoding to ensure characters are read correctly
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- GRAPH GENERATION ---
def create_network_graph(data, min_weight=1, highlight_node=None):
    nodes = data['nodes']
    # Filtering links based on edge weight (Assignment Option 1) [cite: 43]
    links = [l for l in data['links'] if l['value'] >= min_weight]
    
    G = nx.Graph()
    for i, node in enumerate(nodes):
        # Storing original suggested colors and values [cite: 22, 23, 24]
        G.add_node(i, name=node['name'], val=node['value'], color=node['colour'])
    
    for link in links:
        G.add_edge(link['source'], link['target'], weight=link['value'])

    # Node-link diagram layout using spring layout [cite: 35]
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # 1. VISIBLE EDGES
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none', 
        mode='lines')

    # 2. EDGE HOVER POINTS (Details-on-demand for edges) 
    mid_x, mid_y, mid_text = [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Tooltips at the midpoint of each connection
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        # Showing both character names and the shared scene value 
        char1 = G.nodes[edge[0]]['name']
        char2 = G.nodes[edge[1]]['name']
        mid_text.append(f"Connection: {char1} & {char2}<br>Shared Scenes: {edge[2]['weight']}")

    edge_hover_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode='markers',
        hoverinfo='text',
        text=mid_text,
        marker=dict(size=12, color='rgba(0,0,0,0)')) # Transparent trigger

    # 3. NODES (Details-on-demand and Brushing) [cite: 39, 40]
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Showing character name and scene count 
        node_text.append(f"Character: {G.nodes[node]['name']}<br>Scenes: {G.nodes[node]['val']}")
        
        # Brushing and Linking: Highlight character in red if selected 
        if highlight_node and G.nodes[node]['name'] == highlight_node:
            node_color.append('#FF0000') 
            node_size.append(25)
        else:
            # Use suggested character colors from data 
            node_color.append(G.nodes[node]['color'])
            node_size.append(15)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(showscale=False, color=node_color, size=node_size, line_width=2))

    # 4. CREATE FIGURE
    fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                 layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    height=600, # Expanded height for large screens [cite: 38]
                    margin=dict(b=0,l=0,r=0,t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig

# --- DASH APP ---
app = dash.Dash(__name__)

app.layout = html.Div([
    # Required dcc.Store is now inside the layout to avoid AttributeErrors
    dcc.Store(id='selected-character-store'),
    
    html.H1("Star Wars Interaction Network", style={'textAlign': 'center'}),
    
    html.Div([
        # Control Panel [cite: 37, 38]
        html.Div([
            html.H3("Controls"),
            html.Label("Filter Edge Weight (Option 1):"),
            dcc.Slider(id='weight-slider', min=1, max=30, step=1, value=1, 
                       marks={i: str(i) for i in range(1, 31, 5)}),
            html.Br(),
            html.Label("Select Comparison Episodes:"),
            dcc.Dropdown(id='ep-select-1', 
                         options=[{'label': f'Episode {i}', 'value': i} for i in range(1, 8)] + [{'label': 'Full Saga', 'value': 'Full'}], 
                         value=1),
            html.Br(),
            dcc.Dropdown(id='ep-select-2', 
                         options=[{'label': f'Episode {i}', 'value': i} for i in range(1, 8)] + [{'label': 'Full Saga', 'value': 'Full'}], 
                         value=2),
        ], style={'width': '18%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f2f2f2', 'height': '800px'}),
        
        # Two diagram instances for comparison [cite: 36]
        html.Div([
            dcc.Graph(id='network-1', style={'width': '50%', 'display': 'inline-block', 'height': '800px'}),
            dcc.Graph(id='network-2', style={'width': '50%', 'display': 'inline-block', 'height': '800px'}),
        ], style={'width': '80%', 'display': 'inline-block'})
    ], style={'display': 'flex'})
])

@app.callback(
    [Output('network-1', 'figure'), Output('network-2', 'figure'), Output('selected-character-store', 'data')],
    [Input('ep-select-1', 'value'), Input('ep-select-2', 'value'), 
     Input('weight-slider', 'value'), Input('network-1', 'clickData'), Input('network-2', 'clickData')],
    [State('selected-character-store', 'data')]
)
def update_graphs(ep1, ep2, weight, click1, click2, current_store):
    ctx = dash.callback_context
    clicked_char = current_store
    
    # Brushing and Linking logic: matching by character name 
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id']
        if 'network-1.clickData' in trigger_id and click1:
            clicked_char = click1['points'][0]['text'].split('<br>')[0].replace("Character: ", "")
        elif 'network-2.clickData' in trigger_id and click2:
            clicked_char = click2['points'][0]['text'].split('<br>')[0].replace("Character: ", "")

    data1 = load_episode_data(ep1)
    data2 = load_episode_data(ep2)
    
    return create_network_graph(data1, weight, clicked_char), \
           create_network_graph(data2, weight, clicked_char), \
           clicked_char

if __name__ == '__main__':     
    app.run(debug=True)