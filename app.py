import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import dcc, html, dash_table
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy import MemoryCacheHandler
from io import BytesIO
from wordcloud import WordCloud
import base64
import datetime
import os


def create_dfs_from_spotify():
    # Create dictionaries
    for term in ranges:
        global list_length_tracks
        global list_length_artists
        list_length_tracks = min(list_length_tracks, len(spotify.current_user_top_tracks(time_range=term, limit=50)['items']))
        list_length_artists = min(list_length_artists, len(spotify.current_user_top_artists(time_range=term, limit=50)['items']))
        results_tracks = spotify.current_user_top_tracks(time_range=term, limit=list_length_tracks - 1)
        results_artists = spotify.current_user_top_artists(time_range=term, limit=list_length_artists - 1)

        for i, item in enumerate(results_artists['items']):
            dict_top_artists['artist'].append(item['name'])
            dict_top_artists['term'].append(term)
            dict_top_artists['genres'].append(item['genres'])
            dict_top_artists['popularity'].append(item['popularity'])
            dict_top_artists['image'].append(item['images'][0]['url'])

        for i, item in enumerate(results_tracks['items']):
            artist_and_track = item['artists'][0]['name'] + ' - ' + item['name']
            dict_top_tracks['song'].append(artist_and_track)
            dict_top_tracks['term'].append(term)
            dict_audio_features[term].append(spotify.audio_features(item['id']))
            dict_top_tracks['image'].append(item['album']['images'][0]['url'])
            dict_playlists_id[term].append(item['id'])

    for term in ranges:
        for i in range(0, list_length_artists - 1):
            dict_top_artists['Platz'].append(i + 1)

        for i in range(0, list_length_tracks - 1):
            dict_top_tracks['Platz'].append(i + 1)
            dict_top_tracks['danceability'].append(dict_audio_features[term][i][0]['danceability'])
            dict_top_tracks['acousticness'].append(dict_audio_features[term][i][0]['acousticness'])
            dict_top_tracks['loudness'].append(dict_audio_features[term][i][0]['loudness'])
            dict_top_tracks['tempo'].append(dict_audio_features[term][i][0]['tempo'])
            dict_top_tracks['valence'].append(dict_audio_features[term][i][0]['valence'])

    result_playlists = spotify.current_user_playlists()
    for i, item in enumerate(result_playlists['items']):
        dict_playlists['  '].append('')
        dict_playlists['playlist name'].append(item['name'])
        dict_playlists['# tracks'].append(item['tracks']['total'])
        dict_playlists['follower'].append(spotify.playlist(item['id'])['followers']['total'])
        dict_playlists['owner'].append(item['owner']['display_name'])
        dict_playlists['public'].append(item['public'])
        dict_playlists['collaborative'].append(item['collaborative'])


    ' 1./2./3.: Create General Dataframes '
    df_top_tracks = pd.DataFrame(dict_top_tracks)
    df_top_artists = pd.DataFrame(dict_top_artists)
    df_all_playlists = pd.DataFrame(dict_playlists)
    df_all_playlists.sort_values(by=['follower'], ascending=False, inplace=True)
    df_top_tracks['term'].replace(['short_term', 'medium_term', 'long_term'], ['short', 'medium', 'long'], inplace=True)
    df_top_artists['term'].replace(['short_term', 'medium_term', 'long_term'], ['short', 'medium', 'long'], inplace=True)

    ' 4. TRACKS Dataframe'
    df_tracks_html = df_top_tracks.drop(['danceability', 'acousticness', 'loudness', 'tempo', 'valence'], axis=1)
    df_tracks_html = df_tracks_html.pivot(index='Platz', columns='term', values='song')
    df_tracks_html['#'] = list(range(1, list_length_tracks))
    df_tracks_html = df_tracks_html[['#', 'short', 'medium', 'long']]
    df_tracks_html['short'] = df_tracks_html['#'].astype(str) + '. ' + df_tracks_html['short']
    df_tracks_html['medium'] = df_tracks_html['#'].astype(str) + '. ' + df_tracks_html['medium']
    df_tracks_html['long'] = df_tracks_html['#'].astype(str) + '. ' + df_tracks_html['long']
    df_tracks_html = df_tracks_html.drop('#', axis=1)

    ' 5. ARTISTS  dataframe '
    df_art_html = df_top_artists.drop(['genres', 'popularity'], axis=1)
    df_art_html = df_art_html.pivot(index='Platz', columns='term', values='artist')
    df_art_html['#'] = list(range(1, list_length_artists))
    df_art_html = df_art_html[['#', 'short', 'medium', 'long']]
    df_art_html['short'] = df_art_html['#'].astype(str) + '. ' + df_art_html['short']
    df_art_html['medium'] = df_art_html['#'].astype(str) + '. ' + df_art_html['medium']
    df_art_html['long'] = df_art_html['#'].astype(str) + '. ' + df_art_html['long']
    df_art_html = df_art_html.drop('#', axis=1)

    ' 6. TRACKS Timeline dataframe '
    df_tracks_norm = df_top_tracks.copy()
    df_interim = df_tracks_norm[['danceability', 'acousticness', 'loudness', 'tempo', 'valence']]
    df_tracks_norm.loc[:, ['danceability', 'acousticness', 'loudness', 'tempo', 'valence']] = (df_interim - df_interim.min()) / (df_interim.max() - df_interim.min())
    df_tracks_time = pd.DataFrame()
    for term in ['short', 'medium', 'long']:
        df_tracks_time[term] = df_tracks_norm[df_tracks_norm['term'] == term][['danceability', 'acousticness', 'loudness', 'tempo', 'valence']].mean(axis=0)
    df_tracks_time.rename(index={'danceability': 'dance', 'acousticness': 'acoustic', 'loudness': 'loud', 'tempo': 'tempo', 'valence': 'valence'}, inplace=True)

    ' 7. ARTISTS Genre dataframe '
    df_art_genre = df_top_artists.copy()
    df_art_genre.drop(['popularity', 'Platz', 'artist', 'term', 'image'], axis=1, inplace=True)
    df_art_genre = df_art_genre.explode('genres')
    df_art_genre.dropna(inplace=True)
    df_art_genre = df_art_genre.groupby(df_art_genre.columns.tolist(), as_index=False).size().reset_index()
    df_art_genre.sort_values(by=['size'], ascending=True, inplace=True)

    ' 8. KONZERTE '
    df_konz = pd.read_excel(r"C:\Users\msandner\PycharmProjects\spotify\konzert_db.xlsx")

    return df_top_tracks, df_top_artists, df_all_playlists, df_tracks_html, df_art_html, df_tracks_time, df_art_genre, df_konz


def create_playlist(term):
    spotify.user_playlist_create(current_user, str(datetime.datetime.today().strftime('%Y_%m_%d')) + '_Top List - ' + str(term) + ' term', public=False, collaborative=False, description='')
    spotify.playlist_add_items(spotify.current_user_playlists()['items'][0]['id'], dict_playlists_id[term+'_term'])


def get_genre_wordcloud(category):
    #spotify_mask = np.array(Image.open("spotify_mask.png"))
    data = df_artists_genre[['genres', 'size']]
    wc_dict = {a: x for a, x in data.values}
    #wc = WordCloud(background_color=colors['background_hell'], mask=spotify_mask, color_func=lambda *args, **kwargs: colors['spotify_green'])
    if category == 'tracks':
        wc = WordCloud(background_color=colors['background_hell'], width=width_wordcloud, height=height_wordcloud_tracks, color_func=lambda *args, **kwargs: colors['spotify_green'])
    #if type == 'artists':
    else:
        wc = WordCloud(background_color=colors['background_hell'], width=width_wordcloud, height=height_wordcloud_artists, color_func=lambda *args, **kwargs: colors['spotify_green'])
    wc.fit_words(wc_dict)
    img = BytesIO()
    wc.to_image().save(img, format='PNG')
    wordcloud_img = 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    return wordcloud_img


def get_cover_img(category, term, position):
    if category == 'artists':
        return html.Div(html.Img(src=df_artists.loc[(df_artists['term'] == term) & (df_artists['Platz'] == position)].iloc[0]['image'],
                                 style={'width': img_size, 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-bottom': '10pt'}))

    if category == 'tracks':
        return html.Div(html.Img(src=df_tracks.loc[(df_artists['term'] == term) & (df_tracks['Platz'] == position)].iloc[0]['image'],
                                 style={'width': img_size, 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-bottom': '10pt'}))


def get_top_list(df, category, term):
    table = html.Table([
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][term], style={'width': '750px', 'padding-top': '.3em', 'padding-bottom': '.3em'}, id='tooltip_' + str(i) + category + term),
                #dbc.Tooltip(get_cover_img(category, term, i+1), target='tooltip_' + str(i) + category + term, placement='right')
            ], style={'color': 'grey', 'border-bottom': '1px solid ' + colors['background_hell']}) for i in range(len(df))
        ], style={'display': 'block', 'overflow-x': 'hidden', 'overflow-y': 'auto', 'height': height_html_table,
                  'font-family': 'Verdana', 'font-size': fontsize})
    ], style={'border-collapse': 'collapse', 'margin-top': 10})

    return table


def get_playlist_button():
    return html.Div([dbc.Button('> create Spotify playlist <', id='playlist_button',
                    style={'margin-top': 15, 'font-size': fontsize, 'font-family': 'Verdana', 'background': colors['spotify_green'], 'color': colors['background_dunkel'], 'padding': '5px'}),
           html.Div(id="playlist_button_output", style={"text-align": "center", 'font-size': fontsize})])


def get_tracks_circle_graph(x_data, y_data, x_range, y_range):
    return html.Div([
        dcc.Graph(
            figure={
                'data': [
                    dict(
                        x=df_tracks[df_tracks['term'] == i][x_data],
                        y=df_tracks[df_tracks['term'] == i][y_data],
                        text=df_tracks[df_tracks['term'] == i]['song'],
                        mode='markers',
                        marker={'size': marker_size, 'color': colors[i]},
                        name=i
                    ) for i in df_tracks['term'].unique()
                ],
                'layout': dict(
                    xaxis={'title': x_data, 'range': x_range, 'gridcolor': colors['background_hell']},
                    yaxis={'title': y_data, 'range': y_range, 'gridcolor': colors['background_hell']},
                    margin={'l': 45, 'b': 45, 't': 45, 'r': 20},
                    legend={'x': 0.005, 'y': 1.1, 'orientation': 'h'},
                    hovermode='closest',
                    plot_bgcolor=colors['background_dunkel'],
                    paper_bgcolor=colors['background_dunkel'],
                    font={'color': 'grey', 'family': 'Verdana', 'size': fontsize},
                    dragmode='pan',
                )
            },
            config={'displayModeBar': False, 'scrollZoom': True},
            style={'minWidth': '100%', 'height': height_graphs}
        )
    ])


def get_tracks_timeline_graph():
    return html.Div([dcc.Graph(
        figure={
            'data': [
                dict(
                    x=list(df_tracks_timeline.index),
                    y=df_tracks_timeline[i],
                    type='line',
                    name=i,
                    marker={'color': colors[i]}
                ) for i in list(df_tracks_timeline.columns)
            ],
            'layout': dict(
                xaxis={'title': 'feature', 'gridcolor': colors['background_hell']},
                yaxis={'title': 'mean value', 'range': [0, 1], 'tickvals': [0.25, 0.5, 0.75, 1],
                       'gridcolor': colors['background_hell']},
                margin={'l': 45, 'b': 45, 't': 45, 'r': 20},
                legend={'x': 0.005, 'y': 1.1, 'orientation': 'h'},
                hovermode='closest',
                plot_bgcolor=colors['background_dunkel'],
                paper_bgcolor=colors['background_dunkel'],
                font={'color': 'grey', 'family': 'Verdana', 'size': fontsize}
            )
        },
        config={'displayModeBar': False},
        style={'minWidth': '100%', 'height': height_graphs}
    )
    ])


def get_artists_circle_graph(term):
    return html.Div([
        dcc.Graph(
            figure={
                'data': [
                    dict(
                        x=df_artists[df_artists['term'] == term]['Platz'][:10][::-1],
                        y=df_artists[df_artists['term'] == term]['popularity'][:10][::-1],
                        text=df_artists[df_artists['term'] == term]['artist'][:10][::-1],
                        textposition='top center',
                        textfont={'size': 9},
                        hovertext=df_artists[df_artists['term'] == term]['Platz'][:10][::-1].astype(str) + '. ' + df_artists[df_artists['term'] == term]['artist'][:10][::-1] + '<br />popularity: ' + df_artists[df_artists['term'] == term]['popularity'][:10][::-1].astype(str),
                        hoverinfo='text',
                        mode='markers+text',
                        marker={'size': 16, 'color': df_artists[df_artists['term'] == term]['popularity'][:10][::-1], 'colorscale': 'Viridis'},
                        name=term
                    )
                ],
                'layout': dict(
                    xaxis={'title': 'ranking', 'range': [11, 0], 'tickvals': [9, 7, 5, 3, 1], 'gridcolor': colors['background_hell'], 'zeroline': False},
                    yaxis={'title': 'popularity', 'range': [0, 100], 'tickvals': [25, 50, 75, 100], 'gridcolor': colors['background_hell']},
                    margin={'l': 45, 'b': 45, 't': 45, 'r': 20},
                    hovermode='closest',
                    plot_bgcolor=colors['background_dunkel'],
                    paper_bgcolor=colors['background_dunkel'],
                    font={'color': 'grey', 'family': 'Verdana', 'size': fontsize}
                )
            },
            config={'displayModeBar': False},
            style={'minWidth': '100%', 'height': height_graphs}
        )
    ])


def get_playlists_datatable():
    table = dash_table.DataTable(
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": False} for i in df_playlists.columns
        ],
        data=df_playlists.to_dict('records'),
        page_size=10000,
        style_table={'height': height_graphs-20, 'minWidth': '100%'},
        fixed_rows={'headers': True},
        style_as_list_view=True,
        style_header={'font-family': 'Verdana', 'font-size': fontsize, 'color': 'grey', 'fontWeight': 'bold', 'border': '1px solid'},
        style_cell={'font-family': 'Verdana', 'font-size': fontsize, 'color': 'grey', 'backgroundColor': colors['background_dunkel'], 'border-bottom': '1px solid ' + colors['background_hell'], 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'left'},
        style_cell_conditional=[
            {'if': {'column_id': '  '}, 'width': '3%'},
            {'if': {'column_id': '# tracks'}, 'width': '10%', 'textAlign': 'center'},
            {'if': {'column_id': 'follower'}, 'width': '10%', 'textAlign': 'center'},
            {'if': {'column_id': 'owner'}, 'width': '16%', 'textAlign': 'center'},
            {'if': {'column_id': 'public'}, 'width': '10%', 'textAlign': 'center'},
            {'if': {'column_id': 'collaborative'}, 'width': '12%', 'textAlign': 'center'},
        ]
    )
    return html.Div(table, style={'margin-top': '20px'})


#######################################################################

''' CODE start '''
''' Variablen '''
colors = {'short': '#5BC0DE', 'medium': '#01D857', 'long': '#FDE724',
          'background_dunkel': '#222222', 'background_hell': '#303030',
          'spotify_green': '#01D857'}
client_id = os.environ["CLIENT_ID"]
client_secret = os.environ["CLIENT_SECRET"]
redirect_uri = os.environ["REDIRECT_URL"]
callback_uri = os.environ["CALLBACK_URL"]
scope = os.environ["SCOPE"]
token = os.environ["TOKEN"]
username = os.environ["USER_NAME"]
################################################
list_length_tracks = 50
list_length_artists = 50
width_wordcloud = 325
height_wordcloud_tracks = 140
height_wordcloud_artists = 175
height_graphs = 400
height_html_table = 170
img_size = '50px'
fontsize = 11
margin = '30px'
marker_size = 11
playlist_button = 'short'

box_shadow_dunkel = '3px 3px 3px 2px ' + colors['background_dunkel']
tabs_style = {'height': '44px'}
tab_style = {
    'borderRight': '0px',
    'borderLeft': '0px',
    'borderTop': '0px',
    'borderBottom': '0px',
    'backgroundColor': colors['background_dunkel'],
    'fontSize': 16,
    'color': 'grey',
    'padding': '6px'
}
tab_selected = {
    'borderRight': '0px solid ',
    'borderLeft': '0px solid ',
    'borderTop': '3px solid' + colors['spotify_green'],
    'backgroundColor': colors['background_hell'],
    'color': 'white',
    'fontWeight': 'bold',
    'fontSize': 18,
    'padding': '6px'
}
tabs_small_style = {'height': '35px'}
tab_small_style = {
    'borderRight': '0px',
    'borderLeft': '0px',
    'borderTop': '0px',
    'backgroundColor': colors['background_dunkel'],
    'fontSize': 14,
    'color': 'grey',
    'padding': '4px',
}
tab_small_selected = {
    'borderRight': '0px',
    'borderLeft': '0px',
    'borderTop': '1px solid' + colors['spotify_green'],
    'backgroundColor': colors['background_dunkel'],
    'color': 'white',
    'fontWeight': 'bold',
    'fontSize': 14,
    'padding': '4px',
}

################################################
ranges = ['short_term', 'medium_term', 'long_term']
dict_top_artists = {'Platz': [], 'term': [], 'artist': [], 'genres': [], 'popularity': [], 'image': []}
dict_top_tracks = {'Platz': [], 'term': [], 'song': [], 'danceability': [], 'acousticness': [], 'loudness': [], 'tempo': [], 'valence': [], 'image': []}
dict_playlists = {'  ': [], 'playlist name': [], '# tracks': [], 'follower': [], 'owner': [], 'public': [], 'collaborative': []}
dict_audio_features = {'short_term': [], 'medium_term': [], 'long_term': []}
dict_playlists_id = {'short_term': [], 'medium_term': [], 'long_term': []}

''' Spotify Authentication & DataFrame Creation'''
# spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, scope=scope, redirect_uri=redirect_uri))
# spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, username=username, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, show_dialog=False, cache_handler=MemoryCacheHandler(token_info=token)))
spotify = spotipy.Spotify(auth=os.environ["TOKEN"])

df_tracks, df_artists, df_playlists, df_tracks_table, df_artists_table, df_tracks_timeline, df_artists_genre, df_konzerte = create_dfs_from_spotify()

''' Create Spotify fix-values '''
current_user = spotify.current_user()['display_name']
followers = str(spotify.current_user()['followers']['total'])
concerts_sum_cost = '{:,.2f} €'.format(df_konzerte['Kosten'].sum())
concerts_amount = df_konzerte.shape[0]
concerts_average = '{:,.2f}'.format(concerts_amount / (datetime.datetime.now().year - 2004))


#######################################################################

app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True, title='> Hiasn Music Dashboard <')
server = app.server

app.layout = html.Div([
    dcc.Tabs(id='menu-tabs', value='tracks', children=[
        dcc.Tab(label='tracks', value='tracks', style=tab_style, selected_style=tab_selected),
        dcc.Tab(label='artists', value='artists', style=tab_style, selected_style=tab_selected),
    ], style=tabs_style),
    html.Div(id='page_content',
             style={'background': colors['background_hell']}
    )
], style={'margin': '20px'})


#######################################################################


@app.callback(Output('page_content', 'children'), Input('menu-tabs', 'value'))
def render_content(input_value):
    if input_value == 'tracks':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card(dbc.CardBody(html.Div([
                            dcc.Tabs(id='top_lists_tracks_tab', value='short', children=[
                                dcc.Tab(label='short', value='short', style=tab_small_style, selected_style=tab_small_selected, id='t1'),
                                dcc.Tab(label='medium', value='medium', style=tab_small_style, selected_style=tab_small_selected, id='t2'),
                                dcc.Tab(label='long', value='long', style=tab_small_style, selected_style=tab_small_selected, id='t3')
                            ], style=tabs_small_style),
                            html.Div(id='top_lists_tracks_content'),
                        ])), style={'margin': margin, 'background': colors['background_dunkel'], 'box-shadow': box_shadow_dunkel}),

                    html.Div([
                        html.Img(src=get_genre_wordcloud('tracks'), style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    ]),

                    ], width=4),

                dbc.Col(dbc.Card(dbc.CardBody(html.Div([
                    dcc.Tabs(id='tracks_circle_tab', value='schwof', children=[
                        dcc.Tab(label='schwof-factor', value='schwof', style=tab_small_style, selected_style=tab_small_selected),
                        dcc.Tab(label=' power-factor', value='power', style=tab_small_style, selected_style=tab_small_selected),
                        dcc.Tab(label='melancholia-factor', value='melancholia', style=tab_small_style, selected_style=tab_small_selected),
                        dcc.Tab(label='timeline', value='tracks_timeline', style=tab_small_style, selected_style=tab_small_selected),
                        dcc.Tab(label='playlists', value='playlists', style=tab_small_style, selected_style=tab_small_selected)
                    ], style=tabs_small_style),
                    html.Div(id='tracks_tab_content'),
                ])), style={'margin-right': margin, 'margin-bottom': margin, 'margin-top': margin, 'background': colors['background_dunkel'], 'box-shadow': box_shadow_dunkel})),

                # ], justify='center', no_gutters=True),
                ], justify='center'),

        ])

    elif input_value == 'artists':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card(dbc.CardBody(html.Div([
                        dcc.Tabs(id='top_lists_artists_tab', value='short', children=[
                            dcc.Tab(label='short', value='short', style=tab_small_style, selected_style=tab_small_selected, id='t1'),
                            dcc.Tab(label='medium', value='medium', style=tab_small_style, selected_style=tab_small_selected, id='t2'),
                            dcc.Tab(label='long', value='long', style=tab_small_style, selected_style=tab_small_selected, id='t3')
                        ], style=tabs_small_style),
                        html.Div(id='top_lists_artists_content')
                    ])), style={'margin': margin, 'background': colors['background_dunkel'], 'box-shadow': box_shadow_dunkel}),

                    html.Div([
                        html.Img(src=get_genre_wordcloud('artists'), style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
                    ]),

                    ], width=4),

                dbc.Col(dbc.Card(dbc.CardBody(html.Div([
                    dcc.Tabs(id='artists_circle_tab', value='short', children=[
                        dcc.Tab(label='short', value='short', style=tab_small_style, selected_style=tab_small_selected, id='t1'),
                        dcc.Tab(label=' medium', value='medium', style=tab_small_style, selected_style=tab_small_selected, id='t2'),
                        dcc.Tab(label='long', value='long', style=tab_small_style, selected_style=tab_small_selected, id='t3')], style=tabs_small_style),
                    html.Div(id='artists_tab_content'),
                ])), style={'margin-right': margin, 'margin-bottom': margin, 'margin-top': margin, 'background': colors['background_dunkel'], 'box-shadow': box_shadow_dunkel})),

            # ], justify='center', no_gutters=True),
            ], justify='center'),
        ])

    return dbc.Jumbotron([
        html.H1("404: Not found", className="text-danger"),
        html.Hr(),
        html.P(f"The pathname {input_value} was not recognised..."),])


@app.callback(Output('tracks_tab_content', 'children'), [Input('tracks_circle_tab', 'value')])
def tracks_circle_tab(input_value):
    if input_value == 'schwof':
        return get_tracks_circle_graph('danceability', 'valence', [-0.01, 1.01], [-0.01, 1.01])

    if input_value == 'power':
        return get_tracks_circle_graph('loudness', 'tempo', [-25, 0], [40, 220])

    if input_value == 'melancholia':
        return get_tracks_circle_graph('acousticness', 'valence', [-0.01, 1.01], [1.01, -0.01])

    if input_value == 'tracks_timeline':
        return get_tracks_timeline_graph()

    if input_value == 'playlists':
        return get_playlists_datatable()


@app.callback(Output("playlist_button_output", "children"), [Input("playlist_button", "n_clicks")])
def on_button_click(n):
    if n is None:
        return ""
    else:
        create_playlist(playlist_button)
        return 'Playlist erstellt!'


@app.callback(Output('artists_tab_content', 'children'), [Input('artists_circle_tab', 'value')])
def artists_circle_tab(input_value):
        return get_artists_circle_graph(input_value)


@app.callback(Output('top_lists_tracks_content', 'children'), [Input('top_lists_tracks_tab', 'value')])
def top_lists_tracks_tab(input_value):
    global playlist_button

    if input_value == 'short':
        playlist_button = 'short'
        return (get_top_list(df_tracks_table, 'tracks', input_value),
                get_playlist_button())

    if input_value == 'medium':
        playlist_button = 'medium'
        return (get_top_list(df_tracks_table, 'tracks', input_value),
                get_playlist_button())

    if input_value == 'long':
        playlist_button = 'long'
        return (get_top_list(df_tracks_table, 'tracks', input_value),
                get_playlist_button())


@app.callback(Output('top_lists_artists_content', 'children'), [Input('top_lists_artists_tab', 'value')])
def top_lists_artists_tab(input_value):
    if input_value == 'short':
        return get_top_list(df_artists_table, 'artists', input_value)
    if input_value == 'medium':
        return get_top_list(df_artists_table, 'artists', input_value)
    if input_value == 'long':
        return get_top_list(df_artists_table, 'artists', input_value)


#######################################################################


if __name__ == "__main__":
    app.run_server(debug=True)
