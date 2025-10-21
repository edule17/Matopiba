# ==============================================================================
# PROJETO DE ANÁLISE AGRÍCOLA E SOCIOECONÔMICA - MATOPIBA (VERSÃO FINAL)
# ==============================================================================

# --- Passo 0: Carregando bibliotecas necessárias ---
print("Carregando bibliotecas...")
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from libpysal import weights
import esda
from splot.esda import lisa_cluster, plot_moran
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

print("Bibliotecas carregadas com sucesso!")

# ==============================================================================
# --- MÓDULO 1: ETL (Extração, Transformação e Carga) ---
# ==============================================================================
print("\n--- INICIANDO MÓDULO DE ETL ---")

# --- Funções Auxiliares de ETL ---
def carregar_dados(caminho_arquivo, header_row=0):
    """Carrega um arquivo Excel ou CSV, especificando a linha do cabeçalho."""
    try:
        if caminho_arquivo.endswith(('.xls', '.xlsx')):
            return pd.read_excel(caminho_arquivo, header=header_row)
        elif caminho_arquivo.endswith('.csv'):
            return pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8', header=header_row)
        else:
            print(f"ERRO: Formato de arquivo não suportado: {caminho_arquivo}")
            return None
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo não encontrado. Verifique o caminho: '{caminho_arquivo}'")
        return None

def padronizar_codigo_ibge(df, nome_coluna):
    """Padroniza o código do município para 6 dígitos, como string."""
    if nome_coluna in df.columns:
        if pd.api.types.is_numeric_dtype(df[nome_coluna]):
            df[nome_coluna] = df[nome_coluna].astype(int)
        df[nome_coluna] = df[nome_coluna].astype(str).str.replace('.0', '', regex=False).str.slice(0, 6)
    return df

# --- Carregamento dos Dados ---
# ATENÇÃO: Verifique se os caminhos dos arquivos estão corretos para o seu ambiente.
df_pam = carregar_dados('dados/milho_matopiba_sidra.xlsx', header_row=4)
df_gini = carregar_dados('dados/Coeficiente-Gini_2017_TOTAL_MUN.xls')
shapefile_path = 'dados/BRMUE250GC_SIR.shp'
try:
    gdf_brasil = gpd.read_file(shapefile_path, encoding='utf-8')
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar o shapefile: {e}")
    gdf_brasil = None

# --------------------------------------------------------------------------
# BLOCO DE CONFIGURAÇÃO FINAL E CORRETO
# --------------------------------------------------------------------------
if df_pam is not None and df_gini is not None and gdf_brasil is not None:

    # --- BLOCO DE CONFIGURAÇÃO DE COLUNAS ---
    COLUNA_CODIGO_PAM = 'Cód.'
    COLUNA_VALOR_PAM = 'Milho (em grão)'
    COLUNA_CODIGO_GINI = 'COD_MUN'
    COLUNA_VALOR_GINI = 'gini'
    COLUNA_CODIGO_MAPA = 'CD_GEOCMU'
    COLUNA_NOME_MAPA = 'NM_MUNICIP'
    # --- FIM DO BLOCO DE CONFIGURAÇÃO ---

    # Renomeação padronizada
    df_pam.rename(columns={COLUNA_CODIGO_PAM: 'CD_MUN', COLUNA_VALOR_PAM: 'Producao_Milho_Ton'}, inplace=True)
    df_gini.rename(columns={COLUNA_CODIGO_GINI: 'CD_MUN', COLUNA_VALOR_GINI: 'Indice_Gini'}, inplace=True)
    gdf_brasil.rename(columns={COLUNA_CODIGO_MAPA: 'CD_MUN', COLUNA_NOME_MAPA: 'NM_MUN'}, inplace=True)

    # Padronizar códigos de municípios em todas as bases
    df_pam = padronizar_codigo_ibge(df_pam, 'CD_MUN')
    df_gini = padronizar_codigo_ibge(df_gini, 'CD_MUN')
    gdf_brasil = padronizar_codigo_ibge(gdf_brasil, 'CD_MUN')

    # Limpeza e conversão de tipos
    df_pam['Producao_Milho_Ton'] = pd.to_numeric(df_pam['Producao_Milho_Ton'], errors='coerce')
    df_gini['Indice_Gini'] = pd.to_numeric(df_gini['Indice_Gini'], errors='coerce')
    df_pam.dropna(subset=['CD_MUN', 'Producao_Milho_Ton'], inplace=True)
    df_gini.dropna(subset=['CD_MUN', 'Indice_Gini'], inplace=True)
    
    df_pam = df_pam[df_pam['Producao_Milho_Ton'] > 0]

    # Merge das bases de dados
    df_merged = pd.merge(df_pam, df_gini, on='CD_MUN', how='inner')
    gdf_matopiba = pd.merge(gdf_brasil, df_merged, on='CD_MUN', how='inner')

    # Filtro para a região do MATOPIBA
    codigos_estados_matopiba = ['21', '17', '22', '29']
    gdf_matopiba = gdf_matopiba[gdf_matopiba['CD_MUN'].str.startswith(tuple(codigos_estados_matopiba))]

    print("Módulo de ETL concluído com sucesso.")
    dados_prontos = True
else:
    print("Módulo de ETL falhou. Verifique os caminhos e nomes dos arquivos.")
    dados_prontos = False


# ==============================================================================
# --- MÓDULO 2: ANÁLISE ESPACIAL ---
# ==============================================================================
if dados_prontos and not gdf_matopiba.empty:
    print("\n--- INICIANDO MÓDULO DE ANÁLISE ESPACIAL ---")
    variavel_x = 'Producao_Milho_Ton'
    variavel_y = 'Indice_Gini'
    gdf_matopiba.dropna(subset=[variavel_x, variavel_y], inplace=True)
    
    w = weights.Queen.from_dataframe(gdf_matopiba, use_index=True)
    w.transform = 'r'
    
    gdf_matopiba[f'{variavel_x}_std'] = (gdf_matopiba[variavel_x] - gdf_matopiba[variavel_x].mean()) / gdf_matopiba[variavel_x].std()
    gdf_matopiba[f'{variavel_y}_std'] = (gdf_matopiba[variavel_y] - gdf_matopiba[variavel_y].mean()) / gdf_matopiba[variavel_y].std()
    
    # --- CÁLCULO DA CORRELAÇÃO ESPACIAL COM SEMENTE FIXA PARA REPRODUTIBILIDADE ---
    moran_bv = esda.Moran_Local_BV(
        x=gdf_matopiba[f'{variavel_x}_std'], 
        y=gdf_matopiba[f'{variavel_y}_std'], 
        w=w, 
        seed=12345  # Garante que os p-valores da simulação sejam sempre os mesmos
    )
    
    labels = {1: 'Alto-Alto', 2: 'Baixo-Alto', 3: 'Baixo-Baixo', 4: 'Alto-Baixo', 0: 'Não Significativo'}
    p_significativo = 0.05
    quadrantes_significativos = moran_bv.q * (moran_bv.p_sim <= p_significativo)
    gdf_matopiba['quadrante_label'] = [labels[q] for q in quadrantes_significativos]
    
    # Adiciona os resultados ao DataFrame para uso no dashboard
    gdf_matopiba['p_valor'] = moran_bv.p_sim
    gdf_matopiba['moran_I'] = moran_bv.Is 
    gdf_matopiba['abs_moran_I'] = abs(moran_bv.Is)
    
    print("Módulo de Análise Espacial concluído.")


# ==============================================================================
# --- MÓDULO 4: DASHBOARD INTERATIVO COM DASH E PLOTLY ---
# ==============================================================================
if dados_prontos and not gdf_matopiba.empty:
    print("\n--- INICIANDO MÓDULO DE DASHBOARD ---")
    app = dash.Dash(__name__)

    # --- Dicionários e Listas para o Layout ---
    nomes_variaveis = {
        'Producao_Milho_Ton': 'Produção de Milho (Toneladas)',
        'Indice_Gini': 'Índice de Gini',
        'cluster_espacial': 'Correlação Espacial (Clusters LISA)'
    }
    lista_municipios = sorted(gdf_matopiba['NM_MUN'].unique())

    # --- Layout Final do Dashboard ---
    app.layout = html.Div([
        html.H1("Dashboard Agrícola e Socioeconômico do MATOPIBA", style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # --- Seção de Filtros ---
        html.Div([
            html.Div([
                html.Label("Selecione a análise para visualizar no mapa:"),
                dcc.Dropdown(
                    id='dropdown-variavel-mapa',
                    options=[
                        {'label': 'Produção de Milho (Coroplético)', 'value': 'Producao_Milho_Ton'},
                        {'label': 'Índice de Gini (Coroplético)', 'value': 'Indice_Gini'},
                        {'label': 'Correlação Espacial (Clusters LISA)', 'value': 'cluster_espacial'}
                    ],
                    value='Producao_Milho_Ton'
                )
            ], style={'width': '49%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Pesquise por um ou mais municípios:"),
                dcc.Dropdown(
                    id='municipio-search-dropdown',
                    options=[{'label': i, 'value': i} for i in lista_municipios],
                    value=[],
                    multi=True,
                    placeholder="Selecione para filtrar...",
                )
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
        ], style={'width': '98%', 'margin': 'auto', 'marginBottom': '20px'}),

        html.Div([dcc.Graph(id='mapa-interativo')]),

        html.Div([
            html.Div([dcc.Graph(id='grafico-correlacao-interativo')], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='moran-scatter-plot')], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.Hr(style={'marginTop': '20px', 'marginBottom': '20px'}),
        html.H2("Rankings dos Municípios", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([dcc.Graph(id='ranking-milho')], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([dcc.Graph(id='ranking-gini')], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([dcc.Graph(id='ranking-corr')], style={'width': '34%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ])

    # --- Função de Filtro Genérica ---
    def filter_dataframe(searched_municipios):
        if not searched_municipios:
            return gdf_matopiba
        return gdf_matopiba[gdf_matopiba['NM_MUN'].isin(searched_municipios)]

    # --- Callback para o Mapa ---
    @app.callback(
        Output('mapa-interativo', 'figure'),
        [Input('dropdown-variavel-mapa', 'value'),
         Input('municipio-search-dropdown', 'value')]
    )
    def update_map(variavel_selecionada, searched_municipios):
        dff = filter_dataframe(searched_municipios)
        titulo_mapa = nomes_variaveis.get(variavel_selecionada)

        if dff.empty:
            return go.Figure().update_layout(title_text="Nenhum município selecionado", xaxis_visible=False, yaxis_visible=False)

        if variavel_selecionada == 'cluster_espacial':
            color_map = {'Não Significativo': 'lightgrey', 'Alto-Alto': '#d7191c', 'Baixo-Baixo': '#2c7bb6', 'Alto-Baixo': '#fdae61', 'Baixo-Alto': '#abd9e9'}
            fig = px.choropleth_mapbox(dff, geojson=dff.geometry, locations=dff.index, color='quadrante_label', color_discrete_map=color_map,
                                       category_orders={'quadrante_label': ['Alto-Alto', 'Baixo-Baixo', 'Alto-Baixo', 'Baixo-Alto', 'Não Significativo']},
                                       mapbox_style="carto-positron", zoom=5, center={"lat": -9.5, "lon": -45.0}, opacity=0.8, hover_name='NM_MUN')
            fig.update_layout(title_text=titulo_mapa, margin={"r":10,"t":40,"l":10,"b":10}, legend_title_text='<b>Tipo de Cluster</b>', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        else:
            fig = px.choropleth_mapbox(dff, geojson=dff.geometry, locations=dff.index, color=variavel_selecionada, color_continuous_scale="Viridis",
                                       mapbox_style="carto-positron", zoom=5, center={"lat": -9.5, "lon": -45.0}, opacity=0.7, hover_name='NM_MUN', hover_data={variavel_selecionada: True})
            fig.update_layout(title_text=titulo_mapa, margin={"r":10,"t":40,"l":10,"b":10}, coloraxis_colorbar=dict(title=nomes_variaveis.get(variavel_selecionada).split('(')[0].strip()))
        
        if len(dff) == 1:
            center_lat = dff.geometry.centroid.y.iloc[0]
            center_lon = dff.geometry.centroid.x.iloc[0]
            fig.update_layout(mapbox_center={"lat": center_lat, "lon": center_lon}, mapbox_zoom=8)
        else:
             fig.update_geos(fitbounds="locations", visible=False)

        return fig

    # --- Callback para os Gráficos de Dispersão ---
    @app.callback(
        [Output('grafico-correlacao-interativo', 'figure'),
         Output('moran-scatter-plot', 'figure')],
        [Input('municipio-search-dropdown', 'value')]
    )
    def update_scatter_plots(searched_municipios):
        dff = filter_dataframe(searched_municipios)
        
        if dff.empty or len(dff) < 2:
            empty_fig = go.Figure().update_layout(title_text="Selecione 2 ou mais municípios", xaxis_visible=False, yaxis_visible=False)
            return empty_fig, empty_fig

        correlacao = dff['Producao_Milho_Ton'].corr(dff['Indice_Gini'])
        fig_corr = px.scatter(dff, x='Producao_Milho_Ton', y='Indice_Gini', trendline='ols', hover_name='NM_MUN', title=f"Correlação (Pearson: {correlacao:.3f})")
        fig_corr.update_layout(margin={"r":10,"t":40,"l":10,"b":10})
        
        spatial_lag_values = pd.Series(moran_bv.zy, index=gdf_matopiba.index)
        dff['spatial_lag_gini_std'] = spatial_lag_values.loc[dff.index]
        fig_moran = px.scatter(dff, x='Producao_Milho_Ton_std', y='spatial_lag_gini_std', hover_name='NM_MUN', title='Correlação Espacial (Moran Bivariado)',
                               labels={"Producao_Milho_Ton_std": "Produção (Padronizada)", "spatial_lag_gini_std": "Média Gini Vizinhos (Padronizada)"})
        fig_moran.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
        fig_moran.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
        fig_moran.update_layout(margin={"r":10,"t":40,"l":10,"b":10})
        
        return fig_corr, fig_moran

    # --- Callback para as Tabelas de Ranking ---
    @app.callback(
        [Output('ranking-milho', 'figure'),
         Output('ranking-gini', 'figure'),
         Output('ranking-corr', 'figure')],
        [Input('municipio-search-dropdown', 'value')]
    )
    def update_rankings(searched_municipios):
        dff = filter_dataframe(searched_municipios)
        N = 10 
        
        if dff.empty:
            empty_fig = go.Figure().update_layout(title_text="Nenhum município selecionado", xaxis_visible=False, yaxis_visible=False)
            return empty_fig, empty_fig, empty_fig

        # Tabela de Produção de Milho
        top_milho = dff.sort_values(by='Producao_Milho_Ton', ascending=False).head(N)
        fig_milho = go.Figure(data=[go.Table(header=dict(values=['<b>Posição</b>', '<b>Município</b>', '<b>Produção (Ton)</b>'], fill_color='paleturquoise', align='left'),
                                            cells=dict(values=[list(range(1, len(top_milho) + 1)), top_milho['NM_MUN'], top_milho['Producao_Milho_Ton'].map('{:,.0f}'.format)], fill_color='lavender', align='left'))])
        fig_milho.update_layout(title_text=f'Ranking de Produção de Milho', margin={"r":10,"t":40,"l":10,"b":10})

        # Tabela de Índice de Gini
        top_gini = dff.sort_values(by='Indice_Gini', ascending=False).head(N)
        fig_gini = go.Figure(data=[go.Table(header=dict(values=['<b>Posição</b>', '<b>Município</b>', '<b>Índice de Gini</b>'], fill_color='lightcoral', align='left'),
                                            cells=dict(values=[list(range(1, len(top_gini) + 1)), top_gini['NM_MUN'], top_gini['Indice_Gini'].map('{:.3f}'.format)], fill_color='lavender', align='left'))])
        fig_gini.update_layout(title_text=f'Ranking de Índice de Gini', margin={"r":10,"t":40,"l":10,"b":10})
        
        # Tabela de Correlações Espaciais
        significativos = dff[dff['quadrante_label'] != 'Não Significativo']
        top_corr = significativos.sort_values(by=['p_valor', 'abs_moran_I'], ascending=[True, False]).head(N)
        fig_corr = go.Figure(data=[go.Table(header=dict(values=['<b>Posição</b>', '<b>Município</b>', '<b>Cluster</b>', '<b>Moran I</b>', '<b>P-valor</b>'], fill_color='lightgreen', align='left'),
                                            cells=dict(values=[list(range(1, len(top_corr) + 1)), top_corr['NM_MUN'], top_corr['quadrante_label'], top_corr['moran_I'].map('{:.4f}'.format), top_corr['p_valor'].map('{:.4f}'.format)], fill_color='lavender', align='left'))])
        fig_corr.update_layout(title_text=f'Ranking de Correlações Espaciais', margin={"r":10,"t":40,"l":10,"b":10})

        return fig_milho, fig_gini, fig_corr

    if __name__ == '__main__':
        print("\nDashboard pronto! Acesse http://127.0.0.1:8050/ no seu navegador.")
        app.run(debug=False)
        
elif dados_prontos and gdf_matopiba.empty:
    print("\n--- Módulos de Análise e Dashboard não executados: Nenhum dado do MATOPIBA encontrado após o merge ---")