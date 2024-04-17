# Import librairies
import streamlit as st
import numpy as np
import pandas as pd
from dateutil import relativedelta
from pandas.tseries.offsets import DateOffset
from pathlib import Path

from io import BytesIO
import requests

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import scipy.stats
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from empyrical import (sharpe_ratio, calmar_ratio, omega_ratio, sortino_ratio,
                       cagr, annual_volatility, tail_ratio,
                       up_capture, down_capture,
                       alpha_beta, up_alpha_beta, down_alpha_beta,
                       value_at_risk, conditional_value_at_risk,
                       cum_returns_final, cum_returns
                       )

from quantstats.stats import drawdown_details, to_drawdown_series

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

# configuration streamlit
st.set_page_config(layout='wide')

# Fonctions
def sharpe(rdt, risk_free_rdt, period='daily'):
    if period=='daily':
        risk_free_rate = cagr(risk_free_rdt)/252
    elif period=='weelky':
        risk_free_rate = cagr(risk_free_rdt)/52
    elif period=='monthly':
        risk_free_rate = cagr(risk_free_rdt)/12
    return sharpe_ratio(rdt, risk_free=risk_free_rate, period='daily')


def diff_date(date1, date2):
    date_diff = relativedelta.relativedelta(date2, date1)
    date = [f'{abs(date_diff.years)} ans ' if abs(date_diff.years) > 0 else str(),
            f'{abs(date_diff.months)} mois ' if abs(
                date_diff.months) > 0 else str(),
            f'{abs(date_diff.days)} jours' if abs(date_diff.days) > 0 else str()]
    return ''.join(date)


def human_format(num, round_to=1):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num = round(num / 1000.0, round_to)
    return '{:.{}f}{}'.format(num, round_to, ['', 'K', 'M', 'B', 'G'][magnitude])


def custom_styling(val):
    color = "red" if val < 0 else "black"
    return f"color: {color}"


def highlight(s):
    if s.Nom == 'TOTAL':
        return ['background-color: red'] * len(s)
    else:
        ['background-color: white'] * len(s)


def tracking_error(r_a, r_b):
    '''
    Returns the tracking error between two return series. 
    This method is used in Sharpe Analysis minimization problem.
    '''
    return (((r_a - r_b)**2).sum())**(0.5)


def style_analysis_tracking_error(weights, ref_r, bb_r):
    '''
    Sharpe style analysis objective function.
    Returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights. 
    '''
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))


def style_analysis(dep_var, exp_vars):
    '''
    Sharpe style analysis optimization problem.
    Returns the optimal weights that minimizes the tracking error between a portfolio 
    of the explanatory (return) variables and the dependent (return) variable.
    '''
    # dep_var is expected to be a pd.Series
    if isinstance(dep_var, pd.DataFrame):
        dep_var = dep_var[dep_var.columns[0]]

    n = exp_vars.shape[1]
    init_guess = np.repeat(1/n, n)
    weights_const = {
        'type': 'eq',
        'fun': lambda weights: 1 - np.sum(weights)
    }
    solution = minimize(style_analysis_tracking_error,
                        init_guess,
                        method='SLSQP',
                        options={'disp': False},
                        args=(dep_var, exp_vars),
                        constraints=(weights_const,),
                        bounds=((0.0, 1.0),)*n)
    # weights = pd.Series(solution.x, index=exp_vars.columns)
    return solution.x.reshape(1, -1)


def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_style_analysis(dep_var, exp_vars, window_size=52):
    data = pd.concat([dep_var, exp_vars], axis=1)

    data_index = data.iloc[window_size-1:].index
    data_col = exp_vars.columns
    nbre_col = exp_vars.shape[1]+1

    data = data.to_numpy()
    data_rolling = rolling_window(data, window_size=window_size)
    weights = np.concatenate([100*style_analysis(data_roll[:, 0],
                                                 data_roll[:, 1:nbre_col]) for data_roll in data_rolling])
    return pd.DataFrame(weights, columns=data_col, index=data_index)

def roll_cagr(data, window_size=252, period='daily'):
    data_index = data.iloc[window_size-1:].index
    data_col = data.columns
    data = data.to_numpy()
    data_rolling = rolling_window(data, window_size=window_size)
    rolling_cagr = [cagr(data_roll, period=period) for data_roll in data_rolling]
    return pd.DataFrame(rolling_cagr, columns=data_col, index=data_index)

def compound_returns(s, start=100):
    '''
    Compound a pd.Dataframe or pd.Series of returns from an initial default value equal to 100.
    In the former case, the method compounds the returns for every column (Series) by using pd.aggregate. 
    The method returns a pd.Dataframe or pd.Series - using cumprod(). 
    See also the COMPOUND method.
    '''
    if isinstance(s, pd.DataFrame):
        return s.aggregate(compound_returns, start=start)
    elif isinstance(s, pd.Series):
        return start * (1 + s).cumprod()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def drawdown(rets: pd.Series, start=1000):
    '''
    Compute the drawdowns of an input pd.Series of returns. 
    The method returns a dataframe containing: 
    1. the associated wealth index (for an hypothetical starting investment of $1000) 
    2. all previous peaks 
    3. the drawdowns
    '''
    wealth_index = compound_returns(rets, start=start)
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    df = pd.DataFrame(
        {"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns})
    return df

def summary_stats_glissant(s, period='daily'):
    if period == 'daily':
        lag = 252
    elif period == 'weekly':
        lag = 52
    else:
        lag = 12
    stats = pd.DataFrame()
    stats['Perf 2023'] = 100 *s.aggregate(lambda x:  cum_returns_final(x['2023']) if x['2023'].isnull().sum() == 0 else np.nan)
    stats['Perf 2022'] = 100 *s.aggregate(lambda x:  cum_returns_final(x['2022']) if x['2022'].isnull().sum() == 0 else np.nan)
    stats['Perf 2021'] = 100 *s.aggregate(lambda x:  cum_returns_final(x['2021']) if x['2021'].isnull().sum() == 0 else np.nan)
    stats['Perf 1 an'] = 100 *s.aggregate(lambda x:  cum_returns_final(x[-lag:-1]) if x[-260:].isnull().sum() == 0 else np.nan)
    stats['Perf 3 ans'] = 100 *s.aggregate(lambda x: cum_returns_final(x[-3*lag:-1]) if x[-3*260:].isnull().sum() == 0 else np.nan)
    stats['Perf 5 ans'] = 100 *s.aggregate(lambda x: cum_returns_final(x[-5*lag:-1]) if x[-5*260:].isnull().sum() == 0 else np.nan)
    stats["Ann. vol 1 an"] = 100*s.aggregate(lambda x:  annual_volatility(x.iloc[-lag:-1], period=period) if x[-260:].isnull().sum() == 0 else np.nan)
    stats["Ann. vol 3 ans"] = 100 *s.aggregate(lambda x: annual_volatility(x.iloc[-3*lag:-1], period=period) if x[-3*260:].isnull().sum() == 0 else np.nan)
    stats["Ann. vol 5 ans"] = 100 *s.aggregate(lambda x: annual_volatility(x.iloc[-5*lag:-1], period=period) if x[-5*260:].isnull().sum() == 0 else np.nan)
    stats["Sharpe ratio 1 an"] = s.aggregate(lambda x: sharpe(x[-260:], 
                                                              risk_free_rdt=bench['Eonia Capitalization Index 7 Day'].iloc[-260:].pct_change(),
                                                              period=period) if x[-260:].isnull().sum() == 0 else np.nan
                                             )
    stats["Sharpe ratio 3 ans"] = s.aggregate(lambda x: sharpe(x[-3*260:], 
                                                               risk_free_rdt=bench['Eonia Capitalization Index 7 Day'].iloc[-3*260:].pct_change(), 
                                                               period=period) if x[-3*260:].isnull().sum() == 0 else np.nan
                                             )
    stats["Sharpe ratio 5 ans"] = s.aggregate(lambda x: sharpe(x[-5*260:],
                                                               risk_free_rdt=bench['Eonia Capitalization Index 7 Day'].iloc[-5*260:].pct_change(),
                                                               period=period) if x[-5*260:].isnull().sum() == 0 else np.nan
                                             )
    return stats.T


def summary_stats_perf(s, risk_free_rate=0.00, period='daily'):
    if period == 'daily':
        lag = 252
    elif period == 'weekly':
        lag = 52
    else:
        lag = 12
    stats = pd.DataFrame()
    stats["Date début"] = s.aggregate(
        lambda r: r.index[0].strftime("%d/%m/%Y"))
    stats["Date fin"] = s.aggregate(lambda r: r.index[-1].strftime("%d/%m/%Y"))
    stats["Taux sans risque"] = 100*risk_free_rate
    stats["Ann. return"] = 100*s.aggregate(cagr, period=period)
    stats["% Rdt>0"] = s.aggregate(lambda r: 100*r[r > 0].size/r.size)
    stats["Average up"] = s.aggregate(lambda r: 100*r[r >= 0].mean())
    stats["Average Down"] = s.aggregate(lambda r: 100*r[r < 0].mean())
    stats['Up ratio'] = s.aggregate(
        lambda r: 100*up_capture(r, df['indice'], period=period))
    stats['Down ratio'] = s.aggregate(
        lambda r: 100*down_capture(r, df['indice'], period=period))
    stats["Skewness"] = s.aggregate(lambda x: x.skew())
    stats["kurtosis"] = s.aggregate(
        lambda x: x.kurtosis())  # s.aggregate(kurtosis)
    stats['Is Normal ?'] = s.aggregate(
        lambda x: scipy.stats.jarque_bera(x)[1] >= 0.05)
    return stats.T

def summary_stats_risk(s, period='daily', var_level=0.05):
    if period == 'daily':
        lag = 252
    elif period == 'weekly':
        lag = 52
    else:
        lag = 12
    stats = pd.DataFrame()
    stats["Ann. vol"] = 100 * \
        s.aggregate(lambda x: annual_volatility(x, period=period))
    stats["Historic Var"] = 100*s.aggregate(value_at_risk, cutoff=var_level)
    stats["Historic CVar"] = 100 * \
        s.aggregate(conditional_value_at_risk, cutoff=var_level)
    stats["Max Drawdown"] = 100 * \
        s.aggregate(lambda r: drawdown(r)["Drawdown"].min())
    stats["Drawdown average"] = 100*s.aggregate(lambda r: drawdown(
        r).loc[drawdown(r)['Drawdown'] < 0, "Drawdown"].mean())
    stats['Tail'] = s.aggregate(tail_ratio)
    return stats.T


def summary_stats_ratio(s, risk_free_rate=0.00, period='daily'):
    if period == 'daily':
        lag = 252
    elif period == 'weekly':
        lag = 52
    else:
        lag = 12
    stats = pd.DataFrame()
    stats["Sharpe ratio"] = s.aggregate(
        sharpe_ratio, risk_free=risk_free_rate, period=period)
    stats["Calmar Ratio"] = s.aggregate(calmar_ratio)
    stats["Omega Ratio"] = s.aggregate(
        omega_ratio, risk_free=risk_free_rate, required_return=0.0)
    stats["Sortino Ratio"] = s.aggregate(sortino_ratio, required_return=0.0)
    return stats.T


def calcul_perf_glissant(portef, bench, window=52, period='weekly'):
    rdt_portef_glissant = 100*roll_cagr(portef.to_frame(), window_size=window, period=period)
    rdt_bench_glissant = 100*roll_cagr(bench.to_frame(), window_size=window, period=period)
    data = pd.concat([rdt_portef_glissant, rdt_bench_glissant], axis=1)
    ecart_perf_glissant = data.iloc[:,0]-data.iloc[:,1]
    return rdt_portef_glissant, rdt_bench_glissant, ecart_perf_glissant


def graph_bar(y, colors, texte='AuM'):
    fig = go.Figure([go.Bar(name=col,
                            x=y.index,
                            y=y[col],
                            customdata=np.stack((y[col].apply(lambda x: human_format(x, 0)),
                                                 y.sum(1).apply(
                                lambda x: human_format(x, 0))
                            ),
                                axis=-1),
                            xhoverformat="%Y",
                            xperiodalignment="middle",
                            hovertemplate='<br>'.join(['Année: %{x}',
                                                       texte +
                                                       ': %{customdata[0]}€',
                                                       '<b>Total: %{customdata[1]}€</b>',
                                                       ]
                                                      ),
                            marker_color=colors[col]
                            )
                     for col in y.columns]
                    )
    fig.update_layout(barmode='relative',
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="left",
                                  entrywidth=100,
                                  x=0),
                      uniformtext_mode='hide'
                      )
    fig.update_xaxes(ticklabelmode="period",
                     tickformat="%Y",
                     )
    fig.update_yaxes(fixedrange=False)
    return fig

def graph_bar_Nbre_fonds(y, colors, texte='Nombre'):
    fig = go.Figure([go.Bar(name=col,
                            x=y.index,
                            y=y[col],
                            customdata=np.stack((y[col],
                                                 y.sum(1)
                                                 ),
                                                axis=-1),
                            xhoverformat="%Y",
                            xperiodalignment="middle",
                            hovertemplate='<br>'.join(['Année: %{x}',
                                                       texte +
                                                       ': %{customdata[0]}',
                                                       '<b>Total: %{customdata[1]}</b>',
                                                       ]
                                                      ),
                            marker_color=colors[col]
                            )
                     for col in y.columns]
                    )
    fig.update_layout(barmode='relative',
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="left",
                                  entrywidth=100,
                                  x=0),
                      uniformtext_mode='hide'
                      )
    fig.update_xaxes(ticklabelmode="period",
                     tickformat="%Y",
                     )
    fig.update_yaxes(fixedrange=False)
    return fig

def selection_fonds(i, key=''):
    cols = st.columns(5)
    # Choix de l'AM
    selected_am = cols[0].selectbox("Choix de la société de gestion",
                                    desc['FUND_MGMT_COMPANY'].sort_values().unique(),
                                    index=i,
                                    key='selected_am'+str(key)+str(i)
                                    )
    # Choix du type de fonds
    type_fonds = cols[1].multiselect("Nature du fonds",
                                     desc.query(
                                         "FUND_MGMT_COMPANY==@selected_am")['FUND_TYP'].sort_values().unique(),
                                     key='type_fonds'+str(key)+str(i)
                                     )

    if not type_fonds:
        type_fonds = desc.query(
            "FUND_MGMT_COMPANY==@selected_am")['FUND_TYP'].sort_values().unique()

    # Choix de la classe d'actifs
    classes = ['Toutes classes',
               *desc.query("FUND_MGMT_COMPANY==@selected_am & FUND_TYP in @type_fonds")['FUND_ASSET_CLASS_FOCUS'].sort_values().unique()
               ]

    selected_classe = cols[2].selectbox("Choix de la classe d'actif",
                                        classes,
                                        key='selected_classe'+str(key)+str(i)
                                        )

    if selected_classe == 'Toutes classes':
        fonds_sgp = desc.query(
            "FUND_MGMT_COMPANY == @selected_am & FUND_TYP in @type_fonds")['LONG_COMP_NAME']
    else:
        fonds_sgp = desc.query(
            "FUND_ASSET_CLASS_FOCUS==@selected_classe & FUND_MGMT_COMPANY == @selected_am & FUND_TYP in @type_fonds")['LONG_COMP_NAME']

    # Choix du fonds
    selected_fonds = cols[3].selectbox("Choix du fonds",
                                       vl[fonds_sgp].columns.sort_values(),
                                       key='selected_fonds'+str(key)+str(i),
                                       index=i
                                       )
    # Pondérations
    weights = cols[4].number_input("Poids:",
                                   min_value=0.0,
                                   max_value=100.0,
                                   value=100/nombre_fonds,
                                   key='poids'+str(key)+str(i),
                                   step=0.5
                                   )

    return selected_fonds, weights


def selection_indice(i, key=''):
    cols = st.columns(3)
    # Choix de l'indice de référence
    selected_classe_actif = cols[0].selectbox("Classe d'actifs",
                                              desc_indice['security type'].sort_values(
                                              ).unique(),
                                              key='selected_classe_actif' +
                                                  str(key)+str(i)
                                              )
    selected_indices = cols[1].selectbox("Choix de l'indice",
                                         desc_indice.query("`security type`==@selected_classe_actif").sort_values(
                                             ['security type', 'description'])['description'].unique(),
                                         key='selected_indices'+str(key)+str(i)
                                         )
    # Pondérations
    weights = cols[2].number_input("Poids:",
                                   min_value=0.0,
                                   max_value=100.0,
                                   value=100/nombre_fonds,
                                   key='poids'+str(key)+str(i),
                                   step=0.5
                                   )
    return selected_indices, weights


def selection_indice_analyse_style(i, key=''):
    cols = st.columns(2)
    # Choix de l'indice de référence
    selected_classe_actif = cols[0].selectbox("Classe d'actifs",
                                              desc_indice['security type'].sort_values(
                                              ).unique(),
                                              key='selected_classe_actif' +
                                                  str(key)+str(i)
                                              )
    selected_indices = cols[1].selectbox("Choix de l'indice",
                                         desc_indice.query("`security type`==@selected_classe_actif").sort_values(
                                             ['security type', 'description'])['description'].unique(),
                                         key='selected_indices'+str(key)+str(i)
                                         )
    return selected_indices


def update_data(file):
    aum_update = pd.read_excel(file,
                               sheet_name='aum',
                               index_col=0,
                               header=0,
                               skiprows=[1, 2]).sort_index()

    vl_update = pd.read_excel(file,
                              sheet_name='vl',
                              index_col=0,
                              header=0,
                              skiprows=[1, 2]
                              ).sort_index()

    bench_update = pd.read_excel(file,
                                 sheet_name='indices',
                                 index_col=0,
                                 header=0,
                                 skiprows=[1, 2]
                                 ).sort_index().dropna(axis=1, how='all').query('~index.duplicated()')

    desc_update = pd.read_excel(file,
                                sheet_name='desc',
                                index_col=0,
                                ).reset_index()

    desc_update['FUND_ASSET_CLASS_FOCUS'] = desc_update['FUND_ASSET_CLASS_FOCUS'].fillna(
        'Non renseigné')
    desc_update['FUND_MGMT_COMPANY'] = desc_update['FUND_MGMT_COMPANY'].replace(['H2O AM LLP', 'H2O Am Europe SASU', 'Myria Asset Management SAS',
                                                                                 'Myria Asset Management/France', 'Comgest Growth PLC',
                                                                                 'Financiere Arbevel SAS/Fund Parent'],
                                                                                ['H2O AM', 'H2O AM', 'Myria Asset Management',
                                                                                 'Myria Asset Management', 'Comgest SA', 'Financiere Arbevel SAS'],
                                                                                )
    desc_update['FUND_INCEPT_DT'] = pd.to_datetime(
        desc_update['FUND_INCEPT_DT'], dayfirst=True)

    desc_indice_update = pd.read_excel(file,
                                       sheet_name='desc indice',
                                       index_col=0,
                                       ).reset_index()
    return aum_update, vl_update, bench_update, desc_update, desc_indice_update

def load_google(code):
    url = f"https://drive.google.com/uc?export=download&id={code}"
    file = requests.get(url)
    bytesio = BytesIO(file.content)
    return pd.read_parquet(bytesio)


@st.cache_data
def load_df():
    aum = load_google('1-B7Gc12ZcnD-fy_ngS-XQcKyRePERbnw')
    vl = load_google('1-JSSHvILfKERBaV-uaGV6PjheBw0OTRZ')
    bench = load_google('1-FT7EGsFiN6LiKfSJAkVde5ukfa6NIwZ')
    desc = load_google('1-H4arAreH4SioXsLzkkferahTuSzIzcE')
    desc_indice = load_google('1-GdUmwFOAA8hLQvn6cU61Etl-x-b4TKZ')
    # Gestion des données manquantes
    aum = aum.dropna(how='all').dropna(axis=1, how='all')
    vl = vl.dropna(how='all').dropna(axis=1, how='all')
    bench = bench.dropna(how='all').dropna(axis=1, how='all')

    # Nettoyage des fonds en supprimant les fonds non communs entre VL, AUM et desc
    fonds = list(set(vl.columns) & set(aum.columns) & set(desc.LONG_COMP_NAME))
    aum = aum[fonds].asfreq('B').ffill()
    vl = vl[fonds].asfreq('B').ffill()
    bench = bench.asfreq('B').ffill()
    desc = desc.query("LONG_COMP_NAME in @fonds")

    # Construction de la base de données annuelle
    vl_annuel = vl.resample('Y').last()
    aum_annuel = aum.resample('Y').last()
    collecte_annuel = aum_annuel - aum_annuel.shift()*(1+vl_annuel.pct_change())

    # Gestion d'une donnée aberrantes ie AuM multiplié par 1_000_000 et VL mulitipliée par 1_000
    aum_annuel.loc['2002', 'Federal Support Monetaire ESG'] = aum_annuel.loc['2002',
                                                                             'Federal Support Monetaire ESG']/1_000_000
    vl.loc[['2022-02-24', '2022-02-28', '2022-03-07', '2022-03-15'], 'Ecofi Investissements - Epargne Ethique Monetaire'] = vl.loc[['2022-02-24', '2022-02-28', '2022-03-07', '2022-03-15'], 'Ecofi Investissements - Epargne Ethique Monetaire']/1_000
    aum.loc['2012-12-31', 'Ecofi Optim Variance'] = 9_630_000.0    
    
    vl_annuel.columns = pd.MultiIndex.from_tuples(zip(vl_annuel.columns,
                                                      vl_annuel.columns.map(desc.set_index('LONG_COMP_NAME')[
                                                                            'FUND_ASSET_CLASS_FOCUS']),
                                                      vl_annuel.columns.map(desc.set_index(
                                                          'LONG_COMP_NAME')['FUND_MGMT_COMPANY'])
                                                      )
                                                  )

    aum_annuel.columns = pd.MultiIndex.from_tuples(zip(aum_annuel.columns,
                                                       aum_annuel.columns.map(desc.set_index('LONG_COMP_NAME')[
                                                                              'FUND_ASSET_CLASS_FOCUS']),
                                                       aum_annuel.columns.map(desc.set_index(
                                                           'LONG_COMP_NAME')['FUND_MGMT_COMPANY']),
                                                       )
                                                   )

    collecte_annuel.columns = pd.MultiIndex.from_tuples(zip(collecte_annuel.columns,
                                                            collecte_annuel.columns.map(desc.set_index(
                                                                'LONG_COMP_NAME')['FUND_ASSET_CLASS_FOCUS']),
                                                            collecte_annuel.columns.map(desc.set_index(
                                                                'LONG_COMP_NAME')['FUND_MGMT_COMPANY'])
                                                            )
                                                        )
    # desc.fillna('Non renseigné', inplace=True)
    return aum, vl, bench, desc, desc_indice, vl_annuel, aum_annuel, collecte_annuel


# Importation des données
path = Path('/Users/jacques/Library/Mobile Documents/com~apple~CloudDocs/Projets/Analyse fonds/Data')
aum, vl, bench, desc, desc_indice, vl_annuel, aum_annuel, collecte_annuel = load_df()

# Gestion des couleurs vs classes d'actifs
colors = {k: v for k, v in zip(
    desc.FUND_ASSET_CLASS_FOCUS.unique(), px.colors.qualitative.Set1)}

# Gestion des différents onglets
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Paramètres",
                                                          "Liste des sociétés de gestion",
                                                          "Classement des fonds",
                                                          "Analyse société de gestion",
                                                          "Analyse fonds vs indice",
                                                          "Statistiques",
                                                          "Analyse de style",
                                                          "Update base de données"
                                                          ]
                                                         )


##############################################################################
########################### Onglet Paramètres ################################
##############################################################################
with tab1:
    st.header('**Entrez le portefeuille à analyser:**')
    nombre_fonds = st.number_input('**Nombre de fonds pour analyse:**',
                                   min_value=1,
                                   step=1,
                                   key='nombre_fonds1'
                                   )

    # Portefeuille à analyser
    selections = []
    for row in range(nombre_fonds):
        selections.append(selection_fonds(row))

    selection_fonds = pd.DataFrame(selections,
                                   columns=['fonds', 'poids']
                                   )
    portef = pd.DataFrame((selection_fonds['poids'].to_numpy()*vl[selection_fonds['fonds']].pct_change().dropna()).sum(1),
                          columns=['portef']
                          )
    portef = pd.concat([portef,
                        100*vl[selection_fonds['fonds']].pct_change()],
                       axis=1)
    list_fonds = selection_fonds['fonds'].to_list()
    if selection_fonds['poids'].sum() != 100:
        st.subheader("**:red[la somme des poids n'est pas égal à 100%]**")

    st.header("**Choix de l'indice pour comparaison:**")
    indice = st.radio('**Indice ou fonds pour comparaison:**',
                      ('Indice', 'Fonds'),
                      horizontal=True
                      )
    if indice == 'Indice':
        st.subheader("**Entrez l'indice pour comparaison:**")
        nombre_fonds = st.number_input('**Nombre de fonds pour analyse:**',
                                       min_value=1,
                                       value=1,
                                       step=1,
                                       key='nombre_fonds2'
                                       )
        selections = []
        for row in range(nombre_fonds):
            selections.append(selection_indice(row, key=1))

        bench_indice = pd.DataFrame(selections,
                                    columns=['indice', 'poids']
                                    )
        benchmark = pd.DataFrame((bench_indice['poids'].to_numpy()*bench[bench_indice['indice']].pct_change().dropna()).sum(1),
                                 columns=['indice']
                                 )
        benchmark = pd.concat([benchmark,
                               100*bench[bench_indice['indice']].pct_change()
                               ],
                              axis=1)
        list_indice1 = bench_indice['indice'].to_list()
        if bench_indice['poids'].sum() != 100:
            st.subheader("**:red[la somme des poids n'est pas égal à 100%]**")

    else:
        st.subheader("**Entrez les fonds pour comparaison:**")
        nombre_indice = st.number_input('**Nombre indices pour analyse:**',
                                        min_value=1,
                                        value=1,
                                        step=1,
                                        key='nombre_fonds2'
                                        )
        selections = []
        for row in range(nombre_indice):
            selections.append(selection_fonds(row, key=1))

        bench_fonds = pd.DataFrame(selections,
                                   columns=['indice', 'poids']
                                   )
        benchmark = pd.DataFrame((bench_fonds['poids'].to_numpy()*vl[bench_fonds['indice']].pct_change().dropna()).sum(1),
                                 columns=['indice']
                                 )
        benchmark = pd.concat([benchmark,
                               vl[bench_fonds['indice']].pct_change()
                               ],
                              axis=1)
        list_indice2 = bench_fonds['indice'].to_list()
        if bench_fonds['poids'].sum() != 100:
            st.write("**la somme des poids n'est pas égal à 100%**")

    # Base de données contenant le fonds, l'indice et ses composants
    df = pd.concat([portef,
                    benchmark],
                   axis=1).dropna()
    df_base100 = cum_returns(df/100,
                             starting_value=100)

    df['ecart'] = df['portef'] - df['indice']
    df_base100['ecart'] = 100*df_base100['portef']/df_base100['indice']

    df_base100.index = df_base100.index.strftime("%d/%m/%Y")
    st.dataframe(df_base100[['portef', 'indice']],
                 use_container_width=True
                 )
    df_base100.index = pd.to_datetime(df_base100.index, dayfirst=True)

    # Image
    st.image('Img.gif',
             use_column_width=True)


######################################################################################################
##################################### Onglet liste des sociétés ######################################
######################################################################################################
with tab2:
    aum_annuel_sdg = aum_annuel.groupby(
        level=2, axis=1).sum().iloc[-1].to_frame()
    aum_annuel_sdg.columns = ['AuM']
    aum_annuel_sdg = aum_annuel_sdg.sort_values('AuM', ascending=False)

    nbre_sdg, nbre_fds = aum_annuel_sdg.shape[0], vl.shape[1]
    aum_description = pd.concat([desc['FUND_MGMT_COMPANY'].value_counts().to_frame(),
                                 aum_annuel_sdg
                                 ],
                                axis=1).rename({'count': 'Nombre de fonds'}, axis=1)

    AuM_total_couvert = aum_description['AuM'].sum()
    AuM_total_median = aum_description['AuM'].median()

    st.write(f'Nombre de sociétés de gestion: {nbre_sdg}')
    st.write(f'Nombre de fonds: {nbre_fds}')
    st.write(f'Encours total: {human_format(AuM_total_couvert,1)}€')

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribution des AuM par classe d'actifs:**")
        st.write('')
        st.subheader('')
        st.dataframe(pd.concat([aum_annuel.ffill().iloc[-1].groupby(level=1)
                                .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[['count', '10%', '50%', '90%', 'mean']]
                                .assign(AuM=lambda x: x['mean']*x['count'])
                                .rename({'count': 'Nombre de fonds'}, axis=1)
                                .dropna()
                                [['Nombre de fonds', 'AuM', '10%', '50%', '90%']],
                                aum_annuel.ffill().iloc[-1]
                                .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[['count', '10%', '50%', '90%', 'mean']]
                                .rename({'count': 'Nombre de fonds'})
                                .to_frame().T
                                .rename(index=lambda s: 'Total')
                                .assign(AuM=lambda x: x['mean']*x['Nombre de fonds']),
                                ])
                     .assign(**{'AuM en %': lambda x: 2*x['AuM']/x['AuM'].sum()})
                     [['Nombre de fonds', 'AuM', 'AuM en %', '10%', '50%', '90%']]
                     .style.format({'Nombre de fonds': "{:.0f}",
                                    'AuM en %': "{:.1%}",
                                    '10%': lambda x: human_format(x, 1)+'€',
                                    '50%': lambda x: human_format(x, 1)+'€',
                                    '90%': lambda x: human_format(x, 1)+'€',
                                    'AuM': lambda x: human_format(x, 1)+'€',
                                    }
                                   ),
                     use_container_width=True
                     )
    with col2:
        vl_actif = vl.copy()
        vl_actif.columns = pd.MultiIndex.from_tuples(zip(vl_actif.columns,
                                                         vl_actif.columns.map(desc.set_index('LONG_COMP_NAME')[
                                                                              'FUND_ASSET_CLASS_FOCUS']),
                                                         )
                                                     )
        st.write("**Distribution par classe d'actifs:**")
        cols = st.columns(2, gap='small')
        periode = cols[1].radio("**Période pour calcul de performance:**",
                                ('1Y', '3Y', '5Y', '10Y', '20Y'),
                                index=2,
                                label_visibility='collapsed',
                                horizontal=True)
        if periode == '1Y':
            lag = 260
        elif periode == '3Y':
            lag = 3*260
        elif periode == '5Y':
            lag = 5*260
        elif periode == '10Y':
            lag = 10*260
        else:
            lag = 20*260

        risk_free_rate = cagr(bench.iloc[-lag:]['Eonia Capitalization Index Capital 5 Day'].pct_change())/252.
        analyse = cols[0].radio("**Analyse:**",
                                ('Performances', 'Volatilité', 'Sharpe'),
                                index=0,
                                label_visibility='collapsed',
                                horizontal=True
                                )

        if analyse == 'Performances':
            st.dataframe(pd.concat([vl_actif.iloc[-lag:].ffill().apply(lambda x: cagr(x.pct_change()) if x.isnull().sum() == 0 else np.nan)
                                    .groupby(level=[1])
                                    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[['count', '10%',  '50%', '90%']]
                                    .rename({'count': 'Nombre de fonds'}, axis=1)
                                    .dropna(),
                                    vl_actif.iloc[-lag:]
                                    .ffill()
                                    .apply(lambda x: cagr(x.pct_change()) if x.isnull().sum() == 0 else np.nan)
                                    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[['count', '10%',  '50%', '90%']]
                                    .rename({'count': 'Nombre de fonds'}).to_frame().T
                                    .rename(index=lambda s: 'Total')
                                    .dropna()
                                    ])
                         .style.format({'Nombre de fonds': "{:.0f}",
                                        '10%': "{:.2%}",
                                        '50%': "{:.2%}",
                                        '90%': "{:.2%}",
                                        }
                                       ),
                         use_container_width=True
                         )
        elif analyse == 'Volatilité':
            st.dataframe(pd.concat([vl_actif.iloc[-lag:].ffill().apply(lambda x: annual_volatility(x.pct_change()) if x.isnull().sum() == 0 else np.nan)
                                    .groupby(level=[1])
                                    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[['count', '10%',  '50%', '90%']]
                                    .rename({'count': 'Nombre de fonds'}, axis=1)
                                    .dropna(),
                                    vl_actif.iloc[-lag:]
                                    .ffill()
                                    .apply(lambda x: annual_volatility(x.pct_change()) if x.isnull().sum() == 0 else np.nan)
                                    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[['count', '10%',  '50%', '90%']]
                                    .rename({'count': 'Nombre de fonds'}).to_frame().T
                                    .rename(index=lambda s: 'Total')
                                    .dropna()
                                    ])
                         .style.format({'Nombre de fonds': "{:.0f}",
                                        '10%': "{:.2%}",
                                        '50%': "{:.2%}",
                                        '90%': "{:.2%}",
                                        }
                                       ),
                         use_container_width=True
                         )
        else:
            st.dataframe(pd.concat([vl_actif.iloc[-lag:].ffill().apply(lambda x: sharpe_ratio(x.pct_change(),
                                                                                              risk_free=risk_free_rate)
                                                                       if x.isnull().sum() == 0 else np.nan)
                                    .groupby(level=[1])
                                    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[['count', '10%',  '50%', '90%']]
                                    .rename({'count': 'Nombre de fonds'}, axis=1)
                                    .dropna(),
                                    vl_actif.iloc[-lag:]
                                    .ffill()
                                    .apply(lambda x: sharpe_ratio(x.pct_change(), risk_free=risk_free_rate)
                                           if x.isnull().sum() == 0 else np.nan)
                                    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[['count', '10%',  '50%', '90%']]
                                    .rename({'count': 'Nombre de fonds'}).to_frame().T
                                    .rename(index=lambda s: 'Total')
                                    .dropna()
                                    ])
                         .style.format({'Nombre de fonds': "{:.0f}",
                                        '10%': "{:.2f}",
                                        '50%': "{:.2f}",
                                        '90%': "{:.2f}",
                                        }
                                       ),
                         use_container_width=True
                         )

    col1, col2 = st.columns([1, 1], gap='small')
    with col1:
        st.caption("AuM global par classes d'actifs")
        st.plotly_chart(graph_bar(aum_annuel.groupby(
            level=1, axis=1).sum().loc['2005':], texte='AuM', colors=colors))
    with col2:
        st.caption("Collecte global par classes d'actifs")
        st.plotly_chart(graph_bar(collecte_annuel.groupby(
            level=1, axis=1).sum().loc['2006':], texte='collecte', colors=colors))

    col1, col2 = st.columns([1, 1], gap='small')
    with col1:
        st.caption('AuM par société de gestion:')
        aum_description['AuM moyen par fonds'] = aum_description['AuM'] / \
            aum_description['Nombre de fonds']
        aum_description = aum_description.sort_values('AuM', ascending=False)
        st.dataframe(aum_description.style.format({'AuM': lambda x: human_format(x, 1)+'€',
                                                   'AuM moyen par fonds': lambda x: human_format(x, 1)+'€'}
                                                  ),
                     use_container_width=True,
                     height=1250
                     )
    with col2:
        col21, col22 = st.columns(2)
        with col21:
            an = st.number_input('Période en année pour calcul de la collecte:',
                                 min_value=1,
                                 max_value=10,
                                 step=1,
                                 value=3)
        with col22:
            classes = ['Toutes classes',
                       *desc['FUND_ASSET_CLASS_FOCUS'].sort_values().unique()
                       ]
            selected_classe_collecte = st.selectbox("Choix de la classe d'actif",
                                                    classes,
                                                    key='selected_classe_collecte2'
                                                    )
        if selected_classe_collecte == 'Toutes classes':
            collecte_annuel_classe = collecte_annuel.droplevel(level=1, axis=1)
        else:
            collecte_annuel_classe = collecte_annuel.xs(
                selected_classe_collecte, axis=1, level=1)

        AM_Best_Lower_collecte = pd.concat([collecte_annuel_classe.rolling(an).sum().groupby(level=1, axis=1).sum().iloc[-1].nlargest(10).reset_index(),
                                            collecte_annuel_classe.rolling(an).sum().groupby(level=1, axis=1).sum().iloc[-1].nsmallest(10).reset_index()],
                                           axis=1)
        AM_Best_Lower_collecte.columns = [
            'Société de gestion ', 'Top 10 collecte', 'Société de gestion', 'Bottom 10 collecte']
        AM_Best_Lower_collecte[['Top 10 collecte', 'Bottom 10 collecte']] = AM_Best_Lower_collecte[[
            'Top 10 collecte', 'Bottom 10 collecte']].applymap(lambda x: human_format(x, 1)+'€')

        Fds_Best_Lower_collecte = pd.concat([collecte_annuel_classe.rolling(an).sum().groupby(level=0, axis=1).sum().iloc[-1].nlargest(10).reset_index(),
                                             collecte_annuel_classe.rolling(an).sum().groupby(level=0, axis=1).sum().iloc[-1].nsmallest(10).reset_index()],
                                            axis=1)
        Fds_Best_Lower_collecte.columns = [
            'Fonds ', 'Top 10 collecte', 'Fonds', 'Bottom 10 collecte']
        Fds_Best_Lower_collecte[['Top 10 collecte', 'Bottom 10 collecte']] = Fds_Best_Lower_collecte[[
            'Top 10 collecte', 'Bottom 10 collecte']].applymap(lambda x: human_format(x, 1)+'€')

        hide_table_row_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.caption('Société de gestion Top/Bottom 10 collecte')
        st.table(AM_Best_Lower_collecte)
        st.caption('Fonds Top/Bottom 10 collecte')
        st.table(Fds_Best_Lower_collecte)

    # Nombre de fonds crées par an et par classe d'actifs
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Nombre de fonds crée par an et par classe d'actifs")
        fonds_cree = desc.assign(YEAR=desc['FUND_INCEPT_DT'].dt.year)[['YEAR', 'FUND_ASSET_CLASS_FOCUS']] 
        table = pd.crosstab(fonds_cree['YEAR'], fonds_cree['FUND_ASSET_CLASS_FOCUS']).loc['1990':]
        fig = graph_bar_Nbre_fonds(table, colors=colors)        
        st.plotly_chart(fig, 
                        use_container_width=True)
    with col2:
        cols = st.columns(2)
        st.caption("Création par société de gestion")
        lag = cols[0].number_input("Nombre de fonds crée au cours des dernières années:", 
                              min_value=1, 
                              step=1,
                              value=3)
        typ = cols[1].multiselect("Choix type de fonds", 
                             desc.FUND_TYP.unique()
                             )
        if len(typ)>0:
            fonds_cree = desc.query("FUND_TYP==@typ")
        else: 
            fonds_cree = desc
            
        fonds_cree = fonds_cree.assign(YEAR=desc['FUND_INCEPT_DT'].dt.year)[['YEAR', 'FUND_MGMT_COMPANY', 'FUND_ASSET_CLASS_FOCUS']] 
        table = (pd.pivot_table(fonds_cree,
                               index='YEAR', 
                               columns='FUND_MGMT_COMPANY', 
                               aggfunc='count', fill_value=0)
                 .apply(lambda x: x.iloc[-lag:].sum())
                 .sort_values(ascending=False)
                 .droplevel(0)
                 .rename_axis(["Société de gestion"])
                 .rename({0:'Nombre de fonds'})
                 .to_frame()
                 .rename({0:'Nombre de fonds crée'}, axis=1)
                 )
        st.dataframe(table[table['Nombre de fonds crée']!=0],
                    use_container_width=True)

#######################################################################################
################################# classement des fonds     ############################
#######################################################################################
with tab3:
    st.subheader("Classement des fonds")
    cols = st.columns(4)

    selection_classe_actif = cols[0].selectbox("Choix de la classe d'actif",
                                               desc['FUND_ASSET_CLASS_FOCUS'].sort_values(
                                               ).unique(),
                                               index=2
                                               )
    desc2 = desc.query(
        "FUND_ASSET_CLASS_FOCUS==@selection_classe_actif").fillna('Non renseigné')

    selection_geo = cols[1].multiselect("Zone géographique:",
                                        desc2['FUND_GEO_FOCUS'].sort_values(
                                        ).unique(),
                                        )
    if len(selection_geo) != 0:
        desc2 = desc2.query("FUND_GEO_FOCUS in @selection_geo")

    selection_strategy = cols[2].selectbox("Style de gestion:",
                                           ['Tout',
                                            *desc2['FUND_STRATEGY'].sort_values().unique()],
                                           index=0
                                           )
    if selection_strategy != 'Tout':
        desc2 = desc2.query("FUND_STRATEGY in @selection_strategy")

    selection_mkt_cap = cols[3].selectbox("Capitalisation boursière:",
                                          ['Tout',
                                           *desc2['FUND_MKT_CAP_FOCUS'].sort_values().unique()],
                                          index=0
                                          )
    if selection_mkt_cap != 'Tout':
        desc2 = desc2.query("FUND_MKT_CAP_FOCUS==@selection_mkt_cap")

    perf_comparaison = vl[desc2.LONG_COMP_NAME].agg([lambda x: 100*cagr(x[-260:].pct_change()) if x[-260:].isnull().sum() == 0 else np.nan,
                                     lambda x: 100*cagr(x[-3*260:].pct_change()) if x[-3*260:].isnull().sum() == 0 else np.nan,
                                     lambda x: 100*cagr(x[-5*260:].pct_change()) if x[-5 *260:].isnull().sum() == 0 else np.nan,
                                     lambda x: 100 *cagr(x[-10*260:].pct_change()) if x[-10*260:].isnull().sum() == 0 else np.nan,
                                     lambda x: 100 *annual_volatility(x[-1*260:].pct_change()) if x[-260:].isnull().sum() == 0 else np.nan,
                                     lambda x: 100 *annual_volatility(x[-3*260:].pct_change()) if x[-3*260:].isnull().sum() == 0 else np.nan,
                                     lambda x: 100 *annual_volatility(x[-5*260:].pct_change()) if x[-5*260:].isnull().sum() == 0 else np.nan,
                                     lambda x: 100 *annual_volatility(x[-10*260:].pct_change()) if x[-10*260:].isnull().sum() == 0 else np.nan,
                                     lambda x: sharpe(x[-260:].pct_change(), 
                                                      risk_free_rdt=bench['Eonia Capitalization Index 7 Day'].iloc[-260:].pct_change()) 
                                     if x[-260:].isnull().sum() == 0 else np.nan,
                                     lambda x: sharpe(x[-3*260:].pct_change(), 
                                                      risk_free_rdt=bench['Eonia Capitalization Index 7 Day'].iloc[-3*260:].pct_change()) if x[-3*260:].isnull().sum() == 0 else np.nan,
                                     lambda x: sharpe(x[-5*260:].pct_change(), 
                                                      risk_free_rdt=bench['Eonia Capitalization Index 7 Day'].iloc[-5*260:].pct_change()) if x[-5*260:].isnull().sum() == 0 else np.nan,
                                     lambda x: sharpe(x[-10*260:].pct_change(), 
                                                      risk_free_rdt=bench['Eonia Capitalization Index 7 Day'].iloc[-10*260:].pct_change()) if x[-10*260:].isnull().sum() == 0 else np.nan,
                                     ],).set_axis(['Perf 1 an', 'Perf 3 ans', 'Perf 5 ans', 'Perf 10 ans',
                                                   'Volatilité 1 an', 'Volatilité 3 ans', 'Volatilité 5 ans','Volatilité 10 ans',
                                                   'Sharpe 1 an', 'Sharpe 3 ans', 'Sharpe 5 ans', 'Sharpe 10 ans']).T

    st.write(f'**Nombre de fonds analysé: {perf_comparaison.shape[0]}**')
    st.dataframe(perf_comparaison.style
                 .format({'Perf 1 an': "{:.1f}%",
                          'Perf 3 ans': "{:.1f}%",
                          'Perf 5 ans': "{:.1f}%",
                          'Perf 10 ans': "{:.1f}%",
                          'Volatilité 1 an': "{:.1f}%",
                          'Volatilité 3 ans': "{:.1f}%",
                          'Volatilité 5 ans': "{:.1f}%",
                          'Volatilité 10 ans': "{:.1f}%",
                          'Sharpe 1 an': "{:.2f}",
                          'Sharpe 3 ans': "{:.2f}",
                          'Sharpe 5 ans': "{:.2f}",
                          'Sharpe 10 ans': "{:.2f}"
                          }),
                 use_container_width=True
                 )
    cols = st.columns(5)
    period = cols[0].selectbox('Période',
                               ('1 an', '3 ans', '5 ans', '10 ans')
                               )
    if period=='1 an':
        perf_comparaison['Perf'] = perf_comparaison['Perf 1 an']
        perf_comparaison['Volatilité'] = perf_comparaison['Volatilité 1 an']
        perf_comparaison['Sharpe'] = perf_comparaison['Sharpe 1 an']
    elif period=='3 ans':    
        perf_comparaison['Perf'] = perf_comparaison['Perf 3 ans']
        perf_comparaison['Volatilité'] = perf_comparaison['Volatilité 3 ans']
        perf_comparaison['Sharpe'] = perf_comparaison['Sharpe 3 ans']
    elif period=='5 ans':
        perf_comparaison['Perf'] = perf_comparaison['Perf 5 ans']
        perf_comparaison['Volatilité'] = perf_comparaison['Volatilité 5 ans']
        perf_comparaison['Sharpe'] = perf_comparaison['Sharpe 5 ans']
    else:
        perf_comparaison['Perf'] = perf_comparaison['Perf 10 ans']
        perf_comparaison['Volatilité'] = perf_comparaison['Volatilité 10 ans']
        perf_comparaison['Sharpe'] = perf_comparaison['Sharpe 10 ans']

    fig = go.Figure(data=go.Scatter(x=perf_comparaison['Volatilité'],
                                    y=perf_comparaison['Perf'],
                                    mode="markers",
                                    marker_color='blue',
                                    marker=dict(size=12),
                                    customdata=np.stack((perf_comparaison.index,
                                                        perf_comparaison['Sharpe']),
                                                        axis=-1),
                                    hovertemplate='<br>'.join(['<b>%{customdata[0]}</b>',
                                                               'Perf: %{y:,.1f}%',
                                                               'Volat: %{x:,.1f}%',
                                                               'Sharpe: %{customdata[1]:,.2f}',
                                                               '<extra></extra>'
                                                               ]
                                                              ),
                                    showlegend=False
                                    )
                    )
    fig.update_layout(hoverlabel=dict(font_size=16),
                      xaxis_title="Volatilité",
                      yaxis_title="Performance",
                      font=dict(size=20),
                      xaxis_range=[perf_comparaison['Volatilité'].quantile(0.01)-2,
                                   perf_comparaison['Volatilité'].quantile(
                                       0.99)+2
                                   ],
                      yaxis_range=[perf_comparaison['Perf'].quantile(0.01)-2,
                                   perf_comparaison['Perf'].quantile(
                                       0.99)+2
                                   ],
                      )
    st.plotly_chart(fig, use_container_width=True)


################################################################################
######################### Onglet Analyse de la SDG #########################
################################################################################
with tab4:
    selected_am = st.selectbox("Choix de la société de gestion",
                               desc['FUND_MGMT_COMPANY'].sort_values().unique(),
                               )
    st.header(f"Analyse de la société de gestion : {selected_am}")
    col1, col2 = st.columns(2)
    aum_annuel_classe = aum_annuel.xs(selected_am,
                                      level=2,
                                      axis=1).groupby(level=1, axis=1).sum().replace(0, np.nan).dropna(how='all')
    collete_annuel_classe = collecte_annuel.xs(selected_am,
                                               level=2,
                                               axis=1).groupby(level=1, axis=1).sum().replace(0, np.nan).dropna(how='all')
    nom_gerant = pd.concat([desc.query('FUND_MGMT_COMPANY==@selected_am')['FUND_MGR'],
                            desc.query(
                                'FUND_MGMT_COMPANY==@selected_am')['CO_FUND_MGR']
                            ])

    fonds_sgp = desc.query(
        "FUND_MGMT_COMPANY == @selected_am")['LONG_COMP_NAME']
    nom_gerant = (nom_gerant.drop(
        nom_gerant[nom_gerant == '#N/A FIELD NOT APPLICABLE'].index))
    nombre_gerant = nom_gerant.nunique()
    
    with col1:
        st.plotly_chart(graph_bar(aum_annuel_classe,
                        texte='AuM', colors=colors))

    with col2:
        st.plotly_chart(graph_bar(collete_annuel_classe,
                        texte='Collecte', colors=colors))

    col1, col2 = st.columns(2)
    with col1:
        st.caption('Encours par fonds')
        aum_annuel_classe_fonds = aum_annuel.xs(
            selected_am, axis=1, level=2).iloc[-1].sort_index(axis=0, level=1).to_frame()
        aum_annuel_classe_fonds.columns = ['AuM']
        aum_annuel_classe_fonds['AuM en %'] = aum_annuel_classe_fonds['AuM'].div(
            aum_annuel_classe_fonds['AuM'].groupby(level=1).sum(), level=1)
        st.dataframe(aum_annuel_classe_fonds.style.format({'AuM': lambda x: human_format(x),
                                                           'AuM en %': "{:.1%}"}),
                     use_container_width=True
                     )
    with col2:
        aum_annuel_classe_last = aum_annuel_classe.iloc[-1, :].sort_index()
        st.caption("Répartition des encours de " +
                   aum_annuel_classe.index[-1].strftime("%Y"))
        fig = go.Figure(data=[go.Pie(labels=aum_annuel_classe_last.index,
                                     values=aum_annuel_classe_last,
                                     textinfo='label+percent',
                                     customdata=aum_annuel_classe.iloc[-1, :].apply(
                                         lambda x: human_format(x, 1)),
                                     hovertemplate="<b>Classe d'actif</b>: %{label}<br>"
                                     "<b>AuM</b>: %{customdata}€<br>"
                                     "<b>%Total</b>: %{percent:.1%}<br>"
                                     "<extra></extra>",
                                     marker_colors=aum_annuel_classe_last.index.map(
                                         colors),
                                     showlegend=False
                                     )
                              ]
                        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.caption('Nombre de fonds vs AuM')
        aum_last_year = aum_annuel.xs(selected_am, level=2, axis=1).droplevel(level=1, axis=1).iloc[-1]
        Nbre_fonds_encours = pd.cut(aum_last_year,
                                    [0, 50_000_000, 100_000_000, 500_000_000,
                                     1_000_000_000, 5_000_000_000, np.inf],
                                    labels=['<50M€', '50M€-100M€', '100M€-500M€',
                                            '500M€-1Mds€', '1Mds€-5Mds€', '>5Mds€'],
                                    ordered=False
                                    ).value_counts()
        fig = go.Figure(
            data=go.Bar(
                x=Nbre_fonds_encours.index,
                y=Nbre_fonds_encours,
                marker=dict(color='blue'),
                xperiodalignment="middle",
                hovertemplate='<br>'.join(['AuM: %{x}',
                                           'Nombre de fonds : %{y:,.0d}',
                                           '<extra></extra>'
                                           ]
                                          ),
                showlegend=False
            )
        )
        fig.update_xaxes(categoryorder='array',
                         categoryarray=['<50M€', '50M€-100M€', '100M€-500M€', '500M€-1Mds€', '1Mds€-5Mds€', '>5Mds€'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.caption("Nombre de fonds crée par an et par classe d'actifs")
        fonds_cree = desc.query("FUND_MGMT_COMPANY==@selected_am").assign(YEAR=desc['FUND_INCEPT_DT'].dt.year)[['YEAR', 'FUND_ASSET_CLASS_FOCUS']] 
        table = pd.crosstab(fonds_cree['YEAR'], fonds_cree['FUND_ASSET_CLASS_FOCUS']).loc['1994':]
        fig = graph_bar_Nbre_fonds(table, colors=colors)        
        st.plotly_chart(fig, 
                        use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.caption('Top 10 des fonds')
        classe_actif = st.selectbox("Choix de la classe d'actif",
                                    np.append('Total',
                                              desc.query(
                                                  'FUND_MGMT_COMPANY==@selected_am')['FUND_ASSET_CLASS_FOCUS'].sort_values().unique()
                                              )
                                    )

        if classe_actif == 'Total':
            aum_top_5 = aum_annuel.xs(
                selected_am, level=2, axis=1).iloc[-1].nlargest(10).droplevel(1).index
        else:
            aum_top_5 = aum_annuel.xs((selected_am, classe_actif), level=[
                                      2, 1], axis=1).iloc[-1].nlargest(10).index
        stats_best_5 = {
            "AuM": aum_annuel.droplevel([1, 2], axis=1)[aum_top_5].agg(lambda x: human_format(x[-1], 1)+'€'),
            "collecte 2023": collecte_annuel.droplevel([1, 2], axis=1)[aum_top_5].agg(lambda x: human_format(x[-1], 1)+'€'),
            "collecte 2022": collecte_annuel.droplevel([1, 2], axis=1)[aum_top_5].agg(lambda x: human_format(x[-2], 1)+'€'),
            "collecte 2021": collecte_annuel.droplevel([1, 2], axis=1)[aum_top_5].agg(lambda x: human_format(x[-3], 1)+'€'),
            "Collecte cumulée": collecte_annuel.droplevel([1, 2], axis=1)[aum_top_5].agg(lambda x: human_format(x[-1]+x[-2]+x[-3], 1)+'€'),
            "Perf 2023": vl[aum_top_5].resample('Y').last().aggregate(lambda x: (x[-1]/x[-2]-1)).map('{:,.1%}'.format),
            "Perf 2022": vl[aum_top_5].resample('Y').last().aggregate(lambda x: (x[-2]/x[-3]-1)).map('{:,.1%}'.format),
            "Perf 2021": vl[aum_top_5].resample('Y').last().aggregate(lambda x: (x[-3]/x[-4]-1)).map('{:,.1%}'.format),
            "Performance annuelle cumulée": vl[aum_top_5].resample('Y').last().aggregate(lambda x: x[-1]/x[-4]-1).map('{:,.1%}'.format)
        }
        st.dataframe(pd.DataFrame(stats_best_5),
                     use_container_width=True)

    with col2:
        st.caption('Statistiques sur la société de gestion')
        st.write("**Moyens humain :**")
        st.write(f'Nombre de gérants : {nombre_gerant}')
        st.write(f'Nombre de fonds gérés par la SGD : {len(fonds_sgp)}')
        num_format = "{:.2f}".format
        if nombre_gerant != 0:
            st.write(
                f'Nombre de fonds par gérant : {num_format(len(fonds_sgp)/nombre_gerant)}')
        st.write("**Actif sous gestion :**")
        st.write(
            f'AuM min : {human_format(aum_annuel.droplevel([0,1], axis=1)[selected_am].iloc[-1].min())}€ ({aum_annuel.xs(selected_am, axis=1, level=2).droplevel(1,axis=1).columns[aum_annuel.xs(selected_am, axis=1, level=2).droplevel(1,axis=1).iloc[-1].argmin()]})')
        st.write(
            f'AuM moyen : {human_format(aum_annuel.droplevel([0,1], axis=1)[selected_am].iloc[-1].mean())}€')
        st.write(
            f'AuM max : {human_format(aum_annuel.droplevel([0,1], axis=1)[selected_am].iloc[-1].max())}€ ({aum_annuel.xs(selected_am, axis=1, level=2).droplevel(1,axis=1).columns[aum_annuel.xs(selected_am, axis=1, level=2).droplevel(1,axis=1).iloc[-1].argmax()]})')
        st.write("**Collecte sur 3 ans :**")
        st.write(
            f'Collecte min : {human_format(collecte_annuel.droplevel([0,1], axis=1)[selected_am].iloc[-3:].sum().min())}€ ({collecte_annuel.xs(selected_am, axis=1, level=2).droplevel(1,axis=1).columns[collecte_annuel.xs(selected_am, axis=1, level=2).droplevel(1,axis=1).iloc[-1].argmin()]})')
        st.write(
            f'Collecte total : {human_format(collecte_annuel.droplevel([0,1], axis=1)[selected_am].iloc[-3:].sum().sum())}€')
        st.write(
            f'Collecte max : {human_format(collecte_annuel.droplevel([0,1], axis=1)[selected_am].iloc[-3:].sum().max())}€ ({collecte_annuel.xs(selected_am, axis=1, level=2).droplevel(1,axis=1).columns[collecte_annuel.xs(selected_am, axis=1, level=2).droplevel(1,axis=1).iloc[-1].argmax()]})')

#######################################################################################
########################## Onglet Analyse du fonds vs indice ##########################
#######################################################################################
with tab5:
    fonds = st.selectbox('Fonds à analyser',
                         ['Portefeuille',
                          *list_fonds]
                         )
    if fonds != 'Portefeuille':
        # Affichage du desc du fonds (s'il existe)
        st.header(f"**Analyse du fonds : {fonds}**")
        if desc.query("LONG_COMP_NAME==@fonds")['CIE_DES'].isnull().values.any():
            st.write('**Descriptif du fonds :** Non disponible')
        else:
            st.write('**Descriptif du fonds :**')
            st.write(desc.query("LONG_COMP_NAME==@fonds")['CIE_DES'].iloc[0])
        # Affichage du benchmark s'il existe
        if desc.query("LONG_COMP_NAME==@fonds")['FUND_BENCHMARK'].isnull().values.any():
            pass
        else:
            try:
                indice_fonds = desc.query(
                    "LONG_COMP_NAME==@fonds")['FUND_BENCHMARK'].iloc[0]
                indice_fonds = desc_indice.query(
                    "id==@indice_fonds")['description'].iloc[0]
                st.write(f'**Indice de référence du fonds :** {indice_fonds}')
            except:
                st.write("**Indice de référence du fonds :** Non diponible")

        # Affichage des données de AuM et collecte sur le fonds
        with st.expander(f"Encours et collecte du fonds **{fonds}**"):
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Evolution de l'Actif Net du fonds")
                fig = go.Figure(
                    go.Bar(
                        x=aum_annuel.xs(fonds, axis=1, level=0, drop_level=False).droplevel(
                            [1, 2], axis=1).dropna().squeeze().index.year,
                        y=aum_annuel.xs(fonds, axis=1, level=0, drop_level=False).droplevel(
                            [1, 2], axis=1).dropna().squeeze(),
                        marker=dict(color='blue'),
                        customdata=aum_annuel.xs(fonds, axis=1, level=0, drop_level=False).droplevel(
                            [1, 2], axis=1).dropna().squeeze().apply(lambda x: human_format(x, 1)),
                        xhoverformat="%Y",
                        xperiodalignment="middle",
                        hovertemplate='<br>'.join(['Année: %{x}',
                                                   'AuM: %{customdata}€',
                                                   '<extra></extra>'
                                                   ]
                                                  ),
                        showlegend=False
                    )
                )
                fig.update_xaxes(ticklabelmode="period",
                                 tickformat="%Y",
                                 )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.caption("Evolution de la collecte nette du fonds")
                fig = go.Figure(
                    data=go.Bar(
                        x=collecte_annuel.xs(fonds, axis=1, level=0, drop_level=False).droplevel(
                            [1, 2], axis=1).dropna().squeeze().index.year,
                        y=collecte_annuel.xs(fonds, axis=1, level=0, drop_level=False).droplevel(
                            [1, 2], axis=1).dropna().squeeze(),
                        marker=dict(color='blue'),
                        xhoverformat="%Y",
                        customdata=collecte_annuel.xs(fonds, axis=1, level=0, drop_level=False).droplevel(
                            [1, 2], axis=1).dropna().squeeze().apply(lambda x: human_format(x, 1)),
                        xperiodalignment="middle",
                        hovertemplate='<br>'.join(['Année: %{x}',
                                                   'Collecte: %{customdata}€',
                                                   '<extra></extra>'
                                                   ]
                                                  ),
                        showlegend=False
                    )
                )
                fig.update_xaxes(ticklabelmode="period",
                                 tickformat="%Y"
                                 )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(selection_fonds.set_index(
            'fonds').T.style.format('{:,.1f}%'))

    cols = st.columns(2)
    start_date = cols[0].date_input("Date de début",
                                    min_value=df.first_valid_index(),
                                    max_value=df.last_valid_index(),
                                    value=df.first_valid_index()
                                    )
    end_date = cols[1].date_input("Date de fin",
                                  min_value=df.first_valid_index(),
                                  max_value=df.last_valid_index(),
                                  value=df.last_valid_index()
                                  )
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Rebase df sur start_date
    df = df.loc[start_date:end_date]/100
    df_base100 = cum_returns(df, starting_value=100)

    col1, col2 = st.columns(2)
    # Calcul DD du fonds
    dd = to_drawdown_series(df['portef'] if fonds ==
                            'Portefeuille' else df[fonds])
    dd_info_fonds = drawdown_details(dd).sort_values(
        by='max drawdown', ascending=True)
    dd_info_fonds = dd_info_fonds[[
        'start', 'valley', 'end', 'max drawdown', 'days']][:3]
    dd_info_fonds.columns = ['Début', 'Creux', 'Fin',
                             'Drawdown', 'Durée en nombre de jours']
    dd_info_fonds = dd_info_fonds.set_index(['Début', 'Creux', 'Fin'])

    # Calcul DD de l'indice
    dd = to_drawdown_series(df['indice'])
    dd_info_indice = drawdown_details(dd).sort_values(
        by='max drawdown', ascending=True)
    dd_info_indice = dd_info_indice[[
        'start', 'valley', 'end', 'max drawdown', 'days']][:3]
    dd_info_indice.columns = ['Début', 'Creux',
                              'Fin', 'Drawdown', 'Durée en nombre de jours']
    dd_info_indice = dd_info_indice.set_index(['Début', 'Creux', 'Fin'])

    with col1:
        st.caption("Evolution du fonds vs indice")
        fig = go.Figure(
            data=[
                go.Scatter(x=df_base100['portef'].index,
                           y=df_base100['portef'] if fonds == 'Portefeuille' else df_base100[fonds],
                           name='Portefeuille' if fonds == 'Portefeuille' else fonds,
                           marker=dict(color='blue'),
                           ),
                go.Scatter(x=df_base100['indice'].index,
                           y=df_base100['indice'],
                           name="Indice",
                           marker=dict(color='red'),
                           ),
            ])
        fig.update_layout(hovermode='x unified',
                          xaxis_range=(df_base100['portef'].first_valid_index(),
                                       df_base100['portef'].last_valid_index()
                                       )
                          )
        if dd_info_fonds.shape[0] >= 2:
            fig.add_vrect(x0=dd_info_fonds.reset_index()['Début'][0],
                          x1=dd_info_fonds.reset_index()['Creux'][0],
                          annotation_text=str("{0:.1f}%".format(
                              dd_info_fonds.reset_index()['Drawdown'][0])),
                          annotation_position="top left",
                          annotation_font_size=12,
                          annotation_font_color="blue",
                          fillcolor="blue",
                          opacity=0.15,
                          line_width=0)
            fig.add_vrect(x0=dd_info_fonds.reset_index()['Début'][1],
                          x1=dd_info_fonds.reset_index()['Creux'][1],
                          annotation_text=str("{0:.1f}%".format(
                              dd_info_fonds.reset_index()['Drawdown'][1])),
                          annotation_position="top left",
                          annotation_font_size=12,
                          annotation_font_color="blue",
                          fillcolor="blue",
                          opacity=0.15,
                          line_width=0)
        elif dd_info_fonds.shape[0] == 1:
            fig.add_vrect(x0=dd_info_fonds.reset_index()['Début'][0],
                          x1=dd_info_fonds.reset_index()['Creux'][0],
                          annotation_text=str("{0:.1f}%".format(
                              dd_info_fonds.reset_index()['Drawdown'][0])),
                          annotation_position="top left",
                          annotation_font_size=12,
                          annotation_font_color="blue",
                          fillcolor="blue",
                          opacity=0.15,
                          line_width=0)
        if dd_info_indice.shape[0] >= 2:
            fig.add_vrect(x0=dd_info_indice.reset_index()['Début'][0],
                          x1=dd_info_indice.reset_index()['Creux'][0],
                          annotation_text=str("{0:.1f}%".format(
                              dd_info_indice.reset_index()['Drawdown'][0])),
                          annotation_position="bottom left",
                          annotation_font_size=12,
                          annotation_font_color="red",
                          fillcolor="red",
                          opacity=0.15,
                          line_width=0)
            fig.add_vrect(x0=dd_info_indice.reset_index()['Début'][1],
                          x1=dd_info_indice.reset_index()['Creux'][1],
                          annotation_text=str("{0:.1f}%".format(
                              dd_info_indice.reset_index()['Drawdown'][1])),
                          annotation_position="bottom left",
                          annotation_font_size=12,
                          annotation_font_color="red",
                          fillcolor="red",
                          opacity=0.15,
                          line_width=0)
        elif dd_info_indice.shape[0] == 1:
            fig.add_vrect(x0=dd_info_indice.reset_index()['Début'][0],
                          x1=dd_info_indice.reset_index()['Creux'][0],
                          annotation_text=str("{0:.1f}%".format(
                              dd_info_indice.reset_index()['Drawdown'][0])),
                          annotation_position="bottom left",
                          annotation_font_size=12,
                          annotation_font_color="red",
                          fillcolor="red",
                          opacity=0.15,
                          line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # Calcul DD perf relative
    dd = to_drawdown_series(df['ecart'] if fonds == 'Portefeuille' else df[fonds]-df['indice'])
    dd_info = drawdown_details(dd).sort_values(by='max drawdown', ascending=True)
    dd_info = dd_info[['start', 'valley', 'end', 'max drawdown', 'days']][:3]
    dd_info.columns = ['Début', 'Creux', 'Fin',
                       'Drawdown', 'Durée en nombre de jours']
    dd_info = dd_info.set_index(['Début', 'Creux', 'Fin'])
    with col2:
        st.caption("Performance relative fonds vs indice")
        fig = go.Figure(data=go.Scatter(x=df_base100['ecart'].index,
                                        y=100*df_base100['portef']/df_base100['indice'] if fonds == 'Portefeuille' else 100 *
                                        df_base100[fonds]/df_base100['indice'],
                                        marker_color='blue',
                                        hovertemplate='<br>'.join(['Perf. Relative: %{y:,.1f}',
                                                                   '<extra></extra>'
                                                                   ]
                                                                  ),
                                        showlegend=False
                                        )
                        )
        fig.add_vrect(x0=dd_info.reset_index()['Début'][0],
                      x1=dd_info.reset_index()['Creux'][0],
                      annotation_text=str("{0:.1f}%".format(
                          dd_info.reset_index()['Drawdown'][0])),
                      annotation_position="top left",
                      annotation_font_size=12,
                      annotation_font_color="blue",
                      fillcolor="blue",
                      opacity=0.15,
                      line_width=0)
        fig.update_layout(hovermode='x unified',
                          xaxis_range=(df['ecart'].first_valid_index(),
                                       df['ecart'].last_valid_index()
                                       )
                          )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    data_annuel = 100*df_base100.resample('Y').last().pct_change().dropna()
    data_annuel['ecart'] = data_annuel['portef'] - data_annuel['indice']

    with col1:
        st.caption(
            f"Performances annuelles comparées {'portefeuille' if fonds=='Portefeuille' else fonds} vs indice")
        fig = go.Figure(
            data=[go.Bar(x=data_annuel.index.year,
                         y=data_annuel['portef'] if fonds == 'Portefeuille' else data_annuel[fonds],
                         name='Portefeuille' if fonds == 'Portefeuille' else fonds,
                         marker=dict(color='blue'),
                         xhoverformat="%Y",
                         xperiodalignment="middle",
                         hovertemplate='<br>'.join(['Année: %{x}',
                                                   'Portefeuille: %{y:,.1f}%',
                                                    '<extra></extra>'
                                                    ]
                                                   ),
                         showlegend=False),
                  go.Bar(x=data_annuel.index.year,
                         y=data_annuel['indice'],
                         name="Indice",
                         marker=dict(color='red'),
                         xhoverformat="%Y",
                         xperiodalignment="middle",
                         hovertemplate='<br>'.join(['Année: %{x}',
                                                    'Indice: %{y:,.1f}%',
                                                    '<extra></extra>'
                                                    ]
                                                   ),
                         showlegend=False)
                  ]
        )
        fig.update_xaxes(ticklabelmode="period",
                         tickformat="%Y"
                         )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.caption("Ecart de performance annuel")
        fig = go.Figure(data=go.Bar(x=data_annuel.index.year,
                                    y=data_annuel['ecart'] if fonds == 'Portefeuille' else data_annuel[fonds] -
                                    data_annuel['indice'],
                                    marker_color=np.where(data_annuel['ecart'] < 0,
                                                          'red',
                                                          'green'
                                                          ),
                                    xhoverformat="%Y",
                                    xperiodalignment="middle",
                                    hovertemplate='<br>'.join(['Année: %{x}',
                                                               'Ecart: %{y:,.1f}%',
                                                               '<extra></extra>'
                                                               ]
                                                              ),
                                    showlegend=False
                                    )
                        )
        fig.update_xaxes(ticklabelmode="period",
                         tickformat="%Y"
                         )
        st.plotly_chart(fig, use_container_width=True)

    data_annuel = data_annuel.reset_index().rename({'index': 'date'}, axis=1)
    data_annuel['date'] = data_annuel['date'].dt.strftime('%Y')
    data_annuel = data_annuel.set_index('date')

    if fonds != 'Portefeuille':
        data_annuel['portef'] = data_annuel[fonds]
        data_annuel['ecart'] = data_annuel[fonds] - data_annuel['indice']

    col1, col2 = st.columns(2)
    with col1:
        st.caption(
            f"Performance annuelle {'portefeuille' if fonds=='Portefeuille' else fonds}")
        st.dataframe(data_annuel[['portef', 'indice', 'ecart']].style.format('{:,.2f}%').applymap(custom_styling),
                     use_container_width=True,
                     height=600
                     )

    with col2:
        selected_cone = st.selectbox("Cône de performance",
                                     ['Performance absolue',
                                         'Performance relative'],
                                     )
        alpha = st.number_input('Intervalle de confiance (en %)',
                                value=10,
                                min_value=0,
                                max_value=100,
                                step=5)
        if selected_cone == 'Performance absolue':
            cone_perf = 100*pd.DataFrame([np.nanquantile(df_base100['portef'].pct_change(h),
                                                         [alpha/100.0, 0.5, 1-alpha/100.0])
                                          for h in np.arange(1, 3*253)],
                                         columns=['D'+str(alpha)+'%',
                                                  'med',
                                                  'D'+str(100-alpha)+'%']
                                         )
            data = [go.Scatter(name=cone_perf.columns[0],
                               x=cone_perf.index,
                               y=cone_perf.iloc[:, 0],
                               hovertemplate='<br>'.join(['Perf: %{y:,.1f}%']),
                               showlegend=False,
                               line_color='red',
                               ),
                    go.Scatter(name=cone_perf.columns[1],
                               x=cone_perf.index,
                               y=cone_perf.iloc[:, 1],
                               hovertemplate='<br>'.join(['Perf: %{y:,.1f}%']),
                               showlegend=False,
                               line_color='blue',
                               ),
                    go.Scatter(name=cone_perf.columns[2],
                               x=cone_perf.index,
                               y=cone_perf.iloc[:, 2],
                               hovertemplate='<br>'.join(['Perf: %{y:,.1f}%']),
                               showlegend=False,
                               line_color='red',
                               )
                    ]
            fig = go.Figure(data)
            fig.update_layout(xaxis=dict(tickmode='array',
                                         tickvals=[25, 6*25, 252,
                                                   378, 2*252, 3*252],
                                         ticktext=['1M', '6M', '1Y',
                                                   '18M', '2Y', '3Y']
                                         )
                              )
            st.plotly_chart(fig, use_container_width=True)
        else:
            cone_alpha = 100*pd.DataFrame([np.nanquantile(df_base100['ecart'].pct_change(h),
                                                          [alpha/100.0, 0.5, 1-alpha/100.0])
                                           for h in np.arange(1, 3*253)],
                                          columns=['D'+str(alpha)+'%',
                                                   'med',
                                                   'D'+str(100-alpha)+'%'])
            data = [go.Scatter(name=cone_alpha.columns[0],
                               x=cone_alpha.index,
                               y=cone_alpha.iloc[:, 0],
                               hovertemplate='<br>'.join(['Perf: %{y:,.1f}%']),
                               showlegend=False,
                               line_color='red',
                               ),
                    go.Scatter(name=cone_alpha.columns[1],
                               x=cone_alpha.index,
                               y=cone_alpha.iloc[:, 1],
                               hovertemplate='<br>'.join(['Perf: %{y:,.1f}%']),
                               showlegend=False,
                               line_color='blue',
                               ),
                    go.Scatter(name=cone_alpha.columns[2],
                               x=cone_alpha.index,
                               y=cone_alpha.iloc[:, 2],
                               hovertemplate='<br>'.join(['Perf: %{y:,.1f}%']),
                               showlegend=False,
                               line_color='red',
                               )
                    ]
            fig = go.Figure(data)
            fig.update_layout(xaxis=dict(tickmode='array',
                                         tickvals=[25, 6*25, 252,
                                                   378, 2*252, 3*252],
                                         ticktext=['1M', '6M', '1Y',
                                                   '18M', '2Y', '3Y']
                                         )
                              )
            st.plotly_chart(fig, use_container_width=True)

    # Fréquence hebdo pour calcul stat
    df_base100_hebdo = df_base100.resample('W').last()
    df_hebdo = 100*df_base100_hebdo.pct_change().dropna()

    # Calcul de Loi Normale avec même (mu, sigma) sur les données hebdo
    col1, col2 = st.columns(2)
    with col1:
        st.caption('Distribution du fonds vs indice')
        fig = ff.create_distplot([df_hebdo['portef'], df_hebdo['indice']],
                                 ['Portefeuille', 'Indice'],
                                 bin_size=(df_hebdo['portef'].max(
                                 )-df_hebdo['portef'].min())/40,
                                 curve_type='normal',
                                 histnorm='probability',
                                 colors=['blue', 'red']
                                 )
        fig.add_vline(x=df['portef'].mean(),
                      line_dash="dot",
                      annotation_text=f"Moyenne fonds {df_hebdo['portef'].mean():.2}%",
                      annotation_position="top left",
                      annotation_font_size=15,
                      annotation_font_color="black"
                      )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.caption("Distribtion de l'écart de performance")
        fig = ff.create_distplot([df_hebdo['ecart']],
                                 ['Ecart de performance'],
                                 bin_size=(df_hebdo['ecart'].max(
                                 )-df_hebdo['ecart'].min())/40,
                                 curve_type='normal',
                                 histnorm='probability',
                                 colors=['blue']
                                 )
        fig.add_vline(x=df_hebdo['ecart'].mean(),
                      line_dash="dot",
                      annotation_text=f"Moyenne de l'écart {df_hebdo['ecart'].mean():.2f}%",
                      annotation_position="top left",
                      annotation_font_size=15,
                      annotation_font_color="black",
                      )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    # Regression
    with col1:
        Lags = st.number_input('Période glissante (en année)',
                               value=3.,
                               min_value=0.0,
                               max_value=5.0,
                               step=0.5)

    with col2:
        int_conf = st.number_input('Intervalle de confiance (en %)',
                                   value=5.0,
                                   min_value=2.5,
                                   max_value=50.0,
                                   step=2.5)
    Lags = int(52*Lags)

    st.subheader(
        f'Analyse glissante sur {Lags/52} {"an" if Lags/52 <= 1 else "ans"}')
    rdt_portef_glissant, rdt_bench_glissant, ecart_perf_glissant = calcul_perf_glissant(df_hebdo['portef']/100, 
                                                                                        df_hebdo['indice']/100, 
                                                                                        window=Lags, 
                                                                                        period='weekly'
                                                                                        )
    df_hebdo['ecart'] = df_hebdo['portef']-df_hebdo['indice']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption('Alpha glissant')
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ecart_perf_glissant.index,
                                 y=ecart_perf_glissant,
                                 name='Ecart de performance',
                                 marker=dict(color='blue')
                                 )
                      )
        fig.add_trace(go.Scatter(x=rdt_portef_glissant.index,
                                 y=rdt_portef_glissant['portef'],
                                 mode='lines',
                                 name='Performance',
                                 marker=dict(color='red')
                                 )
                      )
        fig.update_layout(legend=dict(orientation="h",
                                      entrywidth=170,
                                      yanchor="bottom",
                                      y=1.02,
                                      xanchor="right",
                                      x=1
                                      )
                          )

        st.plotly_chart(fig, use_container_width=True)

    TE = np.sqrt(52)*df_hebdo['ecart'].rolling(Lags).std().dropna()
    IR = ecart_perf_glissant/TE
    with col2:
        st.caption("Tracking-Error glissant")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=TE.index,
                                 y=TE,
                                 mode='lines',
                                 name='Tracking-Error',
                                 marker=dict(color='blue')
                                 )
                      )
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.caption("Ratio d'information glissant")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=IR.index,
                                 y=IR,
                                 mode='lines',
                                 name="Ratio d'information",
                                 marker=dict(color='blue')
                                 )
                      )
        st.plotly_chart(fig, use_container_width=True)

    # Regression linéaire
    st.caption("Régression linéaire")
    model = sm.OLS(df_hebdo['portef'],
                   sm.add_constant(df_hebdo['indice'])
                   )

    results = model.fit()
    alpha, beta, r_squared = results.params[0], results.params[1], results.rsquared
    x_range = np.linspace(df_hebdo['indice'].min(),
                          df_hebdo['indice'].max(),
                          100)
    y_range = alpha + beta*x_range
    color_range = results.resid.apply(lambda x: 'blue' if x >= 0 else 'red')

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hebdo['indice'],
                                 y=df_hebdo['portef'],
                                 mode='markers',
                                 marker=dict(color=color_range),
                                 showlegend=False,
                                 customdata=df_hebdo['portef'].index.date,
                                 hovertemplate='<br>'.join(['Année: %{customdata}',
                                                            'Portefeuille: %{y:.2f}%',
                                                            'Indice: %{x:.2f}%'
                                                            ]
                                                           ),
                                 )
                      )
        fig.add_traces(go.Scatter(x=x_range,
                                  y=y_range,
                                  showlegend=False,
                                  line=dict(color='green',
                                            width=3
                                            )
                                  )
                       )
        line1 = 'y = ' + str(round(alpha, 2)) + ' + ' + str(round(beta, 2))+'x'
        line2 = 'R^2 = ' + str(round(100*r_squared, 1)) + '%'
        summary = line1 + '<br>' + line2

        fig.add_annotation(
            x=df_hebdo['indice'].min(),
            y=df_hebdo['portef'].max(),
            text=summary,
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
            ),
            align="right",
            arrowcolor="#636363",
            ax=20,
            ay=-30,
            borderwidth=2,
            borderpad=4,
            bgcolor="rgba(100,100,100, 0.6)",
            opacity=0.8
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        alpha_beta = alpha_beta(df_hebdo['portef']/100,
                                df_hebdo['indice']/100,
                                risk_free=0.0,
                                period='daily')
        alpha_beta = pd.DataFrame(
            alpha_beta, index=['alpha annualisé', 'beta'])

        up_alpha_beta = up_alpha_beta(df_hebdo['portef']/100,
                                      df_hebdo['indice']/100,
                                      risk_free=0.0,
                                      period='daily'
                                      )
        up_alpha_beta = pd.DataFrame(
            up_alpha_beta, index=['alpha annualisé', 'beta'])

        down_alpha_beta = down_alpha_beta(df_hebdo['portef']/100,
                                          df_hebdo['indice']/100,
                                          risk_free=0.0,
                                          period='daily')
        down_alpha_beta = pd.DataFrame(down_alpha_beta, index=[
                                       'alpha annualisé', 'beta'])

        tab_alpha_beta = pd.concat([alpha_beta.rename({0: 'Période globale'}, axis=1).T,
                                    up_alpha_beta.rename(
                                        {0: 'Période haussière'}, axis=1).T,
                                    down_alpha_beta.rename({0: 'Période baissière'}, axis=1).T]
                                   )
        st.caption("Regréssion")
        st.dataframe(tab_alpha_beta.style.format({'alpha annualisé': "{:.2%}",
                                                  'beta': "{:.2f}"}
                                                 ),
                     width=350
                     )

    # Rolling regression
    rols = RollingOLS(df_hebdo['portef'],
                      sm.add_constant(df_hebdo['indice']),
                      window=Lags,
                      expanding=False)
    rres = rols.fit()

    # Paramètres et intervalle de confiance des paramètres
    TE = np.sqrt(52*rres.mse_resid).to_frame().rename(columns={0: 'TE'})

    coef_int = rres.conf_int(alpha=int_conf/100.)
    coef_int.columns = ["_".join(col)
                        for col in coef_int.columns.to_flat_index()]

    coef = pd.concat([coef_int,
                      rres.params,
                      100*rres.rsquared.to_frame().rename(columns={0: 'R2'}),
                      TE
                      ],
                     axis=1).dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Evolution du R2")
        fig = go.Figure(data=go.Scatter(
            x=coef.index,
            y=coef['R2'],
            marker=dict(color='blue'),
            xhoverformat="%Y",
            xperiodalignment="middle",
            hovertemplate='<br>'.join(['Année: %{x}',
                                       'R2: %{y:,.2s}%',
                                       '<extra></extra>'
                                       ]
                                      ),
            showlegend=False)
                        )
        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.caption("Evolution de la tracking-error")
            fig = go.Figure(
                data=go.Scatter(
                    x=coef.index,
                    y=coef['TE'],
                    marker=dict(color='blue'),
                    xhoverformat="%Y",
                    xperiodalignment="middle",
                    hovertemplate='<br>'.join(['Année: %{x}',
                                               'TE: %{y:,.2s}%',
                                               '<extra></extra>'
                                               ]
                                              ),
                    showlegend=False
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Evolution du beta")
        fig = go.Figure([
            go.Scatter(
                name='Beta',
                x=coef.index,
                y=coef['indice'],
                mode='lines',
                line=dict(color='blue'),
                showlegend=False
            ),
            go.Scatter(
                name='Upper Bound',
                x=coef.index,
                y=coef['indice_upper'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=coef.index,
                y=coef['indice_lower'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ])        
        fig.update_layout(hovermode="x")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.caption("Evolution de l'alpha")
        fig = go.Figure([
            go.Scatter(
                name='Alpha',
                x=coef.index,
                y=coef['const'],
                mode='lines',
                line=dict(color='blue'),
                showlegend=False
                ),
            go.Scatter(
                name='Borne sup',
                x=coef.index,
                y=coef['const_upper'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Borne inf',
                x=coef.index,
                y=coef['const_lower'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        ])
        fig.update_layout(hovermode="x")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        IR = np.sqrt(52)*coef['const']/coef['TE']
        st.caption("Evolution du ratio d'information")
        fig = go.Figure([go.Scatter(name='IR',
                                    x=IR.index,
                                    y=IR,
                                    mode='lines',
                                    line=dict(color='blue'),
                                    showlegend=False
                                    )
                         ]
                        )
        fig.update_layout(hovermode="x")
        st.plotly_chart(fig, use_container_width=True)

#######################################################################################
############################## Analyse statistiques  #################################
#######################################################################################
with tab6:
    col1, col2, col3 = st.columns(3)
    risk_free = 100*cagr(bench.loc[start_date:end_date,
                         'Eonia Capitalization Index Capital 5 Day'].pct_change())

    with col1:
        risk_free_rate = st.number_input('Taux sans risque (en %)',
                                         min_value=-2.,
                                         max_value=20.,
                                         value=round(risk_free, 2),
                                         format="%f")
        risk_free_rate = risk_free_rate/100
    with col2:
        var_level = st.number_input('seuil pour VaR (en %)',
                                    min_value=1.0,
                                    max_value=10.,
                                    value=5.0,
                                    step=1.,
                                    format="%f")
        var_level = var_level/100.

    with col3:
        frequence_data = st.selectbox('Fréquence des données',
                                      ('Daily', 'Weekly', 'Monthly'),
                                      index=0)
        if frequence_data == 'Daily':
            periods_per_year = 252.
            frequence_data = 'daily'
        elif frequence_data == 'Weekly':
            periods_per_year = 52.
            frequence_data = 'weekly'
            df_base100 = df_base100.resample('W').last()
            df = df_base100.pct_change().dropna()
        elif frequence_data == 'Monthly':
            periods_per_year = 12.
            frequence_data = 'monthly'
            df_base100 = df_base100.resample('M').last()
            df = df_base100.pct_change().dropna()
    risk_free_rate = risk_free_rate/periods_per_year

    col1, col2 = st.columns(2, gap='large')
    with col1:
        sum_stat_perf = pd.concat([summary_stats_perf(df[['portef', 'indice']],
                                                      risk_free_rate=risk_free_rate,
                                                      period=frequence_data),
                                   summary_stats_perf(df['ecart'].to_frame(),
                                                      risk_free_rate=0,
                                                      period=frequence_data)
                                   ],
                                  axis=1)
        sum_stat_perf.loc[['Up ratio', 'Down ratio'], 'ecart'] = 'nd'
        sum_stat_risk = pd.concat([summary_stats_risk(df[['portef', 'indice']],
                                                      period=frequence_data,
                                                      var_level=var_level),
                                   summary_stats_risk(df['ecart'].to_frame(),
                                                      period=frequence_data,
                                                      var_level=var_level)
                                   ],
                                  axis=1)
        sum_stat_ratio = pd.concat([summary_stats_ratio(df[['portef', 'indice']],
                                                        risk_free_rate=risk_free_rate,
                                                        period=frequence_data),
                                   summary_stats_ratio(df['ecart'].to_frame(),
                                                       risk_free_rate=0,
                                                       period=frequence_data)
                                    ],
                                   axis=1)
        sum_stat_glissant = pd.concat([summary_stats_glissant(df[['portef', 'indice']],
                                                              period=frequence_data),
                                       summary_stats_glissant(df['ecart'].to_frame(),
                                                              period=frequence_data)
                                       ],
                                      axis=1)

        st.subheader('Analyse de performance :')
        st.dataframe(sum_stat_perf
                     .style
                     .set_properties(**{"background-color": "white", "color": "black", "border-color": "black", 'text-align': 'center'})
                     .format("{:.2f}%", subset=(['Taux sans risque', 'Ann. return', '% Rdt>0', 'Average up', 'Average Down'],
                                                slice(None))
                             )
                     .format("{:.2f}%", subset=(['Up ratio', 'Down ratio'],
                                                ['portef', 'indice'])
                             )
                     .format("{:.2f}", subset=(['Skewness', 'kurtosis'],
                                               slice(None))
                             ),
                     use_container_width=True,
                     height=450
                     )
        st.divider()
        
        st.subheader('Analyse de performance glissante:')
        st.dataframe(sum_stat_glissant
                    .style
                    .set_properties(**{"background-color": "white", "color": "black", "border-color": "black", 'text-align': 'center'})
                    .format("{:.2f}%", subset=(['Perf 2023', 'Perf 2022', 'Perf 2021', 'Perf 1 an', 'Perf 3 ans', 'Perf 5 ans', 
                                                'Ann. vol 1 an', 'Ann. vol 3 ans', 'Ann. vol 5 ans'],
                                                slice(None))
                             )
                     .format("{:.2f}", subset=(['Sharpe ratio 1 an', 'Sharpe ratio 3 ans', 'Sharpe ratio 5 ans'],
                                               slice(None))
                             ),
                     use_container_width=True,
                     height=450
                     )
        st.divider()
        
        st.subheader('Analyse de risque :')
        st.dataframe(sum_stat_risk.style
                     .format("{:.2f}%", subset=(['Ann. vol', 'Historic Var', 'Historic CVar', 'Max Drawdown', 'Drawdown average'],
                                                slice(None))
                             )
                     .format("{:.2f}", subset=(['Tail'],
                                               slice(None))
                             ),
                     use_container_width=True,
                     height=240
                     )
        st.divider()
        st.subheader('Ratios financiers:')
        st.dataframe(sum_stat_ratio.style.format("{:.2f}"),
                     use_container_width=True,
                     )

    with col2:
        st.subheader('Analyse du drawdown du fonds')
        dd = to_drawdown_series(df['portef'])
        dd_info = drawdown_details(dd).sort_values(
            by='max drawdown', ascending=True)
        dd_info = dd_info[['start', 'valley', 'end', 'max drawdown']][:10]
        dd_info.columns = ['Début', 'Creux', 'Fin', 'Drawdown']
        dd_info["Début -> Creux"] = dd_info.apply(lambda row: diff_date(pd.to_datetime(row['Creux']),
                                                                        pd.to_datetime(
                                                                            row['Début'])
                                                                        ),
                                                  axis=1)

        dd_info["Creux -> Fin"] = dd_info.apply(lambda row: diff_date(pd.to_datetime(row['Creux']),
                                                                      pd.to_datetime(
                                                                          row['Fin'])
                                                                      ),
                                                axis=1)

        dd_info = dd_info.set_index(['Début', 'Creux', 'Fin'])

        st.dataframe(dd_info
                     .style
                     .format("{:.1f}%", subset=(slice(None), 'Drawdown')),
                     use_container_width=True
                     )

        st.subheader("Analyse du drawdown de l'écart de performance")
        dd = to_drawdown_series(df['ecart'])
        dd_info = drawdown_details(dd).sort_values(
            by='max drawdown', ascending=True)
        dd_info = dd_info[['start', 'valley', 'end', 'max drawdown']][:10]
        dd_info.columns = ['Début', 'Creux', 'Fin', 'Drawdown']
        dd_info["Début -> Creux"] = dd_info.apply(lambda row: diff_date(pd.to_datetime(row['Creux']),
                                                                        pd.to_datetime(
                                                                            row['Début'])
                                                                        ),
                                                  axis=1)

        dd_info["Creux -> Fin"] = dd_info.apply(lambda row: diff_date(pd.to_datetime(row['Creux']),
                                                                      pd.to_datetime(
                                                                          row['Fin'])
                                                                      ),
                                                axis=1)

        dd_info = dd_info.set_index(['Début', 'Creux', 'Fin'])

        st.dataframe(dd_info
                     .style
                     .format("{:.1f}%", subset=(slice(None), 'Drawdown')),
                     use_container_width=True
                     )

#######################################################################################
################################# Analyse style   #####################################
#######################################################################################
with tab7:
    st.subheader("**Entrez l'indice pour comparaison:**")
    nombre_fonds = st.number_input('**Nombre de fonds pour analyse:**',
                                   min_value=1,
                                   value=1,
                                   step=1,
                                   key='nombre_fonds3'
                                   )
    selections = []
    for row in range(nombre_fonds):
        selections.append(selection_indice_analyse_style(row, key=2))

    bench_indice = pd.DataFrame(selections,
                                columns=['indice']
                                )
    selected_indices_analyse_style = bench_indice['indice'].to_list()
    data_analyse_style = pd.concat([df_base100['portef'],
                                    bench[selected_indices_analyse_style]],
                                   axis=1).resample('W').last().pct_change().dropna()

    analyse_style_period = st.radio("Période d'analyse :",
                                    ('1 an', '3 ans', '5 ans', 'All'),
                                    horizontal=True)
    if analyse_style_period == '1 an':
        data_analyse_style = 100*data_analyse_style.iloc[-52:]
    elif analyse_style_period == '3 ans':
        data_analyse_style = 100*data_analyse_style.iloc[-3*52:]
    elif analyse_style_period == '5 ans':
        data_analyse_style = 100*data_analyse_style.iloc[-5*52:]
    else:
        data_analyse_style = 100*data_analyse_style

    cols = st.columns(3)
    if len(selected_indices_analyse_style) >= 2:
        weight_analyse_style = style_analysis(data_analyse_style['portef'],
                                              data_analyse_style[selected_indices_analyse_style]
                                              )
        weight = weight_analyse_style
        weight_analyse_style = pd.DataFrame(weight_analyse_style.T,
                                            index=selected_indices_analyse_style,
                                            columns=['poids'])
        cols[0].caption('Analyse de Sharpe')
        fig = go.Figure([go.Bar(x=weight_analyse_style['poids'],
                                y=weight_analyse_style.index,
                                orientation='h',
                                hovertemplate='<br>'.join(['Indice: %{y}',
                                                           'Poids: %{x:,.1%}',
                                                           '<extra></extra>']
                                                          ),
                                showlegend=False
                                )
                         ]
                        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig.update_xaxes(tickformat=".1%")
        cols[0].plotly_chart(fig, use_container_width=True)

        # Stats
        indice_analyse_style = (
            weight*data_analyse_style[selected_indices_analyse_style]).sum(axis=1)
        perf_fonds = cagr(data_analyse_style['portef']/100, period='weekly')
        perf_bench = cagr(indice_analyse_style/100, period='weekly')
        tracking_error_style = annual_volatility(
            data_analyse_style['portef']-indice_analyse_style, period='weekly')/100
        ir_style = (perf_fonds-perf_bench)/tracking_error_style

        # Indice / fonds rebasés
        indice_analyse_style_base100 = 100 * \
            (1+indice_analyse_style/100).cumprod()
        fonds_base100 = 100*(1+data_analyse_style['portef']/100).cumprod()

        cols[1].caption("Evolution du fonds vs indice")
        fig = go.Figure(
            data=[
                go.Scatter(x=fonds_base100.index,
                           y=fonds_base100,
                           name='Portefeuille',
                           marker=dict(color='blue'),
                           ),
                go.Scatter(x=indice_analyse_style_base100.index,
                           y=indice_analyse_style_base100,
                           name='Indice analyse style',
                           marker=dict(color='red')
                           ),
            ])
        fig.update_layout(hovermode='x unified',
                          legend=dict(orientation="h")
                          )
        cols[1].plotly_chart(fig, use_container_width=True)

        cols[2].caption('Statistiques :')
        cols[2].write(f'Performance annuelle du fonds : {perf_fonds:.2%}')
        cols[2].write(f"Performance annuelle de l'indice : {perf_bench:.2%}")
        cols[2].write(f'Tracking-Error : {tracking_error_style:.2%}')
        cols[2].write(f"Ratio d'information : {ir_style:.2f}")
        R2 = r2_score(data_analyse_style['portef'], indice_analyse_style)
        cols[2].write(f"R2 : {R2:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        lag = st.number_input('Période pour calcul glissant (en année):',
                              min_value=1.0,
                              max_value=5.0,
                              value=2.0,
                              step=0.5)
        lag = int(52*lag)

    with col2:
        st.write("")
        st.write("")
        data_analyse_style = pd.concat([df_base100['portef'],
                                        bench[selected_indices_analyse_style]],
                                       axis=1).resample('W').last().pct_change().dropna()
        run_analyse_style = st.button('Lancer analyse de style')

    if run_analyse_style:
        weight = rolling_style_analysis(data_analyse_style['portef'],
                                        data_analyse_style[selected_indices_analyse_style],
                                        window_size=lag
                                        )
        col1, col2 = st.columns(2)
        start_date, end_date = data_analyse_style.first_valid_index(
        )+DateOffset(years=int(lag/52)), weight.last_valid_index()

        with col1:
            st.caption('Analyse de style')
            fig = go.Figure(
                [go.Scatter(
                    name=col,
                    x=weight.loc[start_date:end_date].index,
                    y=weight.loc[start_date:end_date][col],
                    mode='lines',
                    stackgroup='one') for col in weight.columns]
            )
            fig.update_layout(showlegend=True,
                              yaxis=dict(range=[0, 100],
                                         ticksuffix='%'),
                              legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="left",
                                          entrywidth=250,
                                          x=0),
                              )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.caption("R2 de l'analyse de style")
            portefeuille = (
                data_analyse_style[selected_indices_analyse_style]*weight/100).sum(1)
            R2_analyse_style = (
                100*(portefeuille.rolling(lag).corr(data_analyse_style.loc[:, 'portef']))**2).dropna()
            fig = go.Figure(
                [go.Scatter(
                    name='R2',
                    x=R2_analyse_style.loc[start_date:end_date].index,
                    y=R2_analyse_style.loc[start_date:end_date],
                    mode='lines')
                 ]
            )
            fig.update_layout(showlegend=True,
                              yaxis=dict(ticksuffix='%'),
                              )
            st.plotly_chart(fig, use_container_width=True)

        st.caption('Fonds vs benchmark')
        portef_100 = 100 * \
            (1+portefeuille.loc[start_date:end_date].dropna()/100).cumprod()
        fig = go.Figure(
            data=[
                go.Scatter(name='Portefeuille',
                           x=data_analyse_style.loc[start_date:end_date].index,
                           y=100 *
                           (1+data_analyse_style.loc[start_date:end_date]
                            ['portef']/100).cumprod(),
                           mode='lines',
                           marker=dict(color='blue')),
                go.Scatter(name='Indice',
                           x=portef_100.loc[start_date:end_date].index,
                           y=portef_100.loc[start_date:end_date],
                           mode='lines',
                           marker=dict(color='red'))
            ]
        )
        st.plotly_chart(fig, use_container_width=True)

        alpha_analyse_style = 100*(data_analyse_style.loc[:, 'portef'].rolling(lag).apply(lambda x: cagr(x, period='weekly'))
                                   - portefeuille.rolling(lag).apply(lambda x: cagr(x, period='weekly'))
                                   )

        TE = 100 * \
            np.sqrt(
                52)*(data_analyse_style.loc[:, 'portef'] - portefeuille).rolling(lag).std()
        ir_analyse_style = alpha_analyse_style/TE

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Ratio d'information")
            fig = go.Figure(
                [go.Scatter(
                    name="IR",
                    x=ir_analyse_style.loc[start_date:end_date].index,
                    y=ir_analyse_style.loc[start_date:end_date],
                    mode='lines')]
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("Tracking-Error")
            fig = go.Figure(
                [go.Scatter(
                    name='Tracking-Error',
                    x=TE.loc[start_date:end_date].index,
                    y=TE.loc[start_date:end_date],
                    mode='lines')]
            )
            st.plotly_chart(fig, use_container_width=True)


#######################################################################################
################################# Mise à jour des données  ############################
#######################################################################################
with tab8:
    st.write('**Voir J. Tebeka pour mise à jour des données**')
    # if st.button('Mise à jour de la base de données'):
    #     st.write('Mise à jour en cours')

    #     # Chargement des données
    #     st.cache_data.clear()
    #     aum = pd.read_parquet(path / 'aum.parquet')
    #     vl = pd.read_parquet(path / 'vl.parquet')
    #     bench = pd.read_parquet(path / 'bench.parquet')
    #     desc = pd.read_parquet(path / 'desc.parquet').set_index('ID')
    #     desc_indice = pd.read_parquet(path / 'desc_indice.parquet')

    #     # Chargement des données updatés
    #     aum_update, vl_update, bench_update, desc_update, desc_indice_update = update_data(
    #         path / 'update_Societes_de_gestion.xlsx')

    #     # Création des nouveaux fichiers
    #     aum = pd.concat([aum,
    #                      aum_update]).sort_index().dropna(axis=1, how='all').query('~index.duplicated()')
    #     aum.to_parquet(path / 'aum.parquet')

    #     vl = pd.concat([vl,
    #                     vl_update]).sort_index().dropna(axis=1, how='all').query('~index.duplicated()')
    #     vl.to_parquet(path / 'vl.parquet')

    #     bench = pd.concat([bench,
    #                        bench_update]).sort_index().query('~index.duplicated()')
    #     bench.to_parquet(path / 'bench.parquet')

    #     # Affichage pour vérification des nouvelles données
    #     st.dataframe(vl.tail())
    #     st.dataframe(desc)
