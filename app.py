import streamlit as st
import pandas as pd
from screener import get_b3_tickers, run_screener_pro, create_candlestick_chart
import plotly.graph_objects as go

# Configuração Institucional
st.set_page_config(
    page_title="Terminal Quant B3",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Estilo Terminal Bloomberg (Dark & Premium)
st.markdown("""
<style>
    /* Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600;700&display=swap');
    
    .stApp {
        background-color: #050505; /* Fundo hiper escuro */
        color: #e0e0e0;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        color: #ffb703; /* Destaque Ouro/Laranja Institucional */
        font-family: 'Roboto Mono', monospace;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333;
    }
    
    /* Botões */
    .stButton>button {
        background: transparent;
        color: #ffb703;
        font-weight: 700;
        border-radius: 4px;
        border: 1px solid #ffb703;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: rgba(255, 183, 3, 0.1);
        border-color: #fb8500;
        color: #fb8500;
        box-shadow: 0 0 10px rgba(255, 183, 3, 0.2);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #ffffff;
    }
    .metric-card-container {
        background-color: #111111;
        border: 1px solid #222;
        border-left: 4px solid #ffb703;
        padding: 1rem;
        border-radius: 4px;
    }
    
    /* Abas do Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #111111;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #888;
        border: 1px solid #222;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a1a;
        color: #ffb703;
        border-top: 2px solid #ffb703;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background-color: #111;
        border: 1px solid #333;
    }
    
    hr {
        border-color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Título do App
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.markdown("<h1>TERMINAL QUANT B3</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Screener de Ações de Grau Institucional • Alpha Research Engine</p>", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ PARÂMETROS DO SCAN")

default_tickers = get_b3_tickers()
selected_tickers_text = st.sidebar.text_area(
    "Universo de Ações (Vírgula):",
    value=", ".join(default_tickers[:20]), # Limite inicial
    height=200
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size: 0.8rem; color: #888;'>
<strong>PESOS DO SCORE:</strong><br>
• Técnica (50%): Volume, MACD, BB, Tendência<br>
• Fundamentos (50%): P/L, P/VP, ROE, Div Yield<br><br>
<strong>RATING SCALE:</strong><br>
<span style='color:#00ff00'>STRONG BUY</span> > 80 pts<br>
<span style='color:#8fbc8f'>BUY</span> > 60 pts<br>
<span style='color:#aaaaaa'>HOLD</span> > 40 pts<br>
<span style='color:#ff8c00'>UNDERPERFORM</span> > 20 pts<br>
<span style='color:#ff0000'>SELL</span> < 20 pts
</div>
""", unsafe_allow_html=True)

if 'results_pro' not in st.session_state:
    st.session_state['results_pro'] = None

if st.sidebar.button("EXECUTAR SCAN"):
    raw_tickers = [t.strip().upper() for t in selected_tickers_text.split(",") if t.strip()]
    valid_tickers = [t if t.endswith(".SA") else f"{t}.SA" for t in raw_tickers]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def on_progress(pct):
        progress_bar.progress(pct)
        status_text.markdown(f"*Processando modelo multifatorial... {int(pct*100)}%*")
        
    results = run_screener_pro(valid_tickers, progress_callback=on_progress)
    
    if results:
        st.session_state['results_pro'] = results
        status_text.markdown("<span style='color:#00ff00'>✔ Scan Finalizado.</span>", unsafe_allow_html=True)
    else:
        st.error("Falha ao obter dados.")
        
st.markdown("---")

if st.session_state['results_pro']:
    # Separar os dicionários limpos para a tabela (removendo o DF completo)
    table_data = []
    for r in st.session_state['results_pro']:
        clean_r = {k:v for k,v in r.items() if k != 'df'}
        table_data.append(clean_r)
        
    df_results = pd.DataFrame(table_data)
    
    # ABAS PRINCIPAIS
    tab1, tab2 = st.tabs(["RANKING INSTITUCIONAL", "DEEP DIVE (ANÁLISE DETALHADA)"])
    
    with tab1:
        st.markdown("### Visão Geral do Mercado")
        
        # Métricas de Resumo
        strong_buy = len(df_results[df_results['Rating'] == 'STRONG BUY'])
        buy = len(df_results[df_results['Rating'] == 'BUY'])
        sell_under = len(df_results[df_results['Rating'].isin(['SELL', 'UNDERPERFORM'])])
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card-container'>TICKERS<br><b style='font-size:24px;color:#fff'>{len(df_results)}</b></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card-container' style='border-left-color:#00ff00'>STRONG BUY<br><b style='font-size:24px;color:#00ff00'>{strong_buy}</b></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card-container' style='border-left-color:#8fbc8f'>BUY<br><b style='font-size:24px;color:#8fbc8f'>{buy}</b></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card-container' style='border-left-color:#ff0000'>SELL / UNDER.<br><b style='font-size:24px;color:#ff0000'>{sell_under}</b></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Colorização da tabela
        def highlight_rating(val):
            if val == 'STRONG BUY': return 'color: #000; background-color: #00ff00; font-weight: bold'
            elif val == 'BUY': return 'color: #000; background-color: #8fbc8f; font-weight: bold'
            elif val == 'HOLD': return 'color: #fff; background-color: #555555'
            elif val == 'UNDERPERFORM': return 'color: #000; background-color: #ff8c00; font-weight: bold'
            elif val == 'SELL': return 'color: #fff; background-color: #ff0000; font-weight: bold'
            return ''
            
        styled_df = df_results.style.map(highlight_rating, subset=['Rating'])
        st.dataframe(styled_df, use_container_width=True, height=500)
        
    with tab2:
        st.markdown("### Selecione um Ticker para Análise Profunda")
        selected_ticker_for_chart = st.selectbox("Ticker", df_results['Ticker'].tolist())
        
        # Encontrar o resultado completo
        selected_data = next((item for item in st.session_state['results_pro'] if item["Ticker"] == selected_ticker_for_chart), None)
        
        if selected_data:
            c1, c2 = st.columns([3, 1])
            
            with c1:
                # Gráfico interativo
                fig = create_candlestick_chart(selected_data['df'], selected_data['Ticker'])
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                # Painel de Informações Laterais
                st.markdown(f"""
                <div style='background-color:#111; padding:20px; border-radius:4px; border:1px solid #333;'>
                    <h2 style='margin-top:0; color:#fff;'>{selected_data['Ticker']}</h2>
                    <h3 style='color: {"#00ff00" if "BUY" in selected_data["Rating"] else "#ff0000" if "SELL" in selected_data["Rating"] else "#ff8c00"};'>{selected_data['Rating']}</h3>
                    <hr>
                    <p style='margin:0;color:#888'>SCORE TOTAL</p>
                    <h1 style='margin:0;color:#ffb703'>{selected_data['Score Total']}/100</h1>
                    <br>
                    <p style='margin:0;color:#888'>Score Técnico</p>
                    <h3 style='margin:0;color:#fff'>{selected_data['Score Técnico']}/50</h3>
                    <br>
                    <p style='margin:0;color:#888'>Score Fundamentalista</p>
                    <h3 style='margin:0;color:#fff'>{selected_data['Score Fund.']}/50</h3>
                    <hr>
                    <p style='color:#888'>DESTAQUES:</p>
                    <p style='color:#fff; font-size:14px'>{selected_data['Motivos/Destaques']}</p>
                </div>
                """, unsafe_allow_html=True)
else:
    st.info("👈 Insira os tickers e clique em EXECUTAR SCAN no painel lateral para iniciar o terminal.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #444; font-size: 0.8rem;'>QUANT TERMINAL v2.0 • FOR INSTITUTIONAL USE ONLY</p>", unsafe_allow_html=True)
