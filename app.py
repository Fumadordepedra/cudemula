import streamlit as st
import pandas as pd
from screener import get_b3_tickers, run_screener_pro, create_candlestick_chart, optimize_portfolio
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Terminal Quant B3", page_icon="💠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600;700&display=swap');
    .stApp { background-color: #050505; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    h1, h2, h3 { color: #ffb703; font-family: 'Roboto Mono', monospace; letter-spacing: -0.5px; }
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    .stButton>button { background: transparent; color: #ffb703; font-weight: 700; border-radius: 4px; border: 1px solid #ffb703; padding: 0.5rem 1.5rem; width: 100%; text-transform: uppercase; }
    .stButton>button:hover { background: rgba(255, 183, 3, 0.1); border-color: #fb8500; color: #fb8500; box-shadow: 0 0 10px rgba(255, 183, 3, 0.2); }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #ffffff; }
    .metric-card-container { background-color: #111111; border: 1px solid #222; border-left: 4px solid #ffb703; padding: 1rem; border-radius: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: #111111; border-radius: 4px 4px 0px 0px; padding: 10px; color: #888; border: 1px solid #222; border-bottom: none; }
    .stTabs [aria-selected="true"] { background-color: #1a1a1a; color: #ffb703; border-top: 2px solid #ffb703; }
    [data-testid="stDataFrame"] { background-color: #111; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

col_title, col_logo = st.columns([4, 1])
with col_title:
    st.markdown("<h1>TERMINAL QUANT B3</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Fundo Quantitativo: Backtesting, Markowitz e ML Preditivo</p>", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ PARÂMETROS DO SCAN")
selected_tickers_text = st.sidebar.text_area("Universo de Ações (Vírgula):", value=", ".join(get_b3_tickers()[:25]), height=150)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size: 0.8rem; color: #888;'><strong>PESOS DO SCORE (Max 100):</strong><br>• Técnica (Max 50)<br>• Fundamentos (Max 50)<br>• Sentimento IA (± 10)<br>• Previsão ML (± 15)</div>", unsafe_allow_html=True)

if 'results_pro' not in st.session_state: st.session_state['results_pro'] = None

if st.sidebar.button("EXECUTAR SCAN TOTAL"):
    raw_tickers = [t.strip().upper() for t in selected_tickers_text.split(",") if t.strip()]
    valid_tickers = [t if t.endswith(".SA") else f"{t}.SA" for t in raw_tickers]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def on_progress(pct):
        progress_bar.progress(pct)
        status_text.markdown(f"*Lendo Balanços, Extraindo Notícias, Treinando Redes Neurais... {int(pct*100)}%*")
        
    results = run_screener_pro(valid_tickers, progress_callback=on_progress)
    
    if results:
        st.session_state['results_pro'] = results
        status_text.markdown("<span style='color:#00ff00'>✔ Varredura Quantitativa Finalizada.</span>", unsafe_allow_html=True)
    else:
        st.error("Falha ao obter dados.")

st.markdown("---")

if st.session_state['results_pro']:
    table_data = []
    for r in st.session_state['results_pro']:
        clean_r = {k:v for k,v in r.items() if k != 'df'}
        table_data.append(clean_r)
        
    df_results = pd.DataFrame(table_data)
    
    tab1, tab2, tab3 = st.tabs(["RANKING INSTITUCIONAL", "DEEP DIVE (PREDIÇÃO & IA)", "OTIMIZAÇÃO MARKOWITZ"])
    
    with tab1:
        st.markdown("### Visão Geral do Mercado")
        strong_buy = len(df_results[df_results['Rating'] == 'STRONG BUY'])
        buy = len(df_results[df_results['Rating'] == 'BUY'])
        
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='metric-card-container'>ATIVOS ANALISADOS<br><b style='font-size:24px;color:#fff'>{len(df_results)}</b></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card-container' style='border-left-color:#00ff00'>STRONG BUY<br><b style='font-size:24px;color:#00ff00'>{strong_buy}</b></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card-container' style='border-left-color:#8fbc8f'>BUY<br><b style='font-size:24px;color:#8fbc8f'>{buy}</b></div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
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
        selected_data = next((item for item in st.session_state['results_pro'] if item["Ticker"] == selected_ticker_for_chart), None)
        
        if selected_data:
            c1, c2 = st.columns([3, 1])
            with c1:
                fig = create_candlestick_chart(selected_data['df'], selected_data['Ticker'])
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                prob_alta = selected_data['Probabilidade Alta (IA)']
                prob_color = "#00ff00" if prob_alta > 60 else "#ff0000" if prob_alta < 40 else "#ffb703"
                
                st.markdown(f"""
                <div style='background-color:#111; padding:20px; border-radius:4px; border:1px solid #333;'>
                    <h2 style='margin-top:0; color:#fff;'>{selected_data['Ticker']}</h2>
                    <h3 style='color: {"#00ff00" if "BUY" in selected_data["Rating"] else "#ff0000" if "SELL" in selected_data["Rating"] else "#ff8c00"};'>{selected_data['Rating']}</h3>
                    <hr>
                    <p style='margin:0;color:#888'>🧠 PREDIÇÃO MACHINE LEARNING</p>
                    <p style='margin:0;color:#ccc;font-size:12px;'>Prob. de subir nos próximos 5 dias:</p>
                    <h1 style='margin:0;color:{prob_color}'>{prob_alta}%</h1>
                    <br>
                    <p style='margin:0;color:#888'>🤖 SENTIMENTO DAS NOTÍCIAS (NLP)</p>
                    <h3 style='margin:0;color:#fff'>{selected_data['Sentimento (Notícias)']}</h3>
                    <hr>
                    <p style='margin:0;color:#888'>📊 BACKTESTING (12 Meses)</p>
                    <p style='margin:0;color:#fff'>Retorno Estratégia: <b style='color: {"#00ff00" if selected_data["Retorno Estratégia (%)"] > 0 else "#ff0000"};'>{selected_data['Retorno Estratégia (%)']}%</b></p>
                    <p style='margin:0;color:#fff'>Retorno Buy&Hold: <b style='color: {"#00ff00" if selected_data["Retorno 1A (%)"] > 0 else "#ff0000"};'>{selected_data['Retorno 1A (%)']}%</b></p>
                    <p style='margin:0;color:#fff'>Risco (Max Drawdown): <b style='color:#ff0000'>{selected_data['Max Drawdown (%)']}%</b></p>
                    <hr>
                    <p style='color:#888'>DESTAQUES DO ALGORITMO:</p>
                    <p style='color:#fff; font-size:14px'>{selected_data['Motivos/Destaques']}</p>
                </div>
                """, unsafe_allow_html=True)
                
    with tab3:
        st.markdown("### ⚖️ Otimizador de Portfólio (Fronteira Eficiente de Markowitz)")
        st.markdown("O sistema calculou a correlação e volatilidade histórica das ações recomendadas (**STRONG BUY** e **BUY**) para montar a carteira ideal que maximiza o lucro minimizando o risco (Maior Índice Sharpe).")
        
        with st.spinner("Calculando Fronteira Eficiente..."):
            port_stats, error_msg = optimize_portfolio(st.session_state['results_pro'])
            
            if error_msg:
                st.warning(error_msg)
            elif port_stats:
                col_a, col_b = st.columns([1, 1])
                
                with col_a:
                    st.markdown(f"""
                    <div style='background-color:#111; padding:20px; border-radius:4px; border:1px solid #333;'>
                        <h3 style='color:#ffb703'>Projeção da Carteira</h3>
                        <br>
                        <p style='color:#888; font-size:1.2rem; margin:0;'>Retorno Esperado Anualizado</p>
                        <h2 style='color:#00ff00; margin:0;'>{port_stats['Retorno Esperado (A.A.)']}%</h2>
                        <br>
                        <p style='color:#888; font-size:1.2rem; margin:0;'>Risco Histórico (Volatilidade)</p>
                        <h2 style='color:#ff0000; margin:0;'>{port_stats['Volatilidade/Risco (A.A.)']}%</h2>
                        <br>
                        <p style='color:#888; font-size:1.2rem; margin:0;'>Índice Sharpe (Retorno vs Risco)</p>
                        <h2 style='color:#33ccff; margin:0;'>{port_stats['Sharpe Ratio']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_b:
                    labels = list(port_stats['Alocação'].keys())
                    values = list(port_stats['Alocação'].values())
                    
                    fig_pie = px.pie(names=labels, values=values, hole=0.4, color_discrete_sequence=px.colors.sequential.Plasma)
                    fig_pie.update_layout(
                        title="Alocação Sugerida do Capital",
                        template='plotly_dark',
                        plot_bgcolor='#111111', paper_bgcolor='#111111',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("👈 Insira os tickers e clique em EXECUTAR SCAN TOTAL no painel lateral para iniciar.")
