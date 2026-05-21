import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta

def get_b3_tickers():
    """Retorna uma lista estendida de tickers comuns da B3 (Índice Bovespa e Small Caps principais)."""
    return [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", 
        "ABEV3.SA", "WEGE3.SA", "RENT3.SA", "SUZB3.SA", "BPAC11.SA",
        "EQTL3.SA", "RADL3.SA", "RDOR3.SA", "PRIO3.SA", "HAPV3.SA",
        "LREN3.SA", "B3SA3.SA", "GGBR4.SA", "RAIL3.SA", "CSAN3.SA",
        "VIVT3.SA", "SBSP3.SA", "JBSS3.SA", "TOTS3.SA", "ELET3.SA",
        "CPLE6.SA", "CMIG4.SA", "ENEV3.SA", "EGIE3.SA", "KLBN11.SA",
        "CCRO3.SA", "TIMS3.SA", "ASAI3.SA", "HYPE3.SA", "ALOS3.SA",
        "SMTO3.SA", "UGPA3.SA", "CPFE3.SA", "MGLU3.SA", "NTCO3.SA",
        "BBSE3.SA", "CXSE3.SA", "TAEE11.SA", "TRPL4.SA", "CYRE3.SA",
        "EZTC3.SA", "MULT3.SA", "IGTI11.SA", "SOMA3.SA", "ARZZ3.SA"
    ]

def download_data(ticker, period="1y"):
    """Baixa dados históricos e metadados institucionais."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None, None
        
        info = stock.info
        return df, info
    except Exception as e:
        print(f"Erro ao baixar {ticker}: {e}")
        return None, None

def calculate_indicators(df):
    """Calcula um pacote completo de indicadores técnicos institucionais."""
    if df is None or len(df) < 50:
        return None
    
    # Médias Móveis
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Indicadores de Volume
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # RSI / IFR
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Momentum (Rate of Change)
    df['ROC_10'] = ta.momentum.ROCIndicator(close=df['Close'], window=10).roc()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    return df

def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def analyze_ticker(ticker):
    """Realiza a análise profunda (Técnica + Fundamentos) e gera o Rating Institucional."""
    df, info = download_data(ticker)
    df = calculate_indicators(df)
    
    if df is None or df.empty:
        return None
        
    last_row = df.iloc[-1]
    
    # --- DADOS TÉCNICOS ---
    current_price = last_row['Close']
    current_vol = last_row['Volume']
    vol_sma_20 = last_row['Vol_SMA_20']
    rsi = last_row['RSI_14']
    sma_20 = last_row['SMA_20']
    sma_50 = last_row['SMA_50']
    sma_200 = last_row['SMA_200'] if pd.notna(last_row['SMA_200']) else sma_50
    macd = last_row['MACD']
    macd_signal = last_row['MACD_Signal']
    bb_low = last_row['BB_Low']
    momentum = last_row['ROC_10']
    
    # --- DADOS FUNDAMENTALISTAS ---
    pe_ratio = safe_float(info.get('trailingPE', np.nan))
    pb_ratio = safe_float(info.get('priceToBook', np.nan))
    div_yield = safe_float(info.get('dividendYield', np.nan)) * 100 if pd.notna(info.get('dividendYield')) else np.nan
    roe = safe_float(info.get('returnOnEquity', np.nan)) * 100 if pd.notna(info.get('returnOnEquity')) else np.nan
    beta = safe_float(info.get('beta', np.nan))
    
    # --- SCORING TÉCNICO (Máx 50 pts) ---
    tech_score = 0
    reasons = []
    
    # 1. Volume Relativo (até 10 pts)
    if vol_sma_20 > 0:
        vol_ratio = current_vol / vol_sma_20
        if vol_ratio > 2: tech_score += 10; reasons.append("Spike Vol")
        elif vol_ratio > 1.2: tech_score += 5
        
    # 2. Tendência (até 15 pts)
    if current_price > sma_20: tech_score += 5
    if sma_20 > sma_50: tech_score += 5; reasons.append("Uptrend (Curto)")
    if current_price > sma_200: tech_score += 5; reasons.append("Uptrend (Longo)")
    
    # 3. MACD (até 10 pts)
    if macd > macd_signal: tech_score += 10; reasons.append("MACD Bullish Cross")
    elif macd > 0: tech_score += 5
    
    # 4. IFR / Bollinger (até 15 pts)
    if 40 < rsi < 65: tech_score += 5
    elif rsi <= 30: tech_score += 10; reasons.append("Sobrevendido (IFR)")
    
    if current_price <= bb_low * 1.02: tech_score += 5; reasons.append("Sup. Banda Bollinger")
    
    # Filtro de liquidez estrita
    if vol_sma_20 < 100000:
        tech_score = 0
        reasons.append("Rejeitado: Baixa Liquidez")

    # --- SCORING FUNDAMENTALISTA (Máx 50 pts) ---
    fund_score = 0
    
    # 1. Valuation: P/L e P/VP (até 20 pts)
    if pd.notna(pe_ratio) and 0 < pe_ratio < 15: fund_score += 10; reasons.append(f"P/L Descontado")
    elif pd.notna(pe_ratio) and 15 <= pe_ratio < 25: fund_score += 5
    
    if pd.notna(pb_ratio) and 0 < pb_ratio < 1.5: fund_score += 10; reasons.append(f"P/VP Atrativo")
    elif pd.notna(pb_ratio) and 1.5 <= pb_ratio < 3: fund_score += 5
    
    # 2. Eficiência: ROE (até 15 pts)
    if pd.notna(roe) and roe > 15: fund_score += 15; reasons.append(f"Alto ROE")
    elif pd.notna(roe) and roe > 10: fund_score += 8
    
    # 3. Retorno: Dividend Yield (até 15 pts)
    if pd.notna(div_yield) and div_yield > 6: fund_score += 15; reasons.append(f"High Div Yield")
    elif pd.notna(div_yield) and div_yield > 3: fund_score += 8
    
    # Score Total (0 a 100)
    total_score = tech_score + fund_score
    
    # --- RATING INSTITUCIONAL ---
    if total_score >= 80:
        rating = "STRONG BUY"
    elif total_score >= 60:
        rating = "BUY"
    elif total_score >= 40:
        rating = "HOLD"
    elif total_score >= 20:
        rating = "UNDERPERFORM"
    else:
        rating = "SELL"
        
    return {
        "Ticker": ticker,
        "Rating": rating,
        "Score Total": total_score,
        "Score Técnico": tech_score,
        "Score Fund.": fund_score,
        "Preço (R$)": round(current_price, 2),
        "P/L": round(pe_ratio, 2) if pd.notna(pe_ratio) else "-",
        "P/VP": round(pb_ratio, 2) if pd.notna(pb_ratio) else "-",
        "ROE (%)": round(roe, 2) if pd.notna(roe) else "-",
        "Div.Yield (%)": round(div_yield, 2) if pd.notna(div_yield) else "-",
        "RSI": round(rsi, 2),
        "Vol. Real": f"{current_vol/vol_sma_20:.1f}x" if vol_sma_20 > 0 else "-",
        "Motivos/Destaques": ", ".join(reasons[-4:]), # Top 4 reasons
        "df": df # Guardamos o dataframe para o Plotly renderizar na aba detalhada
    }

def run_screener_pro(tickers, progress_callback=None):
    """Executa o screener em uma lista de tickers, retornando um dicionário rico com DF completo."""
    results = []
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        try:
            res = analyze_ticker(ticker)
            if res:
                results.append(res)
        except Exception as e:
            pass
            
        if progress_callback:
            progress_callback((i + 1) / total)
            
    if not results:
        return []
        
    # Sort by Total Score descending
    results.sort(key=lambda x: x['Score Total'], reverse=True)
    return results

def create_candlestick_chart(df, ticker):
    """Gera um gráfico interativo de Candlestick com Plotly padrão Bloomberg."""
    df_chart = df.tail(120) # Últimos ~6 meses
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df_chart.index,
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name='Preço',
                increasing_line_color='#00ff00', decreasing_line_color='#ff0000'))
                
    # SMA 20 e 50
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_20'], line=dict(color='#ff9900', width=1.5), name='SMA 20'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], line=dict(color='#33ccff', width=1.5), name='SMA 50'))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_High'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), name='BB Superior'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Low'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), name='BB Inferior', fill='tonexty', fillcolor='rgba(255,255,255,0.05)'))
    
    # Layout (Estilo Bloomberg/Institucional)
    fig.update_layout(
        title=f"Análise Técnica - {ticker}",
        yaxis_title='Preço (R$)',
        xaxis_title='Data',
        template='plotly_dark',
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig
