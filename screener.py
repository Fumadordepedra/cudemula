import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from textblob import TextBlob
from scipy.optimize import minimize
from datetime import datetime, timedelta

def get_b3_tickers():
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
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None, None, None
        
        info = stock.info
        news = stock.news
        return df, info, news
    except Exception as e:
        print(f"Erro ao baixar {ticker}: {e}")
        return None, None, None

def analyze_sentiment(news):
    """Calcula o sentimento das notícias recentes usando NLP."""
    if not news:
        return 0.0, "Sem notícias recentes"
        
    polarities = []
    for article in news:
        title = article.get('title', '')
        summary = article.get('summary', '')
        # TextBlob analisa o sentimento do texto (em inglês, que é o padrão do YF API)
        text = f"{title}. {summary}"
        blob = TextBlob(text)
        polarities.append(blob.sentiment.polarity)
        
    if not polarities:
        return 0.0, "Neutro"
        
    avg_polarity = np.mean(polarities)
    
    # Classificação
    if avg_polarity > 0.15: sentiment_label = "🟢 Otimista"
    elif avg_polarity < -0.15: sentiment_label = "🔴 Pessimista"
    else: sentiment_label = "⚪ Neutro"
    
    return avg_polarity, sentiment_label

def calculate_backtest(df):
    """Simula uma estratégia simples de Cruzamento de Médias Móveis nos últimos 12 meses."""
    # Estratégia: Comprado quando SMA20 > SMA50, em dinheiro caso contrário.
    df['Signal'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Close'].pct_change()
    
    # Retornos acumulados
    bh_return = (1 + df['Close'].pct_change()).prod() - 1
    strat_return = (1 + df['Strategy_Return']).prod() - 1
    
    # Drawdown Máximo (Buy and Hold)
    roll_max = df['Close'].cummax()
    drawdown = df['Close']/roll_max - 1.0
    max_drawdown = drawdown.min()
    
    return {
        "Buy & Hold Return": bh_return * 100,
        "Strategy Return": strat_return * 100,
        "Max Drawdown": max_drawdown * 100
    }

def calculate_indicators(df):
    if df is None or len(df) < 50: return None
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['ROC_10'] = ta.momentum.ROCIndicator(close=df['Close'], window=10).roc()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    return df

def safe_float(value):
    try: return float(value)
    except (ValueError, TypeError): return np.nan

def analyze_ticker(ticker):
    df, info, news = download_data(ticker)
    df = calculate_indicators(df)
    
    if df is None or df.empty: return None
        
    last_row = df.iloc[-1]
    
    # NLP Sentiment
    polarity_score, sentiment_label = analyze_sentiment(news)
    
    # Backtest Stats
    backtest_stats = calculate_backtest(df.copy())
    
    # Technicals
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
    
    # Fundamentals
    pe_ratio = safe_float(info.get('trailingPE', np.nan))
    pb_ratio = safe_float(info.get('priceToBook', np.nan))
    div_yield = safe_float(info.get('dividendYield', np.nan)) * 100 if pd.notna(info.get('dividendYield')) else np.nan
    roe = safe_float(info.get('returnOnEquity', np.nan)) * 100 if pd.notna(info.get('returnOnEquity')) else np.nan
    
    # Tech Score (50)
    tech_score = 0
    reasons = []
    if vol_sma_20 > 0 and current_vol / vol_sma_20 > 1.5: tech_score += 10
    if current_price > sma_20 and sma_20 > sma_50: tech_score += 10; reasons.append("Uptrend")
    if current_price > sma_200: tech_score += 5
    if macd > macd_signal: tech_score += 10; reasons.append("MACD Bullish")
    if 40 < rsi < 65: tech_score += 5
    elif rsi <= 30: tech_score += 10; reasons.append("Sobrevendido")
    if current_price <= bb_low * 1.02: tech_score += 5; reasons.append("Fundo BB")
    if vol_sma_20 < 100000: tech_score = 0
    
    # Fund Score (50)
    fund_score = 0
    if pd.notna(pe_ratio) and 0 < pe_ratio < 15: fund_score += 10; reasons.append("P/L Baixo")
    if pd.notna(pb_ratio) and 0 < pb_ratio < 1.5: fund_score += 10; reasons.append("P/VP Baixo")
    if pd.notna(roe) and roe > 15: fund_score += 15; reasons.append("ROE Alto")
    if pd.notna(div_yield) and div_yield > 6: fund_score += 15; reasons.append("Alto DivYield")
    
    # IA Sentiment Boost (Ajusta a pontuação em até ±10 pts)
    sentiment_boost = int(polarity_score * 10)
    
    total_score = tech_score + fund_score + sentiment_boost
    total_score = max(0, min(100, total_score)) # Trava entre 0 e 100
    
    if total_score >= 80: rating = "STRONG BUY"
    elif total_score >= 60: rating = "BUY"
    elif total_score >= 40: rating = "HOLD"
    elif total_score >= 20: rating = "UNDERPERFORM"
    else: rating = "SELL"
        
    return {
        "Ticker": ticker,
        "Rating": rating,
        "Score Total": total_score,
        "Preço (R$)": round(current_price, 2),
        "P/L": round(pe_ratio, 2) if pd.notna(pe_ratio) else "-",
        "ROE (%)": round(roe, 2) if pd.notna(roe) else "-",
        "Div.Yield (%)": round(div_yield, 2) if pd.notna(div_yield) else "-",
        "RSI": round(rsi, 2),
        "Sentimento IA": sentiment_label,
        "Polaridade IA": round(polarity_score, 2),
        "Retorno 1A (%)": round(backtest_stats["Buy & Hold Return"], 2),
        "Retorno Estratégia (%)": round(backtest_stats["Strategy Return"], 2),
        "Max Drawdown (%)": round(backtest_stats["Max Drawdown"], 2),
        "Motivos/Destaques": ", ".join(reasons[-4:]),
        "df": df
    }

def run_screener_pro(tickers, progress_callback=None):
    results = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            res = analyze_ticker(ticker)
            if res: results.append(res)
        except Exception: pass
        if progress_callback: progress_callback((i + 1) / total)
    if not results: return []
    results.sort(key=lambda x: x['Score Total'], reverse=True)
    return results

def get_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.105):
    """Calcula o Sharpe Ratio negativo (para o minimizador)."""
    port_return = np.sum(returns.mean() * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - risk_free_rate) / port_volatility
    return -sharpe

def optimize_portfolio(results_list):
    """Fronteira Eficiente de Markowitz para as recomendações de Compra."""
    # Filtrar apenas STRONG BUY e BUY
    buy_results = [r for r in results_list if r['Rating'] in ['STRONG BUY', 'BUY']]
    if len(buy_results) < 2:
        return None, "São necessárias pelo menos 2 ações aprovadas para otimizar o portfólio."
        
    # Limitar a no máximo 10 ações para não diluir muito o portfólio
    buy_results = buy_results[:10]
    tickers = [r['Ticker'] for r in buy_results]
    
    # Extrair séries de preços
    price_dict = {r['Ticker']: r['df']['Close'] for r in buy_results}
    df_prices = pd.DataFrame(price_dict).dropna()
    
    if len(df_prices) < 100:
        return None, "Dados históricos insuficientes para matriz de covariância."
        
    daily_returns = df_prices.pct_change().dropna()
    cov_matrix = daily_returns.cov()
    
    num_assets = len(tickers)
    # Pesos iniciais iguais
    init_guess = np.array(num_assets * [1. / num_assets,])
    # Bounds: peso de cada ativo entre 0 e 1 (0% a 100%)
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    # Constraints: a soma dos pesos deve ser 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    optimized = minimize(get_sharpe_ratio, init_guess, args=(daily_returns, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=constraints)
                         
    weights = optimized.x
    
    # Retornos e Risco Esperados Anualizados
    port_return = np.sum(daily_returns.mean() * weights) * 252
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = -optimized.fun
    
    allocation = {tickers[i]: round(weights[i] * 100, 2) for i in range(num_assets) if weights[i] > 0.01}
    
    return {
        "Alocação": allocation,
        "Retorno Esperado (A.A.)": round(port_return * 100, 2),
        "Volatilidade/Risco (A.A.)": round(port_volatility * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2)
    }, None

def create_candlestick_chart(df, ticker):
    df_chart = df.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Preço', increasing_line_color='#00ff00', decreasing_line_color='#ff0000'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_20'], line=dict(color='#ff9900', width=1.5), name='SMA 20'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], line=dict(color='#33ccff', width=1.5), name='SMA 50'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_High'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), name='BB Superior'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Low'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), name='BB Inferior', fill='tonexty', fillcolor='rgba(255,255,255,0.05)'))
    fig.update_layout(title=f"Gráfico Interativo - {ticker}", yaxis_title='Preço (R$)', template='plotly_dark', plot_bgcolor='#111111', paper_bgcolor='#111111', margin=dict(l=40, r=40, t=40, b=40), xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
