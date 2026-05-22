import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from textblob import TextBlob
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

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
        if df.empty: return None, None, None
        info = stock.info
        news = stock.news
        return df, info, news
    except Exception: return None, None, None

def analyze_sentiment(news):
    if not news: return 0.0, "Sem notícias recentes"
    polarities = [TextBlob(f"{a.get('title', '')}. {a.get('summary', '')}").sentiment.polarity for a in news]
    if not polarities: return 0.0, "Neutro"
    avg = np.mean(polarities)
    return avg, "🟢 Otimista" if avg > 0.15 else "🔴 Pessimista" if avg < -0.15 else "⚪ Neutro"

def calculate_backtest(df):
    df['Signal'] = np.where(df['SMA_20'] > df['SMA_50'], 1, 0)
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Close'].pct_change()
    
    bh_return = (1 + df['Close'].pct_change()).prod() - 1
    strat_return = (1 + df['Strategy_Return']).prod() - 1
    
    drawdown = df['Close']/df['Close'].cummax() - 1.0
    return {"Buy & Hold Return": bh_return * 100, "Strategy Return": strat_return * 100, "Max Drawdown": drawdown.min() * 100}

def calculate_indicators(df):
    if df is None or len(df) < 50: return None
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = ta.trend.ema_indicator(close=df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(close=df['Close'], window=200)
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['ROC_10'] = ta.momentum.ROCIndicator(close=df['Close'], window=10).roc()
    df['CCI_20'] = ta.trend.cci(high=df['High'], low=df['Low'], close=df['Close'], window=20)
    
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'], df['MACD_Signal'] = macd.macd(), macd.macd_signal()
    
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'], df['BB_Low'] = bollinger.bollinger_hband(), bollinger.bollinger_lband()
    return df

def predict_with_ai(df):
    try:
        ml_df = df.copy()
        ml_df['MACD_Diff'] = ml_df['MACD'] - ml_df['MACD_Signal']
        ml_df['Dist_SMA20'] = (ml_df['Close'] - ml_df['SMA_20']) / ml_df['SMA_20']
        ml_df['Vol_Ratio'] = np.where(ml_df['Vol_SMA_20'] > 0, ml_df['Volume'] / ml_df['Vol_SMA_20'], 1.0)
        
        ml_df['Target'] = np.where(ml_df['Close'].shift(-5) > ml_df['Close'], 1, 0)
        
        features = ['RSI_14', 'MACD_Diff', 'Dist_SMA20', 'Vol_Ratio', 'ROC_10', 'CCI_20']
        train_data = ml_df.dropna(subset=features + ['Target']).copy()
        
        X, y = train_data[features][:-5], train_data['Target'][:-5] # Remove os ultimos 5 q n sabemos o futuro
        if len(X) < 30: return 50.0 
            
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X, y)
        
        today_data = ml_df.iloc[-1:][features].fillna(0)
        return model.predict_proba(today_data)[0][1] * 100
    except Exception: return 50.0

def safe_float(value):
    try: return float(value)
    except (ValueError, TypeError): return np.nan

def analyze_ticker(ticker):
    df, info, news = download_data(ticker)
    df = calculate_indicators(df)
    
    if df is None or df.empty: return None
        
    last_row = df.iloc[-1]
    
    current_price, current_vol = last_row['Close'], last_row['Volume']
    vol_sma_20 = last_row['Vol_SMA_20']
    
    # GROK FILTER 1: Liquidez (Volume financeiro menor que 1 milhão BRL/dia = MICO)
    if (vol_sma_20 * current_price) < 1000000:
        return None # Ignora a ação completamente (evita falsos positivos de micos)
    
    pe_ratio = safe_float(info.get('trailingPE', np.nan))
    pb_ratio = safe_float(info.get('priceToBook', np.nan))
    
    # Consertando bug do Div Yield gigante
    raw_dy = safe_float(info.get('dividendYield', np.nan))
    div_yield = (raw_dy * 100) if pd.notna(raw_dy) and raw_dy < 1.0 else (raw_dy if pd.notna(raw_dy) else np.nan)
    
    roe = safe_float(info.get('returnOnEquity', np.nan)) * 100 if pd.notna(info.get('returnOnEquity')) else np.nan
    
    tech_score, reasons = 0, []
    
    # GROK FILTER 2: EMA 200 e Tendência
    ema_50 = last_row['EMA_50']
    ema_200 = last_row['EMA_200'] if pd.notna(last_row['EMA_200']) else last_row['SMA_50']
    
    if current_price > ema_200: tech_score += 15; reasons.append("Tendência Alta (EMA200)")
    else: reasons.append("Abaixo EMA200")
        
    if current_price > ema_50 and ema_50 > ema_200: tech_score += 10
    
    # GROK FILTER 3: Momentum e CCI
    rsi = last_row['RSI_14']
    cci = last_row['CCI_20']
    if 55 <= rsi <= 75: tech_score += 10; reasons.append("RSI Rompimento")
    elif rsi < 30: tech_score += 5; reasons.append("Sobrevendido")
    
    if cci > 0: tech_score += 5; reasons.append("CCI Positivo")
    
    if last_row['MACD'] > last_row['MACD_Signal']: tech_score += 5
    if current_vol > vol_sma_20: tech_score += 5
    
    # Fundamentos (Max 30)
    fund_score = 0
    if pd.notna(pe_ratio) and 0 < pe_ratio < 15: fund_score += 10; reasons.append("P/L Baixo")
    if pd.notna(roe) and roe > 15: fund_score += 10
    if pd.notna(div_yield) and div_yield > 6: fund_score += 10
    
    prob_alta_ia = predict_with_ai(df)
    if prob_alta_ia > 60: tech_score += 15; reasons.append(f"IA Bullish ({prob_alta_ia:.0f}%)")
    elif prob_alta_ia < 40: tech_score -= 10
    
    polarity_score, sentiment_label = analyze_sentiment(news)
    total_score = max(0, min(100, tech_score + fund_score + int(polarity_score * 10)))
    
    # GROK FILTER 4: Trava de Segurança (Nunca dar Strong Buy se estiver abaixo da EMA200)
    if current_price < ema_200 and total_score >= 80:
        total_score = 59 # Rebaixa para HOLD
        reasons.append("Rebaixado: Abaixo EMA200")
        
    if total_score >= 80: rating = "STRONG BUY"
    elif total_score >= 60: rating = "BUY"
    elif total_score >= 40: rating = "HOLD"
    elif total_score >= 20: rating = "UNDERPERFORM"
    else: rating = "SELL"
        
    return {
        "Ticker": ticker, "Rating": rating, "Score Total": total_score, "Probabilidade Alta (IA)": round(prob_alta_ia, 1),
        "Preço (R$)": round(current_price, 2), "P/L": round(pe_ratio, 2) if pd.notna(pe_ratio) else "-",
        "ROE (%)": round(roe, 2) if pd.notna(roe) else "-", "Div.Yield (%)": round(div_yield, 2) if pd.notna(div_yield) else "-",
        "RSI": round(rsi, 2), "Sentimento (Notícias)": sentiment_label,
        "Retorno 1A (%)": round(calculate_backtest(df.copy())["Buy & Hold Return"], 2),
        "Retorno Estratégia (%)": round(calculate_backtest(df.copy())["Strategy Return"], 2),
        "Max Drawdown (%)": round(calculate_backtest(df.copy())["Max Drawdown"], 2),
        "Motivos/Destaques": ", ".join(reasons[-3:]), "df": df
    }

def run_screener_pro(tickers, progress_callback=None):
    results = []
    for i, t in enumerate(tickers):
        try:
            r = analyze_ticker(t)
            if r: results.append(r)
        except Exception: pass
        if progress_callback: progress_callback((i + 1) / len(tickers))
    return sorted(results, key=lambda x: x['Score Total'], reverse=True) if results else []

def get_sharpe_ratio(weights, returns, cov_matrix, rf=0.105):
    return -((np.sum(returns.mean() * weights) * 252 - rf) / np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights))))

def optimize_portfolio(results_list):
    buy_results = [r for r in results_list if r['Rating'] in ['STRONG BUY', 'BUY']][:10]
    if len(buy_results) < 2: return None, "Mínimo de 2 ações recomendadas."
    df_prices = pd.DataFrame({r['Ticker']: r['df']['Close'] for r in buy_results}).dropna()
    if len(df_prices) < 100: return None, "Dados históricos insuficientes."
    daily_returns = df_prices.pct_change().dropna()
    cov_matrix = daily_returns.cov()
    num_assets = len(buy_results)
    optimized = minimize(get_sharpe_ratio, np.array(num_assets * [1./num_assets,]), args=(daily_returns, cov_matrix), method='SLSQP', bounds=tuple((0., 1.) for _ in range(num_assets)), constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))
    weights = optimized.x
    return {"Alocação": {buy_results[i]['Ticker']: round(weights[i] * 100, 2) for i in range(num_assets) if weights[i] > 0.01}, "Retorno Esperado (A.A.)": round(np.sum(daily_returns.mean() * weights) * 252 * 100, 2), "Volatilidade/Risco (A.A.)": round(np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights))) * 100, 2), "Sharpe Ratio": round(-optimized.fun, 2)}, None

def create_candlestick_chart(df, ticker):
    df_chart = df.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Preço', increasing_line_color='#00ff00', decreasing_line_color='#ff0000'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_50'], line=dict(color='#ff9900', width=1.5), name='EMA 50'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_200'], line=dict(color='#33ccff', width=1.5), name='EMA 200'))
    fig.update_layout(title=f"Gráfico Interativo - {ticker}", yaxis_title='Preço (R$)', template='plotly_dark', plot_bgcolor='#111111', paper_bgcolor='#111111', margin=dict(l=40, r=40, t=40, b=40), xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
