# Scanner B3 📈

Um sistema de screening de ações focado exclusivamente no mercado brasileiro (B3).
Ele analisa ações usando dados do Yahoo Finance (`yfinance`) e classifica em:
- ✅ COMPRAR AGORA
- 👀 OBSERVAR
- ❌ EVITAR

## Como usar

1. Crie um ambiente virtual (opcional):
   ```bash
   python -m venv venv
   source venv/Scripts/activate # No Windows
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o aplicativo:
   ```bash
   streamlit run app.py
   ```

4. Abra o navegador em `http://localhost:8501`

## Critérios de Análise
O sistema calcula um **Score** para cada ativo baseado nos seguintes filtros:
- **P/L abaixo de 20** (se disponível)
- **Volume atual acima de 2x a média de 20 dias**
- **Preço acima da média curta (SMA 20)**
- **IFR/RSI saudável (50 - 70)**
- **Tendência de alta (SMA 20 > SMA 50)**
- **Momentum Positivo (ROC)**
- **Liquidez mínima**

Quanto mais critérios a ação preencher, maior sua pontuação (Score), e melhor será sua classificação no ranking.
