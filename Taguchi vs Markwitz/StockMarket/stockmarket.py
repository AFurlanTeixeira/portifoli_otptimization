"""
=============================================================================
ME623 - Planejamento e Pesquisa | Primeiro Semestre de 2026
Projeto 1 - Planeamento Estatístico de um Experimento

Título: Otimização Robusta de Portefólio (Taguchi vs Markowitz)
        Universo Fixo: 3 Ações B3 + 3 Ações S&P 500 + Ouro + CDI

Resumo das Funcionalidades:
  1. Universo fixo com 8 ativos de setores distintos.
  2. Restrição de alocação: mínimo de 2% em cada ativo (garante diversificação).
  3. Otimização via Função de Utilidade (Aversão ao Risco = 0.35).
  4. Teste de Hipótese (Bootstrap) para verificar a superação da meta (IPCA+6%).
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
import requests

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

# =============================================================================
# 1. RECOLHA DE DADOS
# =============================================================================

def coleta_bcb(codigo, nome, data_inicio='01/01/2016', data_fim='31/12/2025'):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json&dataInicial={data_inicio}&dataFinal={data_fim}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df['data']  = pd.to_datetime(df['data'], dayfirst=True)
        df['valor'] = pd.to_numeric(df['valor'].astype(str).str.replace(',', '.'), errors='coerce')
        return df.set_index('data').rename(columns={'valor': nome})
    except Exception as e:
        print(f"  [AVISO] Falha SGS {codigo} ({nome}): {e}")
        return None

def diario_para_mensal(df_diario, col):
    fator = (1 + df_diario[col] / 100)
    mensal = fator.resample('MS').prod() - 1
    return (mensal * 100).rename(col.replace('_diario','').replace('_diaria','')).to_frame()

def carregar_dados_yahoo(ini='2016-01-01', fim='2025-12-31'):
    try:
        import yfinance as yf
    except ImportError:
        print("  [ERRO] A biblioteca 'yfinance' não está instalada. Execute: pip install yfinance")
        return None
        
    # Dicionário de Tickers exatos
    tickers_map = {
        'ITUB4.SA': 'ITUB4',
        'PETR4.SA': 'PETR4',
        'WEGE3.SA': 'WEGE3',
        'AAPL': 'AAPL',
        'JNJ': 'JNJ',
        'XOM': 'XOM',
        'GOLD11.SA': 'GOLD11'
    }
    
    print(f"  A descarregar dados de {len(tickers_map)} ativos via Yahoo Finance...")
    tickers_list = list(tickers_map.keys())
    
    df_yf = yf.download(tickers_list, start=ini, end=fim, interval='1mo', auto_adjust=True, progress=False)
    
    if df_yf.empty:
        return None
        
    df = df_yf['Close']
    df.index = pd.to_datetime(df.index).to_period('M').to_timestamp()
    ret = df.pct_change().dropna() * 100
    ret.rename(columns=tickers_map, inplace=True)
    return ret

def carregar_dados():
    print("\n" + "="*60)
    print("BLOCO 1: RECOLHA DE DADOS (3 B3, 3 S&P500, Ouro, CDI)")
    print("="*60)
    
    df_cdi_d = coleta_bcb(12,  'CDI_diario')
    df_sel_d = coleta_bcb(432, 'Selic_diaria')
    df_ipca  = coleta_bcb(433, 'IPCA')
    df_acoes = carregar_dados_yahoo()

    df_cdi_m = diario_para_mensal(df_cdi_d,  'CDI_diario')
    df_sel_m = diario_para_mensal(df_sel_d,  'Selic_diaria').rename(columns={'Selic':'Selic_mensal'})
    
    for df in [df_cdi_m, df_ipca, df_sel_m]:
        df.index = pd.to_datetime(df.index).to_period('M').to_timestamp()
        
    idx = df_acoes.index.intersection(df_cdi_m.index).intersection(df_ipca.index).intersection(df_sel_m.index)
    return df_acoes.loc[idx], df_cdi_m.loc[idx], df_ipca.loc[idx], df_sel_m.loc[idx]

# =============================================================================
# 2. CENÁRIOS E FUNÇÕES BÁSICAS
# =============================================================================

def classificar_cenarios(df_selic):
    s = df_selic['Selic_mensal'].copy()
    m_cov = (s.index >= '2020-03-01') & (s.index <= '2021-06-01')
    s_nc  = s[~m_cov]
    p33, p67 = s_nc.quantile(0.33), s_nc.quantile(0.67)

    cen = pd.Series(index=s.index, dtype=str)
    cen[m_cov]                            = 'N4_Crise'
    cen[~m_cov & (s > p67)]               = 'N1_Restritivo'
    cen[~m_cov & (s > p33) & (s <= p67)]  = 'N2_Neutro'
    cen[~m_cov & (s <= p33)]              = 'N3_Expansivo'
    return cen

def ret_port(pesos, df_at, ativos):
    return df_at[ativos].values @ np.array(pesos)

def meta_ipca6(df_ipca):
    return df_ipca.values.flatten() + 0.5

def calcular_metricas(pw, df_at, df_ipca, ativos):
    r_real = ret_port(pw, df_at, ativos) - df_ipca.values.flatten()
    mu = np.mean(r_real) * 12
    vol = np.std(r_real, ddof=1) * np.sqrt(12)
    sharpe = mu / vol if vol > 1e-10 else 0
    return mu, vol, sharpe

# =============================================================================
# 3. OTIMIZAÇÕES (UTILIDADE RISCO-RETORNO)
# =============================================================================

AVERSAO_RISCO = 0.35 

def otimizar_taguchi(df_at, df_ipca, cenarios, ativos):
    print("\n" + "="*60)
    print(f"BLOCO 2: OTIMIZAÇÃO ROBUSTA (TAGUCHI) | Aversão ao Risco = {AVERSAO_RISCO}")
    print("="*60)
    
    n_ativos = len(ativos)
    
    def neg_utilidade(pw):
        r_real = ret_port(pw, df_at, ativos) - df_ipca.values.flatten()
        rets_c, vols_c = [], []
        for c in ['N1_Restritivo','N2_Neutro','N3_Expansivo','N4_Crise']:
            y = r_real[cenarios == c]
            if len(y) > 2:
                rets_c.append(np.mean(y) * 12)
                vols_c.append(np.std(y, ddof=1) * np.sqrt(12))
        
        mu_medio = np.mean(rets_c) if rets_c else 0
        vol_media = np.mean(vols_c) if vols_c else 0
        return -(mu_medio - (AVERSAO_RISCO * vol_media))

    rest = [{'type':'eq','fun': lambda x: x.sum()-1}]
    bnds = [(0.02, 1.0)] * n_ativos 
    
    best_util, best_pw = -np.inf, None
    np.random.seed(123)
    for _ in range(50):
        x0  = np.random.dirichlet(np.ones(n_ativos))
        res = minimize(neg_utilidade, x0, method='SLSQP', bounds=bnds, constraints=rest)
        if res.success and (-res.fun) > best_util:
            best_util = -res.fun; best_pw = res.x

    mu, vol, sharpe = calcular_metricas(best_pw, df_at, df_ipca, ativos)

    print(f"\n  Mistura Ótima (Taguchi):")
    for a,p in zip(ativos, best_pw):
        print(f"    {a:10s}: {p*100:6.2f}%")
    print(f"    Retorno Real: {mu:.2f}% | Volatilidade: {vol:.2f}% | Sharpe: {sharpe:.4f}")
    return best_pw

def otimizar_markowitz(df_at, df_ipca, ativos):
    print("\n" + "="*60)
    print(f"BLOCO 3: OTIMIZAÇÃO CLÁSSICA (MARKOWITZ) | Aversão ao Risco = {AVERSAO_RISCO}")
    print("="*60)
    
    n_ativos = len(ativos)

    def neg_utilidade_markowitz(pw):
        r_real = ret_port(pw, df_at, ativos) - df_ipca.values.flatten()
        mu = np.mean(r_real) * 12
        vol = np.std(r_real, ddof=1) * np.sqrt(12)
        return -(mu - (AVERSAO_RISCO * vol))

    rest = [{'type':'eq','fun': lambda x: x.sum()-1}]
    bnds = [(0.02, 1.0)] * n_ativos
    
    best_util, best_pw = -np.inf, None
    np.random.seed(42)
    for _ in range(50):
        x0  = np.random.dirichlet(np.ones(n_ativos))
        res = minimize(neg_utilidade_markowitz, x0, method='SLSQP', bounds=bnds, constraints=rest)
        if res.success and (-res.fun) > best_util:
            best_util = -res.fun; best_pw = res.x

    mu, vol, sharpe = calcular_metricas(best_pw, df_at, df_ipca, ativos)

    print(f"\n  Mistura Ótima (Markowitz):")
    for a,p in zip(ativos, best_pw):
        print(f"    {a:10s}: {p*100:6.2f}%")
    print(f"    Retorno Real: {mu:.2f}% | Volatilidade: {vol:.2f}% | Sharpe: {sharpe:.4f}")
    return best_pw


# =============================================================================
# 4. TESTE DE HIPÓTESE
# =============================================================================

def teste_bootstrap(r_tag, r_mark, df_ipca, B=5000, alpha=0.05):
    print("\n" + "="*60)
    print("BLOCO 4: TESTE DE HIPÓTESE — EXCESSO SOBRE IPCA+6% a.a.")
    print("="*60)
    meta = meta_ipca6(df_ipca)
    
    def bootstrap_array(exc):
        n = len(exc)
        np.random.seed(42)
        return np.array([np.mean(exc[np.random.choice(n, n, replace=True)]) for _ in range(B)])

    mb_tag  = bootstrap_array(r_tag - meta)
    mb_mark = bootstrap_array(r_mark - meta)

    for nome, mb, exc_obs in [("Taguchi", mb_tag, np.mean(r_tag - meta)), 
                              ("Markowitz", mb_mark, np.mean(r_mark - meta))]:
        ic_uni = np.percentile(mb, alpha*100)
        p_val  = np.mean(mb <= 0)
        rejeita = ic_uni > 0
        print(f"\n  --- {nome} ---")
        print(f"  Excesso médio: {exc_obs:.4f}% a.m.")
        print(f"  IC inferior 95%: {ic_uni:.4f} | p-valor: {p_val:.4f}")
        print(f"  Decisão: {'Supera' if rejeita else 'NÃO supera'} IPCA+6% (Rejeita H0: {rejeita})")

    return mb_tag, mb_mark


# =============================================================================
# 5. VISUALIZAÇÕES
# =============================================================================

def plotar(pw_tag, pw_mark, df_at, df_ipca, df_selic, cenarios, ativos, mb_tag, mb_mark):
    r_tag  = ret_port(pw_tag, df_at, ativos)
    r_mark = ret_port(pw_mark, df_at, ativos)
    meta   = meta_ipca6(df_ipca)

    # Cores fixas para os 8 ativos (3 BR, 3 US, 2 Outros)
    cores = ['#2E7D32', '#388E3C', '#66BB6A',  # Tons Verdes (B3)
             '#1565C0', '#1976D2', '#64B5F6',  # Tons Azuis (S&P 500)
             '#F9A825', '#757575']             # Ouro (Amarelo) e CDI (Cinzento)

    # Fig 1 - Gráficos Circulares (Pesos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    fig.suptitle('Composição dos Portefólios (Mínimo de 2% por Ativo)', fontweight='bold')

    ax1.pie(pw_tag*100, labels=ativos, colors=cores, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Robusto (Taguchi)')
    
    ax2.pie(pw_mark*100, labels=ativos, colors=cores, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Tradicional (Markowitz)')
    plt.tight_layout(); plt.savefig('fig1_pesos_dinamicos.png', bbox_inches='tight'); plt.close()

    # Fig 2 - Evolução do Património
    fig, ax = plt.subplots(figsize=(10,5))
    ac_tag  = (1+r_tag/100).cumprod()*100
    ac_mark = (1+r_mark/100).cumprod()*100
    ac_m    = (1+meta/100).cumprod()*100

    ax.plot(df_at.index, ac_tag, color='#1565C0', lw=2, label='Portefólio Taguchi')
    ax.plot(df_at.index, ac_mark, color='#E53935', lw=2, label='Portefólio Markowitz')
    ax.plot(df_at.index, ac_m, color='black', lw=2, ls='--', label='Meta: IPCA+6% a.a.')
    
    ax.set_ylabel('Retorno Acumulado (base 100)')
    ax.set_title('Evolução do Património vs Meta')
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig('fig2_evolucao_dinamica.png', bbox_inches='tight'); plt.close()

    # Fig 3 - Cenários da Selic
    fig, ax = plt.subplots(figsize=(12,4))
    s = df_selic['Selic_mensal']
    cc = {'N1_Restritivo':'#E53935','N2_Neutro':'#FFC107','N3_Expansivo':'#43A047','N4_Crise':'#6A1B9A'}
    for cn, cor in cc.items():
        mask = cenarios == cn
        ax.scatter(df_selic.index[mask], s[mask], c=cor, label=cn.replace('_',' '), s=40, zorder=3)
    ax.plot(df_selic.index, s, color='gray', alpha=0.3, lw=1)
    ax.set_title('Cenários de Selic (Array Externo Taguchi)')
    ax.set_ylabel('Selic Mensal (%)'); ax.legend()
    plt.tight_layout(); plt.savefig('fig3_cenarios.png', bbox_inches='tight'); plt.close()

    # Fig 4 - Distribuição Bootstrap
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(mb_tag, bins=50, density=True, alpha=0.6, color='#1565C0', label='Taguchi (Excesso)')
    ax.hist(mb_mark, bins=50, density=True, alpha=0.6, color='#E53935', label='Markowitz (Excesso)')
    ax.axvline(0, color='black', lw=2, ls='--', label='H0: Excesso = 0')
    ax.set_title('Distribuição Bootstrap do Excesso de Retorno sobre IPCA+6%')
    ax.set_xlabel('Excesso médio mensal (%)'); ax.legend()
    plt.tight_layout(); plt.savefig('fig4_bootstrap_compare.png', bbox_inches='tight'); plt.close()
    
    print("\n  Figuras de análise guardadas com sucesso no diretório atual.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("ME623 — P1 FINAL | 8 ATIVOS FIXOS + TAGUCHI vs MARKOWITZ")
    print("="*60)

    # 1. Carregar os dados reais
    df_acoes, df_cdi, df_ipca, df_selic = carregar_dados()
    cenarios = classificar_cenarios(df_selic)

    # Definição final dos 8 ativos
    ativos_finais = ['ITUB4', 'PETR4', 'WEGE3', 'AAPL', 'JNJ', 'XOM', 'GOLD11', 'CDI']
    
    # Prepara o DataFrame consolidado com os 8 ativos (sincronizado nas mesmas datas)
    df_at = df_acoes.join(df_cdi, how='inner')
    
    # 2. Executar as Otimizações
    pw_tag  = otimizar_taguchi(df_at, df_ipca, cenarios, ativos_finais)
    pw_mark = otimizar_markowitz(df_at, df_ipca, ativos_finais)

    # 3. Executar o Teste de Hipótese
    r_tag  = ret_port(pw_tag, df_at, ativos_finais)
    r_mark = ret_port(pw_mark, df_at, ativos_finais)
    mb_tag, mb_mark = teste_bootstrap(r_tag, r_mark, df_ipca)

    # 4. Criar e guardar os gráficos
    print("\n" + "="*60 + "\nBLOCO 5: A CRIAR FIGURAS\n" + "="*60)
    plotar(pw_tag, pw_mark, df_at, df_ipca, df_selic, cenarios, ativos_finais, mb_tag, mb_mark)

if __name__ == '__main__':
    main()