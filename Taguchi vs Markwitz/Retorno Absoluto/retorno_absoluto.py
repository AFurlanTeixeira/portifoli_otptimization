"""
=============================================================================
ME623 - Planejamento e Pesquisa | Primeiro Semestre de 2026
Projeto 1 - Planejamento Estatístico de um Experimento

Título: Otimização Robusta de Portfólio via Mixture Experiments e Taguchi
        vs. Otimização de Markowitz (Média-Variância)

Ativos : BOVA11.SA, IVVB11.SA, GOLD11.SA, CDI (BCB SGS 12)
Benchmark: IPCA (BCB SGS 433) | Meta: IPCA + 6% a.a.
Período : Janeiro 2016 – Dezembro 2025

Alterações recentes (v8):
  1. Otimização via Função de Utilidade (Retorno - Penalidade de Risco).
  2. Introdução do parâmetro AVERSAO_RISCO (lambda).
  3. Exibição detalhada de Retorno, Volatilidade e Sharpe na saída.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import combinations
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
# 1. COLETA DE DADOS
# =============================================================================

def coleta_bcb(codigo, nome, data_inicio='01/01/2016', data_fim='31/12/2025'):
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
        f"?formato=json&dataInicial={data_inicio}&dataFinal={data_fim}"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        df['data']  = pd.to_datetime(df['data'], dayfirst=True)
        df['valor'] = pd.to_numeric(
            df['valor'].astype(str).str.replace(',', '.'), errors='coerce'
        )
        return df.set_index('data').rename(columns={'valor': nome})
    except Exception as e:
        print(f"  [AVISO] Falha SGS {codigo} ({nome}): {e}")
        return None

def diario_para_mensal(df_diario, col):
    fator   = (1 + df_diario[col] / 100)
    mensal  = fator.resample('MS').prod() - 1
    return (mensal * 100).rename(col.replace('_diario','').replace('_diaria','')).to_frame()

def coleta_etfs(tickers, ini='2016-01-01', fim='2025-12-31'):
    try:
        import yfinance as yf
        dados = {}
        for t in tickers:
            print(f"  Coletando {t}...")
            ativo = yf.download(t, start=ini, end=fim,
                                interval='1mo', auto_adjust=True, progress=False)
            if not ativo.empty:
                dados[t] = ativo['Close'].squeeze()
        if not dados:
            return None
        df = pd.DataFrame(dados)
        df.index = pd.to_datetime(df.index).to_period('M').to_timestamp()
        ret = df.pct_change().dropna() * 100
        return ret.rename(columns={
            'BOVA11.SA':'BOVA11','IVVB11.SA':'IVVB11','GOLD11.SA':'GOLD11'
        })
    except ImportError:
        return None

def gerar_sinteticos(n=120):
    np.random.seed(42)
    datas = pd.date_range('2016-01-01', periods=n, freq='MS')
    corr  = np.array([[1.00, 0.55, 0.18,-0.05],
                      [0.55, 1.00, 0.22,-0.03],
                      [0.18, 0.22, 1.00,-0.08],
                      [-0.05,-0.03,-0.08, 1.00]])
    vols  = np.array([5.50, 4.80, 3.80, 0.12])
    mus   = np.array([0.80, 1.10, 0.95, 0.55])
    L     = np.linalg.cholesky(np.diag(vols) @ corr @ np.diag(vols))
    ret   = np.random.randn(n, 4) @ L.T + mus

    df_etfs = pd.DataFrame(ret, index=datas, columns=['BOVA11','IVVB11','GOLD11','CDI'])
    ipca    = np.random.normal(0.42, 0.22, n)
    selic   = np.random.normal(0.54, 0.14, n)

    return (df_etfs[['BOVA11','IVVB11','GOLD11']], df_etfs[['CDI']],
            pd.DataFrame({'IPCA':  np.clip(ipca,  -0.1, 1.8)}, index=datas),
            pd.DataFrame({'Selic_mensal': np.clip(selic, 0.1, 1.2)}, index=datas))

def carregar_dados():
    print("\n" + "="*60)
    print("BLOCO 1: COLETA DE DADOS (Jan/2016 – Dez/2025)")
    print("="*60)
    df_cdi_d   = coleta_bcb(12,  'CDI_diario')
    df_sel_d   = coleta_bcb(432, 'Selic_diaria')
    df_ipca    = coleta_bcb(433, 'IPCA')
    df_etfs    = coleta_etfs(['BOVA11.SA','IVVB11.SA','GOLD11.SA'])

    if all(x is not None for x in [df_cdi_d, df_sel_d, df_ipca, df_etfs]):
        print("  Dados reais coletados com sucesso.")
        df_cdi_m = diario_para_mensal(df_cdi_d,  'CDI_diario')
        df_sel_m = diario_para_mensal(df_sel_d,  'Selic_diaria').rename(columns={'Selic':'Selic_mensal'})
        for df in [df_etfs, df_cdi_m, df_ipca, df_sel_m]:
            df.index = pd.to_datetime(df.index).to_period('M').to_timestamp()
        idx = (df_etfs.index.intersection(df_cdi_m.index).intersection(df_ipca.index).intersection(df_sel_m.index))
        return df_etfs.loc[idx], df_cdi_m.loc[idx], df_ipca.loc[idx], df_sel_m.loc[idx]
    else:
        print("  A usar dados sintéticos calibrados (modo offline).")
        return gerar_sinteticos()

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
    return cen, p33, p67

def ret_port(pesos, df_at):
    return df_at.values @ np.array(pesos)

def meta_ipca6(df_ipca):
    return df_ipca.values.flatten() + 0.5

def calcular_metricas_finais(pw, df_at, df_ipca, ativos):
    r_real = ret_port(pw, df_at[ativos]) - df_ipca.values.flatten()
    mu = np.mean(r_real) * 12
    vol = np.std(r_real, ddof=1) * np.sqrt(12)
    sharpe = mu / vol if vol > 1e-10 else 0
    return mu, vol, sharpe

# =============================================================================
# 3. OTIMIZAÇÕES: FUNÇÃO UTILIDADE (RETORNO vs RISCO)
# =============================================================================

# Define o quão "medroso" é o algoritmo. 
# 0.0 = Kamikaze (Só retorno) | 0.35 = Arrojado | > 1.0 = Conservador (Foca em CDI)
AVERSAO_RISCO = 0.35 

def otimizar_taguchi(df_at, df_ipca, cenarios):
    print("\n" + "="*60)
    print(f"BLOCO 2: OTIMIZAÇÃO ROBUSTA (TAGUCHI) - Lambda Risco: {AVERSAO_RISCO}")
    print("="*60)
    ativos = ['BOVA11','IVVB11','GOLD11','CDI']

    def neg_utilidade(pw):
        r_real = ret_port(pw, df_at[ativos]) - df_ipca.values.flatten()
        rets_c = []
        vols_c = []
        for c in ['N1_Restritivo','N2_Neutro','N3_Expansivo','N4_Crise']:
            y = r_real[cenarios == c]
            if len(y) > 2:
                rets_c.append(np.mean(y) * 12)
                vols_c.append(np.std(y, ddof=1) * np.sqrt(12))
                
        mu_medio = np.mean(rets_c) if rets_c else 0
        vol_media = np.mean(vols_c) if vols_c else 0
        
        # Função Utilidade: Retorno Médio - (Lambda * Volatilidade Média)
        utilidade = mu_medio - (AVERSAO_RISCO * vol_media)
        return -utilidade

    rest = [{'type':'eq','fun': lambda x: x.sum()-1}]
    bnds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    best_util, best_pw = -np.inf, None
    np.random.seed(123)
    for _ in range(50):
        x0  = np.random.dirichlet(np.ones(4))
        res = minimize(neg_utilidade, x0, method='SLSQP', bounds=bnds, constraints=rest)
        if res.success and (-res.fun) > best_util:
            best_util = -res.fun; best_pw = res.x

    mu, vol, sharpe = calcular_metricas_finais(best_pw, df_at, df_ipca, ativos)

    print(f"\n  Mistura Ótima (Taguchi):")
    for a,p in zip(ativos, best_pw):
        print(f"    {a:8s}: {p*100:6.2f}%")
    print(f"    Retorno Real (Anual): {mu:.2f}% | Volatilidade (Anual): {vol:.2f}%")
    print(f"    >>> Sharpe Anualizado: {sharpe:.4f}")
    return best_pw

def otimizar_markowitz(df_at, df_ipca):
    print("\n" + "="*60)
    print(f"BLOCO 3: OTIMIZAÇÃO CLÁSSICA (MARKOWITZ) - Lambda Risco: {AVERSAO_RISCO}")
    print("="*60)
    ativos = ['BOVA11','IVVB11','GOLD11','CDI']

    def neg_utilidade_markowitz(pw):
        r_real = ret_port(pw, df_at[ativos]) - df_ipca.values.flatten()
        mu = np.mean(r_real) * 12
        vol = np.std(r_real, ddof=1) * np.sqrt(12)
        
        # Função Utilidade: Retorno - (Lambda * Volatilidade)
        utilidade = mu - (AVERSAO_RISCO * vol)
        return -utilidade

    rest = [{'type':'eq','fun': lambda x: x.sum()-1}]
    bnds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    best_util, best_pw = -np.inf, None
    np.random.seed(42)
    for _ in range(50):
        x0  = np.random.dirichlet(np.ones(4))
        res = minimize(neg_utilidade_markowitz, x0, method='SLSQP', bounds=bnds, constraints=rest)
        if res.success and (-res.fun) > best_util:
            best_util = -res.fun; best_pw = res.x

    mu, vol, sharpe = calcular_metricas_finais(best_pw, df_at, df_ipca, ativos)

    print(f"\n  Mistura Ótima (Markowitz):")
    for a,p in zip(ativos, best_pw):
        print(f"    {a:8s}: {p*100:6.2f}%")
    print(f"    Retorno Real (Anual): {mu:.2f}% | Volatilidade (Anual): {vol:.2f}%")
    print(f"    >>> Sharpe Anualizado: {sharpe:.4f}")
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

def plotar(pw_tag, pw_mark, df_etfs, df_cdi, df_ipca, df_selic, cenarios, mb_tag, mb_mark):
    ativos = ['BOVA11','IVVB11','GOLD11','CDI']
    df_at  = df_etfs.join(df_cdi, how='inner')
    r_tag  = ret_port(pw_tag, df_at[ativos])
    r_mark = ret_port(pw_mark, df_at[ativos])
    meta   = meta_ipca6(df_ipca)

    cores  = ['#1565C0','#E53935','#2E7D32','#F9A825']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle(f'Composição das Carteiras (Aversão ao Risco = {AVERSAO_RISCO})', fontweight='bold')
    
    ax1.pie(pw_tag*100, labels=ativos, colors=cores, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Robusto (Taguchi)')
    
    ax2.pie(pw_mark*100, labels=ativos, colors=cores, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Tradicional (Markowitz)')
    plt.tight_layout(); plt.savefig('fig1_comparacao_pesos.png', bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    ac_tag  = (1+r_tag/100).cumprod()*100
    ac_mark = (1+r_mark/100).cumprod()*100
    ac_m    = (1+meta/100).cumprod()*100

    ax.plot(df_at.index, ac_tag, color='#1565C0', lw=2, label='Portfólio Taguchi')
    ax.plot(df_at.index, ac_mark, color='#E53935', lw=2, label='Portfólio Markowitz')
    ax.plot(df_at.index, ac_m, color='black', lw=2, ls='--', label='Meta: IPCA+6% a.a.')
    
    ax.set_ylabel('Retorno Acumulado (base 100)')
    ax.set_title('Evolução do Patrimônio vs Meta')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.tight_layout(); plt.savefig('fig2_evolucao_patrimonio.png', bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(12,4))
    s = df_selic['Selic_mensal']
    cc = {'N1_Restritivo':'#E53935','N2_Neutro':'#FFC107','N3_Expansivo':'#43A047','N4_Crise':'#6A1B9A'}
    for cn, cor in cc.items():
        mask = cenarios == cn
        ax.scatter(df_selic.index[mask], s[mask], c=cor, label=cn.replace('_',' '), s=40, zorder=3)
    ax.plot(df_selic.index, s, color='gray', alpha=0.3, lw=1)
    ax.set_title('Cenários de Selic (Array Externo)')
    ax.set_ylabel('Selic Mensal (%)'); ax.legend()
    plt.tight_layout(); plt.savefig('fig3_cenarios.png', bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(mb_tag, bins=50, density=True, alpha=0.6, color='#1565C0', label='Taguchi (Excesso)')
    ax.hist(mb_mark, bins=50, density=True, alpha=0.6, color='#E53935', label='Markowitz (Excesso)')
    ax.axvline(0, color='black', lw=2, ls='--', label='H0: Excesso = 0')
    ax.set_title('Distribuição Bootstrap do Excesso de Retorno sobre IPCA+6%')
    ax.set_xlabel('Excesso médio mensal (%)'); ax.legend()
    plt.tight_layout(); plt.savefig('fig4_bootstrap_compare.png', bbox_inches='tight'); plt.close()

    print("  Figuras de comparação salvas com sucesso.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("ME623 — P1 v8 | TAGUCHI vs MARKOWITZ (Função de Utilidade Risco-Retorno)")
    print("="*60)

    df_etfs, df_cdi, df_ipca, df_selic = carregar_dados()
    idx = (df_etfs.index.intersection(df_cdi.index).intersection(df_ipca.index).intersection(df_selic.index))
    df_etfs, df_cdi, df_ipca, df_selic = df_etfs.loc[idx], df_cdi.loc[idx], df_ipca.loc[idx], df_selic.loc[idx]

    cenarios, _, _ = classificar_cenarios(df_selic)
    cenarios = cenarios.loc[idx]
    ativos = ['BOVA11','IVVB11','GOLD11','CDI']
    df_at = df_etfs.join(df_cdi, how='inner').loc[idx]

    # Otimizações
    pw_tag  = otimizar_taguchi(df_at, df_ipca, cenarios)
    pw_mark = otimizar_markowitz(df_at, df_ipca)

    # Testes
    r_tag  = ret_port(pw_tag, df_at[ativos])
    r_mark = ret_port(pw_mark, df_at[ativos])
    mb_tag, mb_mark = teste_bootstrap(r_tag, r_mark, df_ipca)

    # Gráficos
    print("\n" + "="*60 + "\nBLOCO 5: A GERAR FIGURAS\n" + "="*60)
    plotar(pw_tag, pw_mark, df_etfs, df_cdi, df_ipca, df_selic, cenarios, mb_tag, mb_mark)

if __name__ == '__main__':
    main()