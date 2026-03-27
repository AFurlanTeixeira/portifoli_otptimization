# portifoli_otptimization

Aqui estou testando a otimização de portifólio, comparando o método de Markowitz com o Taguchi

Fiz a otimização a partir de Sharpe Ratio e também a partir de Valor Absoluto (considerando volatilidade e risco). 

A intenção desses testes era ver se conseguiríamos encontrar uma carteira otimizada com 4 ativos: BOVA11, CDB, OURO e IVVB11 que superasse o IPCA+6 ao longo do tempo. 

Utilizei dados de jan-2016 até dez-2025 para fazer a análise. Pensando em diferentes cenários (justamente para poder aplicar Taguchi), então meu modelo passou por crises (COVID19), recessões e também períodos de bull market. 

Todos os testes foram feitos utilizando $\alpha$ = 0.05 

E em todos os testes nós Rejeitamos  $H_0$ = nosso portifólio supera IPCA+6.

Os dados uilizados foram tirados diretamente da api do BACEN e do YFinance.
