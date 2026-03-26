from config import DATASET, RESULTADOS, EXPERIMENTOS, GERAR_VISAO_GERAL, GERAR_RANKING
from processador import ProcessadorMorfologico

for nome, cfg in EXPERIMENTOS.items():
    p = ProcessadorMorfologico(
        diretorio_dataset = DATASET,
        diretorio_saida   = RESULTADOS,
        tamanhos_kernel   = cfg["kernels"],
        ruido             = cfg["ruido"],
        intensidade_ruido = cfg["intensidade"],
        seed              = cfg["seed"],
    )
    (
        p.selecionar_imagens(nomes=cfg.get("nomes"), categorias=cfg.get("categorias"))
         .selecionar_operacoes(operacoes=cfg["operacoes"])
    )
    p.processar()
    p.visualizar(analise=nome, salvar=True)
    if GERAR_VISAO_GERAL:
        p.visualizar_combinado(analise=nome, salvar=True)
    p.gerar_relatorio_metricas(analise=nome)
    if GERAR_RANKING:
        p.gerar_ranking(analise=nome)

print(f"\nConcluído. Resultados em: {RESULTADOS}/")
