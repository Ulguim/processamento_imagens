from config import DATASET, RESULTADOS, EXPERIMENTOS
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
    sel = p.selecionar_imagens(
        nomes      = cfg.get("nomes"),
        categorias = cfg.get("categorias"),
    )
    ops = cfg["operacoes"]
    if ops == "todas":
        sel.selecionar_operacoes(todas=True)
    else:
        sel.selecionar_operacoes(operacoes=ops)
    p.processar()
    p.visualizar(analise=nome, salvar=True)
    p.gerar_relatorio_metricas(analise=nome)

print(f"\nConcluído. Resultados em: {RESULTADOS}/")
