from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def gerar_ranking(
    resultados:  dict,
    pasta_saida: Path,
    analise:     str = None,
) -> None:
    """
    Calcula score composto = média(jaccard, f1, pixel_accuracy, ssim) para cada
    combinação (imagem, operação, kernel), gera dois CSVs e imprime o resumo.

    Arquivos gerados em pasta_saida/:
        {analise}_metricas.csv  — linha por (imagem, operação, kernel)
        {analise}_ranking.csv   — score médio por (operação, kernel), ordenado
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)
    prefixo = analise or "experimento"

    # ── 1. Flatten: lista de dicts com todas as combinações ─────────────────
    linhas: list[dict] = []
    for chave, dados in resultados.items():
        kernels = [k for k in dados if k.startswith("kernel_")]
        for chave_k in kernels:
            ksize = chave_k.split("_")[1]            # ex: "3x3"
            for op_nome, entry in dados[chave_k].items():
                m     = entry["metricas"]
                score = (
                    m["jaccard"] + m["f1"] + m["pixel_accuracy"] + m["ssim"]
                ) / 4
                linhas.append({
                    "imagem":         chave,
                    "operacao":       op_nome,
                    "kernel":         ksize,
                    "pixel_accuracy": round(m["pixel_accuracy"], 6),
                    "jaccard":        round(m["jaccard"],        6),
                    "f1":             round(m["f1"],             6),
                    "ssim":           round(m["ssim"],           6),
                    "score":          round(score,               6),
                })

    if not linhas:
        print("[RANKING] Nenhum resultado disponível.")
        return

    # ── 2. CSV de métricas completas ────────────────────────────────────────
    campos_metricas = ["imagem", "operacao", "kernel",
                       "pixel_accuracy", "jaccard", "f1", "ssim", "score"]
    path_metricas = pasta_saida / f"{prefixo}_metricas.csv"
    _escrever_csv(path_metricas, campos_metricas, linhas)
    print(f"  CSV de métricas salvo: {path_metricas}")

    # ── 3. Agrupa por (operacao, kernel) para ranking geral ─────────────────
    acumulador: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for l in linhas:
        chave_op = (l["operacao"], l["kernel"])
        for campo in ("pixel_accuracy", "jaccard", "f1", "ssim", "score"):
            acumulador[chave_op][campo].append(l[campo])

    ranking: list[dict] = []
    for (op, kernel), valores in acumulador.items():
        ranking.append({
            "operacao":       op,
            "kernel":         kernel,
            "score_medio":    round(sum(valores["score"])          / len(valores["score"]),          6),
            "jaccard_medio":  round(sum(valores["jaccard"])        / len(valores["jaccard"]),        6),
            "f1_medio":       round(sum(valores["f1"])             / len(valores["f1"]),             6),
            "acc_medio":      round(sum(valores["pixel_accuracy"]) / len(valores["pixel_accuracy"]), 6),
            "ssim_medio":     round(sum(valores["ssim"])           / len(valores["ssim"]),           6),
        })

    ranking.sort(key=lambda x: x["score_medio"], reverse=True)
    for i, r in enumerate(ranking, start=1):
        r["rank"] = i

    campos_ranking = ["rank", "operacao", "kernel",
                      "score_medio", "jaccard_medio", "f1_medio", "acc_medio", "ssim_medio"]
    path_ranking = pasta_saida / f"{prefixo}_ranking.csv"
    _escrever_csv(path_ranking, campos_ranking, ranking)
    print(f"  CSV de ranking salvo:  {path_ranking}")

    # ── 4. Resumo no terminal ────────────────────────────────────────────────
    sep = "═" * 58
    titulo = f"RANKING — {analise}" if analise else "RANKING"
    print(f"\n{sep}")
    print(titulo)
    print(sep)
    print(f"  {'#':<3} {'Operação':<12} {'Kernel':<8} {'Score':>7}  "
          f"{'IoU':>7} {'F1':>7} {'Acc':>7} {'SSIM':>7}")
    print(f"  {'-'*3} {'-'*12} {'-'*8} {'-'*7}  {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for r in ranking:
        print(f"  {r['rank']:<3} {r['operacao']:<12} {r['kernel']:<8} "
              f"{r['score_medio']:>7.4f}  "
              f"{r['jaccard_medio']:>7.4f} {r['f1_medio']:>7.4f} "
              f"{r['acc_medio']:>7.4f} {r['ssim_medio']:>7.4f}")

    melhor = ranking[0]
    print(f"\n  ★ MELHOR GERAL: {melhor['operacao']}  kernel={melhor['kernel']}"
          f"  score={melhor['score_medio']:.4f}")

    for campo, rotulo in [
        ("jaccard_medio",  "IoU "),
        ("f1_medio",       "F1  "),
        ("acc_medio",      "Acc "),
        ("ssim_medio",     "SSIM"),
    ]:
        venc = max(ranking, key=lambda x: x[campo])
        prefixo_str = "├─" if campo != "ssim_medio" else "└─"
        print(f"  {prefixo_str} Melhor {rotulo}: {venc['operacao']}  "
              f"{venc['kernel']}  ({venc[campo]:.4f})")
    print(sep)


def _escrever_csv(caminho: Path, campos: list[str], linhas: list[dict]) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows([{k: l[k] for k in campos} for l in linhas])
