"""
Escaneia o dataset e regenera config.py com um experimento por categoria (defaults).
Use quando quiser resetar o config:  python gerar_config.py
"""

from pathlib import Path

from processador.catalogo import CatalogoImagens

DATASET    = "./dataset/STI"
RESULTADOS = "./resultados"
CONFIG_OUT = Path("config.py")

# Defaults aplicados a todos os experimentos gerados
DEFAULT_KERNELS     = [3, 5]
DEFAULT_RUIDO       = "TipoRuido.SALT_PEPPER"
DEFAULT_INTENSIDADE = 0.05
DEFAULT_SEED        = 42


def _nome_experimento(categoria: str) -> str:
    """Converte nome da categoria em chave válida para o dict."""
    return categoria.lower().replace(" ", "_") + "_todas_operacoes"


def gerar() -> None:
    catalogo = CatalogoImagens(DATASET)
    catalogo.construir()

    # Agrupa por categoria preservando ordem
    categorias: dict[str, list] = {}
    for info in catalogo._itens.values():
        categorias.setdefault(info.categoria, []).append(info)

    if not categorias:
        print(f"[ERRO] Nenhuma imagem encontrada em {DATASET}")
        return

    linhas: list[str] = []

    # Cabeçalho
    linhas += [
        "# GERADO AUTOMATICAMENTE por gerar_config.py",
        "# Edite à vontade. Para restaurar os defaults, rode:  python gerar_config.py",
        "#",
        "# Campos de cada experimento:",
        "#   categorias  : lista de categorias do dataset  (use OU categorias OU nomes)",
        "#   nomes       : lista de nomes de imagens específicas",
        "#   operacoes   : lista de OperacaoMorfologica  ou  \"todas\"",
        "#   kernels     : tamanhos dos elementos estruturantes (ímpares, ex: [3, 5, 7])",
        "#   ruido       : TipoRuido.SALT_PEPPER  ou  TipoRuido.GAUSSIANO",
        "#   intensidade : fração de pixels corrompidos (0.0 – 1.0)",
        "#   seed        : garante reprodutibilidade dos resultados",
        "",
        "from processador import OperacaoMorfologica, TipoRuido",
        "",
        f'DATASET    = "{DATASET}"',
        f'RESULTADOS = "{RESULTADOS}"',
        "",
        "EXPERIMENTOS = {",
    ]

    # Uma entrada por categoria
    for categoria in sorted(categorias):
        n_imagens = len(categorias[categoria])
        chave     = _nome_experimento(categoria)
        linhas += [
            f'    # {categoria} ({n_imagens} imagens)',
            f'    "{chave}": dict(',
            f'        categorias  = ["{categoria}"],',
            f'        operacoes   = "todas",',
            f'        kernels     = {DEFAULT_KERNELS},',
            f'        ruido       = {DEFAULT_RUIDO},',
            f'        intensidade = {DEFAULT_INTENSIDADE},',
            f'        seed        = {DEFAULT_SEED},',
            f'    ),',
        ]

    linhas.append("}")
    linhas.append("")

    CONFIG_OUT.write_text("\n".join(linhas), encoding="utf-8")
    print(f"config.py gerado com {len(categorias)} experimentos:")
    for cat in sorted(categorias):
        print(f"  [{len(categorias[cat]):3d} imgs] {cat}")


if __name__ == "__main__":
    gerar()
