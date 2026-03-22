# GERADO AUTOMATICAMENTE por gerar_config.py
# Edite à vontade. Para restaurar os defaults, rode:  python gerar_config.py
#
# Campos de cada experimento:
#   categorias  : lista de categorias do dataset  (use OU categorias OU nomes)
#   nomes       : lista de nomes de imagens específicas
#   operacoes   : lista de OperacaoMorfologica  ou  "todas"
#   kernels     : tamanhos dos elementos estruturantes (ímpares, ex: [3, 5, 7])
#   ruido       : TipoRuido.SALT_PEPPER  ou  TipoRuido.GAUSSIANO
#   intensidade : fração de pixels corrompidos (0.0 – 1.0)
#   seed        : garante reprodutibilidade dos resultados

from processador import OperacaoMorfologica, TipoRuido

DATASET    = "./dataset/STI"
RESULTADOS = "./resultados"

EXPERIMENTOS = {
    # Additional (28 imagens)
    "additional_todas_operacoes": dict(
        categorias  = ["Additional"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Classic (13 imagens)
    "classic_todas_operacoes": dict(
        categorias  = ["Classic"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Fingerprint (3 imagens)
    "fingerprint_todas_operacoes": dict(
        categorias  = ["Fingerprint"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # High resolution (4 imagens)
    "high_resolution_todas_operacoes": dict(
        categorias  = ["High resolution"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Medical (11 imagens)
    "medical_todas_operacoes": dict(
        categorias  = ["Medical"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Old classic (27 imagens)
    "old_classic_todas_operacoes": dict(
        categorias  = ["Old classic"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Special (4 imagens)
    "special_todas_operacoes": dict(
        categorias  = ["Special"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Sun and Planets (17 imagens)
    "sun_and_planets_todas_operacoes": dict(
        categorias  = ["Sun and Planets"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Texture (6 imagens)
    "texture_todas_operacoes": dict(
        categorias  = ["Texture"],
        operacoes   = "todas",
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
}
