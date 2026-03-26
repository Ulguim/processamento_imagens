# GERADO AUTOMATICAMENTE por gerar_config.py
# Edite à vontade. Para restaurar os defaults, rode:  python gerar_config.py
#
# Campos de cada experimento:
#   categorias  : lista de categorias do dataset  (use OU categorias OU nomes)
#   nomes       : lista de nomes de imagens específicas
#   operacoes   : lista de OperacaoMorfologica
#   kernels     : tamanhos dos elementos estruturantes (ímpares, ex: [3, 5, 7])
#   ruido       : TipoRuido.SALT_PEPPER  ou  TipoRuido.GAUSSIANO
#   intensidade : fração de pixels corrompidos (0.0 – 1.0)
#   seed        : garante reprodutibilidade dos resultados

from processador import OperacaoMorfologica, TipoRuido

DATASET    = "./dataset/STI"
RESULTADOS = "./resultados"

# True  = também salva um arquivo compilado com todas as operações juntas
# False = apenas os arquivos separados por operação (padrão)
GERAR_VISAO_GERAL = False

# True  = gera CSVs de métricas e ranking automático por experimento
GERAR_RANKING = True

EXPERIMENTOS = {
    # Additional (28 imagens)
    "additional_morfologico": dict(
        categorias  = ["Additional"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Classic (13 imagens)
    "classic_morfologico": dict(
        categorias  = ["Classic"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Fingerprint (3 imagens)
    "fingerprint_morfologico": dict(
        categorias  = ["Fingerprint"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # High resolution (4 imagens)
    "high_resolution_morfologico": dict(
        categorias  = ["High resolution"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Medical (11 imagens)
    "medical_morfologico": dict(
        categorias  = ["Medical"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Old classic (27 imagens)
    "old_classic_morfologico": dict(
        categorias  = ["Old classic"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Special (4 imagens)
    "special_morfologico": dict(
        categorias  = ["Special"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Sun and Planets (17 imagens)
    "sun_and_planets_morfologico": dict(
        categorias  = ["Sun and Planets"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
    # Texture (6 imagens)
    "texture_morfologico": dict(
        categorias  = ["Texture"],
        operacoes   = [OperacaoMorfologica.EROSAO, OperacaoMorfologica.DILATACAO, OperacaoMorfologica.ABERTURA, OperacaoMorfologica.FECHAMENTO],
        kernels     = [3, 5],
        ruido       = TipoRuido.SALT_PEPPER,
        intensidade = 0.05,
        seed        = 42,
    ),
}
