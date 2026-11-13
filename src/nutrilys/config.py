"""Paramètres et chemins du projet, centralisés ici pour tout le code."""

from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PRODUCTS_PATH = PROCESSED_DIR / "products.parquet"
CLEANING_REPORT_PATH = REPORTS_DIR / "cleaning_report.json"

# Source Open Food Facts (export Parquet publié sur Hugging Face)
OFF_HF_REPO = "openfoodfacts/product-database"
OFF_HF_FILE = "food.parquet"
OFF_RAW_PATH = RAW_DIR / OFF_HF_FILE
OFF_INTERIM_PATH = INTERIM_DIR / "off_products.parquet"

# Source RappelConso V2 (rappels de produits, un code-barres par ligne)
RAPPELCONSO_URL = (
    "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "rappelconso-v2-gtin-trie/exports/parquet"
)
RAPPELCONSO_RAW_PATH = RAW_DIR / "rappelconso.parquet"

# Taxonomie des additifs d'Open Food Facts (noms, évaluations EFSA, liste ANSES)
ADDITIVES_TAXONOMY_URL = "https://static.openfoodfacts.org/data/taxonomies/additives.json"
ADDITIVES_RAW_PATH = RAW_DIR / "additives_taxonomy.json"

# Tables dérivées de l'étape 5
ADDITIVES_PATH = PROCESSED_DIR / "additives.parquet"
RECALLS_PATH = PROCESSED_DIR / "recalls.parquet"
CONTROVERSIES_PATH = PROCESSED_DIR / "controversies.parquet"

# Modélisation (étapes 6 et 7)
SPLITS_PATH = PROCESSED_DIR / "splits.parquet"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
MODEL_REPORTS_DIR = REPORTS_DIR / "models"
NOVA_ESTIMATED_PATH = PREDICTIONS_DIR / "nova_estimated.parquet"
TFIDF_MAX_FEATURES = 50_000
TFIDF_MIN_DF = 5
# XGBoost : seulement les termes les plus discriminants (chi²), 12 fois plus rapide
XGB_TEXT_FEATURES = 5_000

# Seuils des signaux de controverse
MANY_ADDITIVES = 5  # « beaucoup d'additifs » à partir de ce nombre
MIN_BRAND_PRODUCTS = 30  # produits notés minimum pour classer une marque

# Nutriments extraits, valeurs pour 100 g
NUTRIENTS = [
    "energy-kcal",
    "fat",
    "saturated-fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "proteins",
    "salt",
]

# Bornes de validité des nutriments (pour 100 g). Hors bornes, la valeur est mise à vide.
NUTRIENT_MAX_100G = {"energy_kcal_100g": 900.0}  # 900 kcal : graisse pure
DEFAULT_NUTRIENT_MAX_100G = 100.0
# Tolérance d'arrondi pour les règles « sous-total <= total » (graisses saturées <= graisses)
NUTRIENT_TOLERANCE_G = 1.0

# Valeurs valides des scores
GREEN_SCORE_GRADES = ["a-plus", "a", "b", "c", "d", "e", "f"]

# Ressources allouées à DuckDB (à ajuster selon la mémoire de la machine)
DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_THREADS = 4

# Périmètre géographique (valeurs du champ countries_tags d'Open Food Facts)
PHASE_1_COUNTRIES = ["en:france"]
PHASE_2_COUNTRIES = ["en:belgium", "en:spain", "en:germany", "en:italy"]
COUNTRIES = PHASE_1_COUNTRIES

# Labels considérés comme bio (champ labels_tags)
ORGANIC_LABELS = [
    "en:organic",
    "en:eu-organic",
    "fr:ab-agriculture-biologique",
]

# Cibles
NUTRISCORE_GRADES = ["a", "b", "c", "d", "e"]
NOVA_GROUPS = [1, 2, 3, 4]

# Reproductibilité
RANDOM_SEED = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15


def ensure_dirs() -> None:
    """Crée les dossiers de travail s'ils n'existent pas."""
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
