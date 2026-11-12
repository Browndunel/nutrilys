"""Paramètres et chemins du projet, centralisés ici pour tout le code."""

from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Source Open Food Facts (export Parquet publié sur Hugging Face)
OFF_HF_REPO = "openfoodfacts/product-database"
OFF_HF_FILE = "food.parquet"
OFF_RAW_PATH = RAW_DIR / OFF_HF_FILE

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
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
