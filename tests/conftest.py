"""Outils de test : fabrication d'un extrait Open Food Facts miniature."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

STR = pa.string()
TAGS = pa.list_(pa.string())

# Schéma de data/interim/off_products.parquet tel que produit par extract.py
INTERIM_SCHEMA = pa.schema(
    [
        ("code", STR),
        ("product_name", STR),
        ("generic_name", STR),
        ("ingredients_text", STR),
        ("brands", STR),
        ("brands_tags", TAGS),
        ("categories_tags", TAGS),
        ("food_groups_tags", TAGS),
        ("compared_to_category", STR),
        ("labels_tags", TAGS),
        ("countries_tags", TAGS),
        ("additives_n", pa.int32()),
        ("additives_tags", TAGS),
        ("ingredients_n", pa.int32()),
        ("ingredients_analysis_tags", TAGS),
        ("ingredients_from_palm_oil_n", pa.int32()),
        ("nutriscore_grade", STR),
        ("nutriscore_score", pa.int32()),
        ("nova_group", pa.int32()),
        ("environmental_score_grade", STR),
        ("no_nutrition_data", pa.bool_()),
        ("completeness", pa.float32()),
        ("unique_scans_n", pa.int32()),
        ("last_modified_t", pa.int64()),
        ("energy_kcal_100g", pa.float64()),
        ("fat_100g", pa.float64()),
        ("saturated_fat_100g", pa.float64()),
        ("carbohydrates_100g", pa.float64()),
        ("sugars_100g", pa.float64()),
        ("fiber_100g", pa.float64()),
        ("proteins_100g", pa.float64()),
        ("salt_100g", pa.float64()),
    ]
)

DEFAULT_PRODUCT = {
    "product_name": "Produit",
    "countries_tags": ["en:france"],
    "brands_tags": [],
    "categories_tags": [],
    "food_groups_tags": [],
    "labels_tags": [],
    "additives_tags": [],
    "ingredients_analysis_tags": [],
    "last_modified_t": 1_700_000_000,
}


def write_interim(rows: list[dict], path) -> None:
    """Écrit des produits (champs absents = valeur par défaut ou vide) au format de l'extrait."""
    full_rows = [{**DEFAULT_PRODUCT, **row} for row in rows]
    columns = {name: [row.get(name) for row in full_rows] for name in INTERIM_SCHEMA.names}
    pq.write_table(pa.table(columns, schema=INTERIM_SCHEMA), path)


@pytest.fixture
def interim_factory(tmp_path):
    def _make(rows: list[dict]):
        path = tmp_path / "off_products.parquet"
        write_interim(rows, path)
        return path

    return _make
