# NUTRILYS

Cartographie de la qualité nutritionnelle des produits de supermarché à partir des données ouvertes d'Open Food Facts. Projet de l'entreprise fictive SoGood.

## Objectifs

1. **Qualité nutritionnelle** : Nutri-Score, NOVA, additifs, sucre, sel et graisses saturées.
2. **Controverses** : produits et marques dont les indicateurs se contredisent.
3. **Prédiction** : groupe NOVA et Nutri-Score à partir du texte produit, baseline classique contre transformer.
4. **Web app** : recherche, filtres et fiche produit.
5. **Bio vs conventionnel** : comparaison à catégorie égale, avec tests statistiques.

Périmètre de la phase 1 : produits vendus en France.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/), qui installe aussi la bonne version de Python.

Sous Windows (PowerShell) :

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Sous macOS ou Linux :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Puis, depuis le dossier du projet :

```bash
uv sync
```

Cette commande crée l'environnement `.venv/` et installe toutes les dépendances. Les dépendances du transformer (étape 7) sont à part : `uv sync --extra nlp`.

## Vérifier l'installation

```bash
uv run pytest
uv run ruff check .
```

## Données

```bash
uv run python -m nutrilys.data.download   # Open Food Facts (plusieurs Go) et RappelConso
uv run python -m nutrilys.data.extract    # produits vendus en France -> data/interim/
uv run python -m nutrilys.data.clean      # table de référence -> data/processed/products.parquet
```

Le nettoyage écrit aussi `reports/cleaning_report.json` (effectifs à chaque étape) et `docs/data_dictionary.md` (description de chaque colonne). Le notebook `notebooks/01_qualite_donnees.ipynb` présente ces résultats.

## Analyses

| Notebook | Contenu |
|---|---|
| `01_qualite_donnees` | Entonnoir de nettoyage, taux de remplissage, effectifs par objectif |
| `02_qualite_nutritionnelle` | Nutri-Score, NOVA, croisement des deux, additifs, nutriments par rayon |

Les notebooks s'ouvrent dans VS Code avec le kernel `.venv`. Les figures interactives sont enregistrées dans `reports/figures/`.

Une fois l'extraction faite, le fichier brut `data/raw/food.parquet` peut être supprimé pour libérer de la place : il se retélécharge avec la première commande.

Le téléchargement reprend là où il s'était arrêté en cas de coupure. Les fichiers déjà présents ne sont pas retéléchargés (option `--force` pour forcer).

## Structure

```
nutrilys/
├── data/                  # ignoré par git
│   ├── raw/               # exports bruts, jamais modifiés
│   ├── interim/           # extraits filtrés
│   └── processed/         # tables finales
├── docs/                  # dictionnaire de données
├── notebooks/             # un notebook par étape d'analyse
├── src/nutrilys/
│   ├── config.py          # chemins et paramètres
│   ├── data/              # téléchargement, nettoyage, features
│   ├── analysis/          # statistiques, controverses, bio
│   ├── models/            # baseline, transformer, évaluation
│   └── viz/               # thème et graphiques communs
├── app/                   # application Streamlit
├── models/                # modèles entraînés, ignoré par git
├── reports/figures/
└── tests/
```

## Sources

- [Open Food Facts](https://world.openfoodfacts.org/data), base de données sous licence ODbL.
- RappelConso, EFSA OpenFoodTox, table Ciqual (ANSES).

## Avancement

| Étape | Contenu | Statut |
|---|---|---|
| 0 | Mise en place | Fait |
| 1 | Acquisition des données | Fait |
| 2 | Nettoyage et table de référence | Fait |
| 3 | Qualité nutritionnelle | En cours |
| 4 | Bio vs conventionnel | À faire |
| 5 | Controverses | À faire |
| 6 | Modèles baseline | À faire |
| 7 | Modèle transformer | À faire |
| 8 | Web app | À faire |
| 9 | Finalisation | À faire |
