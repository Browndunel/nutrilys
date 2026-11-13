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
uv run python -m nutrilys.data.download   # Open Food Facts (plusieurs Go), RappelConso, additifs
uv run python -m nutrilys.data.extract    # produits vendus en France -> data/interim/
uv run python -m nutrilys.data.clean      # table de référence -> data/processed/products.parquet
uv run python -m nutrilys.analysis.controversies   # additifs, rappels, produits signalés
```

## Modèles

```bash
uv run python -m nutrilys.models.baseline --sample 50000   # essai rapide
uv run python -m nutrilys.models.baseline --models lr      # régression logistique, 4 tâches
uv run python -m nutrilys.models.baseline --tasks nova_full nutriscore_full --models xgb
```

Quatre tâches (NOVA et Nutri-Score, chacun avec un scénario « texte seul » et un scénario de contrôle), deux modèles (TF-IDF + régression logistique, TF-IDF + XGBoost). XGBoost n'est entraîné que sur les scénarios de contrôle, seuls cas où il fait mieux que la régression logistique. Les métriques sont écrites dans `reports/models/`, les modèles dans `models/`. Le script estime aussi le groupe NOVA des produits sans liste d'ingrédients.

Le nettoyage écrit aussi `reports/cleaning_report.json` (effectifs à chaque étape) et `docs/data_dictionary.md` (description de chaque colonne). Le notebook `notebooks/01_qualite_donnees.ipynb` présente ces résultats.

## Analyses

| Notebook | Contenu |
|---|---|
| `01_qualite_donnees` | Entonnoir de nettoyage, taux de remplissage, effectifs par objectif |
| `02_qualite_nutritionnelle` | Nutri-Score, NOVA, croisement des deux, additifs, nutriments par rayon |
| `03_bio_vs_conventionnel` | Bio contre conventionnel à catégorie égale : tests par sous-groupe, taux standardisés |
| `04_controverses` | Cinq signaux de controverse, additifs à risque, rappels, marques |
| `05_baseline` | Prédiction de NOVA et du Nutri-Score : modèles de référence, erreurs, estimation NOVA |

Les notebooks s'ouvrent dans VS Code avec le kernel `.venv`. Les figures interactives sont enregistrées dans `reports/figures/`.

Une fois l'extraction faite, le fichier brut `data/raw/food.parquet` peut être supprimé pour libérer de la place : il se retélécharge avec la première commande. Pour ne télécharger qu'une source : `--only off`, `--only rappelconso` ou `--only additives`.

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
- [RappelConso](https://data.economie.gouv.fr/explore/dataset/rappelconso-v2-gtin-trie) (DGCCRF), rappels de produits avec code-barres.
- Évaluations des additifs par l'EFSA et liste de l'ANSES, via la taxonomie des additifs d'Open Food Facts.

## Avancement

| Étape | Contenu | Statut |
|---|---|---|
| 0 | Mise en place | Fait |
| 1 | Acquisition des données | Fait |
| 2 | Nettoyage et table de référence | Fait |
| 3 | Qualité nutritionnelle | Fait |
| 4 | Bio vs conventionnel | Fait |
| 5 | Controverses | Fait |
| 6 | Modèles baseline | En cours |
| 7 | Modèle transformer | À faire |
| 8 | Web app | À faire |
| 9 | Finalisation | À faire |
