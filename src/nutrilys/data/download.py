"""Téléchargement des données brutes : Open Food Facts, RappelConso, additifs.

Usage :
    uv run python -m nutrilys.data.download
    uv run python -m nutrilys.data.download --only additives
"""

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from nutrilys import config

logger = logging.getLogger(__name__)


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1024**2


def download_off(force: bool = False) -> Path:
    """Télécharge l'export Parquet d'Open Food Facts (reprise possible si interrompu)."""
    dest = config.OFF_RAW_PATH
    if dest.exists() and not force:
        logger.info("Open Food Facts déjà présent : %s (%.0f Mo)", dest, _size_mb(dest))
        return dest

    logger.info("Téléchargement de %s depuis Hugging Face, plusieurs Go...", config.OFF_HF_FILE)
    path = hf_hub_download(
        repo_id=config.OFF_HF_REPO,
        filename=config.OFF_HF_FILE,
        repo_type="dataset",
        local_dir=config.RAW_DIR,
        force_download=force,
    )
    path = Path(path)
    logger.info("Open Food Facts téléchargé : %s (%.0f Mo)", path, _size_mb(path))
    return path


def _download_url(url: str, dest: Path, name: str, force: bool) -> Path:
    """Téléchargement simple d'un fichier, écrit d'abord sous un nom temporaire."""
    if dest.exists() and not force:
        logger.info("%s déjà présent : %s (%.1f Mo)", name, dest, _size_mb(dest))
        return dest

    logger.info("Téléchargement de %s...", name)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)
    logger.info("%s téléchargé : %s (%.1f Mo)", name, dest, _size_mb(dest))
    return dest


def download_rappelconso(force: bool = False) -> Path:
    """Export complet de RappelConso V2 au format Parquet."""
    return _download_url(config.RAPPELCONSO_URL, config.RAPPELCONSO_RAW_PATH, "RappelConso", force)


def download_additives(force: bool = False) -> Path:
    """Taxonomie des additifs d'Open Food Facts (JSON, moins de 1 Mo)."""
    return _download_url(
        config.ADDITIVES_TAXONOMY_URL, config.ADDITIVES_RAW_PATH, "Taxonomie des additifs", force
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge les données brutes du projet.")
    parser.add_argument(
        "--only", choices=["off", "rappelconso", "additives"], help="une seule source"
    )
    parser.add_argument("--force", action="store_true", help="retélécharge même si présent")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Les bibliothèques HTTP journalisent chaque requête : on ne garde que les avertissements
    for name in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    config.ensure_dirs()

    if args.only in (None, "off"):
        download_off(force=args.force)
    if args.only in (None, "rappelconso"):
        download_rappelconso(force=args.force)
    if args.only in (None, "additives"):
        download_additives(force=args.force)


if __name__ == "__main__":
    main()
