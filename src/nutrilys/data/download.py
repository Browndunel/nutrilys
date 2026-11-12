"""Téléchargement des données brutes : Open Food Facts et RappelConso.

Usage :
    uv run python -m nutrilys.data.download
    uv run python -m nutrilys.data.download --only off --force
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


def download_rappelconso(force: bool = False) -> Path:
    """Télécharge l'export complet de RappelConso V2 au format Parquet."""
    dest = config.RAPPELCONSO_RAW_PATH
    if dest.exists() and not force:
        logger.info("RappelConso déjà présent : %s (%.1f Mo)", dest, _size_mb(dest))
        return dest

    logger.info("Téléchargement de RappelConso...")
    tmp = dest.with_suffix(".part")
    with (
        urllib.request.urlopen(config.RAPPELCONSO_URL, timeout=120) as response,
        tmp.open("wb") as f,
    ):
        shutil.copyfileobj(response, f)
    tmp.replace(dest)
    logger.info("RappelConso téléchargé : %s (%.1f Mo)", dest, _size_mb(dest))
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge les données brutes du projet.")
    parser.add_argument("--only", choices=["off", "rappelconso"], help="une seule source")
    parser.add_argument("--force", action="store_true", help="retélécharge même si présent")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()

    if args.only in (None, "off"):
        download_off(force=args.force)
    if args.only in (None, "rappelconso"):
        download_rappelconso(force=args.force)


if __name__ == "__main__":
    main()
