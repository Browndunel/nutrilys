from nutrilys import config


def test_root_dir_contains_pyproject():
    assert (config.ROOT_DIR / "pyproject.toml").exists()


def test_data_dirs_are_inside_project():
    for path in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR):
        assert config.ROOT_DIR in path.parents


def test_split_sizes_leave_a_training_set():
    assert 0 < config.TEST_SIZE + config.VALID_SIZE < 0.5
