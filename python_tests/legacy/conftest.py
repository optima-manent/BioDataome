def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks slow validation gates (deselect with -m 'not slow')")
