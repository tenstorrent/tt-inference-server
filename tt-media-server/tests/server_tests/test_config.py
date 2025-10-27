from tests.server_tests.base_test import BaseTest


class TestConfig:
    test_timeout: int = 30
    retry_attempts: int = 3
    break_on_failure: bool = True
    
    def __init__(self):
        pass
    
class TestCase:
    test: BaseTest
    config: TestConfig
    targets: object