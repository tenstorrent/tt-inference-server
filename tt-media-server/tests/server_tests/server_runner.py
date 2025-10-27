import time
from typing import List, Any


class TestReport:
    """Represents the result of a test execution"""
    def __init__(self, test_name: str, success: bool, duration: float, error: str = None, targets=None):
        self.test_name = test_name
        self.success = success
        self.duration = duration
        self.error = error
        self.targets = targets
        self.timestamp = time.time()

    def __str__(self):
        status = "✓ PASS" if self.success else "✗ FAIL"
        return f"{status} {self.test_name} ({self.duration:.2f}s)"


class ServerRunner:
    def __init__(self, test_cases: List[Any]):
        self.test_cases = test_cases
        self.reports = []
    
    def run(self):
        self._run_all_tests()
        return self.reports
    
    def _run_all_tests(self):
        for case in self.test_cases:
            test_name = case.test.__class__.__name__
            start_time = time.perf_counter()
            
            try:
                print(f"Running test case: {test_name}")
                result = case.test.run_tests()
                duration = time.perf_counter() - start_time
                
                # Create success report
                report = TestReport(
                    test_name=test_name,
                    success=True,
                    duration=duration,
                    targets=case.targets
                )
                self.reports.append(report)
                print(f"✓ Test case {test_name} passed in {duration:.2f}s")
                
            except SystemExit as e:
                duration = time.perf_counter() - start_time
                report = TestReport(
                    test_name=test_name,
                    success=False,
                    duration=duration,
                    error=f"SystemExit with code {e.code}",
                    targets=case.targets
                )
                self.reports.append(report)
                print(f"Test case {test_name} exited with status {e.code}")
                exit(e.code)
                
            except Exception as e:
                duration = time.perf_counter() - start_time
                report = TestReport(
                    test_name=test_name,
                    success=False,
                    duration=duration,
                    error=str(e),
                    targets=case.targets
                )
                self.reports.append(report)
                print(f"✗ Test case {test_name} failed: {e}")
    
    def _generate_report(self):
        # Placeholder for report generation logic
        pass