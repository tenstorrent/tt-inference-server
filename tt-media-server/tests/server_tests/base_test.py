from abc import abstractmethod
import os
import asyncio
import time
import traceback

from tests.server_tests.test_config import TestConfig


class BaseTest:
    def __init__(self, config: TestConfig, targets):
        self.config = config
        self.targets = targets
        self.service_port = os.getenv("SERVICE_PORT", "8000")
        self.timeout = config.get("timeout", 300)  # Default 5 minutes
        self.max_retries = config.get("max_retries", 3)  # Default 3 retries
        self.retry_delay = config.get("retry_delay", 5)  # Default 5 seconds between retries
        
    def run_tests(self):
        last_exception = None
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                print(f"Running tests (attempt {attempt + 1}/{self.max_retries + 1})")
                
                result = asyncio.run(asyncio.wait_for(
                    self._run_specific_test_async(), 
                    timeout=self.timeout
                ))
                
                print("Tests completed successfully")
                return result  # Success - exit retry loop
                
            except asyncio.TimeoutError as e:
                last_exception = e
                print(f"Tests timed out after {self.timeout} seconds (attempt {attempt + 1}/{self.max_retries + 1})")
                
            except Exception as e:
                last_exception = e
                print(f"Test failed with exception (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}")
                print(f"Exception type: {type(e).__name__}")
                # Optionally print traceback for debugging
                print(f"Traceback: {traceback.format_exc()}")
            
            # Don't wait after the last attempt
            if attempt < self.max_retries:
                print(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        # All retries exhausted
        print(f"All {self.max_retries + 1} attempts failed. Last exception:")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Tests failed after all retry attempts")
    
    @abstractmethod
    async def _run_specific_test_async(self, target):
        pass
