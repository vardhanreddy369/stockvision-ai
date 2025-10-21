"""
Test runner and coverage reporter for StockVision AI
Executes all test suites and generates coverage report
"""
import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests(verbosity=2):
    """
    Run all test suites
    
    Args:
        verbosity: Verbosity level (0=quiet, 1=normal, 2=verbose)
    
    Returns:
        TestResult object
    """
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def print_test_summary(result):
    """
    Print summary of test results
    
    Args:
        result: TestResult object
    """
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped
    
    print(f"\nTotal Tests Run:    {total_tests}")
    print(f"✓ Passed:            {passed}")
    print(f"✗ Failed:            {failures}")
    print(f"⚠ Errors:            {errors}")
    print(f"⊘ Skipped:           {skipped}")
    
    if failures > 0:
        print(f"\n❌ FAILURES ({failures}):")
        for test, traceback in result.failures:
            print(f"\n  {test}:")
            print(f"  {traceback}")
    
    if errors > 0:
        print(f"\n❌ ERRORS ({errors}):")
        for test, traceback in result.errors:
            print(f"\n  {test}:")
            print(f"  {traceback}")
    
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\n📊 Success Rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    print("=" * 70)


if __name__ == '__main__':
    print("\n🚀 Starting StockVision AI Test Suite...")
    print("=" * 70)
    
    result = run_tests(verbosity=2)
    print_test_summary(result)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
