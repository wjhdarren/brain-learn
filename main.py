def main():
    # Run tests if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        from tests.run_tests import run_all_tests
        print("\nRunning all tests:")
        success = run_all_tests()
        print(f"Tests {'passed' if success else 'failed'}!")


if __name__ == "__main__":
    main()
