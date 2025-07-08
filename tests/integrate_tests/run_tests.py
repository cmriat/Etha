import json
import math
import os
import subprocess
import sys

# Test cases definition
TEST_CASES = [
    '{"name": "test_transfer_general_case", "source_mesh": [1, 2, 2, 1, 2], "target_mesh": [1, 4, 1, 1, 2]}',
    '{"name": "test_transfer_same_mesh", "source_mesh": [1, 1, 2, 1, 2], "target_mesh": [1, 1, 2, 1, 2]}',
    '{"name": "test_transfer_2d_to_1d", "source_mesh": [1, 1, 2, 1, 2], "target_mesh": [1, 1, 1, 1, 4]}',
    '{"name": "test_transfer_1d_to_2d", "source_mesh": [1, 1, 1, 1, 4], "target_mesh": [1, 1, 2, 1, 2]}',
    '{"name": "test_transfer_high_dimension", "source_mesh": [1, 1, 4, 1, 4], "target_mesh": [1, 1, 2, 1, 8]}',
    '{"name": "test_transfer_small_to_large", "source_mesh": [1, 1, 1, 1, 2], "target_mesh": [1, 1, 2, 1, 4]}',
    '{"name": "test_transfer_large_to_small", "source_mesh": [1, 1, 4, 1, 4], "target_mesh": [1, 1, 2, 1, 2]}',
    '{"name": "test_transfer_rectangular_mesh", "source_mesh": [1, 1, 2, 1, 4], "target_mesh": [1, 1, 4, 1, 2]}',
    '{"name": "test_transfer_unit_dimensions", "source_mesh": [1, 1, 1, 1, 4], "target_mesh": [1, 1, 4, 1, 1]}',
    '{"name": "test_transfer_large_replica", "source_mesh": [2, 2, 2, 1, 2], "target_mesh": [2, 2, 1, 1, 4]}',
    '{"name": "test_transfer_asymmetric_mesh", "source_mesh": [2, 1, 2, 1, 2], "target_mesh": [1, 2, 2, 1, 2]}',
    '{"name": "test_transfer_multi_replica", "source_mesh": [1, 2, 2, 1, 2], "target_mesh": [2, 1, 2, 1, 2]}',
    '{"name": "test_transfer_different_replica_1", "source_mesh": [2, 1, 2, 1, 2], "target_mesh": [2, 4, 1, 1, 2]}',
    '{"name": "test_transfer_different_replica_2", "source_mesh": [2, 1, 2, 1, 2], "target_mesh": [4, 4, 1, 1, 2]}',
    '{"name": "test_transfer_different_replica_3", "source_mesh": [1, 1, 2, 1, 2], "target_mesh": [4, 4, 1, 1, 2]}',
    '{"name": "test_transfer_different_replica_4", "source_mesh": [1, 1, 2, 1, 2], "target_mesh": [2, 4, 1, 1, 2]}',
    '{"name": "test_transfer_different_replica_5", "source_mesh": [2, 2, 2, 1, 2], "target_mesh": [2, 4, 1, 1, 2]}',
    '{"name": "test_transfer_different_replica_6", "source_mesh": [2, 2, 2, 1, 2], "target_mesh": [4, 4, 1, 1, 2]}',
    '{"name": "test_transfer_different_replica_7", "source_mesh": [4, 2, 2, 1, 2], "target_mesh": [2, 4, 1, 1, 4]}',
    '{"name": "test_transfer_different_replica_8", "source_mesh": [4, 1, 2, 1, 2], "target_mesh": [4, 2, 1, 1, 4]}',
]

def run_tests():
    """
    Runs all integration test cases.
    """
    total_tests = len(TEST_CASES)
    for i, case_json in enumerate(TEST_CASES):
        try:
            test_case = json.loads(case_json)
            
            # Calculate world size
            source_size = math.prod(test_case['source_mesh'])
            target_size = math.prod(test_case['target_mesh'])
            world_size = source_size + target_size
            
            name = test_case['name']
            
            print("=" * 70)
            print(f"({i + 1}/{total_tests}) Starting test: {name} with WORLD_SIZE={world_size}")
            print("=" * 70)
            
            # Set environment variable for the test script
            os.environ['TEST_CASE_JSON'] = case_json
            
            # Define the command to run
            # Note: Assumes the test file is named 'test_integrate_p2p.py'
            cmd = [
                "torchrun",
                f"--nproc_per_node={world_size}",
                "tests/integrate_tests/test_integrate_p2p.py"
            ]
            
            # Run the test
            subprocess.run(cmd, check=True)
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error processing test case: {case_json}", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Test '{name}' failed with exit code {e.returncode}", file=sys.stderr)
            sys.exit(1)
        finally:
            # Clean up environment variable
            if 'TEST_CASE_JSON' in os.environ:
                del os.environ['TEST_CASE_JSON']

    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_tests()
