from rl_comm import get_p2p_map
from torch.distributed.tensor.placement_types import Replicate, Shard

# [cp, dp_replicate, pp, dp_shard, tp]

def test_transfer_general_case():
    """Test complex case with different mesh configurations"""
    source_mesh = (1, 2, 2, 1, 2)
    target_mesh = (1, 4, 1, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 4],
        1: [1, 5],
        2: [0, 4],
        3: [1, 5],
        4: [2, 6],
        5: [3, 7],
        6: [2, 6],
        7: [3, 7],
    }


def test_transfer_same_mesh():
    """Test when source and target meshes are identical"""
    source_mesh = (1, 1, 2, 1, 2)
    target_mesh = (1, 1, 2, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {0: [0], 1: [1], 2: [2], 3: [3]}


def test_transfer_2d_to_1d():
    """Test transfer from 2D to 1D sharding"""
    source_mesh = (1, 1, 2, 1, 2)
    target_mesh = (1, 1, 1, 1, 4)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {0: [0, 1], 1: [2, 3], 2: [0, 1], 3: [2, 3]}


def test_transfer_1d_to_2d():
    """Test transfer from 1D to 2D sharding"""
    source_mesh = (1, 1, 1, 1, 4)
    target_mesh = (1, 1, 2, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {0: [0, 2], 1: [0, 2], 2: [1, 3], 3: [1, 3]}


def test_transfer_high_dimension():
    """Test with higher dimensional sharding"""
    source_mesh = (1, 1, 4, 1, 4)
    target_mesh = (1, 1, 2, 1, 8)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [0, 1],
        5: [2, 3],
        6: [4, 5],
        7: [6, 7],
        8: [8, 9],
        9: [10, 11],
        10: [12, 13],
        11: [14, 15],
        12: [8, 9],
        13: [10, 11],
        14: [12, 13],
        15: [14, 15],
    }


def test_transfer_small_to_large():
    """Test transfer from smaller to larger mesh"""
    source_mesh = (1, 1, 1, 1, 2)
    target_mesh = (1, 1, 2, 1, 4)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {0: [0, 1, 4, 5], 1: [2, 3, 6, 7]}


def test_transfer_large_to_small():
    """Test transfer from larger to smaller mesh"""
    source_mesh = (1, 1, 4, 1, 4)
    target_mesh = (1, 1, 2, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0],
        1: [0],
        2: [1],
        3: [1],
        4: [0],
        5: [0],
        6: [1],
        7: [1],
        8: [2],
        9: [2],
        10: [3],
        11: [3],
        12: [2],
        13: [2],
        14: [3],
        15: [3],
    }


def test_transfer_power_of_two():
    """Test with power of two dimensions"""
    source_mesh = (1, 1, 8, 1, 8)
    target_mesh = (1, 1, 4, 1, 16)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8, 9],
        5: [10, 11],
        6: [12, 13],
        7: [14, 15],
        8: [0, 1],
        9: [2, 3],
        10: [4, 5],
        11: [6, 7],
        12: [8, 9],
        13: [10, 11],
        14: [12, 13],
        15: [14, 15],
        16: [16, 17],
        17: [18, 19],
        18: [20, 21],
        19: [22, 23],
        20: [24, 25],
        21: [26, 27],
        22: [28, 29],
        23: [30, 31],
        24: [16, 17],
        25: [18, 19],
        26: [20, 21],
        27: [22, 23],
        28: [24, 25],
        29: [26, 27],
        30: [28, 29],
        31: [30, 31],
        32: [32, 33],
        33: [34, 35],
        34: [36, 37],
        35: [38, 39],
        36: [40, 41],
        37: [42, 43],
        38: [44, 45],
        39: [46, 47],
        40: [32, 33],
        41: [34, 35],
        42: [36, 37],
        43: [38, 39],
        44: [40, 41],
        45: [42, 43],
        46: [44, 45],
        47: [46, 47],
        48: [48, 49],
        49: [50, 51],
        50: [52, 53],
        51: [54, 55],
        52: [56, 57],
        53: [58, 59],
        54: [60, 61],
        55: [62, 63],
        56: [48, 49],
        57: [50, 51],
        58: [52, 53],
        59: [54, 55],
        60: [56, 57],
        61: [58, 59],
        62: [60, 61],
        63: [62, 63],
    }


def test_transfer_rectangular_mesh():
    """Test rectangular mesh configurations"""
    source_mesh = (1, 1, 2, 1, 4)
    target_mesh = (1, 1, 4, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 2],
        1: [0, 2],
        2: [1, 3],
        3: [1, 3],
        4: [4, 6],
        5: [4, 6],
        6: [5, 7],
        7: [5, 7],
    }


def test_transfer_unit_dimensions():
    """Test with unit dimensions"""
    source_mesh = (1, 1, 1, 1, 4)
    target_mesh = (1, 1, 4, 1, 1)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 1, 2, 3],
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 3],
        3: [0, 1, 2, 3],
    }


def test_transfer_large_replica():
    """Test with larger replica counts"""
    source_mesh = (2, 2, 2, 1, 2)
    target_mesh = (2, 2, 1, 1, 4)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    # Verify structure without exact values
    assert forward_map == {
        0: [0, 1],
        1: [2, 3],
        2: [0, 1],
        3: [2, 3],
        4: [4, 5],
        5: [6, 7],
        6: [4, 5],
        7: [6, 7],
        8: [8, 9],
        9: [10, 11],
        10: [8, 9],
        11: [10, 11],
        12: [12, 13],
        13: [14, 15],
        14: [12, 13],
        15: [14, 15],
    }


def test_transfer_asymmetric_mesh():
    """Test asymmetric mesh configurations"""
    source_mesh = (2, 1, 2, 1, 2)
    target_mesh = (1, 2, 2, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7]}


def test_transfer_multi_replica():
    """Test with multiple replicas"""
    source_mesh = (1, 2, 2, 1, 2)
    target_mesh = (2, 1, 2, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7]}


def test_transfer_different_replica_1():
    source_mesh = (2, 1, 2, 1, 2)
    target_mesh = (2, 4, 1, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 4, 8, 12],
        1: [1, 5, 9, 13],
        2: [0, 4, 8, 12],
        3: [1, 5, 9, 13],
        4: [2, 6, 10, 14],
        5: [3, 7, 11, 15],
        6: [2, 6, 10, 14],
        7: [3, 7, 11, 15],
    }


def test_transfer_different_replica_2():
    source_mesh = (2, 1, 2, 1, 2)
    target_mesh = (4, 4, 1, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 4, 8, 12, 16, 20, 24, 28],
        1: [1, 5, 9, 13, 17, 21, 25, 29],
        2: [0, 4, 8, 12, 16, 20, 24, 28],
        3: [1, 5, 9, 13, 17, 21, 25, 29],
        4: [2, 6, 10, 14, 18, 22, 26, 30],
        5: [3, 7, 11, 15, 19, 23, 27, 31],
        6: [2, 6, 10, 14, 18, 22, 26, 30],
        7: [3, 7, 11, 15, 19, 23, 27, 31],
    }


def test_transfer_different_replica_3():
    source_mesh = (1, 1, 2, 1, 2)
    target_mesh = (4, 4, 1, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    print(forward_map)
    assert forward_map == {
        0: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        1: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        2: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        3: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
    }


def test_transfer_different_replica_4():
    source_mesh = (1, 1, 2, 1, 2)
    target_mesh = (2, 4, 1, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 2, 4, 6, 8, 10, 12, 14],
        1: [1, 3, 5, 7, 9, 11, 13, 15],
        2: [0, 2, 4, 6, 8, 10, 12, 14],
        3: [1, 3, 5, 7, 9, 11, 13, 15],
    }


def test_transfer_different_replica_5():
    source_mesh = (2, 2, 2, 1, 2)
    target_mesh = (2, 4, 1, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 8],
        1: [1, 9],
        2: [0, 8],
        3: [1, 9],
        4: [2, 10],
        5: [3, 11],
        6: [2, 10],
        7: [3, 11],
        8: [4, 12],
        9: [5, 13],
        10: [4, 12],
        11: [5, 13],
        12: [6, 14],
        13: [7, 15],
        14: [6, 14],
        15: [7, 15],
    }


def test_transfer_different_replica_6():
    source_mesh = (2, 2, 2, 1, 2)
    target_mesh = (4, 4, 1, 1, 2)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 8, 16, 24],
        1: [1, 9, 17, 25],
        2: [0, 8, 16, 24],
        3: [1, 9, 17, 25],
        4: [2, 10, 18, 26],
        5: [3, 11, 19, 27],
        6: [2, 10, 18, 26],
        7: [3, 11, 19, 27],
        8: [4, 12, 20, 28],
        9: [5, 13, 21, 29],
        10: [4, 12, 20, 28],
        11: [5, 13, 21, 29],
        12: [6, 14, 22, 30],
        13: [7, 15, 23, 31],
        14: [6, 14, 22, 30],
        15: [7, 15, 23, 31],
    }


def test_transfer_different_replica_7():
    source_mesh = (4, 2, 2, 1, 2)
    target_mesh = (2, 4, 1, 1, 4)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 1],
        1: [2, 3],
        2: [0, 1],
        3: [2, 3],
        4: [4, 5],
        5: [6, 7],
        6: [4, 5],
        7: [6, 7],
        8: [8, 9],
        9: [10, 11],
        10: [8, 9],
        11: [10, 11],
        12: [12, 13],
        13: [14, 15],
        14: [12, 13],
        15: [14, 15],
        16: [16, 17],
        17: [18, 19],
        18: [16, 17],
        19: [18, 19],
        20: [20, 21],
        21: [22, 23],
        22: [20, 21],
        23: [22, 23],
        24: [24, 25],
        25: [26, 27],
        26: [24, 25],
        27: [26, 27],
        28: [28, 29],
        29: [30, 31],
        30: [28, 29],
        31: [30, 31],
    }


def test_transfer_different_replica_8():
    source_mesh = (4, 2, 2, 1, 2)
    target_mesh = (4, 4, 1, 1, 4)
    placements = (Replicate(), Replicate(), Shard(0), Shard(1), Shard(1))
    forward_map, _ = get_p2p_map(source_mesh, placements, target_mesh, placements)
    assert forward_map == {
        0: [0, 1, 32, 33],
        1: [2, 3, 34, 35],
        2: [0, 1, 32, 33],
        3: [2, 3, 34, 35],
        4: [4, 5, 36, 37],
        5: [6, 7, 38, 39],
        6: [4, 5, 36, 37],
        7: [6, 7, 38, 39],
        8: [8, 9, 40, 41],
        9: [10, 11, 42, 43],
        10: [8, 9, 40, 41],
        11: [10, 11, 42, 43],
        12: [12, 13, 44, 45],
        13: [14, 15, 46, 47],
        14: [12, 13, 44, 45],
        15: [14, 15, 46, 47],
        16: [16, 17, 48, 49],
        17: [18, 19, 50, 51],
        18: [16, 17, 48, 49],
        19: [18, 19, 50, 51],
        20: [20, 21, 52, 53],
        21: [22, 23, 54, 55],
        22: [20, 21, 52, 53],
        23: [22, 23, 54, 55],
        24: [24, 25, 56, 57],
        25: [26, 27, 58, 59],
        26: [24, 25, 56, 57],
        27: [26, 27, 58, 59],
        28: [28, 29, 60, 61],
        29: [30, 31, 62, 63],
        30: [28, 29, 60, 61],
        31: [30, 31, 62, 63],
    }


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_transfer_general_case,
        test_transfer_same_mesh,
        test_transfer_2d_to_1d,
        test_transfer_1d_to_2d,
        test_transfer_large_replica,
        test_transfer_asymmetric_mesh,
        test_transfer_high_dimension,
        test_transfer_small_to_large,
        test_transfer_large_to_small,
        test_transfer_multi_replica,
        test_transfer_power_of_two,
        test_transfer_rectangular_mesh,
        test_transfer_unit_dimensions,
    ]

    print("Running comprehensive tests for transfer function...")
    for i, test_func in enumerate(test_functions, 1):
        try:
            test_func()
            print(f"✓ Test {i:2d}: {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ Test {i:2d}: {test_func.__name__} failed - {e}")

    print(f"\nAll {len(test_functions)} tests completed!")