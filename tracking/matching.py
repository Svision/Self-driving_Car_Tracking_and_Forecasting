from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_cost(cost_matrix: np.ndarray, s1: list, s2: list) -> Tuple[float, List, List]:
    total_cost = 0.0
    M, N = len(s1), len(s2)
    row_ids, col_ids = [], []
    if M < N:
        # rotate s1
        s1 = s1[1:] + s1[:1]  # [0, 1, 2, 3] -> [1, 2, 3, 0]
        for i in s1:
            min_cost = float('inf')
            min_i, min_j = -1, -1
            for j in s2:
                if cost_matrix[i, j] < min_cost and (j not in col_ids):
                    min_cost = cost_matrix[i, j]
                    min_i, min_j = i, j
            total_cost += min_cost
            row_ids.append(min_i)
            col_ids.append(min_j)
    else:
        # rotate s2
        s2 = s2[1:] + s2[:1]
        for j in s2:
            min_cost = float('inf')
            min_i, min_j = -1, -1
            for i in s1:
                if cost_matrix[i, j] < min_cost and (i not in row_ids):
                    min_cost = cost_matrix[i, j]
                    min_i, min_j = i, j
            total_cost += min_cost
            row_ids.append(min_i)
            col_ids.append(min_j)

    return total_cost, row_ids, col_ids


def greedy_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the greedy matching algorithm.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # DONE: Replace this stub code.
    M, N = cost_matrix.shape
    row_ids, col_ids = [], []
    min_total_cost = float('inf')

    curr_row_ids, curr_col_ids = [i for i in range(M)], [j for j in range(N)]
    for i in range(min(M, N)):
        curr_total_cost, curr_row_ids, curr_col_ids = greedy_cost(cost_matrix, curr_row_ids, curr_col_ids)
        if curr_total_cost < min_total_cost:
            min_total_cost = curr_total_cost
            row_ids, col_ids = curr_row_ids, curr_col_ids
    return row_ids, col_ids


def hungarian_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the Hungarian matching algorithm.
    For simplicity, we just call the scipy `linear_sum_assignment` function. Please refer to
    https://en.wikipedia.org/wiki/Hungarian_algorithm and
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    for more details of the hungarian matching implementation.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # DONE: Replace this stub code.
    row_ids, col_ids = linear_sum_assignment(cost_matrix)
    return row_ids, col_ids
