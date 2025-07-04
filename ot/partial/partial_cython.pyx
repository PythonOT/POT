# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
import heapq


@cython.boundscheck(False)
@cython.wraparound(False)
def insert_new_chain(np.ndarray[np.int64_t, ndim=1] chains_starting_at, np.ndarray[np.int64_t, ndim=1] chains_ending_at, int i, int j):
    """Insert the `candidate_chain=(i,j)` into the already known chains stored in 
    `chains_starting_at` and `chains_ending_at`.
    `chains_starting_at` and `chains_ending_at` are modified in-place and 
    the chain in which the candidate is inserted is returned.
    """
    cdef int n = chains_starting_at.shape[0]
    assert chains_starting_at.shape == chains_ending_at.shape
    if i - 1 >= 0 and i - 1 < n and chains_ending_at[i - 1] != -1:
        i = chains_ending_at[i - 1]
    if j + 1 >= 0 and j + 1 < n and chains_starting_at[j + 1] != -1:
        j = chains_starting_at[j + 1]
    if i >= 0 and i < n:
        if chains_starting_at[i] != -1:
            chains_ending_at[chains_starting_at[i]] = -1
        chains_starting_at[i] = j
    if j >= 0 and j < n:
        if chains_ending_at[j] != -1:
            chains_starting_at[chains_ending_at[j]] = -1
        chains_ending_at[j] = i
    return i, j

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cost_for_chain(int idx_start, int idx_end, np.ndarray[np.float64_t, ndim=1] chain_costs_cumsum):
    """Compute the associated cost for a chain (set of contiguous points
    included in the solution) ranging from `idx_start` to `idx_end` (both included).
    """
    return chain_costs_cumsum[idx_end] - chain_costs_cumsum[idx_start - 1]

@cython.boundscheck(False)
@cython.wraparound(False)
def precompute_chain_costs_cumsum(np.ndarray[np.int64_t, ndim=1] minimal_chain_ending_at_idx, np.ndarray[np.float64_t, ndim=1] minimal_chain_ending_at_cost, int n):
    """For each position `i` at which a chain could end,
    Compute (using dynamic programming and the costs of minimal chains that have been precomputed)
    the cost of the largest chain ending at `i`.

    This is useful because this can be used later, to compute the cost of any chain in O(1)
    (cf. `compute_cost_for_chain`).
    """
    cdef np.ndarray[np.float64_t, ndim=1] chain_costs_cumsum = np.zeros((n, ), dtype=np.float64)
    cdef int i, start
    cdef double additional_cost
    for i in range(n):
        if minimal_chain_ending_at_idx[i] != -1:
            start = minimal_chain_ending_at_idx[i]
            additional_cost = minimal_chain_ending_at_cost[i]
            if start == 0:
                chain_costs_cumsum[i] = additional_cost
            else:
                chain_costs_cumsum[i] = chain_costs_cumsum[start - 1] + additional_cost
    return chain_costs_cumsum

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cost_w1(np.ndarray[np.float64_t, ndim=1] diff_cum_sum, int idx_start, int idx_end):
    """Compute L1 cost of a chain using the L1 specific trick (O(1) time)."""
    if idx_start == 0:
        return diff_cum_sum[idx_end]
    else:
        return diff_cum_sum[idx_end] - diff_cum_sum[idx_start - 1]

@cython.boundscheck(False)
@cython.wraparound(False)
def get_cost_wp(np.ndarray[np.float64_t, ndim=1] sorted_z, np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator, int idx_start, int idx_end, int p):
    """Compute Lp^p cost of a chain using no specific trick (O(N) time)."""
    cdef np.ndarray[np.float64_t, ndim=1] subset_z = sorted_z[idx_start:idx_end+1]
    cdef np.ndarray[np.int64_t, ndim=1] subset_indicator = sorted_distrib_indicator[idx_start:idx_end+1]
    cdef np.ndarray[np.float64_t, ndim=1] subset_x = subset_z[subset_indicator == 0]
    cdef np.ndarray[np.float64_t, ndim=1] subset_y = subset_z[subset_indicator == 1]
    return np.sum(np.abs(subset_x - subset_y) ** p)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_costs(np.ndarray[np.float64_t, ndim=1] sorted_z,
                 np.ndarray[np.float64_t, ndim=1] diff_cum_sum,
                 np.ndarray[np.int64_t, ndim=1] diff_ranks,
                 np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator,
                 int p=1):
    """For each element in sorted `z`, compute its minimal chain (cf note below).
    Then compute the cost for each minimal chain and sort all minimal chains in increasing 
    cost order.

    Note: the "minimal chain" of a point x_i in x is the minimal set of adjacent 
    points (starting at x_i and extending to the right) that one should 
    take to get a balanced set (ie. a set in which we have as many 
    elements from x as elements from y)
    """
    cdef list l_costs = []
    cdef np.ndarray[np.int64_t, ndim=1] minimal_chain_ending_at_idx = np.full((sorted_z.shape[0], ), -1, dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim=1] minimal_chain_ending_at_cost = np.full((sorted_z.shape[0], ), -1.0, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] last_pos_for_rank_x = np.full((sorted_z.shape[0], ), -1, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] last_pos_for_rank_y = np.full((sorted_z.shape[0], ), -1, dtype=np.int64)
    cdef int n = diff_ranks.shape[0]
    cdef int idx_end, cur_rank, idx_start, target_rank
    cdef double cost
    for idx_end in range(n):
        # For each item in either distrib, find the scope of the smallest
        # "minimal chain" that would end at that point and extend on the left, 
        # if one exists, and store the cost of this "minimal chain" by relying 
        # on differences of cumulative sums
        cur_rank = diff_ranks[idx_end]
        idx_start = -1
        if sorted_distrib_indicator[idx_end] == 0:
            target_rank = cur_rank - 1
            if last_pos_for_rank_y[target_rank] != -1:
                idx_start = last_pos_for_rank_y[target_rank]
            last_pos_for_rank_x[cur_rank] = idx_end
        else:
            target_rank = cur_rank + 1
            if last_pos_for_rank_x[target_rank] != -1:
                idx_start = last_pos_for_rank_x[target_rank]
            last_pos_for_rank_y[cur_rank] = idx_end
        if idx_start != -1:
            if p == 1:
                cost = get_cost_w1(diff_cum_sum, idx_start, idx_end)
            else:
                cost = get_cost_wp(sorted_z, sorted_distrib_indicator, idx_start, idx_end, p)
            if idx_end == idx_start + 1:
                heapq.heappush(l_costs, (abs(cost), idx_start, idx_end))
            minimal_chain_ending_at_idx[idx_end] = idx_start
            minimal_chain_ending_at_cost[idx_end] = abs(cost)
    return l_costs, precompute_chain_costs_cumsum(minimal_chain_ending_at_idx, minimal_chain_ending_at_cost, n)

@cython.boundscheck(False)
@cython.wraparound(False)
def arg_insert_in_sorted(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    """
    Returns the index (similar to argsort) to be used to sort 
    the concatenation of `x` and `y` 
    (supposed to be 1d arrays).

    `x` and `y` are supposed to be sorted and their
    order cannot be changed in the resulting array (which is important
    for our ranking based algo in case of ex-aequos).
    """
    assert x.ndim == 1 and y.ndim == 1
    cdef Py_ssize_t n_x = x.shape[0]
    cdef Py_ssize_t n_y = y.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] arr_out = np.zeros(n_x + n_y, dtype=np.int64)
    cdef Py_ssize_t idx_x = 0
    cdef Py_ssize_t idx_y = 0
    while idx_x + idx_y < n_x + n_y:
        if idx_x == n_x:
            arr_out[idx_x + idx_y] = n_x + idx_y
            idx_y += 1
        elif idx_y == n_y:
            arr_out[idx_x + idx_y] = idx_x
            idx_x += 1
        else:
            if x[idx_x] <= y[idx_y]:
                arr_out[idx_x + idx_y] = idx_x
                idx_x += 1
            else:
                arr_out[idx_x + idx_y] = n_x + idx_y
                idx_y += 1
    return arr_out


@cython.boundscheck(False)
@cython.wraparound(False)
def preprocess(np.ndarray[np.float64_t, ndim=1] x, np.ndarray[np.float64_t, ndim=1] y):
    """Given two 1d distributions `x` and `y`:
    1. `indices_sort_x` sorts `x` (ie. `x[indices_sort_x]` is sorted) and
       `indices_sort_y` sorts `y` (ie. `y[indices_sort_y]` is sorted)
    2. stack them into a single distrib such that:
    * the new distrib is sorted with sort indices (wrt a stack of sorted x and sorted y) `indices_sort_xy`
    * `sorted_distrib_indicator` is a vector of zeros and ones where 0 means 
        "this point comes from x" and 1 means "this point comes from y"
    """
    cdef np.ndarray[np.int64_t, ndim=1] indices_sort_x = np.argsort(x)
    cdef np.ndarray[np.int64_t, ndim=1] indices_sort_y = np.argsort(y)
    cdef np.ndarray[np.int64_t, ndim=1] indices_sort_xy = arg_insert_in_sorted(x[indices_sort_x], y[indices_sort_y])
    cdef np.ndarray[np.int64_t, ndim=1] idx = np.concatenate((np.zeros(x.shape[0], dtype=np.int64), np.ones(y.shape[0], dtype=np.int64)))
    cdef np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator = idx[indices_sort_xy]
    return indices_sort_x, indices_sort_y, indices_sort_xy, sorted_distrib_indicator

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_solution_using_marginal_costs(
    list costs,
    np.ndarray[np.int64_t, ndim=1] ranks_xy,
    np.ndarray[np.float64_t, ndim=1] chain_costs_cumsum,
    int max_iter,
    np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator
):
    """Generate a solution from a sorted list of minimal chain costs.
    See the note in `compute_costs` docs for a definition of minimal chains.

    The solution is a pair of lists. The first list contains the indices from `sorted_x`
    that are in the active set, and the second one contains the indices from `sorted_y`
    that are in the active set. 
    The third returned element is a list of marginal costs induced by each step:
    `arr_marginal_costs[i]` is the marginal cost induced by the `i`-th step of the algorithm, such
    that `np.cumsum(arr_marginal_costs[:i])` gives all intermediate costs up to step `i`.
    """
    cdef np.ndarray[np.int64_t, ndim=1] active_set = np.zeros((sorted_distrib_indicator.shape[0], ), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] chains_starting_at = np.full((sorted_distrib_indicator.shape[0], ), -1, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] chains_ending_at = np.full((sorted_distrib_indicator.shape[0], ), -1, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=2] active_set_inserts = np.zeros((max_iter, 2), dtype=np.int64)
    cdef int n = sorted_distrib_indicator.shape[0]
    cdef int i, j, p_s, p_e
    cdef int n_pairs_in_active_set = 0
    cdef double c
    cdef double marginal_cost
    cdef np.ndarray[np.float64_t, ndim=1] arr_marginal_costs = np.zeros((max_iter, ), dtype=np.float64)

    while len(costs) > 0 and max_iter > n_pairs_in_active_set:
        c, i, j = heapq.heappop(costs)
        if active_set[i] == 1 or active_set[j] == 1:
            continue
        # Case 1: j == i + 1 => "Simple" insert
        if j == i + 1:
            p_s, p_e = insert_new_chain(chains_starting_at, chains_ending_at, i, j)
        # Case 2: insert a chain that contains a chain
        elif chains_ending_at[j - 1] != -1:
            if i + 1 >= 0 and i + 1 < n:
                chains_starting_at[i + 1] = -1
            if j - 1 >= 0 and j - 1 < n:
                chains_ending_at[j - 1] = -1
            p_s, p_e = insert_new_chain(chains_starting_at, chains_ending_at, i, j)
        # There should be no "Case 3"
        else:
            raise ValueError
        active_set_inserts[n_pairs_in_active_set, 0] = i
        active_set_inserts[n_pairs_in_active_set, 1] = j
        arr_marginal_costs[n_pairs_in_active_set] = c
        active_set[i] = 1
        active_set[j] = 1
        n_pairs_in_active_set += 1
        
        # We now need to update the candidate chains wrt the chain we have just created
        if p_s == 0 or p_e == n - 1:
            continue
        if sorted_distrib_indicator[p_s - 1] != sorted_distrib_indicator[p_e + 1]:
            # Insert (p_s - 1, p_e + 1) as a new candidate chain with marginal cost
            marginal_cost = (compute_cost_for_chain(p_s - 1, p_e + 1, chain_costs_cumsum)
                             - compute_cost_for_chain(p_s, p_e, chain_costs_cumsum))
            heapq.heappush(costs, (marginal_cost, p_s - 1, p_e + 1))
    # Generate index arrays in the order of insertion in the active set
    cdef np.ndarray[np.int64_t, ndim=1] indices_sorted_x = np.zeros((max_iter, ), dtype=np.int64) - 1
    cdef np.ndarray[np.int64_t, ndim=1] indices_sorted_y = np.zeros((max_iter, ), dtype=np.int64) - 1
    cdef int a, b
    for i in range(max_iter):
        a = active_set_inserts[i, 0]
        b = active_set_inserts[i, 1]
        if sorted_distrib_indicator[a] == 0:
            indices_sorted_x[i] = ranks_xy[a]
        else:
            indices_sorted_y[i] = ranks_xy[a]
        if sorted_distrib_indicator[b] == 0:
            indices_sorted_x[i] = ranks_xy[b]
        else:
            indices_sorted_y[i] = ranks_xy[b]
    return (
        indices_sorted_x,
        indices_sorted_y,
        arr_marginal_costs
    )

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cumulative_sum_differences(
    np.ndarray[np.float64_t, ndim=1] x_sorted, 
    np.ndarray[np.float64_t, ndim=1] y_sorted, 
    np.ndarray[np.int64_t, ndim=1] indices_sort_xy, 
    np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator
):
    """Computes difference between cumulative sums for both distribs.

    The cumulative sum vector for a sorted x is:

        cumsum_x = [x_0, x_0 + x_1, ..., x_0 + ... + x_n]

    This vector is then extend to reach a length of 2*n by repeating 
    values at places that correspond to an y item.
    In other words, if the order of x and y elements on the real 
    line is something like x-y-y-x..., then the extended vector is 
    (note the repetitions):

        cumsum_x = [x_0, x_0, x_0, x_0 + x_1, ..., x_0 + ... + x_n]

    Overall, this function returns `cumsum_x - cumsum_y` where `cumsum_x`
    and `cumsum_y` are the extended versions.
    """
    cdef np.ndarray[np.float64_t, ndim=1] cum_sum_xs = np.cumsum(x_sorted)
    cdef np.ndarray[np.float64_t, ndim=1] cum_sum_ys = np.cumsum(y_sorted)
    cdef np.ndarray[np.float64_t, ndim=1] cum_sum = np.concatenate((cum_sum_xs, cum_sum_ys))
    cdef np.ndarray[np.float64_t, ndim=1] cum_sum_ordered = cum_sum[indices_sort_xy]

    cdef np.ndarray[np.float64_t, ndim=1] cum_sum_x = insert_constant_values_float(cum_sum_ordered, 0, sorted_distrib_indicator)
    cdef np.ndarray[np.float64_t, ndim=1] cum_sum_y = insert_constant_values_float(cum_sum_ordered, 1, sorted_distrib_indicator)

    return cum_sum_x - cum_sum_y

@cython.boundscheck(False)
@cython.wraparound(False)
def insert_constant_values_float(
    np.ndarray[np.float64_t, ndim=1] arr, 
    np.int64_t distrib_index,
    np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator
):
    """Takes `arr` as input. For each position `i` in `arr`, 
    if `sorted_distrib_indicator[i]==distrib_index`,
    the value from `arr` is copied, 
    otherwise, the previous value that was copied from `arr` is repeated.
    """
    cdef np.ndarray[np.float64_t, ndim=1] arr_insert = np.copy(arr)
    for i in range(len(arr)):
        if sorted_distrib_indicator[i]!=distrib_index:
            if i == 0:
                arr_insert[i] = 0
            else:
                arr_insert[i] = arr_insert[i-1]
    return arr_insert

@cython.boundscheck(False)
@cython.wraparound(False)
def insert_constant_values_int(
    np.ndarray[np.int64_t, ndim=1] arr, 
    np.int64_t distrib_index,
    np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator
):
    """Takes `arr` as input. For each position `i` in `arr`, 
    if `sorted_distrib_indicator[i]==distrib_index`,
    the value from `arr` is copied, 
    otherwise, the previous value that was copied from `arr` is repeated.
    """
    cdef np.ndarray[np.int64_t, ndim=1] arr_insert = np.copy(arr)
    for i in range(len(arr)):
        if sorted_distrib_indicator[i]!=distrib_index:
            if i == 0:
                arr_insert[i] = 0
            else:
                arr_insert[i] = arr_insert[i-1]
    return arr_insert

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_rank_differences(
    np.ndarray[np.int64_t, ndim=1] indices_sort_xy, 
    np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator
):
    """Precompute important rank-related quantities for better minimal chain extraction.
    
    Two quantities are returned:

    * `ranks_xy` is an array that gathers ranks of the elements in 
        their original distrib, eg. if the distrib indicator is
        [0, 1, 1, 0, 0, 1], then `rank_xy` will be:
        [0, 0, 1, 1, 2, 2]
    * `diff_ranks` is computed from `ranks_xy_x_cum` and `ranks_xy_y_cum`.
        For the example above, we would have:
        
        ranks_xy_x_cum = [1, 1, 1, 2, 3, 3]
        ranks_xy_y_cum = [0, 1, 2, 2, 2, 3]

        And `diff_ranks` is just `ranks_xy_x_cum - ranks_xy_y_cum`.
    """
    cdef int n_x = np.sum(sorted_distrib_indicator == 0)
    cdef int n_y = np.sum(sorted_distrib_indicator == 1)
    cdef np.ndarray[np.int64_t, ndim=1] ranks_x = np.arange(n_x)
    cdef np.ndarray[np.int64_t, ndim=1] ranks_y = np.arange(n_y)
    cdef np.ndarray[np.int64_t, ndim=1]ranks_xy = np.concatenate((ranks_x, ranks_y))
    cdef np.ndarray[np.int64_t, ndim=1] ranks_xy_ordered = ranks_xy[indices_sort_xy]

    cdef np.ndarray[np.int64_t, ndim=1] ranks_xy_x = ranks_xy_ordered.copy()
    ranks_xy_x[sorted_distrib_indicator==1] = 0
    ranks_xy_x[sorted_distrib_indicator==0] += 1
    cdef np.ndarray[np.int64_t, ndim=1] ranks_xy_x_cum = insert_constant_values_int(ranks_xy_x, 0, sorted_distrib_indicator)

    cdef np.ndarray[np.int64_t, ndim=1] ranks_xy_y = ranks_xy_ordered.copy()
    ranks_xy_y[sorted_distrib_indicator==0] = 0
    ranks_xy_y[sorted_distrib_indicator==1] += 1
    cdef np.ndarray[np.int64_t, ndim=1] ranks_xy_y_cum = insert_constant_values_int(ranks_xy_y, 1, sorted_distrib_indicator)

    cdef np.ndarray[np.int64_t, ndim=1] diff_ranks = ranks_xy_x_cum - ranks_xy_y_cum

    return ranks_xy_ordered, diff_ranks

@cython.boundscheck(False)
@cython.wraparound(False)
def partial_wasserstein_1d_cy(
    np.ndarray[np.float64_t, ndim=1] x, 
    np.ndarray[np.float64_t, ndim=1] y, 
    int max_iter, 
    int p
):
    """Main routine for the partial Wasserstein problem in 1D.
    
    Does:
    
    1. Preprocessing of the distribs (sorted & co)
    2. Precomputations (ranks, cumulative sums)
    3. Extraction of minimal chains
    4. Generate and return solution

    Note that the indices in `indices_x` and `indices_y` are ordered wrt their order of
    appearance in the solution such that `indices_x[:10]` (resp y) is the set of indices
    from x (resp. y) for the partial problem of size 10.

    Arguments
    ---------
    x : np.ndarray of shape (n, )
        First distrib to be considered (weights are considered uniform)
    y : np.ndarray of shape (m, )
        Second distrib to be considered (weights are considered uniform)
    max_iter : int
        Number of iterations of the algorithm, which is equal to the number of pairs
        in the returned solution.
    p : int (default: 1)
        Order of the partial Wasserstein distance to be computed (p-Wasserstein, or $W_p^p$)

    Returns
    -------
    indices_x : np.ndarray of shape (min(n, m, max_iter), )
        Indices of elements from the x distribution to be included in the partial solutions
        Order of appearance in this array indicates order of inclusion in the solution
    indices_y : np.ndarray of shape (min(n, m, max_iter), )
        Indices of elements from the x distribution to be included in the partial solutions
        Order of appearance in this array indicates order of inclusion in the solution
    list_marginal_costs : list of length min(n, m, max_iter)
        List of marginal costs associated to the intermediate partial problems
        `np.cumsum(list_marginal_costs)` gives the corresponding total costs for intermediate partial problems
    """
    # Sort distribs and keep track of their original indices
    cdef np.ndarray[np.int64_t, ndim=1] indices_sort_x
    cdef np.ndarray[np.int64_t, ndim=1] indices_sort_y
    cdef np.ndarray[np.int64_t, ndim=1] indices_sort_xy
    cdef np.ndarray[np.int64_t, ndim=1] sorted_distrib_indicator
    cdef np.ndarray[np.float64_t, ndim=1] sorted_z
    cdef np.ndarray[np.float64_t, ndim=1] diff_cum_sum
    cdef np.ndarray[np.int64_t, ndim=1] ranks_xy
    cdef np.ndarray[np.int64_t, ndim=1] diff_ranks
    cdef list costs
    cdef np.ndarray[np.float64_t, ndim=1] chain_costs_cumsum
    cdef np.ndarray[np.int64_t, ndim=1] indices_x
    cdef np.ndarray[np.int64_t, ndim=1] indices_y
    cdef np.ndarray[np.int64_t, ndim=1] sol_indices_x_sorted
    cdef np.ndarray[np.int64_t, ndim=1] sol_indices_y_sorted
    cdef np.ndarray[np.float64_t, ndim=1] marginal_costs

    indices_sort_x, indices_sort_y, indices_sort_xy, sorted_distrib_indicator = preprocess(x, y)

    sorted_z = np.concatenate((x[indices_sort_x], y[indices_sort_y]))[indices_sort_xy]

    # Precompute useful quantities
    diff_cum_sum = compute_cumulative_sum_differences(x[indices_sort_x], 
                                                      y[indices_sort_y], 
                                                      indices_sort_xy,
                                                      sorted_distrib_indicator)
    ranks_xy, diff_ranks = compute_rank_differences(indices_sort_xy, sorted_distrib_indicator)

    # Compute costs for "minimal chains"
    costs, chain_costs_cumsum = compute_costs(sorted_z, diff_cum_sum, diff_ranks, sorted_distrib_indicator, p=p)

    # Generate solution from sorted costs
    sol_indices_x_sorted, sol_indices_y_sorted, marginal_costs = generate_solution_using_marginal_costs(costs, 
                                                                                                   ranks_xy, 
                                                                                                   chain_costs_cumsum, 
                                                                                                   max_iter, 
                                                                                                   sorted_distrib_indicator)

    # Convert back into indices in original `x` and `y` distribs
    indices_x = indices_sort_x[sol_indices_x_sorted]
    indices_y = indices_sort_y[sol_indices_y_sorted]
    return indices_x, indices_y, marginal_costs