import numpy as np
import time

def solve(**kwargs):
    nodes = kwargs.get('nodes', [])
    n = len(nodes)

    if n == 0:
        return {'tour': []}
    if n == 1:
        return {'tour': [0]}
    if n == 2:
        return {'tour': [0, 1]}

    start_time = time.time()
    timeout = 9.0  # Leave 1 second margin

    # Compute distance matrix
    coords = np.array(nodes, dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    # For small instances, use exact Held-Karp DP
    if n <= 15:
        tour = held_karp(dist, n)
        return {'tour': tour}

    # For larger instances, use multi-start local search
    best_tour, best_cost = heuristic_solve(dist, n, start_time, timeout)
    return {'tour': best_tour}


def held_karp(dist, n):
    """Exact TSP solver using Held-Karp dynamic programming."""
    m = n - 1  # cities 1..n-1 (city 0 is fixed start)
    INF = float('inf')

    # dp[mask][j]: min cost starting at 0, visiting set 'mask' (over cities 1..n-1), ending at city j+1
    dp = np.full((1 << m, m), INF, dtype=np.float64)
    parent = np.full((1 << m, m), -1, dtype=np.int32)

    # Base cases
    for j in range(m):
        dp[1 << j, j] = dist[0, j + 1]

    # Fill DP table
    for mask in range(1, 1 << m):
        for j in range(m):
            if not (mask & (1 << j)):
                continue
            if mask == (1 << j):
                continue
            prev_mask = mask & ~(1 << j)
            best = INF
            best_i = -1
            for i in range(m):
                if i == j or not (prev_mask & (1 << i)):
                    continue
                val = dp[prev_mask, i] + dist[i + 1, j + 1]
                if val < best:
                    best = val
                    best_i = i
            dp[mask, j] = best
            parent[mask, j] = best_i

    # Find best ending city (must return to 0)
    full_mask = (1 << m) - 1
    best_cost = INF
    best_j = -1
    for j in range(m):
        cost = dp[full_mask, j] + dist[j + 1, 0]
        if cost < best_cost:
            best_cost = cost
            best_j = j

    # Reconstruct tour
    path = []
    mask = full_mask
    j = best_j
    while j != -1:
        path.append(j + 1)
        prev_j = parent[mask, j]
        mask = mask & ~(1 << j)
        j = prev_j
    path.reverse()
    return [0] + path


def heuristic_solve(dist, n, start_time, timeout):
    """Multi-start local search for TSP."""
    INF = float('inf')
    best_tour = None
    best_cost = INF
    rng = np.random.default_rng(42)

    def compute_cost(tour):
        """Compute total tour cost."""
        total = 0.0
        for i in range(n):
            total += dist[tour[i], tour[(i + 1) % n]]
        return total

    def nearest_neighbor(start):
        """Construct tour using nearest neighbor heuristic."""
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        for _ in range(n - 1):
            last = tour[-1]
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            d = dist[last, unvisited]
            idx = unvisited[np.argmin(d)]
            visited[idx] = True
            tour.append(idx)
        return tour

    def random_tour():
        tour = list(range(n))
        rng.shuffle(tour)
        return tour

    def two_opt(tour, time_limit=None):
        """Apply 2-opt improvements until convergence."""
        tour = list(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                if time_limit is not None and time.time() > time_limit:
                    return tour
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a = tour[i]
                    b = tour[i + 1]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d] - 1e-10:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
        return tour

    def or_opt(tour, time_limit=None):
        """Apply Or-opt (segment relocation) improvements."""
        tour = list(tour)
        improved = True
        while improved:
            improved = False
            for seg_len in (1, 2, 3):
                if seg_len >= n:
                    continue
                for i in range(n):
                    if time_limit is not None and time.time() > time_limit:
                        return tour
                    # Segment to move: tour[i], tour[i+1], ..., tour[i+seg_len-1]
                    # (wrapping around)
                    seg = [tour[(i + k) % n] for k in range(seg_len)]

                    # Positions to insert (after position j)
                    for j in range(n):
                        # Skip positions adjacent to the segment
                        if j == i - 1 or j == i + seg_len - 1:
                            continue
                        if j == i and seg_len == 1:
                            continue

                        # Compute cost change
                        # Remove seg from position i
                        # Insert seg after position j (in the modified tour)

                        # Edges affected by removal:
                        # (tour[i-1], tour[i]) and (tour[i+seg_len-1], tour[i+seg_len])
                        # Edges affected by insertion:
                        # (tour[j], tour[j+1]) replaced by (tour[j], seg[0]), (seg[-1], tour[j+1])

                        # Handle wrapping carefully
                        pi = (i - 1) % n  # predecessor of segment
                        si = (i + seg_len) % n  # successor of segment
                        pj = j  # position before insertion
                        sj = (j + 1) % n  # position after insertion (in original)

                        # Check if insertion position overlaps with segment
                        if (j >= i and j < i + seg_len) or (j + 1 >= i and j + 1 < i + seg_len):
                            continue
                        # Also check wrapping
                        if i + seg_len > n:
                            if j < i + seg_len - n or j >= i:
                                continue
                        else:
                            if j >= i and j < i + seg_len:
                                continue

                        # Cost before
                        cost_before = (dist[tour[pi], seg[0]] + dist[seg[-1], tour[si]] +
                                      dist[tour[pj], tour[sj]])
                        # Cost after
                        cost_after = (dist[tour[pi], tour[si]] + dist[tour[pj], seg[0]] +
                                     dist[seg[-1], tour[sj]])

                        if cost_after < cost_before - 1e-10:
                            # Perform the move
                            # Remove segment
                            new_tour = []
                            for k in range(n):
                                if k != i and k != i + 1 and (k != i + 2 or seg_len <= 2):
                                    # Only skip positions that are part of the segment
                                    pass
                            # Simpler: build new tour by removing and inserting
                            # Remove positions i, i+1, ..., i+seg_len-1 (mod n)
                            remaining = []
                            for k in range(n):
                                if k != i and (k != i + 1 or seg_len <= 1) and \
                                   (k != i + 2 or seg_len <= 2):
                                    # Check if k is in the segment (mod n)
                                    in_seg = False
                                    for s in range(seg_len):
                                        if (i + s) % n == k:
                                            in_seg = True
                                            break
                                    if not in_seg:
                                        remaining.append(tour[k])

                            # Find insertion point in remaining
                            # j refers to the position in the ORIGINAL tour
                            # In remaining, find the index of tour[j]
                            insert_idx = None
                            for idx, val in enumerate(remaining):
                                if val == tour[j]:
                                    insert_idx = idx + 1
                                    break

                            if insert_idx is None:
                                continue

                            new_tour = remaining[:insert_idx] + seg + remaining[insert_idx:]
                            tour = new_tour
                            improved = True
                            break
                if improved:
                    break
        return tour

    def double_bridge(tour):
        """Apply double bridge perturbation for diversity."""
        tour = list(tour)
        # Pick 4 distinct cut points
        cuts = sorted(rng.choice(n, size=4, replace=False))
        a, b, c, d = cuts
        # Tour: [0..a] | [a+1..b] | [b+1..c] | [c+1..d] | [d+1..n-1]
        # New:   [0..a] | [b+1..c] | [a+1..b] | [c+1..d] | [d+1..n-1]
        # (swap middle two segments, reversing them)
        seg1 = tour[a + 1:b + 1]
        seg2 = tour[b + 1:c + 1]
        tour = tour[:a + 1] + seg2[::-1] + seg1[::-1] + tour[c + 1:]
        return tour

    # Multi-start local search
    time_limit = start_time + timeout

    # Initial solutions: several nearest neighbor starts + random
    nn_starts = [0]
    if n > 1:
        nn_starts.append(1)
    if n > 2:
        nn_starts.append(n // 2)

    iterations = 0
    while time.time() < time_limit:
        if iterations < len(nn_starts):
            tour = nearest_neighbor(nn_starts[iterations])
        elif iterations < len(nn_starts) + 2:
            tour = random_tour()
        else:
            # Double bridge perturbation of best tour
            tour = double_bridge(best_tour)

        # Improve
        remaining = time_limit - time.time()
        if remaining > 0.1:
            tl = time_limit - 0.05
            tour = two_opt(tour, tl)
            if time.time() < time_limit - 0.05:
                tour = or_opt(tour, time_limit - 0.05)
                tour = two_opt(tour, time_limit - 0.05)

        cost = compute_cost(tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]

        iterations += 1

        # Stop if we've done many iterations and are getting close to timeout
        if iterations > 50 and time.time() > time_limit - 0.5:
            break

    return best_tour, best_cost