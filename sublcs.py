"""
sublcs.py — Reference implementation of Algorithm 1 (SubLCS)
=============================================================
Subsystem Logical Clifford Synthesis via Gauge Orbits.

Implements the two-phase enumeration of Theorem 4:
    Sol(U_L) = M_0 · (H_stab × Sp(2g, F_2))

All arithmetic is over F_2 (binary field).

Dependencies: numpy, itertools (stdlib)

Usage
-----
See __main__ block at the bottom for the [[4,1,1,2]] Bacon-Shor example.

References
----------
[1] Rengaswamy et al., "Logical Clifford Synthesis for Stabilizer Codes",
    IEEE TQE 2020.
[2] This paper: Theorem 4, Algorithm 1.
"""

import numpy as np
from itertools import product as iproduct
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# F_2 linear algebra utilities
# ---------------------------------------------------------------------------

def f2_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiply over F_2."""
    return A @ B % 2


def f2_rref(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Reduced row echelon form over F_2.
    Returns (rref_matrix, pivot_cols).
    """
    M = M.copy() % 2
    nrows, ncols = M.shape
    pivot_cols = []
    row = 0
    for col in range(ncols):
        # find pivot
        found = -1
        for r in range(row, nrows):
            if M[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        M[[row, found]] = M[[found, row]]
        pivot_cols.append(col)
        for r in range(nrows):
            if r != row and M[r, col] == 1:
                M[r] = (M[r] + M[row]) % 2
        row += 1
    return M, pivot_cols


def f2_rank(M: np.ndarray) -> int:
    _, pivots = f2_rref(M)
    return len(pivots)


def f2_null(M: np.ndarray) -> np.ndarray:
    """
    Null space of M over F_2.
    Returns matrix whose rows are a basis for ker(M).
    """
    nrows, ncols = M.shape
    # augment with identity
    aug = np.hstack([M.T, np.eye(ncols, dtype=int)])
    rref, pivots = f2_rref(aug)
    null_rows = []
    for i in range(ncols):
        if i not in pivots:
            null_rows.append(rref[i, nrows:] % 2)
    if not null_rows:
        return np.zeros((0, ncols), dtype=int)
    return np.array(null_rows, dtype=int) % 2


def f2_solve(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """
    Solve Ax = b over F_2. Returns one solution or None.
    b is a 1-D array.
    """
    nrows, ncols = A.shape
    aug = np.hstack([A, b.reshape(-1, 1)]) % 2
    rref, pivots = f2_rref(aug)
    # check consistency
    for i in range(len(pivots), nrows):
        if rref[i, ncols] == 1:
            return None
    x = np.zeros(ncols, dtype=int)
    for idx, col in enumerate(pivots):
        x[col] = rref[idx, ncols]
    return x % 2


def f2_inv(M: np.ndarray) -> np.ndarray:
    """Inverse of an invertible square matrix over F_2."""
    n = M.shape[0]
    aug = np.hstack([M, np.eye(n, dtype=int)]) % 2
    rref, _ = f2_rref(aug)
    return rref[:, n:] % 2


def f2_col_space(M: np.ndarray) -> np.ndarray:
    """Returns matrix whose rows span the column space of M (= row space of M^T)."""
    _, pivots = f2_rref(M.T)
    return M.T[pivots] % 2


# ---------------------------------------------------------------------------
# Symplectic utilities
# ---------------------------------------------------------------------------

def omega_matrix(m: int) -> np.ndarray:
    """
    Symplectic form Omega = [[0, I], [I, 0]] for m qubits (2m x 2m).
    Convention: <u,v>_s = u Omega v^T mod 2.
    """
    I = np.eye(m, dtype=int)
    return np.block([[np.zeros((m, m), dtype=int), I],
                     [I, np.zeros((m, m), dtype=int)]])


def symp_inner(u: np.ndarray, v: np.ndarray, m: int) -> int:
    """Symplectic inner product <u,v>_s mod 2."""
    Om = omega_matrix(m)
    return int(u @ Om @ v % 2)


def is_symplectic(M: np.ndarray, m: int) -> bool:
    """Check M Omega M^T = Omega over F_2."""
    Om = omega_matrix(m)
    return np.array_equal(f2_matmul(f2_matmul(M, Om), M.T) % 2, Om)


def symp_gram_schmidt(vecs: List[np.ndarray], m: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Given a list of vectors spanning a non-degenerate symplectic subspace,
    return a list of symplectic pairs (e_i, f_i) with <e_i, f_j>_s = delta_ij.

    Uses the standard symplectic Gram-Schmidt procedure (Wilde 2009).
    Handles redundant (linearly dependent) vectors gracefully.
    """
    # First reduce to a linearly independent set
    independent = []
    for v in vecs:
        v = v.copy() % 2
        if not np.any(v):
            continue
        if independent:
            M = np.array(independent, dtype=int)
            test = np.vstack([M, v.reshape(1, -1)])
            if f2_rank(test) == f2_rank(M):
                continue  # linearly dependent, skip
        independent.append(v)

    pairs = []
    remaining = list(independent)

    while remaining:
        e = remaining.pop(0)
        if not np.any(e):
            continue
        # find f in remaining with <e, f> = 1
        f_idx = None
        for i, v in enumerate(remaining):
            if symp_inner(e, v, m) == 1:
                f_idx = i
                break
        if f_idx is None:
            # e has trivial symplectic inner product with all remaining —
            # it belongs to the radical; skip (shouldn't happen for non-degenerate subspace)
            continue
        f = remaining.pop(f_idx)
        # project out e, f from remaining
        new_remaining = []
        for v in remaining:
            v = (v + symp_inner(v, f, m) * e + symp_inner(v, e, m) * f) % 2
            new_remaining.append(v)
        remaining = [v for v in new_remaining if np.any(v)]
        pairs.append((e, f))
    return pairs


# ---------------------------------------------------------------------------
# Sp(2g, F_2) enumeration
# ---------------------------------------------------------------------------

def enumerate_sp2g(g: int) -> List[np.ndarray]:
    """
    Enumerate all elements of Sp(2g, F_2) by brute force (for small g).

    For g=1: |Sp(2,F_2)| = 6
    For g=2: |Sp(4,F_2)| = 720
    For g=3: |Sp(6,F_2)| = 1,451,520

    For larger g, use generate_sp2g_transvections instead.
    """
    n = 2 * g
    Om = omega_matrix(g)
    result = []
    # iterate over all 2^(n^2) binary matrices
    for bits in iproduct([0, 1], repeat=n * n):
        M = np.array(bits, dtype=int).reshape(n, n)
        if np.array_equal(f2_matmul(f2_matmul(M, Om), M.T) % 2, Om):
            result.append(M)
    return result


def sp2g_transvection(h: np.ndarray, g: int) -> np.ndarray:
    """Symplectic transvection T_h in Sp(2g, F_2): v -> v + <v,h> h."""
    n = 2 * g
    T = np.eye(n, dtype=int)
    Om = omega_matrix(g)
    for i in range(n):
        ei = np.zeros(n, dtype=int)
        ei[i] = 1
        T[i] = (ei + int(ei @ Om @ h % 2) * h) % 2
    return T


def generate_sp2g_transvections(g: int) -> List[np.ndarray]:
    """
    Generate all transvections in Sp(2g, F_2).
    There are 2^(2g) - 1 transvections (one per nonzero vector).
    """
    n = 2 * g
    transvections = []
    for bits in iproduct([0, 1], repeat=n):
        h = np.array(bits, dtype=int)
        if np.any(h):
            transvections.append(sp2g_transvection(h, g))
    return transvections


def enumerate_sp2g_via_transvections(g: int, max_elements: Optional[int] = None) -> List[np.ndarray]:
    """
    Enumerate Sp(2g, F_2) by multiplying transvections (BFS/DFS).
    More memory-efficient than brute force for larger g.
    Stops early if max_elements is set (for heuristic search).
    """
    n = 2 * g
    identity = np.eye(n, dtype=int)
    transvections = generate_sp2g_transvections(g)

    seen = set()
    result = []
    queue = [identity]

    def mat_key(M):
        return tuple(M.flatten().tolist())

    seen.add(mat_key(identity))
    result.append(identity)

    while queue:
        if max_elements and len(result) >= max_elements:
            break
        M = queue.pop(0)
        for T in transvections:
            N = f2_matmul(T, M)
            k = mat_key(N)
            if k not in seen:
                seen.add(k)
                result.append(N)
                queue.append(N)
                if max_elements and len(result) >= max_elements:
                    break

    return result


# ---------------------------------------------------------------------------
# Stabilizer sector: H_stab enumeration (Phase 1)
# ---------------------------------------------------------------------------

def enumerate_hstab(WS_basis: np.ndarray, m: int) -> List[np.ndarray]:
    """
    Enumerate H_stab = group generated by transvections {T_s : s in WS}.

    WS_basis: (r x 2m) matrix whose rows are a basis for W_S.
    Returns list of 2^(r(r+1)/2) symplectic matrices.

    H_stab is the group of all products of subsets of
    {T_{s_i} : s_i basis of W_S} together with
    T_{s_i + s_j} style combinations — equivalently, all
    products of transvections along WS vectors.

    We enumerate by BFS over the generating set.
    """
    r = WS_basis.shape[0]
    n = 2 * m
    identity = np.eye(n, dtype=int)

    # generators: transvections along basis vectors and their sums
    generators = []
    rows = list(WS_basis)
    for v in rows:
        generators.append(_transvection_full(v, m))
    # also add transvections along pairwise sums (needed for full group)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            vij = (rows[i] + rows[j]) % 2
            generators.append(_transvection_full(vij, m))

    seen = set()
    result = []
    queue = [identity]

    def mat_key(M):
        return tuple(M.flatten().tolist())

    seen.add(mat_key(identity))
    result.append(identity)

    while queue:
        M = queue.pop(0)
        for T in generators:
            N = f2_matmul(T, M)
            k = mat_key(N)
            if k not in seen:
                seen.add(k)
                result.append(N)
                queue.append(N)

    expected = 2 ** (r * (r + 1) // 2)
    assert len(result) == expected, (
        f"H_stab has {len(result)} elements but expected {expected} = 2^({r}*{r+1}/2)."
    )
    return result


def _transvection_full(h: np.ndarray, m: int) -> np.ndarray:
    """Symplectic transvection T_h in Sp(2m, F_2)."""
    n = 2 * m
    Om = omega_matrix(m)
    T = np.eye(n, dtype=int)
    for i in range(n):
        ei = np.zeros(n, dtype=int)
        ei[i] = 1
        T[i] = (ei + int(ei @ Om @ h % 2) * h) % 2
    return T


# ---------------------------------------------------------------------------
# Base solution M_0 (Phase 1)
# ---------------------------------------------------------------------------

def find_base_solution(
    WS_basis: np.ndarray,
    WG_basis: np.ndarray,
    logical_basis: np.ndarray,
    UL_matrix: np.ndarray,
    m: int
) -> np.ndarray:
    """
    Construct a base solution M_0 in Sp(2m, F_2) satisfying (G), (S), (L).

    Strategy: build M_0 column by column in a symplectic basis
    (V_gauge | W_S | V_log | V_rem) adapted to the decomposition.

    WS_basis  : (r  x 2m) stabilizer generators
    WG_basis  : (r+2g x 2m) gauge group generators
    logical_basis : (2k x 2m) rows = [X_L1, Z_L1, ..., X_Lk, Z_Lk]
    UL_matrix : (2k x 2k) symplectic matrix of U_L action on logical sector
    m         : number of physical qubits

    Returns M_0 as a (2m x 2m) symplectic matrix.
    """
    n = 2 * m
    Om = omega_matrix(m)

    # --- Extract V_gauge basis from W_G ---
    Vgauge_pairs = _extract_vgauge(WS_basis, WG_basis, m)
    g = len(Vgauge_pairs)
    k = logical_basis.shape[0] // 2
    r = WS_basis.shape[0]

    # Build full ordered basis B = [v_gauge | w_stab | v_log | v_rem]
    basis_vecs = []

    # gauge vectors (e_1, f_1, ..., e_g, f_g)
    for (e, f) in Vgauge_pairs:
        basis_vecs.extend([e, f])

    # stabilizer vectors
    for s in WS_basis:
        basis_vecs.append(s)

    # logical vectors
    for v in logical_basis:
        basis_vecs.append(v)

    # remainder: complete to a full symplectic basis
    rem = _complete_symplectic_basis(basis_vecs, m)
    basis_vecs.extend(rem)

    assert len(basis_vecs) == n, f"Expected {n} basis vectors, got {len(basis_vecs)}"

    # --- Build M_0 ---
    # M_0 acts as:
    #   identity on V_gauge (fix gauge vectors)
    #   identity on W_S     (fix stabilizers)
    #   U_L on V_log        (implement logical)
    #   identity on V_rem
    #
    # Since basis_vecs are the *rows* we want M_0 to map to themselves
    # (for gauge/stab/rem) or to UL-transformed versions (for logical),
    # we build M_0 as the matrix that maps standard basis -> image.

    # Collect images
    images = []

    # V_gauge: fixed
    for (e, f) in Vgauge_pairs:
        images.extend([e, f])

    # W_S: fixed
    for s in WS_basis:
        images.append(s)

    # V_log: apply UL_matrix (2k x 2k) to pairs
    log_vecs = list(logical_basis)
    # UL_matrix acts on the 2k-dim logical sector;
    # column j of UL_matrix gives the image of the j-th logical basis vector
    for j in range(2 * k):
        img = np.zeros(n, dtype=int)
        for i in range(2 * k):
            img = (img + int(UL_matrix[i, j]) * log_vecs[i]) % 2
        images.append(img)

    # V_rem: fixed
    for v in rem:
        images.append(v)

    # M_0 maps basis_vecs[i] -> images[i]
    # Expressed as matrix: columns of M_0 in standard basis
    B = np.array(basis_vecs, dtype=int).T   # 2m x 2m, columns = basis vecs
    Im = np.array(images, dtype=int).T      # 2m x 2m, columns = images
    B_inv = f2_inv(B)
    M0 = f2_matmul(Im, B_inv) % 2

    assert is_symplectic(M0, m), "M_0 is not symplectic — check basis construction."
    return M0


def _extract_vgauge(WS_basis: np.ndarray, WG_basis: np.ndarray, m: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract symplectic pairs for V_gauge from W_G.
    Applies symplectic Gram-Schmidt to W_G / W_S.
    """
    r = WS_basis.shape[0]
    # find vectors in WG_basis not in WS span
    # quotient: WG / WS — find complement vectors
    gauge_complement = []
    for v in WG_basis:
        # check if v is in span of WS_basis
        combined = np.vstack([WS_basis, v.reshape(1, -1)]) % 2
        if f2_rank(combined) > r:
            gauge_complement.append(v)

    # now apply symplectic Gram-Schmidt to gauge_complement
    pairs = symp_gram_schmidt(gauge_complement, m)
    return pairs


def _complete_symplectic_basis(partial: List[np.ndarray], m: int) -> List[np.ndarray]:
    """
    Complete a partial set of vectors to a full symplectic basis of F_2^{2m}.
    Returns the additional vectors needed.
    """
    n = 2 * m
    Om = omega_matrix(m)
    current = [v.copy() % 2 for v in partial]
    added = []

    while len(current) + len(added) < n:
        # find a vector not in span of current + added
        all_vecs = current + added
        span_matrix = np.array(all_vecs, dtype=int)
        rank_so_far = f2_rank(span_matrix)

        new_v = None
        for bits in iproduct([0, 1], repeat=n):
            candidate = np.array(bits, dtype=int)
            if not np.any(candidate):
                continue
            test = np.vstack([span_matrix, candidate.reshape(1, -1)])
            if f2_rank(test) > rank_so_far:
                new_v = candidate
                break

        if new_v is None:
            break

        # find its symplectic partner: w s.t. <new_v, w> = 1, w not in span
        partner = None
        for bits in iproduct([0, 1], repeat=n):
            w = np.array(bits, dtype=int)
            if not np.any(w):
                continue
            if symp_inner(new_v, w, m) != 1:
                continue
            test = np.vstack([span_matrix, new_v.reshape(1, -1), w.reshape(1, -1)])
            if f2_rank(test) == rank_so_far + 2:
                partner = w
                break

        if partner is None:
            added.append(new_v)
        else:
            added.extend([new_v, partner])

    return added


# ---------------------------------------------------------------------------
# Gauge sector lift sigma(N): Phase 2
# ---------------------------------------------------------------------------

def lift_sigma(N: np.ndarray, Vgauge_pairs: List[Tuple[np.ndarray, np.ndarray]], m: int) -> np.ndarray:
    """
    Construct sigma(N) in Sp(2m, F_2) from N in Sp(2g, F_2).

    sigma(N) acts as N on V_gauge and as identity on V_gauge^perp.

    N            : (2g x 2g) symplectic matrix
    Vgauge_pairs : list of (e_i, f_i) pairs forming V_gauge basis
    m            : number of physical qubits
    """
    n = 2 * m
    g = len(Vgauge_pairs)

    # V_gauge basis vectors in order: e_1, f_1, ..., e_g, f_g
    vgauge_vecs = []
    for (e, f) in Vgauge_pairs:
        vgauge_vecs.extend([e, f])

    # Build iota: embedding F_2^{2g} -> F_2^{2m}
    # Standard basis of F_2^{2g} maps to vgauge_vecs
    iota = np.array(vgauge_vecs, dtype=int).T  # 2m x 2g

    # sigma(N) = I + iota (N - I) iota^+ where iota^+ is left-inverse
    # Simpler: build as a 2m x 2m matrix by explicit action on a basis

    # Find a full basis of F_2^{2m} = V_gauge ⊕ V_gauge^perp
    # V_gauge^perp: vectors orthogonal to all of V_gauge
    # Build sigma(N) column by column

    sigma = np.eye(n, dtype=int)

    # For each standard basis vector, compute sigma(N) * e_i
    # sigma(N) fixes V_gauge^perp and acts as N on V_gauge

    # Express each standard basis vector in terms of V_gauge and V_gauge^perp components
    # sigma(N)(v) = sigma(N)(v_gauge + v_perp) = N*v_gauge + v_perp

    # Build projection onto V_gauge along V_gauge^perp
    # Since V_gauge is non-degenerate, use symplectic projection:
    # v_gauge component = sum_i [ <v, f_i> e_i + <v, e_i> f_i ]  (symplectic dual)

    Om = omega_matrix(m)

    for j in range(n):
        ej = np.zeros(n, dtype=int)
        ej[j] = 1

        # compute gauge component coefficients
        coords_gauge = np.zeros(2 * g, dtype=int)
        for i, (e, f) in enumerate(Vgauge_pairs):
            # <ej, f_i> gives e_i coefficient, <ej, e_i> gives f_i coefficient
            coords_gauge[2 * i]     = int(ej @ Om @ f % 2)
            coords_gauge[2 * i + 1] = int(ej @ Om @ e % 2)

        # apply N to gauge coordinates
        new_coords = f2_matmul(N, coords_gauge.reshape(-1, 1)).flatten() % 2

        # reconstruct gauge part
        v_gauge_new = np.zeros(n, dtype=int)
        for i, (e, f) in enumerate(Vgauge_pairs):
            v_gauge_new = (v_gauge_new + int(new_coords[2 * i]) * e
                           + int(new_coords[2 * i + 1]) * f) % 2

        # original gauge part
        v_gauge_old = np.zeros(n, dtype=int)
        for i, (e, f) in enumerate(Vgauge_pairs):
            v_gauge_old = (v_gauge_old + int(coords_gauge[2 * i]) * e
                           + int(coords_gauge[2 * i + 1]) * f) % 2

        # sigma(N)(ej) = ej - v_gauge_old + v_gauge_new
        sigma[:, j] = (ej - v_gauge_old + v_gauge_new) % 2

    assert is_symplectic(sigma, m), "sigma(N) is not symplectic."
    return sigma


# ---------------------------------------------------------------------------
# Main Algorithm 1: SubLCS
# ---------------------------------------------------------------------------

def sublcs(
    WS_basis: np.ndarray,
    WG_basis: np.ndarray,
    logical_basis: np.ndarray,
    UL_matrix: np.ndarray,
    m: int,
    enumerate_gauge: str = "brute",
    max_gauge_elements: Optional[int] = None
) -> List[np.ndarray]:
    """
    Algorithm 1: Subsystem Logical Clifford Synthesis (SubLCS).

    Parameters
    ----------
    WS_basis       : (r x 2m) basis for stabilizer space W_S (isotropic)
    WG_basis       : ((r+2g) x 2m) basis for gauge space W_G
    logical_basis  : (2k x 2m) symplectic basis for logical sector V_log
                     ordered as [X_L1, Z_L1, ..., X_Lk, Z_Lk]
    UL_matrix      : (2k x 2k) symplectic matrix of target logical Clifford U_L
                     in the basis given by logical_basis
    m              : number of physical qubits
    enumerate_gauge: "brute"        — enumerate all of Sp(2g,F_2) by brute force
                     "transvection" — enumerate via transvection BFS
    max_gauge_elements : if set, stop gauge enumeration early (heuristic mode)

    Returns
    -------
    List of symplectic matrices in Sol(U_L), each of shape (2m x 2m).

    Complexity
    ----------
    Phase 1: O(m^3) preprocessing + O(m^2) per stabilizer solution
    Phase 2: O(g^3) gauge complement + O(m^2) per gauge lift
    Total  : O(m^3) preprocessing + O(m^2) per solution output
    """
    r = WS_basis.shape[0]
    k = logical_basis.shape[0] // 2
    g = (WG_basis.shape[0] - r) // 2

    print(f"[SubLCS] m={m}, k={k}, g={g}, r={r}")
    print(f"[SubLCS] Expected |Sol(U_L)| = 2^{r*(r+1)//2} * |Sp({2*g},F_2)|")

    # -----------------------------------------------------------------------
    # Phase 1: Stabilizer solutions
    # -----------------------------------------------------------------------
    print("[SubLCS] Phase 1: computing H_stab ...")
    H_stab = enumerate_hstab(WS_basis, m)
    print(f"[SubLCS] |H_stab| = {len(H_stab)}")

    print("[SubLCS] Phase 1: finding base solution M_0 ...")
    M0 = find_base_solution(WS_basis, WG_basis, logical_basis, UL_matrix, m)
    print("[SubLCS] M_0 found and verified symplectic.")

    # Stabilizer coset: M0 * H_stab
    stab_solutions = [f2_matmul(M0, h) % 2 for h in H_stab]

    # -----------------------------------------------------------------------
    # Phase 2: Gauge orbit
    # -----------------------------------------------------------------------
    print("[SubLCS] Phase 2: extracting V_gauge ...")
    Vgauge_pairs = _extract_vgauge(WS_basis, WG_basis, m)
    assert len(Vgauge_pairs) == g, f"Expected {g} gauge pairs, got {len(Vgauge_pairs)}"

    print(f"[SubLCS] Phase 2: enumerating Sp({2*g}, F_2) ...")
    if g == 0:
        sp2g_elements = [np.eye(0, dtype=int)]  # trivial group
    elif enumerate_gauge == "brute" and g <= 3:
        sp2g_elements = enumerate_sp2g(g)
    else:
        sp2g_elements = enumerate_sp2g_via_transvections(g, max_elements=max_gauge_elements)

    print(f"[SubLCS] |Sp({2*g},F_2)| = {len(sp2g_elements)}")

    print("[SubLCS] Phase 2: lifting gauge elements ...")
    gauge_lifts = []
    for N in sp2g_elements:
        if g == 0:
            sigma_N = np.eye(2 * m, dtype=int)
        else:
            sigma_N = lift_sigma(N, Vgauge_pairs, m)
        gauge_lifts.append(sigma_N)

    # -----------------------------------------------------------------------
    # Combine
    # -----------------------------------------------------------------------
    print("[SubLCS] Combining ...")
    solutions = []
    for Mi in stab_solutions:
        for sigma_N in gauge_lifts:
            sol = f2_matmul(Mi, sigma_N) % 2
            solutions.append(sol)

    print(f"[SubLCS] Total solutions: {len(solutions)}")
    return solutions


# ---------------------------------------------------------------------------
# Verification utilities
# ---------------------------------------------------------------------------

def verify_solution(
    M: np.ndarray,
    WS_basis: np.ndarray,
    WG_basis: np.ndarray,
    logical_basis: np.ndarray,
    UL_matrix: np.ndarray,
    m: int
) -> bool:
    """
    Verify that M satisfies conditions (G), (S), (L).

    (G) M * W_G = W_G  (gauge normalization)
    (S) M * s = s for all s in W_S  (stabilizer centralization)
    (L) M acts as UL on logical sector
    """
    n = 2 * m

    # (S): M fixes each stabilizer generator
    for s in WS_basis:
        Ms = f2_matmul(M, s.reshape(-1, 1)).flatten() % 2
        if not np.array_equal(Ms, s):
            return False

    # (G): M maps W_G to itself
    # Check that each row of WG_basis, when hit by M, lands in span of WG_basis
    for v in WG_basis:
        Mv = f2_matmul(M, v.reshape(-1, 1)).flatten() % 2
        test = np.vstack([WG_basis, Mv.reshape(1, -1)]) % 2
        if f2_rank(test) > f2_rank(WG_basis):
            return False

    # (L): M acts as UL_matrix on logical sector
    k = logical_basis.shape[0] // 2
    for j in range(2 * k):
        v = logical_basis[j]
        Mv = f2_matmul(M, v.reshape(-1, 1)).flatten() % 2
        # compute expected image via UL_matrix
        expected = np.zeros(n, dtype=int)
        for i in range(2 * k):
            expected = (expected + int(UL_matrix[i, j]) * logical_basis[i]) % 2
        # Mv and expected should be equal mod W_G
        diff = (Mv + expected) % 2
        test = np.vstack([WG_basis, diff.reshape(1, -1)]) % 2
        if f2_rank(test) > f2_rank(WG_basis):
            return False

    return True


def count_distinct(solutions: List[np.ndarray]) -> int:
    """Count distinct matrices in a list."""
    seen = set()
    for M in solutions:
        key = tuple(M.flatten().tolist())
        seen.add(key)
    return len(seen)


# ---------------------------------------------------------------------------
# [[4,1,1,2]] Bacon-Shor Code Example
# ---------------------------------------------------------------------------

def bacon_shor_4112():
    """
    Reproduce the [[4,1,1,2]] Bacon-Shor example from Section V of the paper.

    Expected: |Sol(U_L)| = 48 for any logical Clifford U_L.
    """
    print("=" * 60)
    print("[[4,1,1,2]] Bacon-Shor Code Example")
    print("=" * 60)

    m = 4  # physical qubits
    # Notation: vectors are [a | b] in F_2^8 = F_2^{2*4}
    # First 4 bits = X part, last 4 bits = Z part

    # Gauge generators (Eq. 29-32 of paper)
    gX1 = np.array([1,1,0,0, 0,0,0,0], dtype=int)  # X1X2
    gX2 = np.array([0,0,1,1, 0,0,0,0], dtype=int)  # X3X4
    gZ1 = np.array([0,0,0,0, 1,0,1,0], dtype=int)  # Z1Z3
    gZ2 = np.array([0,0,0,0, 0,1,0,1], dtype=int)  # Z2Z4

    # Stabilizers (center Z(G))
    SX  = np.array([1,1,1,1, 0,0,0,0], dtype=int)  # X1X2X3X4
    SZ  = np.array([0,0,0,0, 1,1,1,1], dtype=int)  # Z1Z2Z3Z4

    # Logical operators
    XL  = np.array([1,0,1,0, 0,0,0,0], dtype=int)  # X1X3
    ZL  = np.array([0,0,0,0, 1,1,0,0], dtype=int)  # Z1Z2

    WS_basis      = np.array([SX, SZ],             dtype=int)  # r=2
    WG_basis      = np.array([gX1, gX2, gZ1, gZ2], dtype=int)  # r+2g=4
    logical_basis = np.array([XL, ZL],             dtype=int)  # 2k=2

    # Verify symplectic inner products
    Om = omega_matrix(m)
    print(f"<gX1, gZ1> = {int(gX1 @ Om @ gZ1 % 2)}  (should be 1, non-Abelian)")
    print(f"<SX,  SZ>  = {int(SX  @ Om @ SZ  % 2)}  (should be 0, Abelian center)")
    print(f"<XL,  ZL>  = {int(XL  @ Om @ ZL  % 2)}  (should be 1, logical pair)")
    print()

    # Target logical Clifford: identity on logical qubit (U_L = I)
    # UL_matrix is 2x2 identity acting on (XL, ZL)
    UL_identity = np.eye(2, dtype=int)

    print("Running SubLCS for U_L = logical identity ...")
    solutions = sublcs(
        WS_basis, WG_basis, logical_basis,
        UL_identity, m,
        enumerate_gauge="brute"
    )

    n_distinct = count_distinct(solutions)
    print(f"\nDistinct solutions: {n_distinct}  (expected 48)")
    assert n_distinct == 48, f"Expected 48, got {n_distinct}"

    # Verify all solutions satisfy (G), (S), (L)
    print("Verifying all solutions satisfy (G), (S), (L) ...")
    all_valid = all(
        verify_solution(M, WS_basis, WG_basis, logical_basis, UL_identity, m)
        for M in solutions
    )
    print(f"All solutions valid: {all_valid}")
    assert all_valid

    # Target logical Clifford: logical Hadamard H_L: XL <-> ZL
    # In the basis (XL, ZL), H_L maps XL -> ZL and ZL -> XL
    # Symplectic matrix: [[0,1],[1,0]]
    UL_hadamard = np.array([[0,1],[1,0]], dtype=int)

    print("\nRunning SubLCS for U_L = logical Hadamard ...")
    solutions_H = sublcs(
        WS_basis, WG_basis, logical_basis,
        UL_hadamard, m,
        enumerate_gauge="brute"
    )

    n_distinct_H = count_distinct(solutions_H)
    print(f"Distinct solutions: {n_distinct_H}  (expected 48)")
    assert n_distinct_H == 48

    all_valid_H = all(
        verify_solution(M, WS_basis, WG_basis, logical_basis, UL_hadamard, m)
        for M in solutions_H
    )
    print(f"All solutions valid: {all_valid_H}")
    assert all_valid_H

    # Demonstrate: solutions for identity and Hadamard are disjoint cosets
    keys_I = {tuple(M.flatten().tolist()) for M in solutions}
    keys_H = {tuple(M.flatten().tolist()) for M in solutions_H}
    print(f"\nCosets disjoint (Corollary 5): {len(keys_I & keys_H) == 0}")

    print("\n[PASS] All assertions passed.")
    return solutions, solutions_H


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    solutions_I, solutions_H = bacon_shor_4112()

    print("\n--- Sample solution (U_L = identity, index 0) ---")
    M = solutions_I[0]
    print(f"Shape: {M.shape}")
    print("Matrix (8x8 over F_2):")
    print(M)

    # Verify symplecticity
    print(f"Is symplectic: {is_symplectic(M, 4)}")
