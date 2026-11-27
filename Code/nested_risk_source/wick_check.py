"""Finite Wick-pairing audit for the rooted/marked Wishart formulas.

This module is intentionally independent of the formulas it checks.  One
side enumerates all pairings of the 2q Gaussian factors and obtains row and
feature classes by union--find.  The other side enumerates all set partitions
and retains the noncrossing ones.  Their rooted covariance monomials are then
compared coefficient by coefficient, including every possible marked feature
position.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations
from math import comb
from pathlib import Path
import json


Signature = tuple[int, int, tuple[int, ...]]


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left

    def blocks(self) -> tuple[tuple[int, ...], ...]:
        groups: dict[int, list[int]] = {}
        for value in range(len(self.parent)):
            groups.setdefault(self.find(value), []).append(value)
        return tuple(sorted((tuple(group) for group in groups.values()), key=lambda x: x[0]))


def pairings(items: tuple[int, ...]):
    """Yield all perfect pairings in a deterministic canonical order."""
    if not items:
        yield ()
        return
    first = items[0]
    for offset in range(1, len(items)):
        second = items[offset]
        rest = items[1:offset] + items[offset+1:]
        for tail in pairings(rest):
            yield ((first, second),) + tail


def set_partitions(size: int):
    """Yield canonical set partitions of range(size), each exactly once."""
    blocks: list[list[int]] = []

    def visit(value: int):
        if value == size:
            yield tuple(tuple(block) for block in blocks)
            return
        for block in blocks:
            block.append(value)
            yield from visit(value+1)
            block.pop()
        blocks.append([value])
        yield from visit(value+1)
        blocks.pop()

    yield from visit(0)


def is_noncrossing(partition: tuple[tuple[int, ...], ...]) -> bool:
    for left_index, left in enumerate(partition):
        for right in partition[left_index+1:]:
            for a, c in combinations(left, 2):
                for b, d in combinations(right, 2):
                    if a < b < c < d or b < a < d < c:
                        return False
    return True


def noncrossing_partitions(size: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(partition for partition in set_partitions(size) if is_noncrossing(partition))


def pairing_classes(q: int, pairing: tuple[tuple[int, int], ...]) -> dict:
    """Return row/feature quotient classes for one Gaussian Wick pairing.

    Factors 2j and 2j+1 are u_j and v_j.  Both carry row alpha_j;
    u_j carries feature position j and v_j position j+1 modulo q.
    """
    row = UnionFind(q)
    feature = UnionFind(q)
    all_uv = True
    for left, right in pairing:
        left_j, left_kind = divmod(left, 2)
        right_j, right_kind = divmod(right, 2)
        row.union(left_j, right_j)
        left_feature = left_j if left_kind == 0 else (left_j+1) % q
        right_feature = right_j if right_kind == 0 else (right_j+1) % q
        feature.union(left_feature, right_feature)
        all_uv = all_uv and left_kind != right_kind
    row_blocks = row.blocks()
    feature_blocks = feature.blocks()
    row_classes = len(row_blocks)
    free_feature_classes = len(feature_blocks)-1  # class containing position 0 is rooted
    exponent = row_classes + free_feature_classes - q
    return {
        "row_blocks": row_blocks,
        "feature_blocks": feature_blocks,
        "row_classes": row_classes,
        "free_feature_classes": free_feature_classes,
        "power_of_n": exponent,
        "all_pairs_u_v": all_uv,
    }


def rooted_signature(
    partition: tuple[tuple[int, ...], ...], mark_position: int | None = None
) -> Signature:
    root_index = next(index for index, block in enumerate(partition) if 0 in block)
    powers = [len(block) for block in partition]
    if mark_position is not None:
        marked_index = next(index for index, block in enumerate(partition) if mark_position in block)
        powers[marked_index] += 1
    root_power = powers[root_index]
    trace_powers = tuple(sorted(powers[index] for index in range(len(powers)) if index != root_index))
    return len(partition)-1, root_power, trace_powers


def _terms(partitions, mark_position: int | None = None) -> Counter[Signature]:
    return Counter(rooted_signature(partition, mark_position) for partition in partitions)


def _serialize_terms(terms: Counter[Signature]) -> list[dict]:
    return [
        {"c_power": signature[0], "nu_power": signature[1],
         "mu_powers": list(signature[2]), "coefficient": coefficient}
        for signature, coefficient in sorted(terms.items())
    ]


def evaluate_terms(
    terms: Counter[Signature], c: float, mu: dict[int, float], nu: dict[int, float]
) -> float:
    total = 0.0
    for (c_power, root_power, trace_powers), coefficient in terms.items():
        value = coefficient*c**c_power*nu[root_power]
        for power in trace_powers:
            value *= mu[power]
        total += value
    return total


def catalan(q: int) -> int:
    return comb(2*q, q)//(q+1)


def double_factorial_odd(q: int) -> int:
    result = 1
    for value in range(1, 2*q, 2):
        result *= value
    return result


def validate_wick_pairings(max_q: int = 6) -> dict:
    levels = []
    global_ok = True
    for q in range(1, max_q+1):
        digest = sha256()
        exponent_counts: Counter[int] = Counter()
        pairing_records = []
        leading_partitions = []
        leading_all_uv = True
        pairing_count = 0
        for pairing in pairings(tuple(range(2*q))):
            pairing_count += 1
            classes = pairing_classes(q, pairing)
            exponent_counts[classes["power_of_n"]] += 1
            digest.update(repr((pairing, classes)).encode("ascii"))
            signature = rooted_signature(classes["feature_blocks"])
            pairing_records.append({
                "pairs_zero_based_factor_ids": [list(pair) for pair in pairing],
                "row_blocks_zero_based": [list(block) for block in classes["row_blocks"]],
                "feature_blocks_zero_based": [list(block) for block in classes["feature_blocks"]],
                "row_classes": classes["row_classes"],
                "free_feature_classes": classes["free_feature_classes"],
                "power_of_n": classes["power_of_n"],
                "all_pairs_u_v": classes["all_pairs_u_v"],
                "rooted_covariance_monomial": {
                    "free_feature_classes_c_power": signature[0],
                    "nu_power": signature[1],
                    "mu_powers": list(signature[2]),
                },
            })
            if classes["power_of_n"] == 0:
                leading_partitions.append(classes["feature_blocks"])
                leading_all_uv = leading_all_uv and classes["all_pairs_u_v"]

        nc = noncrossing_partitions(q)
        pairing_partition_set = set(leading_partitions)
        nc_partition_set = set(nc)
        rooted_pairing_terms = _terms(leading_partitions)
        rooted_nc_terms = _terms(nc)
        marked_checks = []
        for position in range(q):
            pairing_terms = _terms(leading_partitions, position)
            nc_terms = _terms(nc, position)
            match = pairing_terms == nc_terms
            marked_checks.append({
                "feature_position_1based": position+1,
                "is_E_ab_position": 1 <= position <= q-1,
                "coefficients_match": match,
                "terms": _serialize_terms(nc_terms),
            })
            global_ok = global_ok and match

        checks = {
            "pairing_count_matches_double_factorial": pairing_count == double_factorial_odd(q),
            "leading_count_matches_catalan": len(leading_partitions) == catalan(q),
            "nc_count_matches_catalan": len(nc) == catalan(q),
            "leading_pairs_are_all_u_v": leading_all_uv,
            "leading_feature_partitions_equal_NC": pairing_partition_set == nc_partition_set,
            "rooted_coefficients_match": rooted_pairing_terms == rooted_nc_terms,
            "all_marked_coefficients_match": all(item["coefficients_match"] for item in marked_checks),
        }
        level_ok = all(checks.values())
        global_ok = global_ok and level_ok
        levels.append({
            "q": q,
            "pairings_total": pairing_count,
            "expected_pairings": double_factorial_odd(q),
            "leading_pairings": len(leading_partitions),
            "noncrossing_partitions": len(nc),
            "power_of_n_counts": {str(key): value for key, value in sorted(exponent_counts.items(), reverse=True)},
            "pairing_audit_sha256": digest.hexdigest(),
            "factor_id_convention": "factor 2*j is u_j and 2*j+1 is v_j, with j zero-based",
            "pairing_records": pairing_records,
            "checks": checks,
            "rooted_terms": _serialize_terms(rooted_nc_terms),
            "marked_positions": marked_checks,
        })
    return {
        "schema": "pls-wick-noncrossing-audit-v1",
        "description": "Independent Wick-pairing/union-find and noncrossing-partition enumeration",
        "max_q": max_q,
        "all_checks_ok": global_ok,
        "levels": levels,
    }


def write_wick_validation(path: Path, max_q: int = 6) -> dict:
    validation = validate_wick_pairings(max_q)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validation, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return validation
