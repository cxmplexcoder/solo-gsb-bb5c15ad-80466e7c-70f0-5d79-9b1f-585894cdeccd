"""片段拼接链规划接口测试。"""

import itertools
import random

import pytest
from fastapi.testclient import TestClient

from app.assembly import plan_assembly_chain
from app.ligation import match_ends
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


# 5' 突出端酶：识别 GAAGTG，切口 1/5 -> 5prime，突出端 AAGT
FIVE_PRIME_ENZYME = {"name": "F5", "pattern": "GAAGTG", "cut_top": 1, "cut_bottom": 5}
# 3' 突出端酶：识别 GACTTG，切口 5/1 -> 3prime，突出端 ACTT = 反向互补(AAGT)
THREE_PRIME_ENZYME = {"name": "F3", "pattern": "GACTTG", "cut_top": 5, "cut_bottom": 1}
# 平端酶：识别 GATATC，切口 3/3 -> blunt
BLUNT_ENZYME = {"name": "Blunt", "pattern": "GATATC", "cut_top": 3, "cut_bottom": 3}
# 识别 GAATTC，在全 A 序列上无位点，用于构造"不切割"的记录
ECORI = {"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5}


def _create(client, sequence, enzymes, topology="linear"):
    resp = client.post(
        "/digests",
        json={"sequence": sequence, "topology": topology, "enzymes": enzymes},
    )
    assert resp.status_code == 201
    return resp.json()


def _create_five_prime(client):
    """5' AAGT 记录：单切口；f0 右端 / f1 左端为 5prime(AAGT)，另一端为 none。"""
    return _create(client, "AAAAGAAGTGAAAA", [FIVE_PRIME_ENZYME])


def _create_three_prime(client):
    """3' ACTT 记录：单切口；f0 右端 / f1 左端为 3prime(ACTT)，另一端为 none。"""
    return _create(client, "AAAAGACTTGAAAA", [THREE_PRIME_ENZYME])


def _create_blunt(client, sequence="AAAGATATCAAA"):
    """blunt 记录：单切口；f0 右端 / f1 左端为平端，另一端为 none。"""
    return _create(client, sequence, [BLUNT_ENZYME])


def _create_blunt_bridge(client):
    """双平端切口记录：f1 两端均为 blunt，可作拼接链的内部桥接片段。"""
    return _create(client, "AAAGATATCAAAGATATCAAA", [BLUNT_ENZYME])


def _create_uncut(client):
    """无切口记录：唯一片段两端均为 none。"""
    return _create(client, "AAAAAAAAAA", [ECORI])


def _plan(client, digest_ids, include_blunt=None, max_chain_length=8):
    payload = {"digest_ids": digest_ids}
    if include_blunt is not None:
        payload["include_blunt"] = include_blunt
    if max_chain_length is not None:
        payload["max_chain_length"] = max_chain_length
    return client.post("/ligation-assembly", json=payload)


def _end(digest_id, fragment_index, end, end_type, overhang):
    return {
        "digest_id": digest_id,
        "fragment_index": fragment_index,
        "end": end,
        "end_type": end_type,
        "overhang": overhang,
    }


# ---------- 黏端最长链 ----------


def test_sticky_chain_uses_bridge_fragment_and_all_three_records(client):
    rec5 = _create_five_prime(client)
    # 桥记录：f0(右端5'AAGT) - f1(左端5'AAGT, 右端3'ACTT) - f2(左端3'ACTT)
    bridge = _create(
        client, "AAAAGAAGTGAAAAAGACTTGAAAA",
        [FIVE_PRIME_ENZYME, THREE_PRIME_ENZYME],
    )
    rec3 = _create_three_prime(client)
    assert len(bridge["fragments"]) == 3

    resp = _plan(client, [rec5["id"], bridge["id"], rec3["id"]])
    assert resp.status_code == 200
    data = resp.json()
    assert data["record_count"] == 3
    assert data["node_count"] == 3
    assert data["connection_count"] == 2
    a, b, c = rec5["id"], bridge["id"], rec3["id"]

    # 链：rec5 f1 反向(入端=原右端 none；出端=原左端5'AAGT→换算 ACTT)
    #     -> bridge f1 反向(入端=原右端3'ACTT→换算 AAGT；
    #        出端=原左端5'AAGT→换算 ACTT)
    #     -> rec3 f0 反向(入端=原右端3'ACTT→换算 AAGT；出端=原左端 none)
    # 反向节点的朝向端突出端为原序列的反向互补，极性保持 5'/3' 不变：
    # ACTT(5') 与 AAGT(3') 满足 5'==revcomp(3') 方可连。
    assert data["nodes"] == [
        {
            "digest_id": a, "fragment_index": 1, "orientation": "reverse",
            "left_end": _end(a, 1, "right", "none", ""),
            "right_end": _end(a, 1, "left", "5prime", "ACTT"),
        },
        {
            "digest_id": b, "fragment_index": 1, "orientation": "reverse",
            "left_end": _end(b, 1, "right", "3prime", "AAGT"),
            "right_end": _end(b, 1, "left", "5prime", "ACTT"),
        },
        {
            "digest_id": c, "fragment_index": 0, "orientation": "reverse",
            "left_end": _end(c, 0, "right", "3prime", "AAGT"),
            "right_end": _end(c, 0, "left", "none", ""),
        },
    ]
    assert data["connections"] == [
        {
            "index": 0,
            "from_end": _end(a, 1, "left", "5prime", "ACTT"),
            "to_end": _end(b, 1, "right", "3prime", "AAGT"),
            "compatibility": "sticky",
            "ligated_overhang": "ACTT",
        },
        {
            "index": 1,
            "from_end": _end(b, 1, "left", "5prime", "ACTT"),
            "to_end": _end(c, 0, "right", "3prime", "AAGT"),
            "compatibility": "sticky",
            "ligated_overhang": "ACTT",
        },
    ]
    assert data["used_digest_ids"] == [a, b, c]
    assert data["unused_digest_ids"] == []
    assert data["stop_reason"] == "all_records_used"
    assert data["stop_reason_detail"]


def test_simple_two_record_sticky_chain_orientation(client):
    rec5 = _create_five_prime(client)
    rec3 = _create_three_prime(client)
    data = _plan(client, [rec5["id"], rec3["id"]]).json()
    # 字典序最小的可行链：rec5 f0 forward -> rec3 f1 forward。
    # rec3 f0 reverse 的入端是原右端 3'ACTT 换算后的 AAGT(3')，与
    # 5'AAGT 不兼容（5'==revcomp(3')? AAGT != ACTT）；f1 forward
    # 入端即原左端 3'ACTT，5'AAGT == revcomp(ACTT) 兼容。
    assert [(n["digest_id"], n["fragment_index"], n["orientation"]) for n in data["nodes"]] == [
        (rec5["id"], 0, "forward"),
        (rec3["id"], 1, "forward"),
    ]
    assert data["connection_count"] == 1
    assert data["connections"][0]["compatibility"] == "sticky"
    assert data["connections"][0]["ligated_overhang"] == "AAGT"
    assert data["stop_reason"] == "all_records_used"


def test_reverse_request_order_flips_chain_records(client):
    rec5 = _create_five_prime(client)
    rec3 = _create_three_prime(client)
    # 请求顺序反转：择优以请求位置为键，链首改属请求顺序靠前的 3' 记录
    data = _plan(client, [rec3["id"], rec5["id"]]).json()
    assert [n["digest_id"] for n in data["nodes"]] == [rec3["id"], rec5["id"]]
    # rec3 片段：f0 左 none / 右 3'ACTT；f1 左 3'ACTT / 右 none。
    # 链首出端必须是 3'（才能连 5'）：最小键节点是 f0 forward
    # （入端=left(none)，出端=right(3')）。
    first = data["nodes"][0]
    assert first["fragment_index"] == 0
    assert first["orientation"] == "forward"
    assert first["left_end"]["end_type"] == "none"
    assert first["right_end"]["end_type"] == "3prime"
    # 次节点最小键：rec5 f1 forward（入端=left(5')；f0 reverse 的入端为
    # 原右端 5'AAGT 换算后的 TGAA(5')，与 3'ACTT 不兼容）
    second = data["nodes"][1]
    assert second["fragment_index"] == 1
    assert second["orientation"] == "forward"
    assert data["connections"][0]["from_end"]["end_type"] == "3prime"
    assert data["connections"][0]["to_end"]["end_type"] == "5prime"


def test_non_compatible_sticky_overhangs_yield_empty_chain(client):
    # 两条 5' 记录：5'-5' 不兼容 -> 不存在任何拼接方案
    rec_a = _create_five_prime(client)
    rec_b = _create_five_prime(client)
    data = _plan(client, [rec_a["id"], rec_b["id"]]).json()
    assert data["node_count"] == 0
    assert data["connection_count"] == 0
    assert data["nodes"] == []
    assert data["connections"] == []
    assert data["used_digest_ids"] == []
    assert data["unused_digest_ids"] == [rec_a["id"], rec_b["id"]]
    assert data["stop_reason"] == "no_compatible_ends"


def _single_right_end_record(digest_id, end_type, overhang):
    """合成"只有右端可连接"的单片段记录（左端为天然末端）。"""
    return {
        "digest_id": digest_id,
        "fragments": [
            {
                "index": 0,
                "left_end": {"type": "none", "overhang": ""},
                "right_end": {"type": end_type, "overhang": overhang},
            }
        ],
    }


def test_reversed_right_end_5a_vs_3t_does_not_assemble(client):
    # 反向节点突出端必须反向互补换算：两条"只有右端"的记录，
    # 5'-A 与 3'-T 在旧逻辑（仅交换 left/right）下被误判兼容
    # （A == revcomp(T)）；换算后反向节点入端为 3'-revcomp(T)=A，
    # A != revcomp(A)=T，不成链。
    records = [
        _single_right_end_record(1, "5prime", "A"),
        _single_right_end_record(2, "3prime", "T"),
    ]
    plan = plan_assembly_chain(records, include_blunt=True, max_chain_length=8)
    assert plan["nodes"] == []
    assert plan["connections"] == []
    assert plan["stop_reason"] == "no_compatible_ends"


def test_reversed_right_end_5a_vs_3a_assembles_with_converted_overhang(client):
    # 同一对朝向（两条"只有右端"的记录）：5'-A 与 3'-A 在旧逻辑下漏报
    # （A != revcomp(A)=T）；换算后反向节点入端为 3'-revcomp(A)=T，
    # 5'-A == revcomp(T)=A，恰可连接。
    records = [
        _single_right_end_record(1, "5prime", "A"),
        _single_right_end_record(2, "3prime", "A"),
    ]
    plan = plan_assembly_chain(records, include_blunt=True, max_chain_length=8)
    assert len(plan["nodes"]) == 2
    first, second = plan["nodes"]
    # 首节点正向：出端即原右端 5'-A
    assert (first["fragment_index"], first["orientation"]) == (0, "forward")
    # 次节点反向摆入：入端标识仍为原片段右端，突出端为换算后的 3'-T
    assert (second["fragment_index"], second["orientation"]) == (0, "reverse")
    assert plan["connections"] == [
        {
            "index": 0,
            "from_end": _end(1, 0, "right", "5prime", "A"),
            "to_end": _end(2, 0, "right", "3prime", "T"),
            "compatibility": "sticky",
            "ligated_overhang": "A",
        }
    ]
    # 对外节点端点同样使用换算后的朝向端，端标识保留原片段 right
    from app.assembly import public_node
    assert public_node(first)["right_end"] == _end(1, 0, "right", "5prime", "A")
    assert public_node(second)["left_end"] == _end(2, 0, "right", "3prime", "T")
    assert public_node(second)["right_end"] == _end(2, 0, "left", "none", "")


def test_reversed_overhang_conversion_uses_iupac_complement(client):
    # 简并碱基同样按 IUPAC 表换算：反向节点 3'-R 的入端换算为 3'-Y
    # （COMPLEMENT[R]=Y），与 5'-R 兼容（R == revcomp(Y)=R）；
    # 旧逻辑直接拿 3'-R 比较，R != revcomp(R)=Y 会漏报。
    records = [
        _single_right_end_record(1, "5prime", "R"),
        _single_right_end_record(2, "3prime", "R"),
    ]
    plan = plan_assembly_chain(records, include_blunt=True, max_chain_length=8)
    assert len(plan["nodes"]) == 2
    assert plan["connections"][0]["from_end"] == _end(1, 0, "right", "5prime", "R")
    assert plan["connections"][0]["to_end"] == _end(2, 0, "right", "3prime", "Y")
    assert plan["connections"][0]["ligated_overhang"] == "R"


def test_forward_nodes_keep_original_overhangs(client):
    # 正向节点不换算：5'-AAGT 正向出端与原右端 3'-ACTT 正向入端
    # （AAGT == revcomp(ACTT)）照常兼容，回显突出端为原始序列。
    records = [
        _single_right_end_record(1, "5prime", "AAGT"),
        {
            "digest_id": 2,
            "fragments": [
                {
                    "index": 0,
                    "left_end": {"type": "3prime", "overhang": "ACTT"},
                    "right_end": {"type": "none", "overhang": ""},
                }
            ],
        },
    ]
    plan = plan_assembly_chain(records, include_blunt=True, max_chain_length=8)
    assert [(n["fragment_index"], n["orientation"]) for n in plan["nodes"]] == [
        (0, "forward"), (0, "forward"),
    ]
    assert plan["connections"][0]["from_end"] == _end(1, 0, "right", "5prime", "AAGT")
    assert plan["connections"][0]["to_end"] == _end(2, 0, "left", "3prime", "ACTT")



def test_uncut_record_does_not_participate(client):
    rec5 = _create_five_prime(client)
    rec3 = _create_three_prime(client)
    uncut = _create_uncut(client)
    data = _plan(client, [rec5["id"], uncut["id"], rec3["id"]]).json()
    # 无切口记录两端 none，链仍是 rec5 -> rec3；uncut 未入选
    assert data["node_count"] == 2
    assert data["used_digest_ids"] == [rec5["id"], rec3["id"]]
    assert data["unused_digest_ids"] == [uncut["id"]]
    # 链右端是 none 天然末端，与未用记录无法连接
    assert data["stop_reason"] == "no_compatible_extension"


def test_only_uncut_records_yield_no_chain(client):
    a = _create_uncut(client)
    b = _create_uncut(client)
    data = _plan(client, [a["id"], b["id"]]).json()
    assert data["nodes"] == []
    assert data["stop_reason"] == "no_compatible_ends"


# ---------- 平端 ----------


def test_blunt_three_record_chain_through_bridge(client):
    edge1 = _create_blunt(client)
    bridge = _create_blunt_bridge(client)
    edge3 = _create_blunt(client)
    assert len(bridge["fragments"]) == 3
    data = _plan(client, [edge1["id"], bridge["id"], edge3["id"]]).json()
    assert data["node_count"] == 3
    assert [(n["digest_id"], n["fragment_index"]) for n in data["nodes"]] == [
        (edge1["id"], 0),
        (bridge["id"], 1),  # 两端皆 blunt 的桥接片段
        (edge3["id"], 0),
    ]
    assert all(c["compatibility"] == "blunt" for c in data["connections"])
    assert all(c["ligated_overhang"] == "" for c in data["connections"])
    # 三条记录全部入选
    assert data["stop_reason"] == "all_records_used"
    assert data["unused_digest_ids"] == []


def test_blunt_excluded_removes_blunt_chain(client):
    a = _create_blunt(client)
    b = _create_blunt(client)
    data = _plan(client, [a["id"], b["id"]], include_blunt=False).json()
    assert data["nodes"] == []
    assert data["stop_reason"] == "no_compatible_ends"


def test_include_blunt_false_keeps_sticky_chain(client):
    rec5 = _create_five_prime(client)
    rec3 = _create_three_prime(client)
    data = _plan(
        client, [rec5["id"], rec3["id"]], include_blunt=False
    ).json()
    assert data["node_count"] == 2
    assert all(c["compatibility"] == "sticky" for c in data["connections"])


# ---------- 最长链长 ----------


def test_max_chain_length_limits_chain_and_reports_reason(client):
    # edge(单切口) -> bridge(双切口, f1 两端皆 blunt) -> circ(环形单切口,
    # 唯一片段两端皆 blunt，仍可继续与 edge4 相连)：上限 3 截断在 circ，
    # 其右端是平端且 edge4 有平端入端，确为可延伸 -> max_chain_length_reached
    edge1 = _create_blunt(client)
    bridge = _create_blunt_bridge(client)
    circ = _create(client, "AAAGATATCAAA", [BLUNT_ENZYME], topology="circular")
    edge4 = _create_blunt(client)
    data = _plan(
        client,
        [edge1["id"], bridge["id"], circ["id"], edge4["id"]],
        max_chain_length=3,
    ).json()
    assert data["node_count"] == 3
    assert data["connection_count"] == 2
    assert data["max_chain_length"] == 3
    assert [n["digest_id"] for n in data["nodes"]] == [
        edge1["id"], bridge["id"], circ["id"]
    ]
    assert data["used_digest_ids"] == [edge1["id"], bridge["id"], circ["id"]]
    assert data["unused_digest_ids"] == [edge4["id"]]
    assert data["stop_reason"] == "max_chain_length_reached"

    # 对照：放开上限后 4 条记录全部入选
    full = _plan(
        client,
        [edge1["id"], bridge["id"], circ["id"], edge4["id"]],
    ).json()
    assert full["node_count"] == 4
    assert full["stop_reason"] == "all_records_used"


def test_max_chain_length_reason_requires_real_extension(client):
    # 两条 5' 记录不兼容，链为空；即使限制为 2 也不应报 max_length
    a = _create_five_prime(client)
    b = _create_five_prime(client)
    data = _plan(client, [a["id"], b["id"]], max_chain_length=2).json()
    assert data["stop_reason"] == "no_compatible_ends"


def test_records_used_reason_takes_precedence_over_limit(client):
    rec5 = _create_five_prime(client)
    rec3 = _create_three_prime(client)
    # 上限 8 远大于可用记录数 2 -> all_records_used
    data = _plan(
        client, [rec5["id"], rec3["id"]], max_chain_length=8
    ).json()
    assert data["node_count"] == 2
    assert data["stop_reason"] == "all_records_used"


# ---------- 记录不重复使用 ----------


def test_each_record_used_at_most_once(client):
    # 两个切口、4 片段的 blunt 记录有大量内部可连面，但同一记录不能在链中重复：
    # 只提交 2 条记录时链最多 2 节点。
    a = _create_blunt_bridge(client)
    b = _create_blunt_bridge(client)
    data = _plan(client, [a["id"], b["id"]]).json()
    assert data["node_count"] == 2
    assert data["connection_count"] == 1
    used = [n["digest_id"] for n in data["nodes"]]
    assert len(set(used)) == 2


def test_eight_blunt_records_chain_each_once(client):
    # 单切口 blunt 记录只有一个可连接面：不能做内部桥，最多 2 节点；
    # 改用双切口桥结构的记录，8 条记录全部入选。
    records = [_create_blunt_bridge(client) for _ in range(8)]
    ids = [r["id"] for r in records]
    data = _plan(client, ids).json()
    assert data["node_count"] == 8
    assert data["connection_count"] == 7
    assert [n["digest_id"] for n in data["nodes"]] == ids
    assert data["stop_reason"] == "all_records_used"


# ---------- 环形记录 ----------


def test_circular_single_cut_fragment_has_two_ligatable_ends(client):
    # 环形单切口开环：唯一全长片段两端均为 5'(AAGT)，本身可正向/反向摆入，
    # 与 3' 记录形成 2 节点链
    circ = _create(
        client, "AAAAGAAGTGAAAA", [FIVE_PRIME_ENZYME], topology="circular"
    )
    assert len(circ["fragments"]) == 1
    rec3 = _create_three_prime(client)
    data = _plan(client, [circ["id"], rec3["id"]]).json()
    assert data["node_count"] == 2
    assert data["nodes"][0]["digest_id"] == circ["id"]
    assert data["nodes"][0]["fragment_index"] == 0


# ---------- 同长度稳定择优 ----------


def test_tie_break_prefers_request_order_then_fragment_then_direction(client):
    # 三条结构相同的 blunt 单切口记录 + 一个双切口桥：存在多种 3 节点方案，
    # 选请求位置最小的 (edge1, bridge, edge3)；节点方向按 left<right 取最小
    edge1 = _create_blunt(client)
    edge2 = _create_blunt(client)
    edge3 = _create_blunt(client)
    bridge = _create_blunt_bridge(client)
    data = _plan(
        client, [edge1["id"], edge2["id"], bridge["id"], edge3["id"]],
        max_chain_length=3,
    ).json()
    assert [n["digest_id"] for n in data["nodes"]] == [
        edge1["id"], bridge["id"], edge2["id"]
    ]
    # 首节点最小键：f0 forward（入端 left=none、出端 right=blunt）
    assert (data["nodes"][0]["fragment_index"], data["nodes"][0]["orientation"]) == (
        0, "forward"
    )


def test_result_is_deterministic(client):
    edge1 = _create_blunt(client)
    bridge = _create_blunt_bridge(client)
    edge3 = _create_blunt(client)
    payload = {
        "digest_ids": [edge1["id"], bridge["id"], edge3["id"]],
        "include_blunt": True,
        "max_chain_length": 8,
    }
    first = client.post("/ligation-assembly", json=payload).json()
    second = client.post("/ligation-assembly", json=payload).json()
    assert first == second


# ---------- 校验与状态码 ----------


def test_duplicate_digest_ids_return_422(client):
    rec = _create_blunt(client)
    resp = _plan(client, [rec["id"], rec["id"]])
    assert resp.status_code == 422
    assert "duplicate digest ids" in resp.text


def test_digest_id_count_limits_return_422(client):
    rec = _create_blunt(client)
    assert _plan(client, [rec["id"]]).status_code == 422
    assert _plan(client, list(range(1, 10))).status_code == 422


def test_max_chain_length_limits_return_422(client):
    rec_a = _create_blunt(client)
    rec_b = _create_blunt(client)
    assert _plan(client, [rec_a["id"], rec_b["id"]], max_chain_length=1).status_code == 422
    assert _plan(client, [rec_a["id"], rec_b["id"]], max_chain_length=9).status_code == 422
    assert _plan(client, [rec_a["id"], rec_b["id"]], max_chain_length=0).status_code == 422


def test_missing_max_chain_length_returns_422(client):
    rec = _create_blunt(client)
    resp = client.post(
        "/ligation-assembly", json={"digest_ids": [rec["id"], rec["id"] + 1]}
    )
    assert resp.status_code == 422


def test_invalid_argument_types_return_422(client):
    rec = _create_blunt(client)
    assert _plan(client, ["x", "y"]).status_code == 422
    resp = _plan(
        client, [rec["id"], rec["id"] + 1],
        include_blunt="notabool", max_chain_length=3,
    )
    assert resp.status_code == 422
    resp = _plan(
        client, [rec["id"], rec["id"] + 1], max_chain_length="long"
    )
    assert resp.status_code == 422


def test_missing_record_returns_404(client):
    rec = _create_blunt(client)
    resp = _plan(client, [rec["id"], 999], max_chain_length=3)
    assert resp.status_code == 404
    assert "999" in resp.json()["detail"]


def test_first_missing_id_in_request_order_is_reported(client):
    resp = _plan(client, [999, 998], max_chain_length=3)
    assert resp.status_code == 404
    assert "999" in resp.json()["detail"]


def test_validation_precedes_existence_check(client):
    # 重复 id / 非法链长直接 422，即使 id 不存在也不报 404
    assert _plan(client, [999, 999], max_chain_length=3).status_code == 422
    assert _plan(client, [999, 998], max_chain_length=1).status_code == 422


def test_too_many_fragments_returns_422(client):
    # 一条含 65 个片段的记录（64 个 blunt 切口）+ 一条普通记录：
    # 总片段数 65 + 2 > 64 -> 422（记录均存在）
    big = _create_blunt(client, "AAA" + "GATATCAAA" * 64)
    small = _create_blunt(client)
    assert len(big["fragments"]) == 65
    resp = _plan(client, [big["id"], small["id"]], max_chain_length=2)
    assert resp.status_code == 422
    assert "64" in resp.json()["detail"]


def test_fragment_limit_boundary_accepted(client):
    # 64 个片段（单记录 33 切口 + 单记录 31 切口）恰在上界内 -> 200
    big = _create_blunt(client, "AAA" + "GATATCAAA" * 32)
    other = _create_blunt(client, "AAA" + "GATATCAAA" * 30)
    assert len(big["fragments"]) + len(other["fragments"]) == 64
    resp = _plan(client, [big["id"], other["id"]], max_chain_length=2)
    assert resp.status_code == 200


# ---------- 只读 ----------


def test_planner_is_read_only(client):
    rec5 = _create_five_prime(client)
    rec3 = _create_three_prime(client)
    before = client.get("/digests").json()
    resp = _plan(client, [rec5["id"], rec3["id"]])
    assert resp.status_code == 200
    assert client.get(f"/digests/{rec5['id']}").json() == rec5
    assert client.get(f"/digests/{rec3['id']}").json() == rec3
    assert client.get("/digests").json() == before


# ---------- 规划器：小规模随机暴力交叉校验 ----------


def test_planner_matches_brute_force_on_random_graphs():
    from app.assembly import _candidate_nodes, _node_key, _end_key, _public_end

    rng = random.Random(20260923)
    end_pool = [
        ("blunt", ""),
        ("5prime", "AATT"),
        ("3prime", "AATT"),  # AATT 回文：5'AATT 与 3'AATT 兼容
        ("5prime", "AAGT"),
        ("3prime", "ACTT"),
        ("5prime", "GGGG"),
        ("3prime", "CCCC"),
        ("none", ""),
    ]

    def synthetic_record(did, fragment_count):
        fragments = []
        for index in range(fragment_count):
            left_type, left_oh = rng.choice(end_pool)
            right_type, right_oh = rng.choice(end_pool)
            fragments.append(
                {
                    "index": index,
                    "left_end": {"type": left_type, "overhang": left_oh},
                    "right_end": {"type": right_type, "overhang": right_oh},
                }
            )
        return {"digest_id": did, "fragments": fragments}

    for case in range(60):
        record_count = rng.randint(2, 5)
        records = [
            synthetic_record(i + 1, rng.randint(1, 3)) for i in range(record_count)
        ]
        include_blunt = rng.choice([True, False])
        max_chain_length = rng.randint(2, min(8, record_count + 1))

        plan = plan_assembly_chain(records, include_blunt, max_chain_length)

        # 暴力：展开节点后枚举记录子集的全排列
        nodes_by_record = [
            _candidate_nodes(pos, r["digest_id"], r["fragments"])
            for pos, r in enumerate(records)
        ]
        expected_len = 0
        expected_keys = None
        for size in range(1, min(max_chain_length, record_count) + 1):
            for positions in itertools.permutations(range(record_count), size):
                pools = [
                    [_node_key(n) for n in nodes_by_record[p]] for p in positions
                ]
                key_to_node = {
                    _node_key(n): n
                    for group in nodes_by_record
                    for n in group
                }
                for keys in itertools.product(*pools):
                    nodes = [key_to_node[k] for k in keys]
                    if all(
                        match_ends(
                            nodes[i]["out_end"], nodes[i + 1]["in_end"],
                            include_blunt,
                        )
                        is not None
                        for i in range(len(nodes) - 1)
                    ):
                        if len(keys) > expected_len or (
                            len(keys) == expected_len
                            and (expected_keys is None or keys < expected_keys)
                        ):
                            expected_len, expected_keys = len(keys), keys

        if expected_len < 2:
            assert plan["nodes"] == [], case
            assert plan["stop_reason"] == "no_compatible_ends", case
            continue

        actual_keys = tuple(_node_key(n) for n in plan["nodes"])
        assert actual_keys == expected_keys, (case, actual_keys, expected_keys)

        # 链内相邻端点确实兼容（规划器自检之外再独立校验一次）
        for i in range(len(plan["nodes"]) - 1):
            assert plan["connections"][i]["from_end"] == _public_end(
                plan["nodes"][i], plan["nodes"][i]["out_end"]
            )
            assert plan["connections"][i]["to_end"] == _public_end(
                plan["nodes"][i + 1], plan["nodes"][i + 1]["in_end"]
            )

        # 记录不重复
        used = [n["record_position"] for n in plan["nodes"]]
        assert len(used) == len(set(used))
