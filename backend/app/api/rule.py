import json

from flask import Blueprint, request

from app.extensions import db
from app.models.rule import HazardRule
from app.utils.auth import get_current_user_id, login_required
from app.utils.response import error, success, table_data

rule_bp = Blueprint("rule", __name__)


@rule_bp.route("/list", methods=["GET"])
@login_required
def list_rules():
    page = request.args.get("pageNum", 1, type=int)
    page_size = request.args.get("pageSize", 10, type=int)
    rule_type = request.args.get("ruleType", "")
    status = request.args.get("status", "")

    query = HazardRule.query
    if rule_type:
        query = query.filter_by(rule_type=rule_type)
    if status:
        query = query.filter_by(status=status)
    query = query.order_by(HazardRule.priority.desc(), HazardRule.create_time.desc())

    pagination = query.paginate(page=page, per_page=page_size, error_out=False)
    return table_data([row.to_dict() for row in pagination.items], pagination.total)


@rule_bp.route("", methods=["POST"])
@login_required
def create_rule():
    data = request.get_json() or {}
    if not data.get("ruleCode") or not data.get("ruleName"):
        return error("缺少 ruleCode 或 ruleName", 400)

    row = HazardRule(
        rule_code=data["ruleCode"],
        rule_name=data["ruleName"],
        rule_type=data.get("ruleType", "context_policy"),
        context_id=data.get("contextId"),
        relation_type=data.get("relationType"),
        match_mode=data.get("matchMode", "any"),
        priority=int(data.get("priority", 0)),
        required_ppe_json=json.dumps(data.get("requiredPpeJson", []), ensure_ascii=False),
        trigger_relations_json=json.dumps(data.get("triggerRelationsJson", []), ensure_ascii=False),
        rule_text=data.get("ruleText"),
        status=data.get("status", "active"),
        version=data.get("version", "1.0.0"),
        remark=data.get("remark"),
        creator_id=get_current_user_id(),
    )
    db.session.add(row)
    db.session.commit()
    return success(row.to_dict(), msg="规则创建成功")


@rule_bp.route("/<int:rule_id>", methods=["GET"])
@login_required
def get_rule(rule_id):
    row = HazardRule.query.get_or_404(rule_id)
    return success(row.to_dict())


@rule_bp.route("/<int:rule_id>", methods=["PUT"])
@login_required
def update_rule(rule_id):
    row = HazardRule.query.get_or_404(rule_id)
    data = request.get_json() or {}
    row.rule_name = data.get("ruleName", row.rule_name)
    row.rule_type = data.get("ruleType", row.rule_type)
    row.context_id = data.get("contextId", row.context_id)
    row.relation_type = data.get("relationType", row.relation_type)
    row.match_mode = data.get("matchMode", row.match_mode)
    row.priority = int(data.get("priority", row.priority))
    row.required_ppe_json = json.dumps(data.get("requiredPpeJson", json.loads(row.required_ppe_json) if row.required_ppe_json else []), ensure_ascii=False)
    row.trigger_relations_json = json.dumps(data.get("triggerRelationsJson", json.loads(row.trigger_relations_json) if row.trigger_relations_json else []), ensure_ascii=False)
    row.rule_text = data.get("ruleText", row.rule_text)
    row.status = data.get("status", row.status)
    row.version = data.get("version", row.version)
    row.remark = data.get("remark", row.remark)
    db.session.commit()
    return success(row.to_dict(), msg="规则更新成功")


@rule_bp.route("/<int:rule_id>", methods=["DELETE"])
@login_required
def delete_rule(rule_id):
    row = HazardRule.query.get_or_404(rule_id)
    db.session.delete(row)
    db.session.commit()
    return success(msg="规则删除成功")


@rule_bp.route("/import-json", methods=["POST"])
@login_required
def import_rules_from_json():
    data = request.get_json() or {}
    rows = data.get("rules") or data.get("contexts") or []
    created = 0

    for item in rows:
        rule_code = item.get("ruleCode") or item.get("rule_code") or item.get("context_id")
        rule_name = item.get("ruleName") or item.get("rule_name") or item.get("context_text") or rule_code
        if not rule_code:
            continue
        row = HazardRule.query.filter_by(rule_code=rule_code).first()
        if row is None:
            row = HazardRule(rule_code=rule_code, rule_name=rule_name, creator_id=get_current_user_id())
            db.session.add(row)
            created += 1
        row.rule_type = item.get("ruleType") or item.get("rule_type") or "context_policy"
        row.context_id = item.get("contextId") or item.get("context_id")
        row.relation_type = item.get("relationType") or item.get("relation_type")
        row.match_mode = item.get("matchMode") or item.get("match_mode") or row.match_mode
        row.priority = int(item.get("priority", row.priority or 0))
        row.required_ppe_json = json.dumps(item.get("requiredPpeJson") or item.get("required_ppe") or [], ensure_ascii=False)
        row.trigger_relations_json = json.dumps(item.get("triggerRelationsJson") or item.get("trigger_relations") or [], ensure_ascii=False)
        row.rule_text = item.get("ruleText") or item.get("rule_text") or item.get("description")
        row.status = item.get("status", row.status or "active")
        row.version = item.get("version", row.version or "1.0.0")
    db.session.commit()
    return success({"created": created, "total": len(rows)}, msg="规则导入成功")


@rule_bp.route("/export-json", methods=["GET"])
@login_required
def export_rules_to_json():
    rows = HazardRule.query.order_by(HazardRule.priority.desc(), HazardRule.create_time.asc()).all()
    payload = {
        "rules": [row.to_dict() for row in rows]
    }
    return success(payload)
