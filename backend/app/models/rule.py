from datetime import datetime
import json

from app.extensions import db


class HazardRule(db.Model):
    __tablename__ = "hazard_rule"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rule_code = db.Column(db.String(64), unique=True, nullable=False)
    rule_name = db.Column(db.String(128), nullable=False)
    rule_type = db.Column(db.String(32), default="context_policy")
    context_id = db.Column(db.String(64))
    relation_type = db.Column(db.String(64))
    match_mode = db.Column(db.String(32), default="any")
    priority = db.Column(db.Integer, default=0)
    required_ppe_json = db.Column(db.Text)
    trigger_relations_json = db.Column(db.Text)
    rule_text = db.Column(db.Text)
    status = db.Column(db.String(16), default="active")
    version = db.Column(db.String(32), default="1.0.0")
    remark = db.Column(db.Text)
    creator_id = db.Column(db.Integer, default=0)
    create_time = db.Column(db.DateTime, default=datetime.now)
    update_time = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "ruleCode": self.rule_code,
            "ruleName": self.rule_name,
            "ruleType": self.rule_type,
            "contextId": self.context_id,
            "relationType": self.relation_type,
            "matchMode": self.match_mode,
            "priority": self.priority,
            "requiredPpeJson": json.loads(self.required_ppe_json) if self.required_ppe_json else [],
            "triggerRelationsJson": json.loads(self.trigger_relations_json) if self.trigger_relations_json else [],
            "ruleText": self.rule_text,
            "status": self.status,
            "version": self.version,
            "remark": self.remark,
            "creatorId": self.creator_id,
            "createTime": self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
            "updateTime": self.update_time.strftime("%Y-%m-%d %H:%M:%S") if self.update_time else None,
        }
