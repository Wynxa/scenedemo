from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text
from werkzeug.security import generate_password_hash

from app.extensions import db

LOGGER = logging.getLogger(__name__)


class SchemaSyncError(RuntimeError):
    pass


def initialize_database(flask_app) -> None:
    """Bootstrap database tables, sync columns, and seed default records."""
    auto_init = flask_app.config.get("AUTO_INIT_DB", True)
    auto_sync = flask_app.config.get("AUTO_SYNC_DB_COLUMNS", True)
    auto_seed = flask_app.config.get("AUTO_SEED_DB", True)

    if not _is_enabled(auto_init):
        LOGGER.info("AUTO_INIT_DB disabled, skip database bootstrap.")
        return

    with flask_app.app_context():
        import app.models  # noqa: F401  # ensure model metadata is registered
        from app.models.model_registry import ModelRegistry
        from app.models.rule import HazardRule
        from app.models.user import User

        db.create_all()
        LOGGER.info("Database tables checked/created.")

        if _is_enabled(auto_sync):
            synced_columns = sync_missing_columns()
            if synced_columns:
                LOGGER.info("Database columns auto-added: %s", ", ".join(synced_columns))

        if _is_enabled(auto_seed):
            changed = seed_default_records(
                app=flask_app,
                User=User,
                ModelRegistry=ModelRegistry,
                HazardRule=HazardRule,
            )
            if changed:
                db.session.commit()
                LOGGER.info("Database seed completed.")
            else:
                db.session.rollback()
                LOGGER.info("Database seed already up to date.")


def _is_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def sync_missing_columns() -> list[str]:
    inspector = inspect(db.engine)
    synced: list[str] = []

    for table in db.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue
            ddl = build_add_column_ddl(table.name, column)
            db.session.execute(text(ddl))
            synced.append(f"{table.name}.{column.name}")

    if synced:
        db.session.commit()
    return synced


def build_add_column_ddl(table_name: str, column) -> str:
    compiled_type = column.type.compile(dialect=db.engine.dialect)
    parts = [f"ALTER TABLE `{table_name}` ADD COLUMN `{column.name}` {compiled_type}"]

    default_clause = build_default_clause(column)
    if default_clause:
        parts.append(default_clause)

    if column.nullable:
        parts.append("NULL")
    else:
        parts.append("NOT NULL")
        if not default_clause:
            raise SchemaSyncError(
                f"Column {table_name}.{column.name} is NOT NULL but has no default; auto sync would be unsafe."
            )

    return " ".join(parts)


def build_default_clause(column) -> str:
    if column.server_default is not None:
        raw = str(column.server_default.arg)
        return f"DEFAULT {raw}"

    if column.default is None or not getattr(column.default, "is_scalar", False):
        return ""

    value = column.default.arg
    if value is None:
        return "DEFAULT NULL"
    if isinstance(value, bool):
        return f"DEFAULT {1 if value else 0}"
    if isinstance(value, (int, float)):
        return f"DEFAULT {value}"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"DEFAULT '{escaped}'"
    return ""


def seed_default_records(*, app, User, ModelRegistry, HazardRule) -> bool:
    changed = False

    if not User.query.filter_by(username="admin").first():
        db.session.add(
            User(
                username="admin",
                password=generate_password_hash("admin123"),
                real_name="系统管理员",
                role="admin",
                status="0",
            )
        )
        changed = True

    if not User.query.filter_by(username="inspector").first():
        db.session.add(
            User(
                username="inspector",
                password=generate_password_hash("inspector123"),
                real_name="安全检查员",
                role="inspector",
                status="0",
            )
        )
        changed = True

    default_model = ModelRegistry.query.filter_by(name="Helmet-YOLO-Default").first()
    if not default_model:
        db.session.add(
            ModelRegistry(
                name="Helmet-YOLO-Default",
                version="1.0.0",
                checkpoint_path="checkpoints/helmet_yolo.pt",
                config_json=json.dumps(
                    {
                        "confidenceThreshold": 0.5,
                        "iouThreshold": 0.45,
                        "imageSize": 640,
                        "classNames": ["helmet", "person", "vest"],
                    },
                    ensure_ascii=False,
                ),
                service_type="yolo_detection",
                runtime_type="local_python",
                metrics_json=json.dumps(
                    {
                        "mAP50": 0.912,
                        "mAP50_95": 0.687,
                        "precision": 0.894,
                        "recall": 0.876,
                    },
                    ensure_ascii=False,
                ),
                status="active",
            )
        )
        changed = True

    if sync_hazard_rules_from_builtin(app=app, HazardRule=HazardRule):
        changed = True

    return changed


def sync_hazard_rules_from_builtin(*, app, HazardRule) -> bool:
    project_root = Path(app.root_path).parent
    rule_file = project_root / "resources" / "hazard_reasoning" / "hazard_context_policies_v4.json"
    if not rule_file.exists():
        LOGGER.warning("Hazard rule seed file not found: %s", rule_file)
        return False

    payload = json.loads(rule_file.read_text(encoding="utf-8"))
    contexts = payload.get("contexts", [])
    changed = False

    for item in contexts:
        rule_code = item.get("context_id")
        if not rule_code:
            continue

        row = HazardRule.query.filter_by(rule_code=rule_code).first()
        if row is None:
            row = HazardRule(rule_code=rule_code, creator_id=0)
            db.session.add(row)
            changed = True

        required_ppe = json.dumps(item.get("required_ppe", []), ensure_ascii=False)
        trigger_relations = json.dumps(item.get("trigger_relations", []), ensure_ascii=False)
        rule_name = item.get("context_text") or rule_code
        rule_text = item.get("description") or item.get("context_text") or rule_code

        changed = _assign_if_changed(row, "rule_name", rule_name) or changed
        changed = _assign_if_changed(row, "rule_type", "context_policy") or changed
        changed = _assign_if_changed(row, "context_id", item.get("context_id")) or changed
        changed = _assign_if_changed(row, "relation_type", item.get("relation_type")) or changed
        changed = _assign_if_changed(row, "match_mode", item.get("match_mode", "any")) or changed
        changed = _assign_if_changed(row, "priority", int(item.get("priority", 0))) or changed
        changed = _assign_if_changed(row, "required_ppe_json", required_ppe) or changed
        changed = _assign_if_changed(row, "trigger_relations_json", trigger_relations) or changed
        changed = _assign_if_changed(row, "rule_text", rule_text) or changed
        changed = _assign_if_changed(row, "status", "active") or changed
        changed = _assign_if_changed(row, "version", payload.get("meta", {}).get("version", "1.0.0")) or changed
        changed = _assign_if_changed(row, "remark", "auto synced from hazard_context_policies_v4.json") or changed

    return changed


def _assign_if_changed(row, field_name: str, new_value: Any) -> bool:
    if getattr(row, field_name) != new_value:
        setattr(row, field_name, new_value)
        return True
    return False
