"""Unified API response format matching RuoYi frontend expectations."""

from flask import jsonify


def success(data=None, msg="操作成功"):
    """Success response: {code: 200, msg, data}"""
    return jsonify({"code": 200, "msg": msg, "data": data})


def error(msg="操作失败", code=500):
    """Error response: {code, msg}"""
    return jsonify({"code": code, "msg": msg})


def table_data(rows, total, msg="查询成功"):
    """Paginated table response: {code: 200, msg, rows, total}"""
    return jsonify({"code": 200, "msg": msg, "rows": rows, "total": total})


def unauthorized(msg="未登录或登录已过期"):
    return jsonify({"code": 401, "msg": msg})
