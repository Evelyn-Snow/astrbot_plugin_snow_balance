"""余额与可用性感知的纯逻辑：状态分级、拟人化文案、余额解析。

设计要点：
- 余额低 ≠ 不可用：智谱的 GLM-Flash 系列是免费模型，余额 0 也能正常调用。
  所以「余额」和「实际可用性」分开判断，避免误报。
- 文案按「能力 + 严重程度」分级，带具体数值，不消耗 LLM 生成。
"""

from __future__ import annotations

import re

# 能力标识
CAP_LLM = "llm"
CAP_VOICE = "voice"
CAP_VISION = "vision"

# 状态
STATE_OK = "ok"
STATE_LOW = "low"
STATE_EMPTY = "empty"
STATE_DOWN = "down"

# 每类能力在各级别下的人格化说法（第一人称，符合她的语气）
_MESSAGES: dict[str, dict[str, str]] = {
    CAP_LLM: {
        # 主模型：脑子/精神力
        STATE_LOW: "肚子有点饿了，脑子转得比平时慢半拍…（对话额度还剩 {detail}）",
        STATE_EMPTY: "饿得厉害，快没力气思考了…（对话额度只剩 {detail}，得先充点钱）",
        STATE_DOWN: "脑袋空空的，什么都想不起来…（对话服务连不上了）",
    },
    CAP_VOICE: {
        # 语音：嗓子
        STATE_LOW: "嗓子有点干，说话有点费劲…（语音额度剩 {detail}）",
        STATE_EMPTY: "嗓子哑了，暂时说不出话…（语音额度见底了，{detail}）",
        STATE_DOWN: "嗓子完全哑了，一句话都说不出来…（语音服务连不上了）",
    },
    CAP_VISION: {
        # 视觉：眼睛
        STATE_LOW: "眼睛有点花，看东西得凑很近…（视觉额度剩 {detail}）",
        STATE_EMPTY: "眼前一片模糊，看不太清楚了…（视觉额度见底了）",
        STATE_DOWN: "眼睛睁不开了，什么都看不见…（视觉服务连不上了）",
    },
}

# 恢复时的说法
_RECOVERED = {
    CAP_LLM: "休息了一会儿，脑子清爽多了，能好好聊天啦。",
    CAP_VOICE: "嗓子润开了，又能正常说话啦。",
    CAP_VISION: "眼睛缓过来了，看东西清楚了。",
}

_CNY_LOW_RE = re.compile(r"\d+")


def parse_deepseek_balance(payload: dict) -> tuple[bool, float, str]:
    """解析 DeepSeek /user/balance → (是否可用, 余额数值, 币种)。"""
    if not isinstance(payload, dict):
        return False, 0.0, "CNY"
    infos = payload.get("balance_infos") or []
    info = infos[0] if isinstance(infos, list) and infos else {}
    currency = str(info.get("currency") or "CNY")
    try:
        total = float(info.get("total_balance") or 0)
    except (TypeError, ValueError):
        total = 0.0
    return bool(payload.get("is_available")), total, currency


def parse_zhipu_balance(payload: dict) -> tuple[bool, float, str]:
    """解析智谱账户报告 → (接口是否成功, 可用余额, 币种)。"""
    if not isinstance(payload, dict):
        return False, 0.0, "CNY"
    ok = str(payload.get("code")) == "200" or payload.get("msg") == "操作成功"
    data = payload.get("data") or {}
    raw = data.get("availableBalance")
    if raw is None:
        raw = data.get("balance") or 0
    try:
        value = float(raw or 0)
    except (TypeError, ValueError):
        value = 0.0
    return ok, value, "CNY"


def grade_balance(balance: float, low_threshold: float) -> str:
    """按余额分级：>阈值 ok，>0 low… 见下方规则。"""
    if balance > float(low_threshold):
        return STATE_OK
    if balance > 0:
        return STATE_LOW
    return STATE_EMPTY


def grade_voice(recent_results: list[bool], down: bool = False) -> str:
    """按最近的语音合成结果分级：连续失败→down，多数失败→empty，偶发失败→low。"""
    if down:
        return STATE_DOWN
    if not recent_results:
        return STATE_OK
    recent = recent_results[-6:]
    failures = sum(1 for ok in recent if not ok)
    if failures == len(recent):
        return STATE_DOWN
    if failures * 2 > len(recent):
        return STATE_EMPTY
    if failures:
        return STATE_LOW
    return STATE_OK


def should_notify(
    prev_state: str | None, new_state: str, last_notify_ts: float, now: float,
    cooldown_hours: float,
) -> bool:
    """状态恶化要通知；恢复也通知一次；同一状态在冷却期内不重复打扰。"""
    if new_state == STATE_OK:
        # 只有从异常恢复时才说一句
        if prev_state in (STATE_LOW, STATE_EMPTY, STATE_DOWN):
            return (now - last_notify_ts) >= 600  # 恢复消息 10 分钟冷却即可
        return False
    if prev_state == new_state:
        return (now - last_notify_ts) >= max(0.1, float(cooldown_hours)) * 3600
    return True


def severity(state: str) -> int:
    return {STATE_OK: 0, STATE_LOW: 1, STATE_EMPTY: 2, STATE_DOWN: 3}.get(state, 0)


def compose_message(capability: str, state: str, detail: str = "") -> str:
    """生成人格化提醒文本；恢复状态返回恢复语。"""
    if state == STATE_OK:
        return _RECOVERED.get(capability, "恢复了。")
    table = _MESSAGES.get(capability) or {}
    template = table.get(state)
    if not template:
        return f"（{capability} 状态异常：{state}）"
    return template.format(detail=detail or "不多了")


def format_balance(value: float, currency: str = "CNY") -> str:
    symbol = "¥" if str(currency).upper() in ("CNY", "RMB") else f"{currency} "
    if value >= 100:
        return f"{symbol}{value:.0f}"
    if value >= 1:
        return f"{symbol}{value:.2f}"
    return f"{symbol}{value:.3f}"
