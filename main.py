"""Bot 余额感知：对话/语音/视觉额度与可用性监控，异常时用自己的口吻私聊主人。

检查三类能力（各自独立，缺配置就跳过）：
- **对话**（DeepSeek）：`GET /user/balance` 查余额，低于预警线说"饿了"；
- **视觉**（智谱）：账户报告接口 + 一次免费模型探活。余额 0 但探活成功不报
  （GLM-Flash 系列免费，余额为 0 也能用，只报警额会误伤）；
- **语音**（火山）：没有余额查询接口，改看最近合成记录，连续失败即"嗓子哑了"。

只在状态**变化**时提醒（并带同类冷却），避免反复打扰；恢复时也说一句。
提醒发到 owner_session 或管理员私聊，走 context.send_message（主动消息通道）。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import aiohttp

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.config.astrbot_config import AstrBotConfig

from .library import (
    CAP_LLM,
    CAP_VISION,
    CAP_VOICE,
    STATE_OK,
    compose_message,
    default_voice_log_path,
    format_balance,
    grade_balance,
    grade_voice,
    parse_deepseek_balance,
    parse_zhipu_balance,
    severity,
    should_notify,
)

NAME = "astrbot_plugin_snow_balance"
DEEPSEEK_BALANCE = "https://api.deepseek.com/user/balance"
ZHIPU_REPORT = "https://open.bigmodel.cn/api/biz/account/query-customer-account-report"
STATE_FILE = "balance_state.json"


class SnowBalance(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self._task: asyncio.Task | None = None
        self._states: dict[str, str] = {}
        self._last_notify: dict[str, float] = {}
        self._last_detail: dict[str, str] = {}
        self._loaded = False

    # ---------------- 配置 ----------------
    def _enabled(self) -> bool:
        return bool(self.conf.get("enabled", True))

    def _cfg(self, key, default):
        return self.conf.get(key, default)

    def _data_dir(self) -> Path:
        try:
            return Path(str(StarTools.get_data_dir(NAME)))
        except Exception:
            path = Path("data") / "plugin_data" / NAME
            path.mkdir(parents=True, exist_ok=True)
            return path

    # ---------------- 状态持久化 ----------------
    def _load(self) -> None:
        if self._loaded:
            return
        try:
            data = json.loads((self._data_dir() / STATE_FILE).read_text(encoding="utf-8"))
            self._states = dict(data.get("states") or {})
            self._last_notify = {k: float(v) for k, v in (data.get("last_notify") or {}).items()}
            self._last_detail = dict(data.get("last_detail") or {})
        except Exception:
            self._states, self._last_notify, self._last_detail = {}, {}, {}
        self._loaded = True

    def _save(self) -> None:
        try:
            self._data_dir().mkdir(parents=True, exist_ok=True)
            (self._data_dir() / STATE_FILE).write_text(
                json.dumps(
                    {
                        "states": self._states,
                        "last_notify": self._last_notify,
                        "last_detail": self._last_detail,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug(f"[{NAME}] 状态保存失败：{exc}")

    # ---------------- 凭证 ----------------
    def _provider_key(self, source_id: str) -> str:
        if not source_id:
            return ""
        try:
            cfg = self.context.get_config()
            for source in cfg.get("provider_sources", []) or []:
                if isinstance(source, dict) and source.get("id") == source_id:
                    key = source.get("key")
                    if isinstance(key, list) and key:
                        return str(key[0])
                    return str(key or "")
        except Exception as exc:
            logger.debug(f"[{NAME}] 读取凭证失败：{exc}")
        return ""

    def _owner_umo(self) -> str:
        configured = str(self._cfg("owner_session", "") or "").strip()
        if configured:
            return configured
        try:
            cfg = self.context.get_config()
            admins = cfg.get("admins_id") or []
            platform = ""
            for p in cfg.get("platform", []) or []:
                if isinstance(p, dict) and p.get("type") == "aiocqhttp":
                    platform = str(p.get("id") or "")
                    break
            if admins and platform:
                return f"{platform}:FriendMessage:{admins[0]}"
        except Exception:
            pass
        return ""

    # ---------------- 检查项 ----------------
    async def _check_llm(self, session) -> tuple[str, str]:
        source = str(self._cfg("llm_provider_source", "deepseek_default_source") or "").strip()
        if not source:
            return STATE_OK, ""
        key = self._provider_key(source)
        if not key:
            return STATE_OK, ""
        timeout = aiohttp.ClientTimeout(total=25)
        try:
            async with session.get(
                DEEPSEEK_BALANCE,
                headers={"Authorization": f"Bearer {key}"},
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    return "down", ""
                payload = await resp.json(content_type=None)
        except Exception as exc:
            logger.info(f"[{NAME}] 对话余额查询失败：{exc}")
            return "down", ""
        available, total, currency = parse_deepseek_balance(payload)
        if not available and total <= 0:
            return "empty", format_balance(total, currency)
        state = grade_balance(total, float(self._cfg("llm_low_balance", 5.0)))
        return state, format_balance(total, currency)

    async def _check_vision(self, session) -> tuple[str, str]:
        source = str(self._cfg("vision_provider_source", "snowbot-caption_source") or "").strip()
        if not source:
            return STATE_OK, ""
        key = self._provider_key(source)
        if not key:
            return STATE_OK, ""
        detail = ""
        balance = None
        try:
            async with session.get(
                ZHIPU_REPORT,
                headers={"Authorization": f"Bearer {key}"},
                timeout=aiohttp.ClientTimeout(total=25),
            ) as resp:
                if resp.status == 200:
                    payload = await resp.json(content_type=None)
                    ok, value, currency = parse_zhipu_balance(payload)
                    if ok:
                        balance = value
                        detail = format_balance(value, currency)
        except Exception as exc:
            logger.debug(f"[{NAME}] 视觉余额查询失败：{exc}")

        # 探活：免费模型余额 0 也能用，所以**探活结果优先于余额**
        alive, kind = await self._vision_probe(session, key)
        if not alive and kind == "rate_limited":
            # 429 / 1305「访问量过大」是免费模型的临时限流，不代表不可用，
            # 短暂重试一次，仍被限流就保持上一次状态，不报警。
            await asyncio.sleep(3)
            alive, kind = await self._vision_probe(session, key)
            if not alive and kind == "rate_limited":
                logger.info(f"[{NAME}] 视觉探活被限流（429），本次跳过判定")
                return self._states.get(CAP_VISION, STATE_OK), detail
        if not alive:
            return ("down" if balance is None else "empty"), detail
        # 探活成功 = 视觉真的可用。余额再低也不报（否则免费模型会天天误报"看不清"）
        return STATE_OK, detail

    async def _vision_probe(self, session, key: str) -> tuple[bool, str]:
        """用免费视觉模型发一个最小请求，判断视觉链路是否真的可用。

        返回 (是否可用, 失败类型)；失败类型 "rate_limited" 表示临时限流（不是真故障）。
        """
        try:
            async with session.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "glm-4v-flash",
                    "messages": [{"role": "user", "content": "1"}],
                    "max_tokens": 1,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    return True, ""
                body = (await resp.text())[:200]
                logger.info(f"[{NAME}] 视觉探活失败 HTTP {resp.status}: {body}")
                if resp.status == 429 or "1305" in body or "访问量过大" in body:
                    return False, "rate_limited"
                if resp.status in (401, 403):
                    return False, "auth"
                return False, "error"
        except Exception as exc:
            logger.info(f"[{NAME}] 视觉探活异常：{exc}")
            return False, "error"

    def _check_voice(self) -> tuple[str, str]:
        if not bool(self._cfg("voice_enabled", True)):
            return STATE_OK, ""
        raw_path = str(self._cfg("voice_log_path", "") or "").strip()
        path = (
            Path(raw_path)
            if raw_path
            else default_voice_log_path(
                str(StarTools.get_data_dir("astrbot_plugin_seed_audio_tts"))
            )
        )
        try:
            if not path.exists():
                return STATE_OK, ""
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
            rows = [json.loads(line) for line in lines if line.strip()]
        except Exception:
            return STATE_OK, ""
        if not rows:
            return STATE_OK, ""
        recent = rows[-12:]
        results = [bool(r.get("ok")) for r in recent]
        state = grade_voice(results)
        if state == STATE_OK:
            return STATE_OK, ""
        last_err = next((str(r.get("error") or "") for r in reversed(recent) if not r.get("ok")), "")
        detail = "连续失败" if state == "down" else "多数失败"
        if last_err:
            detail = detail + "：" + last_err[:60]
        return state, detail

    # ---------------- 通知 ----------------
    async def _notify(self, capability: str, state: str, detail: str) -> None:
        umo = self._owner_umo()
        if not umo:
            logger.info(f"[{NAME}] 没有可用的主人会话，跳过提醒（{capability}/{state}）")
            return
        text = compose_message(capability, state, detail if bool(self._cfg("include_detail", True)) else "")
        label = {CAP_LLM: "对话", CAP_VOICE: "语音", CAP_VISION: "视觉"}.get(capability, capability)
        try:
            delivered = await self.context.send_message(
                umo, MessageChain(chain=[Plain(text)])
            )
            if delivered is False:
                logger.warning(
                    f"[{NAME}] 提醒未送达：找不到可用平台（QQ 可能未登录）→ {umo}"
                )
                return
            logger.info(f"[{NAME}] 已提醒主人：{label} {state}（{detail or '无细节'}）")
        except Exception as exc:
            logger.warning(f"[{NAME}] 提醒发送失败：{exc or type(exc).__name__}")

    # ---------------- 主循环 ----------------
    async def initialize(self) -> None:
        self._load()
        self._task = asyncio.create_task(self._loop())

    async def terminate(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _loop(self) -> None:
        await asyncio.sleep(60)  # 启动后缓一分钟，等 provider 就绪
        while True:
            try:
                await self._check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[{NAME}] 检查失败：{exc}")
            await asyncio.sleep(max(5, int(self._cfg("check_interval_minutes", 30))) * 60)

    async def _check_once(self) -> None:
        if not self._enabled():
            return
        self._load()
        now = time.time()
        cooldown = float(self._cfg("notify_cooldown_hours", 6.0))
        async with aiohttp.ClientSession() as session:
            checks = [
                (CAP_LLM, await self._check_llm(session)),
                (CAP_VISION, await self._check_vision(session)),
                (CAP_VOICE, self._check_voice()),
            ]
        for capability, (state, detail) in checks:
            previous = self._states.get(capability)
            # 只在恶化或恢复时提醒，避免"好→好"抖动
            if state != STATE_OK and previous and severity(state) < severity(previous):
                state = previous  # 不因短暂好转就报恢复，等真恢复
            self._states[capability] = state
            self._last_detail[capability] = detail
            if should_notify(
                previous, state, self._last_notify.get(capability, 0.0), now, cooldown
            ):
                await self._notify(capability, state, detail)
                self._last_notify[capability] = now
        self._save()
