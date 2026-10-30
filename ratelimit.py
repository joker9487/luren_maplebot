"""限流（純邏輯，時間可注入方便測試）。

- 依使用者 IP：冷卻秒數、每小時上限（滑動視窗）
- 全站：每日上限（台灣時間每天 0 點歸零）

IP 只存在記憶體，不寫入任何檔案或資料庫；超過一小時沒來的 IP 會被清掉。
"""

from __future__ import annotations

import ipaddress
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Callable, Mapping

TAIPEI = timezone(timedelta(hours=8))
HOUR = 3600
CLEANUP_INTERVAL = 60

MESSAGES = MappingProxyType({
    "cooldown": "問太快了，{n} 秒後再問。",
    "hourly": "這個小時問了不少題了，休息一下，約 {n} 分鐘後再來。",
    "daily": "今天的問答次數已經用完了，明天再來。",
})

# 許願（想問的問題）用的訊息
SUGGEST_MESSAGES = MappingProxyType({
    "cooldown": "送太快了，{n} 秒後再送。",
    "hourly": "這個小時已經送了不少了，謝謝！約 {n} 分鐘後再來。",
    "daily": "今天收到的許願已經滿了，明天再來。",
})


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str | None = None  # cooldown / hourly / daily
    retry_after: int = 0  # 秒
    messages: Mapping[str, str] = field(default=MESSAGES, compare=False, repr=False)

    @property
    def message(self) -> str:
        if self.reason is None:
            return ""
        minutes = max(1, math.ceil(self.retry_after / 60))
        return self.messages[self.reason].format(n=self.retry_after if self.reason == "cooldown" else minutes)


@dataclass
class DayStats:
    day: str = ""
    asked: int = 0
    blocked: dict[str, int] = field(default_factory=lambda: {"cooldown": 0, "hourly": 0, "daily": 0})


def client_ip(headers: Mapping[str, str], peer: str | None) -> str:
    """經過 Cloudflare Tunnel 時真正的 IP 在 CF-Connecting-IP；沒有就用連線來源。"""
    ip = (headers.get("cf-connecting-ip") or "").strip()
    return ip or peer or "unknown"


def rate_key(ip: str) -> str:
    """限流用的 key：IPv4 一個 IP 算一個人；IPv6 整個 /64 算一個人。

    家用和手機網路通常一戶分到一整段 /64，裡面的位址可以隨便換，
    如果每個 IPv6 位址分開算，有心人換個位址就能繞過冷卻和每小時上限。
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if addr.version == 6:
        if addr.ipv4_mapped:  # ::ffff:1.2.3.4 其實是 IPv4
            return str(addr.ipv4_mapped)
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


def _day_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, TAIPEI).strftime("%Y-%m-%d")


def _seconds_to_midnight(ts: float) -> int:
    now = datetime.fromtimestamp(ts, TAIPEI)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, math.ceil((tomorrow - now).total_seconds()))


class RateLimiter:
    def __init__(
        self,
        cooldown_seconds: float,
        hourly_limit: int,
        daily_limit: int,
        clock: Callable[[], float] = time.time,
        messages: Mapping[str, str] = MESSAGES,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.messages = messages
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}  # IP → 最近一小時內被允許的時間
        self._last_cleanup = clock()
        self.today = DayStats(day=_day_of(clock()))

    def check(self, ip: str) -> Decision:
        """判斷這次能不能問；允許的話順便記下來。被擋的不算次數。"""
        now = self._clock()
        self._roll_day(now)
        if now - self._last_cleanup >= CLEANUP_INTERVAL:
            self.cleanup(now)

        decision = self._decide(ip, now)
        if decision.allowed:
            self._hits.setdefault(ip, deque()).append(now)
            self.today.asked += 1
        else:
            self.today.blocked[decision.reason] += 1
        return decision

    def _decide(self, ip: str, now: float) -> Decision:
        if self.today.asked >= self.daily_limit:
            return Decision(False, "daily", _seconds_to_midnight(now), self.messages)

        hits = self._hits.get(ip)
        if hits:
            while hits and now - hits[0] >= HOUR:
                hits.popleft()
        if hits:
            since_last = now - hits[-1]
            if since_last < self.cooldown_seconds:
                return Decision(False, "cooldown", max(1, math.ceil(self.cooldown_seconds - since_last)), self.messages)
            if len(hits) >= self.hourly_limit:
                return Decision(False, "hourly", max(1, math.ceil(hits[0] + HOUR - now)), self.messages)
        return Decision(True)

    def _roll_day(self, now: float) -> None:
        day = _day_of(now)
        if day != self.today.day:
            self.today = DayStats(day=day)

    def cleanup(self, now: float | None = None) -> None:
        """清掉一小時內都沒來的 IP，避免記憶體一直長。"""
        now = self._clock() if now is None else now
        for ip in [ip for ip, hits in self._hits.items() if not hits or now - hits[-1] >= HOUR]:
            del self._hits[ip]
        self._last_cleanup = now

    @property
    def tracked_ips(self) -> int:
        return len(self._hits)
