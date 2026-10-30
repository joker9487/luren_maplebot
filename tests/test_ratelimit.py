from datetime import datetime

import pytest

from ratelimit import TAIPEI, RateLimiter, client_ip
from scripts.load_test import percentile


class Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, seconds: float):
        self.t += seconds


def ts(*args) -> float:
    return datetime(*args, tzinfo=TAIPEI).timestamp()


@pytest.fixture
def clock():
    return Clock(ts(2026, 10, 30, 12, 0, 0))


def limiter(clock, cooldown=5, hourly=30, daily=1000):
    return RateLimiter(cooldown, hourly, daily, clock=clock)


def test_cooldown(clock):
    rl = limiter(clock)
    assert rl.check("a").allowed
    clock.tick(2)
    d = rl.check("a")
    assert (d.allowed, d.reason, d.retry_after) == (False, "cooldown", 3)
    assert d.message == "問太快了，3 秒後再問。"
    assert rl.check("b").allowed  # 別人不受影響
    clock.tick(3)
    assert rl.check("a").allowed


def test_blocked_requests_do_not_extend_cooldown(clock):
    rl = limiter(clock)
    rl.check("a")
    clock.tick(4)
    assert not rl.check("a").allowed
    clock.tick(1)
    assert rl.check("a").allowed  # 從上次「被允許」開始算 5 秒


def test_hourly_sliding_window(clock):
    rl = limiter(clock, cooldown=0, hourly=3)
    for _ in range(3):
        assert rl.check("a").allowed
        clock.tick(600)  # 0、10、20 分各問一次
    d = rl.check("a")  # 第 30 分
    assert d.reason == "hourly"
    assert d.retry_after == 1800  # 要等到第一題滿一小時（第 60 分）
    assert d.message == "這個小時問了不少題了，休息一下，約 30 分鐘後再來。"
    clock.tick(1800)
    assert rl.check("a").allowed  # 第一題滑出視窗
    assert rl.check("a").reason == "hourly"  # 10、20、60 分 → 又滿了


def test_daily_limit_is_site_wide(clock):
    rl = limiter(clock, daily=2)
    assert rl.check("a").allowed
    assert rl.check("b").allowed
    d = rl.check("c")
    assert d.reason == "daily"
    assert d.message == "今天的問答次數已經用完了，明天再來。"
    assert d.retry_after == 12 * 3600  # 中午 12 點 → 台灣時間午夜


def test_daily_resets_at_taipei_midnight():
    clock = Clock(ts(2026, 10, 30, 23, 59, 50))
    rl = limiter(clock, daily=1)
    assert rl.check("a").allowed
    assert rl.check("b").reason == "daily"
    assert rl.today.blocked["daily"] == 1
    clock.tick(10)  # 10/31 00:00:00
    assert rl.check("b").allowed
    assert rl.today.day == "2026-10-31"
    assert rl.today.asked == 1 and rl.today.blocked["daily"] == 0


def test_day_boundary_uses_taipei_not_utc():
    # 台灣 07:59 = UTC 前一天 23:59；換到 08:00 不應該歸零
    clock = Clock(ts(2026, 10, 30, 7, 59, 0))
    rl = limiter(clock, daily=1)
    rl.check("a")
    clock.tick(120)
    assert rl.check("b").reason == "daily"


def test_blocked_counts(clock):
    rl = limiter(clock, daily=1)
    rl.check("a")
    rl.check("a")
    rl.check("b")
    assert rl.today.asked == 1
    assert rl.today.blocked == {"cooldown": 0, "hourly": 0, "daily": 2}  # 每日上限優先


def test_cleanup_drops_idle_ips(clock):
    rl = limiter(clock)
    rl.check("a")
    clock.tick(1800)
    rl.check("b")
    assert rl.tracked_ips == 2
    clock.tick(1800)  # a 滿一小時沒來
    rl.check("c")  # 距上次清理超過 60 秒，會順便清
    assert rl.tracked_ips == 2  # b、c


def test_client_ip_prefers_cloudflare_header():
    assert client_ip({"cf-connecting-ip": "1.2.3.4"}, "127.0.0.1") == "1.2.3.4"
    assert client_ip({"cf-connecting-ip": "  "}, "127.0.0.1") == "127.0.0.1"
    assert client_ip({}, "127.0.0.1") == "127.0.0.1"
    assert client_ip({}, None) == "unknown"


def test_percentile():
    values = [5, 1, 4, 2, 3]
    assert percentile(values, 50) == 3
    assert percentile(values, 95) == 5
    assert percentile([7], 95) == 7


def test_rate_key_groups_ipv6_by_64():
    from ratelimit import rate_key

    assert rate_key("1.2.3.4") == "1.2.3.4"
    a = rate_key("2001:db8:abcd:12:1::1")
    assert a == "2001:db8:abcd:12::/64"
    assert rate_key("2001:db8:abcd:12:ffff:ffff:ffff:ffff") == a  # 同一段 /64 換位址 → 同一個人
    assert rate_key("2001:db8:abcd:13::1") != a  # 隔壁那段 → 不同人
    assert rate_key("::ffff:1.2.3.4") == "1.2.3.4"  # IPv4-mapped
    assert rate_key("unknown") == "unknown"
    assert rate_key("testclient") == "testclient"
