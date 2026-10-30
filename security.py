"""安全相關的 HTTP header。

- 所有回應：不准被嵌進別人的網頁（點擊劫持）、瀏覽器不准亂猜檔案類型、不帶完整來源網址、關掉用不到的裝置權限
- 首頁（HTML）另外加 CSP：只准執行首頁本身那段 inline script / style（用 sha256 雜湊指定），
  其他來源的腳本、被插進頁面的腳本都不會執行
"""

from __future__ import annotations

import base64
import hashlib
import re

from starlette.datastructures import MutableHeaders

COMMON_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    # 瀏覽器只會在 HTTPS 下採用（經過 Cloudflare 時）；本機 http://localhost 不受影響
    "Strict-Transport-Security": "max-age=15552000",
}

_SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.S)
_STYLE_RE = re.compile(r"<style>(.*?)</style>", re.S)


def _hashes(blocks: list[str]) -> str:
    return " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(b.encode("utf-8")).digest()).decode() + "'" for b in blocks
    )


def build_csp(html: str) -> str:
    """（純函式）依首頁內容產生 CSP；改了 index.html 重啟就會重算，不用手動維護雜湊。"""
    scripts = _hashes(_SCRIPT_RE.findall(html)) or "'none'"
    styles = _hashes(_STYLE_RE.findall(html)) or "'none'"
    return "; ".join(
        [
            "default-src 'self'",
            f"script-src {scripts}",
            f"style-src {styles}",
            "img-src 'self' data:",
            "connect-src 'self'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ]
    )


class SecurityHeadersMiddleware:
    """純 ASGI middleware：只在送出 response header 時補上 header，不碰 body，串流（SSE）不受影響。"""

    def __init__(self, app, csp: str, csp_paths: tuple[str, ...] = ("/",)):
        self.app = app
        self.csp = csp
        self.csp_paths = csp_paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        add_csp = scope.get("path") in self.csp_paths

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in COMMON_HEADERS.items():
                    headers.setdefault(name, value)
                if add_csp:
                    headers["Content-Security-Policy"] = self.csp
            await send(message)

        await self.app(scope, receive, send_with_headers)
