from security import build_csp


def test_build_csp_hashes_inline_blocks():
    csp = build_csp("<style>a{}</style><script>alert(1)</script>")
    assert "script-src 'sha256-" in csp and "style-src 'sha256-" in csp
    assert "unsafe-inline" not in csp
    assert build_csp("<script>x</script>") != build_csp("<script>y</script>")  # 內容變了雜湊就變


def test_build_csp_without_inline_blocks():
    csp = build_csp("")
    assert "script-src 'none'" in csp and "style-src 'none'" in csp
