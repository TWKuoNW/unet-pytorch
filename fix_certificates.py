import ssl
import certifi
import urllib.request

print("🌐 正在測試 HTTPS 憑證下載功能...")

# 建立一個 SSL context 使用 certifi 內建的憑證
ssl_context = ssl.create_default_context(cafile=certifi.where())

try:
    with urllib.request.urlopen("https://www.google.com", context=ssl_context) as response:
        print("✅ 憑證配置成功，連線 HTTPS 成功")
except Exception as e:
    print("❌ 憑證錯誤：", e)
