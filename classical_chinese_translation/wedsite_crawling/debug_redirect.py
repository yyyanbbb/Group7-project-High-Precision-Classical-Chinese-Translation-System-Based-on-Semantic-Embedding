import requests
from bs4 import BeautifulSoup

BASE = "https://www.gushiwen.cn"
HEADERS = {"User-Agent": "Mozilla/5.0"}

url = f"{BASE}/shiwens/default.aspx?page=1"
r = requests.get(url, headers=HEADERS, timeout=10)
print("实际访问地址：", r.url, " 状态码：", r.status_code)

soup = BeautifulSoup(r.text, "lxml")
links = soup.select("div.sons a[href*='shiwenv_']")
print(f"共找到 {len(links)} 个诗文链接")
for a in links[:10]:
    print(a.get_text(strip=True), "->", BASE + a["href"])
