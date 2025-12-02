# -*- coding: utf-8 -*-
import os, re, time, json, argparse, random
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.gushiwen.cn/shiwens/default.aspx"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/118.0.0.0 Safari/537.36",
    "Referer": "https://www.gushiwen.cn/",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# Ensure all data is saved to 诗文数据 directory (absolute path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "诗文数据"))
os.makedirs(ROOT_DIR, exist_ok=True)

def get_save_directory() -> str:
    """Return the absolute path where all poem data will be saved."""
    return ROOT_DIR

# 会话 + 基础重试
session = requests.Session()
retries = Retry(total=5, backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"])
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

def safe_name(s: str) -> str:
    return "".join(c for c in s.strip() if c not in r'\/:*?"<>|')

def get_with_retry(url: str, max_retries: int = 5, base_sleep: float = 3.0) -> requests.Response:
    for i in range(1, max_retries+1):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            if len(r.content) < 512:
                raise Exception("响应过短")
            return r
        except Exception as e:
            wait = base_sleep + random.random()*2
            print(f"⚠️ 列表页第 {i}/{max_retries} 次失败：{e}，等待 {wait:.1f}s")
            time.sleep(wait)
    raise Exception(f"多次重试失败：{url}")

def fetch_ajax_translation(fid: str) -> str:
    for url in [f"https://www.gushiwen.cn/nocdn/ajaxfanyi.aspx?id={fid}",
                f"https://so.gushiwen.cn/nocdn/ajaxfanyi.aspx?id={fid}"]:
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and r.text.strip():
                return BeautifulSoup(r.text, "lxml").get_text("\n", strip=True)
        except: pass
    return ""

def parse_inline_translation(soup: BeautifulSoup) -> str:
    for block in soup.select("div.contyishang"):
        text = block.get_text("\n", strip=True)
        if text and len(text) > 10:
            return text
    return ""

def fetch_detail(detail_url: str):
    r = session.get(detail_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = soup.select_one("h1").get_text(strip=True)
    author = soup.select_one("p.source").get_text(strip=True) if soup.select_one("p.source") else ""
    content = soup.select_one("div.contson").get_text("\n", strip=True)

    translation = parse_inline_translation(soup)
    if not translation:
        btn = soup.find("a", onclick=re.compile(r"fanyiShow\(\d+\)"))
        if btn and btn.has_attr("onclick"):
            m = re.search(r"fanyiShow\((\d+)\)", btn["onclick"])
            if m:
                fid = m.group(1)
                translation = fetch_ajax_translation(fid)
    if not translation:
        translation = "暂无翻译"

    return title, author, content, translation

def write_three_files(base_path: str, title: str, author: str, original: str, translation: str):
    poem_dir = os.path.join(base_path, safe_name(title))
    os.makedirs(poem_dir, exist_ok=True)

    with open(os.path.join(poem_dir, "原文.txt"), "w", encoding="utf-8") as f:
        f.write(f"标题：{title}\n作者：{author}\n\n{original}\n")

    with open(os.path.join(poem_dir, "译文.txt"), "w", encoding="utf-8") as f:
        f.write(f"标题：{title}\n作者：{author}\n\n{translation}\n")

    with open(os.path.join(poem_dir, "原文译文穿插.txt"), "w", encoding="utf-8") as f:
        f.write(f"标题：{title}\n作者：{author}\n\n")
        ori_lines = [ln for ln in original.split("\n") if ln.strip()]
        trans_lines = [ln for ln in translation.split("\n") if ln.strip()] if translation != "暂无翻译" else []
        max_len = max(len(ori_lines), len(trans_lines))
        for i in range(max_len):
            if i < len(ori_lines):
                f.write(ori_lines[i] + "\n")
            if i < len(trans_lines):
                f.write("译文：" + trans_lines[i] + "\n")

def state_file(category: str) -> str:
    return os.path.join(SCRIPT_DIR, f"crawl_state_{category}.json")

def failed_file(category: str) -> str:
    return os.path.join(SCRIPT_DIR, f"failed_links_{category}.json")

def load_state(category: str) -> dict:
    f = state_file(category)
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"page": 1}

def save_state(category: str, state: dict):
    with open(state_file(category), "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)

def load_failed(category: str) -> list:
    f = failed_file(category)
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []

def save_failed(category: str, failed_links: list):
    with open(failed_file(category), "w", encoding="utf-8") as fh:
        json.dump(failed_links, fh, ensure_ascii=False)

def crawl_pages(category: str, start_page: int = 1):
    state = load_state(category)
    page = max(start_page, state.get("page", 1))
    failed_links = load_failed(category)

    while True:
        list_url = f"{BASE}?xstr={category}&page={page}"
        try:
            r = get_with_retry(list_url)
        except Exception as e:
            print(f"❌ 列表页请求失败 {list_url} - {e}")
            break

        soup = BeautifulSoup(r.text, "lxml")
        links = [a["href"] for a in soup.select("div.sons a[href*='shiwenv_']")]
        if not links:
            print("没有更多诗文，爬取结束。")
            break

        for href in links:
            detail_url = href if href.startswith("http") else "https://www.gushiwen.cn" + href
            try:
                title, author, original, translation = fetch_detail(detail_url)
                write_three_files(ROOT_DIR, title, author, original, translation)
                print(f"✅ {title}")
            except Exception as e:
                print(f"❌ {detail_url} - {e}")
                failed_links.append(detail_url)
            time.sleep(2)

        save_state(category, {"page": page})
        save_failed(category, failed_links)
        print(f"--- 第 {page} 页完成，已保存断点和失败链接 ---")

        next_btn = soup.find("a", string="下一页")
        if next_btn:
            page += 1
            time.sleep(3)
        else:
            print("📖 已到最后一页，爬取完成。")
            break

    if failed_links:
        print(f"🔄 开始重试 {len(failed_links)} 个失败链接...")
        still_failed = []
    for url in failed_links:
        try:
            title, author, original, translation = fetch_detail(url)
            write_three_files(ROOT_DIR, title, author, original, translation)
            print(f"✅ 重试成功：{title}")
        except Exception as e:
            print(f"❌ 重试失败 {url} - {e}")
            still_failed.append(url)
        time.sleep(2)
        save_failed(category, still_failed)

def main():
    # Show save directory at startup
    print("=" * 60)
    print(f"📁 数据保存目录: {ROOT_DIR}")
    print("=" * 60)
    
    print("\n请选择要爬取的类型：")
    print("1. 诗")
    print("2. 词")
    print("3. 曲")
    print("4. 文言文")
    print("5. 全部（依次爬取所有类型）")
    print("0. 退出")
    
    choice = input("\n请输入选项 (0-5): ").strip()
    
    category_map = {
        "1": "诗",
        "2": "词", 
        "3": "曲",
        "4": "文言文"
    }
    
    if choice == "0":
        print("已退出。")
        return
    elif choice == "5":
        # Crawl all categories
        for cat_name in ["诗", "词", "曲", "文言文"]:
            print(f"\n{'='*60}")
            print(f"🔄 开始爬取：{cat_name}")
            print(f"📁 保存到: {ROOT_DIR}")
            print(f"{'='*60}")
            crawl_pages(cat_name)
    elif choice in category_map:
        cat_name = category_map[choice]
        print(f"\n{'='*60}")
        print(f"🔄 开始爬取：{cat_name}")
        print(f"📁 保存到: {ROOT_DIR}")
        print(f"{'='*60}")
        crawl_pages(cat_name)
    else:
        print("无效选项，请重新运行程序。")
        return
    
    # Summary
    if os.path.exists(ROOT_DIR):
        poem_count = len([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
        print(f"\n{'='*60}")
        print(f"✅ 爬取完成！")
        print(f"📁 数据目录: {ROOT_DIR}")
        print(f"📚 共有 {poem_count} 篇诗文")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
