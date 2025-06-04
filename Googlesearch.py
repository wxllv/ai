#search.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
import time
import random
from bs4 import BeautifulSoup

import warnings
# 1. 禁用所有Python警告
warnings.filterwarnings("ignore")

class GoogleSearcher:
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None

    def _init_driver(self):
        options = Options()

        # 反检测配置
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        # 随机用户代理
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        ]
        options.add_argument(f"user-agent={random.choice(user_agents)}")

        # 其他设置
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

        # 隐藏自动化特征
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
            window.navigator.chrome = {
                runtime: {},
            };
            """
        })

    def _human_interaction(self, element, text):
        """模拟人类输入行为"""
        for char in text:
            element.send_keys(char)
            time.sleep(random.uniform(0.05, 0.2))
        time.sleep(random.uniform(0.5, 1.5))


    def extract_page_content(self, url, timeout=10):
        """提取网页正文内容"""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(random.uniform(1, 2))

            html = self.driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            # 提取段落文本
            paragraphs = soup.find_all(['p'])
            text = "\n".join(p.get_text() for p in paragraphs if p.get_text(strip=True))
            return text.strip()[:1000] + "..." if len(text) > 1000 else text.strip()

        except Exception as e:
            return f"❌ 无法提取页面内容: {str(e)}"
        
    def search(self, query, num_results=3, timeout=15):
        try:
            self._init_driver()

            # 访问Google
            self.driver.get("https://www.google.com/ncr")
            time.sleep(random.uniform(1, 3))

            # 定位搜索框（多策略兼容）
            try:
                search_box = WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.NAME, "q"))
                )
            except:
                search_box = WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.XPATH, "//textarea[@name='q']"))
                )

            # 模拟人类输入
            self._human_interaction(search_box, query)
            search_box.send_keys(Keys.RETURN)

            # 等待结果（多策略兼容）
            try:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
                )
            except:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class,'g ')]"))
                )

            # 获取结果
            results = []
            search_results = self.driver.find_elements(By.XPATH, "//div[contains(@class,'g ') or @class='g']")[:num_results]
            
            result_list = []
            for i, result in enumerate(search_results, 1):
                try:
                    title = result.find_element(By.XPATH, ".//h1|.//h2|.//h3").text
                    link = result.find_element(By.XPATH, ".//a[@href]").get_attribute("href")
                    result_list.append( (title, link) )
                except Exception as e:
                    print(f"提取结果时出错: {str(e)}")
                    continue
            
            for i, (title, link) in enumerate(result_list, 1):
                try:
                    content = self.extract_page_content(link)
                    results.append(f"结果 {i}:\n标题: {title}\n链接: {link}\n内容摘要:\n{content}\n")
                except Exception as e:
                    print(f"提取内容时出错: {str(e)}")
                continue
            
            return "\n".join(results) if results else "未找到结果"

        except TimeoutException:
            self.driver.save_screenshot("search_timeout.png")
            return "错误：页面加载超时（截图已保存）"
        except Exception as e:
            return f"搜索出错: {str(e)}"
        finally:
            if self.driver:
                self.driver.quit()

# 测试函数
def test_google_search():
    print("=== Google搜索测试 ===")

    # 测试1: 基础搜索
    print("\n测试1: 基础搜索")
    searcher = GoogleSearcher(headless=True)  # 关闭无头模式便于调试
    results = searcher.search("Python最新特性", 2)
    print(results)

    # 测试2: 长尾关键词
    print("\n测试2: 长尾关键词")
    results = searcher.search("Python 3.12有什么新特性", 1)
    print(results)

    # 测试3: 特殊字符
    print("\n测试3: 特殊字符")
    results = searcher.search("Python+机器学习", 1)
    print(results)

    # 测试4: 无结果查询
    print("\n测试4: 无结果查询")
    results = searcher.search("asdfghjkl123456", 1)
    print(results)


def functional_test():
    searcher = GoogleSearcher(headless=False)

    test_cases = [
        ("Python", 3, "正常短关键词"),
        ("Python 3.12的新特性有哪些", 2, "长尾关键词"),
        ("2023年最佳编程语言", 1, "趋势性查询"),
        ("asdfghjkl123456", 1, "无结果查询")
    ]

    for query, num, desc in test_cases:
        print(f"\n测试用例: {desc}")
        print(f"查询: '{query}' | 预期结果数: {num}")
        start = time.time()
        results = searcher.search(query, num)
        elapsed = time.time() - start

        print(f"实际结果 (耗时{elapsed:.2f}s):")
        print(results if results else "无结果返回")

        # 简单验证
        if "错误" in results or ("未找到" in results and "无结果查询" not in desc):
            print("❌ 测试失败")
        else:
            print("✅ 测试通过")

def robustness_test():
    searcher = GoogleSearcher()

    # 边界值测试
    edge_cases = [
        ("", 1, "空查询"),
        ("a"*1000, 1, "超长查询"),
        ("@#$%^&*", 1, "特殊字符"),
        ("Python\n换行", 1, "非常规字符")
    ]

    for query, num, desc in edge_cases:
        print(f"\n边界测试: {desc}")
        results = searcher.search(query, num)
        print(f"输入: '{query[:20]}...' | 结果: {results[:50]}...")

        if "错误" not in results:
            print("✅ 处理成功")
        else:
            print("❌ 处理失败")

def performance_test():
    searcher = GoogleSearcher()
    queries = ["机器学习", "数据分析", "人工智能", "大数据", "深度学习"]

    total_time = 0
    success_count = 0

    for i, query in enumerate(queries, 1):
        print(f"\n测试 {i}/{len(queries)}: '{query}'")
        start = time.time()
        results = searcher.search(query, 1)
        elapsed = time.time() - start

        if "错误" not in results and "未找到" not in results:
            success_count += 1
            status = "✅"
        else:
            status = "❌"

        print(f"状态: {status} | 耗时: {elapsed:.2f}s | 结果: {results[:30]}...")
        total_time += elapsed

        # 随机延迟避免封禁
        time.sleep(random.uniform(3, 10))

    print(f"\n测试完成 | 成功率: {success_count}/{len(queries)}")
    print(f"平均耗时: {total_time/len(queries):.2f}s")
if __name__ == "__main__":
    test_google_search()
    #functional_test()
    #robustness_test()
    #performance_test()