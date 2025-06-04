#baidu_search.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

class BaiduSearcher:
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
            # 使用新标签页打开链接以防止页面状态丢失
            main_window = self.driver.current_window_handle
            self.driver.execute_script(f"window.open('{url}', '_blank');")
            self.driver.switch_to.window(self.driver.window_handles[-1])
            
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(random.uniform(1, 2))

            html = self.driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            # 关闭当前标签页并切换回主窗口
            self.driver.close()
            self.driver.switch_to.window(main_window)

            # 提取段落文本
            paragraphs = soup.find_all(['p', 'article', 'div'])
            text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return text.strip()[:1000] + "..." if len(text) > 1000 else text.strip()

        except Exception as e:
            return f"❌ 无法提取页面内容: {str(e)}"
        
    def search(self, query, num_results=1, timeout=15):
        try:
            self._init_driver()

            # 访问百度
            self.driver.get("https://www.baidu.com")
            time.sleep(random.uniform(1, 3))

            # 定位百度搜索框
            try:
                search_box = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#kw"))
                )
            except Exception as e:
                return f"定位搜索框失败: {str(e)}"

            # 模拟人类输入
            self._human_interaction(search_box, query)
            search_box.send_keys(Keys.RETURN)

            # 等待结果加载
            try:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'result')]"))
                )
            except TimeoutException:
                return "结果加载超时"

            # 获取搜索结果
            search_results = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.result.c-container.new-pmd:not(.c-container-ad)"))
            )[:num_results]

            # 第一阶段：收集所有结果的基本信息
            results_info = []
            for result in search_results:
                try:
                    # 使用相对定位获取元素
                    title_element = result.find_element(By.CSS_SELECTOR, "h3.t>a")
                    title = title_element.text
                    link = title_element.get_attribute("href")
                    
                    try:
                        abstract = result.find_element(By.CSS_SELECTOR, "div.c-span-last span.content-right").text
                    except NoSuchElementException:
                        abstract = "无摘要内容"
                    
                    results_info.append({
                        "title": title,
                        "link": link,
                        "abstract": abstract
                    })
                except Exception as e:
                    print(f"解析结果时出错: {str(e)}")
                    continue

            # 第二阶段：逐个提取页面内容
            final_results = []
            for idx, info in enumerate(results_info[:num_results], 1):
                full_content = self.extract_page_content(info["link"])
                final_results.append(
                    f"结果 {idx}:\n标题: {info['title']}\n链接: {info['link']}\n摘要: {info['abstract']}\n详细内容:\n{full_content}\n"
                )

            return "\n".join(final_results) if final_results else "未找到相关结果"

        except Exception as e:
            return f"搜索出错: {str(e)}"
        finally:
            if self.driver:
                self.driver.quit()

# 测试函数
def test_baidu_search():
    print("=== 百度搜索测试 ===")

    # 测试1: 基础搜索
    print("\n测试1: 基础搜索")
    searcher = BaiduSearcher(headless=False)
    results = searcher.search("pagerank", 3)
    print(results)

if __name__ == "__main__":
    test_baidu_search()