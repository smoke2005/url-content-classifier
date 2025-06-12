from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
import os

def scrape_and_save(url, output_text="page_text.txt", output_image="screenshot.png"):
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.7151.56 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
    )

    driver.get(url)
    WebDriverWait(driver, 15).until(lambda d: d.execute_script('return document.readyState') == 'complete')

    os.makedirs("screenshots", exist_ok=True)
    driver.save_screenshot(output_image)
    print(f"Screenshot saved to: {output_image}")

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    page_text = soup.get_text(separator="\n", strip=True)

    print("Page Title:")
    print(soup.title.string if soup.title else "No title found.")

    print("\nPage Text Preview:")
    print(page_text[:1000])  

    with open(output_text, "w", encoding="utf-8") as file:
        file.write(page_text)

    print(f"Text content saved to: {output_text}")

    driver.quit()

#if __name__ == "__main__":
#    url = "https://stripchat.global/"
 #   scrape_and_save(url, output_text="page_text.txt", output_image="page_ss.png")
