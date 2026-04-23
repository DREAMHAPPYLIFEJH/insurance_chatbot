"""
동양생명 보험 상품 PDF 자동 다운로드
실행: python crawl_pdfs.py

다운로드 대상: 상품요약서 PDF (전체 상품)
저장 경로: ./pdfs/
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DOWNLOAD_DIR = os.path.abspath("./pdfs")
BASE_URL     = "https://pbano.myangel.co.kr/paging/WE_AC_WEPAAP020100L"
TOTAL_PAGES  = 11  # 109개 상품 / 10개씩 = 11페이지

# ─────────────────────────────────────────
# 드라이버 초기화
# ─────────────────────────────────────────
def init_driver():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # PDF 자동 다운로드 설정 (뷰어 대신 바로 저장)
    prefs = {
        "download.default_directory":         DOWNLOAD_DIR,
        "download.prompt_for_download":        False,
        "download.directory_upgrade":          True,
        "plugins.always_open_pdf_externally":  True,  # PDF 뷰어 안 열고 바로 다운로드
    }
    options.add_experimental_option("prefs", prefs)

    # 크롬 146 이상 → 내장 드라이버 자동 사용 (별도 설치 불필요)
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    return driver


# ─────────────────────────────────────────
# 페이지에서 상품요약서 버튼 클릭
# ─────────────────────────────────────────
def download_page(driver, page_num):
    print(f"\n  📄 {page_num}페이지 처리 중...")

    # 1페이지는 기본 URL, 2페이지부터 JS로 페이지 이동
    if page_num == 1:
        driver.get(BASE_URL)
        time.sleep(2)
    else:
        driver.execute_script(f"PP_Query('mainform', 'dw_99', '{page_num}');")
        time.sleep(2)

    # 테이블 행 수집
    rows = driver.find_elements(By.CSS_SELECTOR, "table tr")
    count = 0

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) < 5:
            continue

        product_name = cols[3].text.strip()
        if not product_name:
            continue

        # alt="상품요약서 다운로드" 이미지의 부모 a 태그로 찾기
        try:
            summary_img = row.find_element(By.CSS_SELECTOR, "img[alt='상품요약서 다운로드']")
            summary_btn = summary_img.find_element(By.XPATH, "..")  # 부모 a 태그
            href        = summary_btn.get_attribute("href")         # javascript:MasFiledownload(...)

            if href and "MasFiledownload" in href:
                print(f"    ⬇️  {product_name} 다운로드 중...")
                driver.execute_script(href.replace("javascript:", ""))
                time.sleep(1.5)  # 다운로드 간격
                count += 1

        except Exception as e:
            print(f"    ⚠️  {product_name} 상품요약서 없음: {e}")
            continue

    print(f"  ✅ {page_num}페이지 완료 ({count}개)")
    return count


# ─────────────────────────────────────────
# 다운로드 완료 대기
# ─────────────────────────────────────────
def wait_for_downloads(timeout=30):
    """모든 .crdownload(다운로드 중) 파일이 사라질 때까지 대기"""
    start = time.time()
    while time.time() - start < timeout:
        downloading = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".crdownload")]
        if not downloading:
            break
        time.sleep(1)


# ─────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print("  동양생명 보험 상품 PDF 크롤러 시작")
    print(f"  저장 경로: {DOWNLOAD_DIR}")
    print("=" * 45)

    driver      = init_driver()
    total_count = 0

    try:
        # 1페이지 먼저 접속
        driver.get(BASE_URL)
        time.sleep(2)

        for page in range(1, TOTAL_PAGES + 1):
            count        = download_page(driver, page)
            total_count += count

        # 마지막 다운로드 완료 대기
        print("\n  ⏳ 다운로드 완료 대기 중...")
        wait_for_downloads()

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise

    finally:
        driver.quit()

    # 결과 확인
    pdf_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".pdf")]
    print("\n" + "=" * 45)
    print(f"  ✅ 크롤링 완료!")
    print(f"  📁 저장된 PDF: {len(pdf_files)}개")
    print(f"  📂 저장 경로: {DOWNLOAD_DIR}")
    print("=" * 45)