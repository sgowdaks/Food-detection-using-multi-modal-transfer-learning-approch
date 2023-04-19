#!/usr/bin/env python
# coding: utf-8

###  !pip install selenium
#!conda install -y -c conda-forge geckodriver firefox selenium

import os
import time
import csv
# import pyautogui

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.common.keys import Keys
from pathlib import Path

with open("categories.txt", "r") as input:
    categories = csv.reader(input)
    for category in categories:
        print(category[0].split(" ")[1])
        path = "/home/sg/work/scrapy_selenium/"
        download_dir = path + "/"+ category[0].split(" ")[1]

        download_dir = Path(download_dir)

        download_dir.mkdir(parents=True, exist_ok=True)
        download_dir.touch()
        
        time.sleep(10)
        print(str(download_dir))
        options = Options()
        options.add_argument("--disable-notifications")
        options.add_argument('--no-sandbox')
        options.add_argument('--verbose')
        options.add_argument('--headless=new')
        options.add_argument("--disable-extensions")
        options.add_argument('--disable-application-cache')
        options.add_argument("--disable-setuid-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument('--disable-gpu')
        options.add_experimental_option("prefs", {
                "download.default_directory": str(download_dir),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing_for_trusted_sources_enabled": False,
                "safebrowsing.enabled": False
        })
        options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(options = options)
        driver.maximize_window()        
        
        driver.get(<website_name>)
        driver.set_window_size(2000, 1500)

        time.sleep(30)
        elem = driver.find_element(By.ID, "food_query")
        elem.clear()
        elem.send_keys(category[0])
        elem.send_keys(Keys.RETURN)
        
        driver.execute_script("document.body.style.transform='scale(0.5)';")
        
        time.sleep(20)
        
        link = []
             
        def links():
            driver.execute_script("document.body.style.transform='scale(0.5)';")
            elems = driver.find_elements(By.XPATH, "//a[contains(@class, 'table_item_name')]")
            if len(elems) == 0:
                return -1
            for elem in elems:
                # driver.execute_script("arguments[0].scrollIntoView();", elem)
                link.append(elem.get_attribute('href'))
            return 1
        
        while links() > 0:
            try:
                driver.find_element(By.XPATH, "//a[text()='Next']").click()
            except ElementClickInterceptedException:
                print("element not found")
                break
                
        output_file = download_dir / "names.tsv"
        
        with open(output_file, "w") as output:
            csv_writer = csv.writer(output, delimiter="\t")
            for i, l in enumerate(link):
                # print(l)
                driver.get(l)
                name = driver.find_element(By.ID, "food-name").get_attribute("textContent")
                driver.find_element(By.XPATH, "//option[contains(@value, '1 kg = 1000 g')]").click()
                element = driver.find_element(By.XPATH, "//a[text()='Download spreadsheet (CSV)']")
                ActionChains(driver).move_to_element(element).click(element).perform()
                time.sleep(10)
                csv_file_path = max([os.path.join(download_dir,f) for f in os.listdir(download_dir)], key=os.path.getctime)
                file_name = csv_file_path.split("/")[-1]
                # print(file_name)
                csv_writer.writerow([name, file_name])
                time.sleep(10)
            driver.close()
