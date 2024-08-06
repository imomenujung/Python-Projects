import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

url = "https://www.tokopedia.com/search?st=product&q=handphone&srp_component_id=02.01.00.00&srp_page_id=&srp_page_title=&navsource="

driver = webdriver.Firefox()
driver.get(url)

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#zeus-root")))
time.sleep(2)

produk_list = []
harga_list = []
lokasi_list = []
toko_list = []
rating_list = []
terjual_list = []

for i in range(10):
    # Scroll down to load more products
    for _ in range(17):
        driver.execute_script("window.scrollBy(0, 250)")
        time.sleep(1)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    for item in soup.findAll('div', class_='KeRtO-dE1UuZjZ6nmihLZw=='):
        produk = item.find('span', class_='OWkG6oHwAppMn1hIBsC3pQ==').text
        harga = item.find('div', class_='_8cR53N0JqdRc+mQCckhS0g==').text
        
        lokasi = item.find('span', class_='-9tiTbQgmU1vCjykywQqvA==').text if item.find('span', class_='-9tiTbQgmU1vCjykywQqvA==') else ''
        toko = item.find('span', class_='X6c-fdwuofj6zGvLKVUaNQ==').text if item.find('span', class_='X6c-fdwuofj6zGvLKVUaNQ==') else ''
        rating = item.find('span', class_='nBBbPk9MrELbIUbobepKbQ==').text if item.find('span', class_='nBBbPk9MrELbIUbobepKbQ==') else ''
        terjual = item.find('span', class_='eLOomHl6J3IWAcdRU8M08A==').text if item.find('span', class_='eLOomHl6J3IWAcdRU8M08A==') else ''
        
        produk_list.append(produk)
        harga_list.append(harga)
        lokasi_list.append(lokasi)
        toko_list.append(toko)
        rating_list.append(rating)
        terjual_list.append(terjual)

    # Scroll down to the bottom to find the "Next Page" button
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    # Wait for the "Next Page" button to be clickable
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Laman berikutnya']"))).click()
    time.sleep(3)

driver.close()

data = {
    'Produk': produk_list,
    'Harga': harga_list,
    'Lokasi': lokasi_list,
    'Toko': toko_list,
    'Rating': rating_list,
    'Terjual': terjual_list
}

df = pd.DataFrame(data)

# Tampilkan DataFrame
print(df)

# Simpan DataFrame ke file CSV
df.to_csv("produk_tokopedia.csv", index=False)
