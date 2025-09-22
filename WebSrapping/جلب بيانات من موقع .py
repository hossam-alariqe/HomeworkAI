#!/usr/bin/env python
# coding: utf-8

# In[19]:



import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

# رابط الصفحة الرئيسية للقسم
base_url = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops"

# إنشاء ملف Excel
wb = Workbook()
ws = wb.active
ws.title = "Laptops"
ws.append(["الرقم", "اسم المنتج", "السعر"])

product_count = 0
page = 1

while True:
    # تحديد الرابط مع رقم الصفحة
    if page == 1:
        url = base_url
    else:
        url = f"{base_url}?page={page}"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # البحث عن المنتجات
    products = soup.find_all("div", class_="thumbnail")
    if not products:
        break
    
    # معالجة المنتجات
    for product in products:
        product_count += 1
        
        # اسم المنتج
        name = product.find("a", class_="title")
        name = name.get_text(strip=True) if name else "غير متوفر"
        
        # السعر (الوسم الصحيح)
        price = product.find("h4", class_="price")
        price = price.get_text(strip=True) if price else "غير متوفر"
        
        print(f"{product_count}. {name} - {price}")
        ws.append([product_count, name, price])
    
    # الانتقال للصفحة التالية
    page += 1

# حفظ الملف
wb.save("all_laptops.xlsx")
print(f"تم حفظ {product_count} منتج في ملف all_laptops.xlsx")


# In[ ]:





# In[ ]:




