#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[4]:


import pandas as pd
from sklearn.linear_model import LinearRegression

# قراءة البيانات
df = pd.read_csv("C:/Users/ComputerWorld/Desktop/mobiles.csv")

# تقسيم X و y
X = df[['RAM', 'Storage', 'Camera', 'Battery']]
y = df['Price']

# تدريب النموذج
model = LinearRegression()
model.fit(X, y)

# إدخال المواصفات من المستخدم
ram = int(input("ادخل حجم الرام (RAM بالجيجا): "))
storage = int(input("ادخل حجم التخزين (Storage بالجيجا): "))
camera = int(input("ادخل دقة الكاميرا (Camera بالميجا بكسل): "))
battery = int(input("ادخل حجم البطارية (mAh): "))

# التوقع
pred_price = model.predict([[ram, storage, camera, battery]])

print(f"\nالسعر المتوقع للجوال هو: {pred_price[0]:.2f} دولار")


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.xlabel('Storage')
plt.ylabel('price')
plt.scatter(df.Storage,df.Price,color='red',marker='.')


# In[ ]:




