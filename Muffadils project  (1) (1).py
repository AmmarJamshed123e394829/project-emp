
# coding: utf-8

# In[74]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[75]:


pwd


# In[76]:


cd


# # Company summary 
# df = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_1.xlsx')

# In[78]:


df = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_1.xlsx')
#first 5 rows
df.head()


# In[79]:


#last 5 rows 
df.tail()


# In[80]:


#pre-prcocess the data
df.dropna()


# In[81]:


df.dtypes


# In[82]:


df['TOTAL_NO_DAYS'].max() # highest no of leaves taken inn the company


# In[83]:


df['TOTAL_NO_DAYS'].min() # highest no of leaves taken inn the company


# In[84]:


x ='TOTAL_NO_DAYS'
y = 'DEPARTMENT'
plt.scatter(x,y)
plt.title('TOTAL NUMBER OF LEAVES PER DEPARTMENT')
plt.show()


# # TM- department employees
# df2 = pd.read_excel('C:\\Users\\Hp\\Documents\\TM-E_leave.xlsx')

# In[87]:


df2 = pd.read_excel('C:\\Users\\Hp\\Documents\\TM-E_leave.xlsx')
# first 5 rows
df2.head()


# In[88]:


df2.dtypes


# In[89]:


df2['TOTAL_NO_DAYS'].max() # highest no of leaves taken inn the Tm department 


# In[90]:


df2['TOTAL_NO_DAYS'].min() # highest no of leaves taken inn the TM department


# # Accounts Department Employees
# df3 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_Accounts_dep.xlsx')

# In[91]:


df3 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_Accounts_dep.xlsx')


# In[92]:


#first 4 rows
df3.head()


# In[93]:


df3.tail()


# In[94]:


df3['TOTAL_NO_DAYS'].max() # highest no of leaves taken inn the Accounts department 


# In[95]:


df3['TOTAL_NO_DAYS'].min() # highest no of leaves taken inn the Tm department 


# # Adminsitration Department Employees
# df4 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_Administration.xlsx')

# In[97]:


df4 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_Administration.xlsx')
#first 5 rows
df4.head()


# In[98]:


df4.tail()


# In[99]:


df4['TOTAL_NO_DAYS'].max() # highest no of leaves taken inn the Tm department 


# In[100]:


df4['TOTAL_NO_DAYS'].min() # lowest no of leaves taken inn the Tm department 


# # Area managers
# df5 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_Area manager.xlsx')

# In[101]:


df5 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_Area manager.xlsx')


# In[102]:


df5.head()


# In[103]:


df5.tail()


# In[104]:


df5['TOTAL_NO_DAYS'].max() # highest no of leaves taken inn the Tm department 


# In[105]:


df4['TOTAL_NO_DAYS'].min() # lowest no of leaves taken inn the Tm department 


# # Assistant Managers

# In[106]:


df5 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_Assistant managers.xlsx')


# In[107]:


df5.head()


# In[108]:


df5.tail()


# In[109]:


df5['TOTAL_NO_DAYS'].max() # highest no of leaves taken inn the Tm department 


# In[110]:


df5['TOTAL_NO_DAYS'].min() # Lowest no of leaves taken inn the Tm department 


# # warehouse staff
# df6 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_workshop_staff.xlsx')

# In[112]:


df6 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_data_workshop_staff.xlsx')


# In[113]:


df6.head()


# In[114]:


df6.tail()


# In[115]:


df6['TOTAL_NO_DAYS'].max() # highest no of leaves taken inn the Tm department 


# In[116]:


df5['TOTAL_NO_DAYS'].min() # highest no of leaves taken inn the Tm department 


# # IT and Business Intelligence staff
# df6 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_IT and BI.xlsx')

# In[117]:


df6 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_IT and BI.xlsx')


# In[118]:


df6.head()


# In[119]:


df6.tail()


# In[120]:


df6['TOTAL_NO_DAYS'].max() # highest no of leaves taken in the department 


# In[121]:


df6['TOTAL_NO_DAYS'].min() # Lowest no of leaves taken in the department 


# # Engineering staff
# df7 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_IT and BI.xlsx')

# In[122]:


df7.head()


# In[123]:


df7.tail()


# In[124]:


df7['TOTAL_NO_DAYS'].max() # highest no of leaves taken in the department 


# In[125]:


df7['TOTAL_NO_DAYS'].min() # lowest no of leaves taken in the department 


# # Finance Department 

# In[126]:


df8 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_Finance.xlsx')


# In[127]:


df8.head()


# In[128]:


df8.tail()


# In[129]:


df8['TOTAL_NO_DAYS'].max() # Highest no of leaves taken in the department 


# In[130]:


df8['TOTAL_NO_DAYS'].min() # lowest no of leaves taken in the department 


# # Human Resource Department 

# In[131]:


df9 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_Human resource.xlsx')


# In[132]:


df9.head()


# In[133]:


df9.tail()


# In[134]:


df9['TOTAL_NO_DAYS'].max() # Highest no of leaves taken in the department 


# In[135]:


df9['TOTAL_NO_DAYS'].min() # Lowest no of leaves taken in the department 


# # Other Specialist Services - Operations, shift managers, logistics, shipping, regulatory officers, Maintainance etc

# In[136]:


df10 = pd.read_excel('C:\\Users\\Hp\\Downloads\\Eleave_other.xlsx')


# In[137]:


df10.head()


# In[138]:


df10.tail()


# In[139]:


df10['TOTAL_NO_DAYS'].max() # Highest no of leaves taken in the department - Taken By Supervisor Muhammad Atif Hussain


# In[161]:


df10['TOTAL_NO_DAYS'].min() # Lowest no of leaves taken in the department 


# In[148]:


#Polynomial regression is a special case of linear regression
#where we fit a polynomial equation on the data with a curvilinear relationship between the target variable and the independent variables
#We create the following function for X and Y along with 3 as a factor (depending upon the case).
import numpy as np
f = np.polyfit('TOTAL_NO_DAYS','APPLICANT_ID', 3)
P = np.polyld(f)
from sklearn.preprocessing Import PolynomialFeatures
pv= PolynomialFeatures (degree = 2 , include_bias =
False)
X_polly = pv.sit_transform (x ['TOTAL_NO_DAYS', 'APPLICANT_ID'])


# In[151]:


#Setup a StandardScaler
SCALE = StandardScaler ()
SCALE.fit(x_data[['TOTAL_NO_DAYS', 'APPLICANT_ID']])


# In[156]:


#we input the standard scaler, Polynomial Features and Linear Regression in the pipe line constructor
Input = [ ('scale',StandardScale()), ('Polynomial'),
PolynomialFeatures (degree 2), ('model',
LinearRegression)]


# In[157]:


#How we make predictions using the input.
Pipe = Pipeline(Input)
#Input X and Y values in the model
Pipetrain(x['TOTAL_NO_DAYS', 'APPLICANT_ID'])
#Design a regression Line
yhat = Pipe.predict (X ['TOTAL_NO_DAYS', 'APPLICANT_ID'])


# In[158]:


#How to place a graph on the data calculated for
evaluation.
• Lm.fit (df['TOTAL_NO_DAYS','APPLICANT_ID'] , df ['APPLICANT_ID'])
#Predict how much will the data calculate in the future based on constant variables calculated in the evaluation data above assuming the rate of change to be the same throughout.
Lm.predict()


# In[159]:


#Derive the coefficient of ‘lm’
Lm.coef
#Create a Regression line for the predicted data
Import numpy as np
New_input = np.arange (1 , 101 , 1) , reshape(-1 , 1)
Yhat = lm.predict(new_input)

