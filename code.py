#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # **Part I: Research Question**
# 
# ## Research Question
# 
# My data set for this data mining exercise includes data on a telecommunications company’s sales history, dating back to 2 years prior, with a focus on technology related merchandise.  Data analysis performed on the dataset will be aimed with this research question in mind: what are the top 3 association rules we can determine based on the raw sales data?  The telecommunications company's data is only loosely organized and will require some cleaning and restructuring.

# ---
# 
# ## Objectives and Goals
# 
# The goal of my data analysis will be to determine 3 rules best suited to illustrate the relationships between items frequently purchased together, and offer advice on how those rules and other insights might be actionable by the telecommunications company.

# ---
# 
# # **Part II: Market Basket Justification**
# 
# ## Market Basket Analysis
# 
# Market Basket Analysis is a technique used by retailers to determine associations between items. The algorithm discovers associations between different items and products that may be purchased together.  This helps retailers to make business and marketing decision, such as the right product placement or promotions likely to succeed. The algorithm presents this information as association rules, which can be thought of as "if, then" type rules.  The two components of these rules are the antecedent (the "if" component) and the consequent (the "then" component) (Deb, 2019).
# 
# The quality of the association rules mined by the algorithm is determined by three metrics:
# 
# * Support - the fraction of transactions which contain item "A" and "B". Support reveals the frequently bought items or combinations of items.
# * Confidence - how often the items "A" and "B" are purchased together, based on the number times "A" is purchased.
# * Lift - the strength of a rule over random instances of "A" and "B". Lift is commonly used as the authoritative indicator of how strong a rule is.
# 
# The Apriori algorithm, which I'll be using for this market basket analysis, begins by identifying frequently purchased individual items in a data set of transactions.  Each item is assigned a "support" measure, which again is determined by how frequently the item is purchased.  It then proceeds to take items that meet a minimum support threshold and looks for frequent item combinations, grouping them into item sets.  This process continues until the algorithm can no longer find larger item sets that meet the minimum support threshold.  Association rules can then be created using minimum threshold values for the other metrics, "confidence" and "lift" (Deb, 2019).
# 
# The expected outcome of this exercise will be a group of association rules that exhibit the strongest values for support, confidence and lift.  The insights provided by these rules can be used to drive business related decisions.
# 
# 
# ## Transactions
# 
# 
# Transactions within the transformed data set will be more clearly shown further on in this document, but one example of a transaction that would appear in the data is shown here, where a customer purchased three items: "Apple Lightning to Digital AV Adapter", "Apple Pencil", and "TP-Link AC1750 Smart WiFi Router", denoted by the "TRUE" values in the columns for those items.
# 
# ---
# 
# 

# 1	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	<span style="color:green">TRUE</span>	FALSE	FALSE	<span style="color:green">TRUE</span>	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	<span style="color:green">TRUE</span>	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE	FALSE
# 

# ---
# 
# ## Assumption
# 
# 
# Market basket analysis assumes that if an item set is frequent, then the subsets of that item set (whether that consists of an individual item or multiple items) must also be frequent (Ranjan, 2020).

# ---
# 
# # **Part III: Data Preparation and Analysis**
# 
# C.  Prepare and perform market basket analysis by doing the following:
# 
# 1.  Transform the dataset to make it suitable for market basket analysis. Include a copy of the cleaned dataset.
# 
# 2.  Execute the code used to generate association rules with the Apriori algorithm. Provide screenshots that demonstrate the error-free functionality of the code.
# 
# 3.  Provide values for the support, lift, and confidence of the association rules table.
# 
# 4.  Identify the top three rules generated by the Apriori algorithm. Include a screenshot of the top rules along with their summaries.
# 
# 
# My first steps will be to import the Python libraries needed for my data analysis and then import the complete data set and execute functions that will give me information on its size and the data types of its variables.

# In[1]:


# Imports and housekeeping
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[2]:


# Import the main dataset
df = pd.read_csv('teleco_market_basket.csv')


# In[3]:


# Column names, non-null counts and dtypes
df.info()


# In[4]:


# Preview top 5 rows
df.head()


# In[5]:


# Show column data types
print(df.dtypes)


# In[6]:


# Dimensions of data set
df.shape


# ---
# 
# Once this is done, I determine whether null data points exist in the data set, and if so, I remove them.

# In[7]:


# Check the data frame for null values
print(df.isnull().sum())


# In[8]:


# Drop null values from the data frame
df = df.dropna(how = 'all')


# ---
# 
# Reviewing the changes made to the data set by removing the null data points, I see that the data set size has been reduced from 15001 rows to 7501.

# In[9]:


# Dimensions of data set with no nulls
df.shape


# ---
# 
# With my null values removed, I can proceed with transactionalizing the data set.  This is done by creating an array of data points from the data set, then fitting and transforming the array of data points using mlxtend's TransactionEncoder function.  I will create a new data frame from the transactionalized data, named "prep_df".

# In[10]:


# Initialize "trans" array and populate with data points
trans = []
for i in range (0, 7501):
    trans.append([str(df.values[i, j]) for j in range (0,20)])


# In[11]:


# Transactionalize the data in the "trans" array
te = TransactionEncoder()
array = te.fit(trans).transform(trans)


# In[12]:


# Create a data frame from the transactionalized data
prep_df = pd.DataFrame(array, columns = te.columns_)
prep_df


# ---
# 
# With my data transformed, I will check the columns of the new dataframe to see if any null (nan) columns are present.

# In[13]:


# List columns in data frame
for col in prep_df.columns:
    print(col)


# ---
# 
# There is one null column (third from the bottom), so I will remove it and check my columns once more to ensure no nulls exist.

# In[14]:


# Remove null columns from data frame
prep_df = prep_df.drop(['nan'], axis = 1)


# In[15]:


# List columns in data frame
for col in prep_df.columns:
    print(col)


# ---
# 
# ## Copy of Prepared Data Set
# 
# With my data set cleaned and prepared I will export the data frame.  Below is the code used to export the prepared data set to CSV format.

# In[16]:


# Export prepared dataframe to csv
prep_df.to_csv(r'C:\Users\wstul\d212\transactions_cleaned.csv')


# ---
# 
# I can now begin data mining using the Apriori algorithm.  The first step will be to determine which items within the transactionalized data meet a minimum "support" threshold, in this case 0.05, meaning the items are included in no fewer than 5% of purchases.

# In[17]:


# Narrow the data set using a support value of 0.05 as the cutoff
fi = apriori(prep_df, min_support = 0.05, use_colnames = True)
fi


# ---
# 
# With this new set of "frequent items", I can use Apriori to data mine association rules.  I want a view of the strongest rules only, so I set a minimum "lift" threshold of 1.

# In[18]:


# Mine association rules using a lift value of 1 as the cutoff
rules = association_rules(fi, metric = 'lift', min_threshold = 1)
rules


# ---
# 
# With a small set of association rules to work with, I can determine the strongest rules by eliminating those rules that have a "lift" value less than 1.15 and a "confidence" value less than 0.26, then list the top 3 rules resulting from the data mining.

# In[19]:


# List the top 3 rules using a lift threshold of 1.15 and a confidence threshold of 0.26
rules[(rules['lift'] >= 1.15) &
      (rules['confidence'] >= 0.26)].nlargest(n = 3, columns = 'lift')


# ---
# 
# The rules can be summarized as follows (values rounded to 2 decimals):
# 
# 1. IF **"VIVO Dual LCD Monitor Desk mount"** is purchased THEN **"Dust-Off Compressed Gas 2 pack"** is also purchased
# 
#     lift = 1.44, confidence = 0.34, support = 0.06
#     
# 2. IF **"HP 61 ink"** is purchased THEN **"Dust-Off Compressed Gas 2 pack"** is also purchased
# 
#     lift = 1.35, confidence = 0.32, support = 0.05
#     
# 3. IF **"Apple Pencil"** is purchased THEN **"Dust-Off Compressed Gas 2 pack"** is also purchased
# 
#     lift = 1.19, confidence = 0.28, support = 0.05
#     

# ---
# 
# # **Part IV: Data Summary and Implications**
# 
# ## Support, Lift, and Confidence
# 
# To recap from my earlier explanation of Apriori association rule mining:
# 
# * Support - the fraction of transactions which contain item "A" and "B". Support reveals the frequently bought items or combinations of items.
# * Confidence - how often the items "A" and "B" are purchased together, based on the number times "A" is purchased.
# * Lift - the strength of a rule over random instances of "A" and "B". Lift is commonly used as the authoritative indicator of how strong a rule is.
# 
# Based on my results, in order of "lift" measure descending:
# 
# 1. Customers purchase a VIVO Dual LCD Monitor Desk mount together with a Dust-Off Compressed Gas 2 pack 6% of the time, and if a customer purchases a VIVO Dual LCD Monitor Desk mount, there is a 34% likelihood they will also purchase a Dust-Off Compressed Gas 2 pack.
# 2. Customers purchase an HP 61 ink together with a Dust-Off Compressed Gas 2 pack 5% of the time, and if a customer purchases an HP 61 ink, there is a 32% likelihood they will also purchase a Dust-Off Compressed Gas 2 pack.
# 3. Customers purchase an Apple Pencil together with a Dust-Off Compressed Gas 2 pack 5% of the time, and if a customer purchases an Apple Pencil, there is a 28% likelihood they will also purchase a Dust-Off Compressed Gas 2 pack.
# 
# "Dust-Off Compressed Gas 2 pack" appears to be not only a frequently purchased item (28% of purchases, regardless of associations), but is frequently purchased with other products.  Due to its position as the "consequent" in each of these rules, we might conclude that customers purchase these as an "add-on" while they are already shopping for other items.
# 
# Based upon these insights it might prove advantageous to position the "Dust-Off Compressed Gas 2 pack" in multiple locations, such as on endcaps throughout the store.  Additionally the retailer could position items they would like to sell more of near the "Dust-Off Compressed Gas 2 pack" in its department of the store.

# ---
# 
# # **Part V: Demonstration**
# 
# **Panopto Video Recording**
# 
# A link for the Panopto video has been provided separately.  The demonstration includes the following:
# 
# •  Demonstration of the functionality of the code used for the analysis
# 
# •  Identification of the version of the programming environment
# 

# ---
# 
# # **Web Sources**
# 
# https://www.section.io/engineering-education/apriori-algorithm-in-python/
# 
# https://medium.com/edureka/apriori-algorithm-d7cc648d4f1e
# 

# ---
# 
# # **References**
# 
# 
# Deb, S.  (2019, June 20).  *Apriori Algorithm — Know How to Find Frequent Itemsets.*  Medium. https://medium.com/edureka/apriori-algorithm-d7cc648d4f1e
# 
# 
# Ranjan, A.  (2020, December 3).  *Apriori Algorithm in Association Rule Learning.*  Medium.  https://medium.com/analytics-vidhya/apriori-algorithm-in-association-rule-learning-9287fe17e944
# 
