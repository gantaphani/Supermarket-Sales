# Databricks notebook source
#Importing necessary libraries
from pyspark.sql import types
from pyspark.sql.types import DoubleType,StructType,StructField,StringType,DateType,FloatType,DecimalType
from pyspark.sql import functions as f,Window

# COMMAND ----------

#Defining the schema for the dataframes
items_schema=StructType([StructField("Item Code",DoubleType(),True),
                         StructField("Item Name",StringType(),True),
                         StructField("Category Code",DoubleType(),True),
                         StructField("Category Name",StringType(),True)])

daily_sales_schema=StructType([StructField("Date",DateType(),True),
                         StructField("Time",StringType(),True),
                         StructField("Item Code",DoubleType(),True),
                         StructField("Quantity Sold(kilo)",FloatType(),True),
                         StructField("Unit Selling Price (RMB/kg)",FloatType(),True),
                         StructField("Sale or Return",StringType(),True),
                         StructField("Discount (Yes/No)",StringType(),True)])

daily_wholesaleprice_schema=StructType([StructField("Date",DateType(),True),
                         StructField("Item Code",DoubleType(),True),
                         StructField("Wholesale Price (RMB/kg)",FloatType(),True)])

item_loss_schema=StructType([StructField("Item Code",DoubleType(),True),
                              StructField("Item Name",StringType(),True),
                         StructField("Loss Rate (%)",FloatType(),True)])

# COMMAND ----------

#Loading data from files to the dataframes
items=spark.read.format("csv").option("inferschema",True).option("header",True).schema(items_schema).load("/FileStore/tables/Items.csv")
daily_sales=spark.read.format("csv").option("inferschema",True).option("header",True).schema(daily_sales_schema).load("/FileStore/tables/Daily_Sales.csv")
daily_wholesaleprice=spark.read.format("csv").option("inferschema",True).option("header",True).schema(daily_wholesaleprice_schema).load("/FileStore/tables/Daily_Wholesaleprice.csv")
item_loss=spark.read.format("csv").option("inferschema",True).option("header",True).schema(item_loss_schema).load("/FileStore/tables/Item_Loss.csv")

# COMMAND ----------

#Selecting data from dataframes
display(items)
display(daily_sales)
display(daily_wholesaleprice)
display(item_loss)

# COMMAND ----------

#Total number of items in each category
total_items_category=items.groupBy('Category Code').agg(f.count('Item Code').alias('item counts'))
display(total_items_category)

# COMMAND ----------

#Total quantity sold and revenue generated for each item
daily_sales=daily_sales.withColumn('Revenue',daily_sales['Quantity Sold(kilo)']*daily_sales['Unit Selling Price (RMB/kg)'])
total_quantities_sold_revenue_item=daily_sales.groupBy('Item Code').agg(f.sum('Quantity Sold(kilo)').cast(DecimalType(20,2)).alias('Total quantities sold'),
                                                                        f.sum('Revenue').cast(DecimalType(20,2)).alias('Total revenue'))
display(total_quantities_sold_revenue_item)

# COMMAND ----------

#Monthly sales per year
daily_sales=daily_sales.withColumn('Year',f.year('Date')).withColumn('Month',f.month('Date'))
sales_year_month=daily_sales.groupBy('year','month').agg(f.sum('Revenue').cast(DecimalType(20,2)).alias('Revenue'))
sales_year_month=sales_year_month.orderBy('year','month')
display(sales_year_month)

# COMMAND ----------

#Top selling category sold per each month in year
top_selling_category=daily_sales.join(items,daily_sales['Item Code']==items['Item Code'],'inner')
top_selling_category=top_selling_category.groupBy('year','month','category name').agg(f.sum('Revenue').cast(DecimalType(20,2)).alias('Revenue'))
w=Window.partitionBy(f.column('year'),f.column('month')).orderBy(f.column('Revenue').desc())
top_selling_category=top_selling_category.withColumn('rank',f.dense_rank().over(w))
top_selling_category=top_selling_category.orderBy('year','month')
top_selling_category=top_selling_category[top_selling_category['rank']==1]
display(top_selling_category.select('year','month','category name','Revenue'))

# COMMAND ----------

#Month with Highest number of item returns
returns_month=daily_sales[daily_sales['Sale or Return']=='return']
returns_month=returns_month.groupBy('month').agg(f.countDistinct('Item Code').alias('return counts'))
returns_month=returns_month.orderBy('return counts',ascending=False)
returns_month=returns_month.show(1)
display(returns_month)

# COMMAND ----------

#Total number of items sold above the wholesaleprice per each month in year
wholesale=daily_sales.alias('s').join(daily_wholesaleprice.alias('w'),(daily_sales.Date==daily_wholesaleprice.Date)&
                                      (daily_sales['Item Code']==daily_wholesaleprice['Item Code']),"inner").select(f.col('s.Date'),f.col('s.year'),
                                                                                                                    f.col('s.month'),f.col('s.Item Code'),f.col('s.Unit Selling Price (RMB/kg)'),
                                              f.column('w.Wholesale Price (RMB/kg)'))
wholesale=wholesale[wholesale['Unit Selling Price (RMB/kg)']>wholesale['Wholesale Price (RMB/kg)']]
wholesale=wholesale.groupBy('year','month').agg(f.countDistinct('Item Code').alias('count of Items sold above wholesale'))
wholesale=wholesale.orderBy('year','month')
display(wholesale)

# COMMAND ----------

#Total number of items per category given a discount in a year
discount=items.alias('i').join(daily_sales.alias('s'),items['Item Code']==daily_sales['Item Code'],'inner').select(
    f.column('i.Item Code'),f.column('i.Category Name'),f.column('year'),f.column('Discount (Yes/No)'))
discount=discount[discount['Discount (Yes/No)']=='Yes']
discount=discount.groupBy('year','Category Name').agg(f.countDistinct('Item Code').alias('count of discounted items'))
discount=discount.orderBy('year')
display(discount)

# COMMAND ----------

#Total number of sale and return items per each month in a year
sale_returns=daily_sales.groupBy('year','month','Sale or Return').agg(f.countDistinct('Item Code').alias('items count'))
sale_returns=sale_returns.orderBy('year','month')
display(sale_returns)

# COMMAND ----------

#Top 3 categories with the highest average loss percentage
loss=items.alias('i').join(item_loss.alias('l'),items['Item Code']==item_loss['Item Code'],'inner').select(f.column('i.Category Name'),f.column('l.Loss Rate (%)'))
loss=loss.groupBy('Category Name').agg(f.mean('Loss Rate (%)').cast(DecimalType(20,2)).alias('Loss Percentage'))
loss=loss.orderBy('Loss Percentage',ascending=False).show(3)
display(loss)

# COMMAND ----------

#Identify the item with the highest wholesale price in a year.
daily_wholesaleprice=daily_wholesaleprice.withColumn('year',f.year('Date'))
highestwholesale=daily_wholesaleprice.groupBy('year').agg(f.max('Wholesale Price (RMB/kg)').alias('highest wholesale price'))
highestwholesale=daily_wholesaleprice.alias('w').join(highestwholesale.alias('h'),(daily_wholesaleprice['year']==highestwholesale['year'])&
                                                      (daily_wholesaleprice['Wholesale Price (RMB/kg)']==highestwholesale['highest wholesale price']),'inner').select(f.column('w.year'),f.column('w.item Code'),f.column('h.highest wholesale price'))
highestwholesale=highestwholesale.distinct()
display(highestwholesale)
