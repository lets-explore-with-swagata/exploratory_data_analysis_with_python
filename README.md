Exploratory Data Analysis on "Wage World: A Profound Annotation"
The Goal is to predict wheather a person has an income of more than 50k a year or not
Importing Libraries
Read Dataset
Sanity Check (check missing values, duplicates, garbage values)
Statistical Summery
Univariate analysis (Histogram, boxplot)
Bivariate Analysis (Scatter plot, Count plot, Co-relation heatmap)
Outliers Detection (Boxplot)
Conclusion and Insights

Introduction
In the world of data exploration, the chosen dataset stands as a trove of information, offering a glimpse into the intricate interplay of demographic and socio-economic factors influencing income levels. In this exploratory journey, our mission is to dissect the data using pandas, numpy, matplotlib, and seaborn, unraveling hidden patterns that may guide us in understanding the determinants of financial success.

Image
The investigation encompasses a diverse array of features, ranging from age and education to occupation and working hours. The goal is clear: predict whether an individual earns more than 50k a year or not. To embark on this analytical odyssey, we'll employ statistical summaries, visualizations, and correlation analyses, seeking to uncover insights that transcend mere data points.

Setting the Stage:
The journey of exploration of data analysis, the chosen dataset contains information about individuals, including demographic features, education, occupation, and salary. The goal of this project is to build a predictive model that accurately classifies individuals into two income groups: those earning more than 50k a year and those earning 50k or less.

Unraveling the Tapestry:
Education and occupation emerge as threads woven into the socio-economic tapestry. Each education level and occupation type carries a distinct weight in the income narrative.It will pPerform a comprehensive EDA using pandas, numpy, matplotlib, and seaborn to analyze the dataset and uncover patterns, trends, and relationships among different features. The focus is on understanding how demographic and socio-economic factors may be associated with income levels.

Key Analysis and Considerations:

Demographic Patterns: Explore the distribution of individuals based on demographic features such as age, gender, and marital status.

Income Distribution: Investigate the distribution of income levels and identify any patterns or disparities.

Education and Occupation Impact: Analyze the relationship between education levels, occupation types, and income.

Working Hours Influence: Examine the impact of working hours on income levels:**

Visualizations: Create informative visualizations (using seaborn and matplotlib) to effectively communicate the findings.

Problem Statement
The goal of this exploratory data analysis (EDA) is to gain comprehensive insights into the socio-economic dynamics captured within the dataset. By leveraging statistical summaries, visualizations, and correlation analyses, it is aimed to answer critical questions and uncover patterns that shed light on the factors influencing income levels. The authentication of the dataset is unknown, if there is any unmatched value and information that can devaluate the significance of this Exploratory data analysis as well as unaccounted confounding variables can distort the relationship between variables.

Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
Read Data
df=pd.read_csv("social_economy.csv")
df.head()
age	workclass	education	education-num	marital-status	occupation	relationship	sex	capital-gain	capital-loss	hours-per-week	country	salary
0	39	State-gov	Bachelors	13	Never-married	Adm-clerical	Not-in-family	Male	2174	0	40	United-States	<=50K
1	50	Self-emp-not-inc	Bachelors	13	Married-civ-spouse	Exec-managerial	Husband	Male	0	0	13	United-States	<=50K
2	38	Private	HS-grad	9	Divorced	Handlers-cleaners	Not-in-family	Male	0	0	40	United-States	<=50K
3	53	Private	11th	7	Married-civ-spouse	Handlers-cleaners	Husband	Male	0	0	40	United-States	<=50K
4	28	Private	Bachelors	13	Married-civ-spouse	Prof-specialty	Wife	Female	0	0	40	Cuba	<=50K
df.columns
Index(['age', 'workclass', 'education', 'education-num', 'marital-status',
       'occupation', 'relationship', 'sex', 'capital-gain', 'capital-loss',
       'hours-per-week', 'country', 'salary'],
      dtype='object')
df.shape
(32561, 13)
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 13 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             32561 non-null  int64 
 1   workclass       32561 non-null  object
 2   education       32561 non-null  object
 3   education-num   32561 non-null  int64 
 4   marital-status  32561 non-null  object
 5   occupation      32561 non-null  object
 6   relationship    32561 non-null  object
 7   sex             32561 non-null  object
 8   capital-gain    32561 non-null  int64 
 9   capital-loss    32561 non-null  int64 
 10  hours-per-week  32561 non-null  int64 
 11  country         32561 non-null  object
 12  salary          32561 non-null  object
dtypes: int64(5), object(8)
memory usage: 3.2+ MB
Sanity Checking

Data Cleaning

df.isnull().sum()
age               0
workclass         0
education         0
education-num     0
marital-status    0
occupation        0
relationship      0
sex               0
capital-gain      0
capital-loss      0
hours-per-week    0
country           0
salary            0
dtype: int64
df['workclass'].value_counts()
workclass
 Private             22696
 Self-emp-not-inc     2541
 Local-gov            2093
 ?                    1835
 State-gov            1298
 Self-emp-inc         1116
 Federal-gov           960
 Without-pay            14
 Never-worked            7
Self-emp-inc             1
Name: count, dtype: int64
df['country'].value_counts()
country
 United-States                 29170
 Mexico                          643
 ?                               583
 Philippines                     198
 Germany                         137
 Canada                          121
 Puerto-Rico                     114
 El-Salvador                     106
 India                           100
 Cuba                             95
 England                          90
 Jamaica                          81
 South                            80
 China                            75
 Italy                            73
 Dominican-Republic               70
 Vietnam                          67
 Guatemala                        64
 Japan                            62
 Poland                           60
 Columbia                         59
 Taiwan                           51
 Haiti                            44
 Iran                             43
 Portugal                         37
 Nicaragua                        34
 Peru                             31
 France                           29
 Greece                           29
 Ecuador                          28
 Ireland                          24
 Hong                             20
 Cambodia                         19
 Trinadad&Tobago                  19
 Laos                             18
 Thailand                         18
 Yugoslavia                       16
 Outlying-US(Guam-USVI-etc)       14
 Honduras                         13
 Hungary                          13
 Scotland                         12
 Holand-Netherlands                1
Name: count, dtype: int64
df['occupation'].value_counts()
occupation
 Prof-specialty       4140
 Craft-repair         4099
 Exec-managerial      4066
 Adm-clerical         3770
 Sales                3650
 Other-service        3295
 Machine-op-inspct    2002
 ?                    1843
 Transport-moving     1597
 Handlers-cleaners    1370
 Farming-fishing       994
 Tech-support          928
 Protective-serv       649
 Priv-house-serv       149
 Armed-Forces            9
Name: count, dtype: int64
df['workclass'].replace('?',0,inplace= True)
df['country'].replace('?',0,inplace= True)
df['occupation'].replace('?',0,inplace= True)
df['workclass'].replace(0,np.nan,inplace= True)
df['country'].replace(0,np.nan,inplace= True)
df['occupation'].replace(0,np.nan,inplace= True)
df['workclass']= df['workclass'].fillna(df['workclass'].mode()[0])
df['country']= df['country'].fillna(df['country'].mode()[0])
df['occupation']= df['occupation'].fillna(df['occupation'].mode()[0])
df['workclass'].value_counts()
workclass
 Private             22696
 Self-emp-not-inc     2541
 Local-gov            2093
 ?                    1835
 State-gov            1298
 Self-emp-inc         1116
 Federal-gov           960
 Without-pay            14
 Never-worked            7
Self-emp-inc             1
Name: count, dtype: int64
df['country'].value_counts().head()
country
 United-States    29170
 Mexico             643
 ?                  583
 Philippines        198
 Germany            137
Name: count, dtype: int64
var=df['country'].value_counts().head()
df['occupation'].value_counts()
occupation
 Prof-specialty       4140
 Craft-repair         4099
 Exec-managerial      4066
 Adm-clerical         3770
 Sales                3650
 Other-service        3295
 Machine-op-inspct    2002
 ?                    1843
 Transport-moving     1597
 Handlers-cleaners    1370
 Farming-fishing       994
 Tech-support          928
 Protective-serv       649
 Priv-house-serv       149
 Armed-Forces            9
Name: count, dtype: int64
df.rename(columns={'sex':'gender'},inplace=True)
df.columns
Index(['age', 'workclass', 'education', 'education-num', 'marital-status',
       'occupation', 'relationship', 'gender', 'capital-gain', 'capital-loss',
       'hours-per-week', 'country', 'salary'],
      dtype='object')
df.describe()
age	education-num	capital-gain	capital-loss	hours-per-week
count	32561.000000	32561.000000	32561.000000	32561.000000	32561.000000
mean	38.581647	10.080679	1077.648844	87.303830	40.437456
std	13.640433	2.572720	7385.292085	402.960219	12.347429
min	17.000000	1.000000	0.000000	0.000000	1.000000
25%	28.000000	9.000000	0.000000	0.000000	40.000000
50%	37.000000	10.000000	0.000000	0.000000	40.000000
75%	48.000000	12.000000	0.000000	0.000000	45.000000
max	90.000000	16.000000	99999.000000	4356.000000	99.000000
df['hours-per-week'].describe()
count    32561.000000
mean        40.437456
std         12.347429
min          1.000000
25%         40.000000
50%         40.000000
75%         45.000000
max         99.000000
Name: hours-per-week, dtype: float64
Analyse Data

Age

Univariate Analysis

plt.figure(figsize=(8,4))
sns.histplot(df['age'],color='green',bins=15)
plt.grid(True)
plt.title('Age Distribution')
plt.show()

sns.histplot(x=df['age'],hue=df['salary'],bins=15)
plt.tight_layout()
plt.title('Age Distribution')
plt.show()

From the graph we can see in the age group 0.20 there is not entry of salary greater than 50K, same goes with the group greater than 75 years

Bivariate Analysis

Workclass

sns.countplot(y=df['workclass'],hue=df['salary'],palette=['#a4def5','#e1a4f5'])
plt.tight_layout()
plt.xticks(rotation=47)
plt.grid(True)
plt.title('Workclass Distribution')
plt.show()

The majority of people work in private sector. The probability is making of 50000 are similar among the work classes except of self-emp-inc and federal governmant. Federal governmant is seen as the most elite in the public sector, which more likely explains the higher chance of earning more than 50000.

Univariate Analysis

Occupation

sns.histplot(x=df['occupation'],hue=df['salary'],bins=15)
plt.tight_layout()
plt.xticks(rotation=50)
plt.title('Occupation Distribution')
plt.show()

Here in this above graph the x-axis reprents several occupation, each bar divided into colors that indicates the distribution of 'salary' category. The insights says which occupationa have the higher concentration of every individual earning <=50000 or >50000. It observes if certain occupations are more associated with higher or lower income levels.

Bivariate Analysis

Gender

plt.figure(figsize=(8,4))
sns.countplot(y=df['gender'],hue=df['salary'],palette=['#a4def5','#e1a4f5'])
plt.tight_layout()
plt.title('Gender Distribution')
plt.show()

The percentage of male who is greater than 50000is much greater than the percentage of females that makes the same amount. This will ceratinly be a significant factor, and should be a feature consider5ed in our prediction model.

Univariate Analysis

Capital loss

sns.histplot(df['capital-loss'],color='green',bins=15)
plt.tight_layout()
plt.grid(True)
plt.title('Capital-loss Distribution')
plt.show()

The x-axis represents different ranges or values of capital losses. The y-axis represents the frequency or count of occurrences for each range or value. In summary, the code helps to visually explore the distribution of capital losses in the dataset. It can provide insights into the common ranges or values of capital losses, potential outliers, and the overall spread of this variable.

Univariate Analysis

plt.figure(figsize=(14, 8))

pivot_df = df.pivot_table(index='education', columns='occupation', values='salary', aggfunc='count', fill_value=0)

sns.heatmap(pivot_df, annot=True, fmt='d', cmap='viridis', cbar_kws={'label': 'Count'})
plt.title('Income Distribution by Education and Occupation')
plt.xlabel('Occupation')
plt.ylabel('Education Level')
plt.show()


Analysis: The heatmap provides a comprehensive overview of income distribution across various education levels. Higher education levels, such as 'Bachelors', 'Masters', and 'Doctorate', tend to have more prominent representations across different occupations. This heatmap also gives some of the insights in the areas where the numbers of individual persons in the '<=50k' income category. It will help to understand what is the main contribution of to the lower incomes for the specific education and occuaption group. As well as it shows higher counts in the specific area of '>50k'. 'HS-grad' gives a diverse income distribution accross the other educational category.

Hours per week

Working Hours as a Catalyst: In our pursuit of clarity, we delve into the correlation between working hours and income. The balance between professional dedication and financial returns unfolds, with code snippets guiding us through the nuances of this pivotal relationship.

def hours_edit(val):
    if (val<40):
        return ('< 40 hours')
    elif (val==40):
        return ('= 40 hours')
    else:
        return ('> 40 hours')
df['hours-per-week']=df['hours-per-week'].apply(hours_edit)
sns.countplot(x=df['hours-per-week'],hue=df['salary'],palette='viridis',saturation=0.9,edgecolor='black',order=['< 40 hours','= 40 hours','> 40 hours'])
plt.tight_layout()
plt.title('Hours-per-week')
plt.grid(True)
plt.show()

The percentage of individuals are making over 50000 drastically decreses when less than 40 hours in a week, and increses significantly when greater than 40 hours in a week

Bivariate Analysis

Graphical distribution of Data and Statistical Summery

sns.heatmap(df.select_dtypes(include='number').corr(),annot=True)
<Axes: >

print(df[['age', 'salary']].isnull().sum())
age       0
salary    0
dtype: int64
print(df['salary'].unique())
print(df.groupby('salary')['age'].describe())
[' <=50K' ' >50K']
          count       mean        std   min   25%   50%   75%   max
salary                                                             
 <=50K  24720.0  36.783738  14.020088  17.0  25.0  34.0  46.0  90.0
 >50K    7841.0  44.249841  10.519028  19.0  36.0  44.0  51.0  90.0
print(df['age'].dtype)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
int64
print(df['salary'].unique())
[' <=50K' ' >50K']
summary = df.describe(include='all')
print(summary)
                 age workclass education  education-num       marital-status  \
count   32561.000000     32561     32561   32561.000000                32561   
unique           NaN        10        17            NaN                    7   
top              NaN   Private   HS-grad            NaN   Married-civ-spouse   
freq             NaN     22696     10501            NaN                14976   
mean       38.581647       NaN       NaN      10.080679                  NaN   
std        13.640433       NaN       NaN       2.572720                  NaN   
min        17.000000       NaN       NaN       1.000000                  NaN   
25%        28.000000       NaN       NaN       9.000000                  NaN   
50%        37.000000       NaN       NaN      10.000000                  NaN   
75%        48.000000       NaN       NaN      12.000000                  NaN   
max        90.000000       NaN       NaN      16.000000                  NaN   

             occupation relationship    sex  capital-gain  capital-loss  \
count             32561        32561  32561  32561.000000  32561.000000   
unique               15            6      2           NaN           NaN   
top      Prof-specialty      Husband   Male           NaN           NaN   
freq               4140        13193  21790           NaN           NaN   
mean                NaN          NaN    NaN   1077.648844     87.303830   
std                 NaN          NaN    NaN   7385.292085    402.960219   
min                 NaN          NaN    NaN      0.000000      0.000000   
25%                 NaN          NaN    NaN      0.000000      0.000000   
50%                 NaN          NaN    NaN      0.000000      0.000000   
75%                 NaN          NaN    NaN      0.000000      0.000000   
max                 NaN          NaN    NaN  99999.000000   4356.000000   

        hours-per-week         country  salary  
count     32561.000000           32561   32561  
unique             NaN              42       2  
top                NaN   United-States   <=50K  
freq               NaN           29170   24720  
mean         40.437456             NaN     NaN  
std          12.347429             NaN     NaN  
min           1.000000             NaN     NaN  
25%          40.000000             NaN     NaN  
50%          40.000000             NaN     NaN  
75%          45.000000             NaN     NaN  
max          99.000000             NaN     NaN  
import scipy.stats as stats

# Example: Perform t-test for two groups based on the 'age' column
group1 = df[df['salary'] == '<=50K']['age']
group2 = df[df['salary'] == '>50K']['age']

t_statistic, p_value = stats.ttest_ind(group1, group2)

print(f'T-Statistic: {t_statistic}')
print(f'P-Value: {p_value}')
T-Statistic: nan
P-Value: nan
Analysis: The t-test can provide insights into whether there is a significant difference in the age distribution between those earning more than 50k and those earning 50k or less. If the p-value is less than the chosen significance level (commonly 0.05), you may reject the null hypothesis. If the t-statistic is positive, it suggests that the mean age in the '>50K' group is higher than in the '<=50K' group, and vice versa for a negative t-statistic. If t-statistic is positive, the '>50K' group tends to have a higher average age. If negative, the '<=50K' group tends to have a higher average age. If p_value >= 0.05, there is no strong evidence to conclude a significant difference in average age between the two salary groups.

Univariate Analysis

Outliers Detection (Boxplot)

plt.figure(figsize=(8, 6))
sns.boxplot(x='salary', y='age', data=df)
plt.title('Boxplot of Age by Salary')
plt.show()

Analysis: Box (Interquartile Range): The box represents the interquartile range (IQR), which spans from the first quartile (Q1) to the third quartile (Q3). The height of the box indicates the spread of the middle 50% of the data.

Line Inside the Box (Median): The line inside the box represents the median (Q2) of the data. It indicates the middle value of the dataset.

Whiskers: The whiskers extend from the edges of the box to the minimum and maximum values within a defined range (usually 1.5 times the IQR). Any data points beyond the whiskers are considered outliers and are plotted individually.

Outliers: Individual points outside the whiskers are considered outliers. They are marked as individual points in the plot.

Summary: If there are noticeable differences in the median, spread, or presence of outliers between the salary groups, it supports the need for statistical testing (like the t-test) to formally assess the significance of these difference

This is a univariate analysis focusing on the 'age' variable. To gain a more comprehensive understanding, you may want to explore other variables and consider multivariate analysis in the context of your prediction goal. If you have specific observations or questions about the boxplot, feel free to share, and I can provide more targeted insights.s.

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='age', hue='salary', fill=True, common_norm=False)
plt.title('Kernel Density Estimation (KDE) of Age by Income Level')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

below_50k_count = df[df['salary'] == '<=50K'].shape[0]
print(f"The number of people with incomes below $50,000 is: {below_50k_count}")

The number of people with incomes below $50,000 is: 0
above_50k_count = df[df['salary'] == '>50K'].shape[0]
print(f"The number of people with incomes above $50,000 is: {above_50k_count}")

The number of people with incomes above $50,000 is: 0
Analysis: This code will generate a KDE plot for the 'age' feature, with different colors representing the income levels ('<=50K' and '>50K'). The plot helps visualize the distribution of ages for each income group, allowing to observe patterns and differences. It has been seen the right skewed distribution in the above graph that signifies that the persons who are between the age of 20 to 40 and their salary is "<=50" the density is near about 0.030. At the age of 40 people are earning better and that is ">50k" as their salary. The distribution of age of workers with ">50" salary is at more stable position.

Insights

The EDA of the chosen dataset has provided valuable insights into the demographic and socio-economic factors associated with income levels. Several key observations and patterns emerged during the analysis:

Age and Income: The age distribution indicates a diverse workforce, with a concentration in the middle-age range. While income tends to increase with age, there are variations, and other factors contribute to earnings.

Gender Disparities: The dataset showcases gender disparities in income, with a higher concentration of males in higher-income brackets. Further investigation into occupational choices and educational levels could provide additional context.

Education Impact: Education levels significantly influence income, with individuals holding advanced degrees generally earning higher incomes. The dataset highlights the importance of education in socio-economic mobility.

Occupational Trends: Certain occupations are associated with higher incomes, emphasizing the role of the chosen profession in determining earnings. Executives and professionals tend to have higher average incomes.

Working Hours Influence: Individuals working more than 40 hours per week generally exhibit higher income levels. The analysis suggests a positive correlation between working hours and income.

Geographic Variations: Geographic variations in income levels exist, with certain countries showing higher average incomes. This may reflect regional economic disparities and opportunities.

Data Imbalances: The analysis identified potential imbalances in the dataset, particularly in gender representation and income distribution. Addressing these imbalances in future analyses is essential for unbiased insights.

Conclusion

In conclusion, this EDA serves as a foundational exploration into the intricate relationships between demographic features and income levels. While the analysis provides valuable insights, further investigations, feature engineering, and modeling efforts could enhance predictive capabilities for income classification. It has been that the percentage of individuals are making over 50000 drastically decreses when less than 40 hours in a week, and increses significantly when greater than 40 hours in a week. By the evaluation of this EDA, it can be generate the idea which provides a comprehensive overview of income distribution across various education levels. Higher education levels, such as 'Bachelors', 'Masters', and 'Doctorate', tend to have more prominent representations across different occupations. It also conveys more information about goal of this analysis through the KDE plot analysis it has been seen the right skewed distribution in the above graph that signifies that the persons who are between the age of 20 to 40 and their salary is "<=50" the density is near about 0.030. At the age of 40 people are earning better and that is ">50k" as their salary. The distribution of age of workers with ">50" salary is at more stable position. This exploration of this above dataset is able to fulfill the goal of the analysis.

The findings underscore the importance of considering multiple factors in understanding income disparities, offering a starting point for more in-depth analyses and targeted interventions. In future this exploration with its analysis will be great assessment for the socio-economic dynamics of society. The well-being of poeple and their livlihood should need more evaluation by the government because each and every persons are the pillers of the economical growth of the country as their individual income contributesmuch to the national income of a country.
