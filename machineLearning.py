import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import utils
from sklearn.svm import SVC
from sklearn import tree
# import graphviz

# Read Files
consoleSales = pd.read_csv('consoleSales.csv')
videoSales = pd.read_csv('Video_Game_Sales_as_of_Jan_2017.csv')

# Drop the NA values
videoSales = videoSales.dropna()

# Select all cases where PUBLISHER are nintende, microsoft, sony
videoSales = videoSales.loc[videoSales['Publisher'].isin(['Nintendo','Microsoft Game Studios'
                            ,'Sony Computer Entertainment'])]

# Select all cases where yeaR is greater than 2007
videoSales = videoSales [videoSales['Year_of_Release'] > 2007.0]
videoSales.drop(['Rating', 'Other_Sales', 'JP_Sales', 'EU_Sales', 'NA_Sales','Genre'], axis=1, inplace=True)

# Group by year and publisher and sum of Global Sales
videoSalesSum = videoSales.groupby(['Year_of_Release','Publisher'])['Global_Sales'].sum().reset_index()
# bar1=sns.barplot(x=videoSalesSum['Year_of_Release'], 
#            y=videoSalesSum['Global_Sales'],
#            hue=videoSalesSum['Publisher'], data=videoSalesSum,
#            hue_order=['Nintendo','Sony Computer Entertainment','Microsoft Game Studios'])
# plt.show(bar1)

# Console sales bar chart
# bar2=sns.barplot(x=consoleSales['Year'],
#            y=consoleSales['ConsoleUnitSold'], 
#            hue=consoleSales['Company'])
# plt.show(bar2)

# Group by two year and puslisher mean and sum
videoSalesMean = videoSales.groupby(['Year_of_Release','Publisher']).mean().reset_index()

# Merge the two file based on year and publisher
merged = pd.concat([videoSalesMean, consoleSales], axis=1)
merged.drop(['Company','Year'] ,axis=1, inplace=True)
merged.isnull().sum()

# Merge the global sales and console unit sales
# videoMean = videoSalesMean[['Year_of_Release','Publisher','Global_Sales']]
videoMean = videoSalesMean
videoMean = videoMean.rename(columns={'Year_of_Release':'Year', 'Publisher': 'Company' })
console_and_Global_Sale = pd.merge(consoleSales,videoMean,
                                   how='inner',
                                   on=['Year','Company'])
console_and_Global_Sale.drop(['Year','Company'], axis=1, inplace=True)

console_and_Global_Sale_2 = pd.merge(consoleSales,console_and_Global_Sale,
                                   how='inner',
                                   on=['ConsoleUnitSold'])

console_and_Global_Sale_2.drop(['Critic_Count','User_Count'], axis=1, inplace=True)
le = LabelEncoder()

cont_attributes = console_and_Global_Sale_2.select_dtypes(exclude=['object'])
obj_attributes = console_and_Global_Sale_2.select_dtypes(include=['object'])

obj_attributes_labelled = obj_attributes.apply(le.fit_transform)
merged_label = pd.concat([cont_attributes,obj_attributes_labelled],axis=1)
def sold_check(x):
    if 0<=x and x<10:
        return 0
    elif 10<=x and x<15:
        return 1
    elif 15<=x and x<20:
        return 2
    elif 20<=x and x<27:
        return 3
    elif 27<=x and x<35:
        return 3
    elif 35<=x:
        return 4
   

merged_label['ConsoleUnitSold'] = merged_label['ConsoleUnitSold'].apply(sold_check)
# merged_label = merged_label_2.apply(le.fit_transform)

# ML Decision Tree
features = merged_label.drop(['ConsoleUnitSold', "Company" , "Year"], axis=1)
target = merged_label[['ConsoleUnitSold']]
target = np.ravel(target)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.5) #Split the data

c = DecisionTreeClassifier(random_state=42)#DecisionTreeClassifier(min_samples_split=2, random_state=1); #Now splits in every 100 so that overfitting doesnt occur

# Round For ML Tech
# X_Train = (np.round(train[features],5)).astype('int')
# Y_Train = (np.round(train[trains],5)).astype('int')

# X_Test = (np.round(test[features],5)).astype('int')
# Y_Test = (np.round(test[trains],5)).astype('int')

decThree = c.fit(X_train,Y_train) #Produce Decision Tree
featuresNames = ["Global_Sales", "Critic_Score", "User_Score"];
tree.export_graphviz(decThree, out_file = "assas.dot", feature_names = featuresNames) #Print DecTree
 
y_pred = c.predict(X_test)

print(accuracy_score(Y_test,y_pred))

svc_model = SVC()
svc_model.fit(X_train,Y_train)

svc_y_pred = c.predict(X_test)

print(accuracy_score(Y_test,svc_y_pred))
from sklearn.linear_model import Perceptron

# Create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train,Y_train)

ppn_predict = ppn.predict(X_test)

print (accuracy_score(Y_test,ppn_predict))
