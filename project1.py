from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
train_data = pd.read_csv('/Users/dany/Desktop/Новая папка/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/Users/dany/Desktop/Новая папка/house-prices-advanced-regression-techniques/test.csv')


#check for empty values
# print(train_data.isnull().sum())
# print(test_data.isnull().sum())

missing_values = train_data.isnull().sum()
missing_values = missing_values[missing_values>0].sort_values(ascending=False)


# Sales price visualization
# plt.figure(figsize=(10,6))
# sns.histplot(train_data['SalePrice'], kde=True)
# plt.xlabel('Price')
# plt.ylabel('Frequency')
# plt.show()


#check for correlation on SalePrice
# num_data = train_data.select_dtypes(include=['number'])
# cor = num_data.corr()
# cor_price = cor['SalePrice'].sort_values(ascending=False)
# print(cor_price)


# Work with missing values

num_features = train_data.select_dtypes(include=['int64', 'float64']).drop(columns=['SalePrice']).columns
cat_features = train_data.select_dtypes(include=['object']).columns
# print(num_features)
# print(cat_features)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
] )

preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])

# split data
from sklearn.model_selection import train_test_split
x = train_data.drop(columns = ['SalePrice'])
y = train_data['SalePrice']
X_train, X_valid, Y_train, Y_valid = train_test_split(x, y, test_size= 0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)


# gradient boosting model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

model = GradientBoostingRegressor(n_estimators=100, random_state= 42, learning_rate=0.1).fit(X_train, Y_train)
y_pred = model.predict(X_valid)
print(mean_absolute_error(Y_valid, y_pred))

#model's result visualization
# plt.figure(figsize=(10,6))
# plt.scatter(Y_valid, y_pred, alpha=0.5)
# plt.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], 'r--')
# plt.show()

X_test = test_data
X_test = preprocessor.transform(X_test)
test_pred = model.predict(X_test)
submission = pd.DataFrame({'Id':test_data['Id'], 'SalePrice':test_pred})
submission.to_csv('submission.csv', index=False)