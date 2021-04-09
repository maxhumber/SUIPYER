import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load
df = pd.read_csv('data/games.csv')

# fix
df["time"] = df["time"].apply(pd.to_numeric, errors="coerce")
df["age"] = df["age"].apply(lambda x: pd.to_numeric(x.replace("+", ""), errors="coerce"))
df = pd.concat([df, pd.get_dummies(df["category"])], axis=1)
df.columns = [c.replace("'", "").lower() for c in df.columns]
df = df.dropna()

# split
target = 'rating'
predictors = ['time', 'age', 'complexity', 'abstract', 'childrens',
    'customizable', 'family', 'party', 'strategy', 'thematic', 'wargames']
y = df[target]
X = df[predictors]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# model
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train), model.score(X_test, y_test))

# convert
import coremltools as ct
coreml_model = ct.converters.sklearn.convert(model, predictors, target)
coreml_model.save('models/BoardGameRegressor.mlmodel')
