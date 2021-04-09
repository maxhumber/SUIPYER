import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

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
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1])),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=tf.keras.losses.mean_squared_error,
    metrics=tf.keras.metrics.mean_absolute_error
)
model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

from sklearn.metrics import r2_score
r2_score(y_test, model.predict(X_test).flatten())

# convert
import coremltools as ct
coreml_model = ct.convert(model, )
coreml_model.save('models/BoardGameRegressor2.mlmodel')
