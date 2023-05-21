import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("Hunger Games survival analysis data set.csv")

df.head()

df.fillna(0, inplace=True)

X = df.drop(columns=["survival_days", "name"])
y = df["survival_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2694
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# model evaluation
print("mean_squared_error : ", mean_squared_error(y_test, predictions))
print("mean_absolute_error : ", mean_absolute_error(y_test, predictions))

# Plot predictions against actual
figure = sns.scatterplot(x=predictions, y=y_test)
figure.set(xlabel="predictions", ylabel="actual")
figure.figure.savefig("predictions.png")
