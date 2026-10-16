import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read the dataset
data = pd.read_csv("data/Housing.csv")

# Show first 5 rows
print(data.head())

# Select features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check model performance
print("\nModel Performance")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 4))

# Take input from user
print("\nEnter House Details")

area = float(input("Area (sq ft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))

house = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms]
})

price = model.predict(house)

print("\nEstimated House Price")
print("₹", round(price[0], 2))

# Plot graph
plt.figure(figsize=(7,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")
plt.grid(True)

# Save graph
plt.savefig("images/house_prediction.png", dpi=300)

# Show graph
plt.show()