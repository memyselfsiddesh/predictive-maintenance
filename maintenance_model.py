import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

class PredictiveMaintenanceModel:

    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()

    # Load dataset
    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    # Preprocess data
    def preprocess(self, df):
        X = df.drop('machine_failure', axis=1)
        y = df['machine_failure']

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    # Train model
    def train(self):
        df = self.load_data()
        X, y = self.preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return accuracy, report

    # Predict new data
    def predict(self, input_data):
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)
        return prediction

    # AI Rule Engine (Non-ML logic)
    def rule_engine(self, air_temp, process_temp, torque, wear):
        if air_temp > 320 or process_temp > 330:
            return "High Temperature Warning"
        if wear > 180:
            return "Tool Wear Critical"
        if torque > 65:
            return "High Torque Load"
        return "Normal Condition"


# Initialize and train model
model_object = PredictiveMaintenanceModel("data/maintenance.csv")
accuracy, report = model_object.train()

print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Model object created successfully")