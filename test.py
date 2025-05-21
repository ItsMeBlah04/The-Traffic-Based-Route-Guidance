# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R² Score: {r2:.4f}")
