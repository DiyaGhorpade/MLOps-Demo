import pandas as pd
import tensorflow as tf
import os

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/iris.csv")

print("\nDataset loaded successfully!")

print("\nColumns in dataset:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. CONVERT SPECIES INTO NUMBERS
# ==========================================

# Convert:
# Iris-setosa     -> 0
# Iris-versicolor -> 1
# Iris-virginica  -> 2

species_mapping = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

df["target"] = df["species"].map(species_mapping)

print("\nSpecies converted to numerical labels:")
print(df[["species", "target"]].head())


# ==========================================
# 3. SEPARATE INPUTS (X) AND OUTPUT (y)
# ==========================================

X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
].values

y = df["target"].values


print("\nInput shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ==========================================
# 4. CREATE NEURAL NETWORK
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])


# ==========================================
# 5. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 6. DISPLAY MODEL
# ==========================================

print("\nModel architecture:")
model.summary()


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\n========================================")
print("Starting model training...")
print("========================================")

history = model.fit(
    X,
    y,
    epochs=20,
    validation_split=0.2
)


# ==========================================
# 8. CREATE SAVED MODEL DIRECTORY
# ==========================================

os.makedirs("saved_model/1", exist_ok=True)


# ==========================================
# 9. EXPORT MODEL
# ==========================================

model.export("saved_model/1")


# ==========================================
# 10. FINISHED
# ==========================================

print("\n========================================")
print("MODEL TRAINING COMPLETED!")
print("========================================")

print("\nModel saved to:")
print("saved_model/1")

print("\nYou can now use this model with")
print("TensorFlow Serving.")