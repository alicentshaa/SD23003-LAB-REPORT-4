import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Activation Function Visualizer",
    layout="centered",
)

st.title("Neural Network Activation Functions")
st.write(
    """
    This application visualizes the three core activation functions:
    **ReLU, Sigmoid, and Tanh**.
    """
)

# ---------------------------------------------------------
# Sidebar for Function Selection
# ---------------------------------------------------------
st.sidebar.header("Navigation")
# Requirement: Create visualizations for ReLU, Sigmoid, and Tanh 
option = st.sidebar.selectbox(
    "Select Activation Function",
    ["ReLU", "Sigmoid", "Tanh"]
)

# ---------------------------------------------------------
# Define Activation Functions
# ---------------------------------------------------------
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Generate X values
x = np.linspace(-10, 10, 200)

# ---------------------------------------------------------
# Visualization Logic
# ---------------------------------------------------------
st.subheader(f"Visualization: {option}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, linestyle='--', alpha=0.6)

if option == "ReLU":
    y = relu(x)
    ax.plot(x, y, label="ReLU(x) = max(0, x)", color="red", linewidth=2)
    st.write("**ReLU (Rectified Linear Unit)**: Outputs the input directly if it is positive; otherwise, it outputs zero[cite: 23].")

elif option == "Sigmoid":
    y = sigmoid(x)
    ax.plot(x, y, label="Sigmoid(x) = 1 / (1 + e^-x)", color="blue", linewidth=2)
    st.write("**Sigmoid**: Squashes input values into a range between 0 and 1[cite: 24].")

elif option == "Tanh":
    y = tanh(x)
    ax.plot(x, y, label="Tanh(x) = (e^x - e^-x) / (e^x + e^-x)", color="green", linewidth=2)
    st.write("**Tanh (Hyperbolic Tangent)**: Squashes input values into a range between -1 and 1[cite: 25].")

ax.legend()
ax.set_title(f"{option} Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output (f(x))")

st.pyplot(fig)
