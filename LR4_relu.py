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
    This application visualizes the activation functions:
    **ReLU**.
    """
)

x = np.linspace(-10, 10, 200)
y = np.maximum(0, x)

fig, ax = plt.subplots()
ax.plot(x, y, color="red", label="ReLU")
ax.grid(True)
ax.set_title("Rectified Linear Unit")
st.pyplot(fig)

st.write("ReLU outputs the input if positive, otherwise zero.")

