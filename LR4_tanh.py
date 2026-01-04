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
    **Tanh**.
    """
)

st.title("Application 3: Tanh")

x = np.linspace(-10, 10, 200)
y = np.tanh(x)

fig, ax = plt.subplots()
ax.plot(x, y, color="green", label="Tanh")
ax.grid(True)
ax.set_title("Hyperbolic Tangent")
st.pyplot(fig)

st.write("Tanh squashes values between -1 and 1.")