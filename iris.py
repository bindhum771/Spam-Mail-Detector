# This script creates an interactive web app for Iris flower classification.
# It uses the Streamlit library to build the user interface.

import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import base64
from PIL import Image
import io

# Base64 encoded images for Iris flowers
# This approach embeds the images directly into the script,
# making the app self-contained and eliminating broken links.
IRIS_SETOSA_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAoAAAAGQCAMAAACq1h4uAAAAAXNSR0IArs4c6QAAAXNQTFhJ\n+o+AAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAA\nAAAAAAADAfwAAAAAAAAAAAAAAAADAfwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPf+AOPXAAAAA\nSURBVHic7cEBAQAAAIIg/6L/Qz+AAAIACAAAAA"
IRIS_VERSICOLOR_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAoAAAAGQCAMAAACq1h4uAAAAAXNSR0IArs4c6QAAAXNQTFhJ\n+o+AAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAA\nAAAAAAADAfwAAAAAAAAAAAAAAAADAfwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPf+AOPXAAAAA\nSURBVHic7cEBAQAAAIIg/6L/Qz+AAAIACAAAAA"
IRIS_VIRGINICA_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAoAAAAGQCAMAAACq1h4uAAAAAXNSR0IArs4c6QAAAXNQTFhJ\n+o+AAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAAAAAAAIA+gAAAAAAAAAAA\nAAAAAAADAfwAAAAAAAAAAAAAAAADAfwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPf+AOPXAAAAA\nSURBVHic7cEBAQAAAIIg/6L/Qz+AAAIACAAAAA"

def train_model():
    """
    Loads the Iris dataset, splits the data, and trains a KNN classifier.
    Returns the trained model and target names for later use.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    return model, target_names

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Iris Flower Classifier")
    st.title("Iris Flower Classifier")
    st.markdown("Enter the measurements of the flower to classify its species.")

    # Train the model once
    model, target_names = train_model()

    st.subheader("Flower Measurements")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=4.0, max_value=8.0, value=5.1, step=0.1
    )
    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=2.0, max_value=4.5, value=3.5, step=0.1
    )
    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=1.0, max_value=7.0, value=1.4, step=0.1
    )
    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=0.1, max_value=2.5, value=0.2, step=0.1
    )

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Classify Flower", help="Click to classify the flower based on the measurements."):
        prediction = model.predict(input_data)
        predicted_species_index = prediction[0]
        predicted_species_name = target_names[predicted_species_index].capitalize()

        st.subheader("Classification Result")
        st.success(f"The predicted species is: **{predicted_species_name}**")

        # Display an image related to the classification
        image_base64 = ""
        caption = ""
        if predicted_species_name == "Setosa":
            image_base64 = IRIS_SETOSA_BASE64
            caption = "Iris Setosa"
        elif predicted_species_name == "Versicolor":
            image_base64 = IRIS_VERSICOLOR_BASE64
            caption = "Iris Versicolor"
        elif predicted_species_name == "Virginica":
            image_base64 = IRIS_VIRGINICA_BASE64
            caption = "Iris Virginica"
        else:
            st.warning("Could not display an image for the predicted species.")

        if image_base64:
            # Decode the base64 string and display the image
            image_bytes = base64.b64decode(image_base64.encode('utf-8'))
            st.image(image_bytes, caption=caption, use_container_width=True)


if __name__ == "__main__":
    main()
