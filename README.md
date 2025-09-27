
# AI Crop Health Monitoring System

## Overview
This project leverages **Artificial Intelligence (AI)** and **Machine Learning (ML)** to monitor crop health using image analysis. The system can **detect crop diseases early**, helping farmers reduce losses and improve yield.

---

## Features
- Upload images of crops via a web interface.
- Detect whether a crop is **healthy** or **diseased**.
- Provides **confidence scores** for predictions.
- Built with **Python, Streamlit, and ML models**.

---

## Step-by-Step Process

### 1. Data Collection
- Collected images of healthy and diseased crops.
- Organized dataset into folders, e.g., `Healthy/` and `Diseased/`.

### 2. Data Preprocessing
- Resized all images to a uniform size (e.g., 224x224).
- Normalized pixel values (0–1).
- Labeled the data (`0 = Healthy`, `1 = Diseased`).
- Split dataset into **training** and **testing** sets (80/20).

### 3. Model Development
- Used a **Convolutional Neural Network (CNN)** for image classification.
- Compiled the model with:
  - Loss: `categorical_crossentropy`
  - Optimizer: `adam`
  - Metric: `accuracy`

### 4. Model Training
- Trained the model on the training dataset.
- Validated using the testing dataset.
- Saved the trained model for inference (`train_model.py`).

### 5. Web Application
- Built a **Streamlit app** (`app.py`) for user interaction.
- Upload crop images and view predictions in real time.

### 6. Testing & Deployment
- Tested the model with new images for accuracy.
- Run locally:
  ```bash
  streamlit run app.py
````

* Optional: Deploy online using **Streamlit Cloud**.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Mohanraj24palliyandur/crop-health.git
```

2. Navigate into the project directory:

```bash
cd crop-health
```

3. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
```

4. Activate the virtual environment:

```powershell
# Windows
.\.venv\Scripts\activate
```

5. Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open the URL provided by Streamlit in your browser (usually `http://localhost:8501`).

3. Upload an image of a crop to get a prediction.

---

## Technologies Used

* Python
* Streamlit
* OpenCV
* TensorFlow / Keras
* Pandas, NumPy
* Git & GitHub

---

## Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add feature"`.
4. Push to your branch: `git push origin feature-name`.
5. Open a Pull Request on GitHub.

---

## License

This project is licensed under the MIT License.

---

## Demo Screenshot (Optional)

Add a screenshot of your Streamlit app here for a better visual impression.

---

If you want, I can also **write a `.gitignore` file for this project** so you don’t accidentally push your `.venv`, `.pyc`, or other unnecessary files.  

Do you want me to do that?
```
# crop-health
