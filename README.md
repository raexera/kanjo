# Kanjo (感情) - AI-Powered Sentiment Analysis Tool

## Project Overview

**Kanjo (感情)** is an AI-powered sentiment analysis tool designed to analyze and predict the sentiment of Japanese text data. Whether you're working with movie reviews, customer feedback, or any other text data, Kanjo enables you to assess sentiment with ease. The tool offers flexibility with model selection, allowing users to choose from Naive Bayes, SVM, Random Forest, or Ensemble methods. It also provides options for using pre-trained models or training models with custom datasets.

## Technologies Used

- **Programming Languages**: Python
- **Frameworks**: Streamlit
- **Libraries**:
  - **Machine Learning**: Scikit-learn
  - **Text Processing**: NLTK, TfidfVectorizer
  - **Data Handling**: Pandas, NumPy
  - **Model Persistence**: Joblib

## Key Features

- **Sentiment Analysis**: Analyze text to determine if the sentiment is positive or negative.
- **Model Selection**: Choose between Naive Bayes, SVM, Random Forest, and Ensemble methods.
- **Custom Dataset Training**: Train the model using your own dataset in CSV format.
- **Model Persistence**: Save and load trained models for reuse.
- **Interactive Interface**: Streamlit-based user interface for easy interaction.
- **Visualization**: Real-time sentiment distribution, confidence levels, and probability scores.
- **Analysis History**: Track and display the history of your sentiment analyses.

## Installation

To install and run Kanjo locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/rxyhn/kanjo.git
   cd kanjo
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Using Pre-Trained Model

1. **Start the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

2. **Select a model** from the dropdown menu (e.g., Naive Bayes, SVM).

3. **Enter text** for sentiment analysis in the provided text area.

4. **Click "Analyze"** to see the sentiment prediction along with confidence levels and other details.

### Training with Custom Dataset

1. **Upload your CSV dataset** containing text and sentiment labels (positive as `pos`, negative as `neg`).

2. The model will train automatically on the uploaded data, and results will be displayed, including accuracy.

### Visualization and History

- The tool displays sentiment distribution, prediction confidence levels, and probability scores in real-time.
- You can view the history of your analyses in a tabular format, complete with the model used, text input, sentiment prediction, and confidence levels.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
