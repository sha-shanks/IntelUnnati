# IntelUnnati

Unnati is a comprehensive chatbot application designed to assist Class 10 CBSE students with their studies. It leverages a powerful Retrieval-Augmented Generation (RAG) pipeline to answer subject-specific queries accurately. The application also includes an innovative emotion detection feature to analyze the user's facial expressions, providing a more interactive and personalized learning experience.

## Features

- **Multi-Subject Chatbot**: Get answers to your questions across a wide range of subjects, including History, Geography, Civics, Economics, Mathematics, and Science.
- **RAG-Powered Accuracy**: The chatbot uses a sophisticated RAG pipeline with a Pinecone vector store and Google's Gemini model to provide contextually relevant and accurate answers.
- **Emotion Detection**: The application uses your webcam to detect your emotions, allowing for a more engaging and responsive interaction.
- **User-Friendly Interface**: A simple and intuitive graphical user interface (GUI) built with Tkinter makes the application easy to use.

## Submission Files
[Report - Google Drive](https://drive.google.com/file/d/18BJJsYyGO_fQZx9Acyk--3cjZEi9UUPK/view?usp=sharing) 
[Demo Video - YouTube](https://youtu.be/5H1ulgyYNnQ)

## Tech Stack

- **Core Application**: Python
- **GUI**: Streamlit
- **Chatbot and RAG**:
  - LangChain
  - Google Gemini
  - Pinecone (Vector Store)
  - Hugging Face Transformers (Embeddings)
- **Emotion Detection**:
  - OpenCV
  - Haarcascade Classifiers

## Project Structure

```
IntelUnnati/
├── .gitignore
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
├── icons/
│   ├── ico.jpg
│   └── intel_logo.png
├── src/
│   ├── retriever_chain.py
│   ├── system_prompt.py
│   ├── haarcascade_files/
│   │   ├── haarcascade_eye.xml
│   │   └── haarcascade_frontalface_default.xml
│   ├── model/
│   │   ├── emotion_detection_model.bin
│   │   └── emotion_detection_model.xml
│   └── vector_store/
│       └── db_civics/
│       └── db_economics/
│       ├── db_geography/
│       ├── db_history/
│       └── db_mathematics/
│       └── db_science/
└── utils/
    └── pinecone_create_index.py
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- An active internet connection
- API keys for Google and Pinecone

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/IntelUnnati.git
    cd IntelUnnati
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add your API keys:
    ```
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    ```

## Usage

To start the application, run the `main.py` file:

```bash
python main.py
```

This will open the main application window. From there, you can select a subject and start asking questions. The chatbot will provide answers based on the selected subject.

## Modules

### `main.py`

This is the main entry point of the application. It initializes the GUI, handles user input, and integrates the chatbot and emotion detection modules.

### `src/retriever_chain.py`

This module contains the core logic for the RAG pipeline. It uses LangChain to create a retrieval chain that fetches relevant documents from the Pinecone vector store and generates an answer using the Gemini model.

### `utils/pinecone_create_index.py`

This script is used to create the Pinecone index and upload the vector embeddings of the knowledge base.

### `src/system_prompt.py`

This file contains the system prompt that is used to instruct the chatbot on how to behave and respond to user queries.

### `src/haarcascade_files/`

This directory contains the pre-trained Haar Cascade classifier files for detecting faces and eyes, which are used by the emotion detection feature.

## Contributing

Contributions to IntelUnnati are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
