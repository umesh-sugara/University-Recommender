# University Recommender

This is a Streamlit application for recommending universities:  https://university-recommender.streamlit.app/

---
Product Plan Document:
 - https://docs.google.com/document/d/1uJgJz-d96-3PkXwXziEZG6Kkf4c03k3AhEZz4NsxXjo/edit?usp=sharing
---
## Running the Project Locally

To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/umesh-sugara/University-Recommender.git
    cd University-Recommender
    ```

2.  **Set up a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create your secrets file:**
    This project requires an API key to connect to the LLM. You need to create a secrets file for Streamlit.

    Create a folder named `.streamlit` and inside it, a file named `secrets.toml`:
    ```bash
    mkdir .streamlit
    nano .streamlit/secrets.toml
    ```
    Paste the following into the file, replacing the placeholder with your own API key:
    ```toml
    # .streamlit/secrets.toml

    GOOGLE_API_KEY = "PASTE_YOUR_SECRET_API_KEY_HERE"
    ```
5.  **Run the Setup app:**
    ```bash
    python setup.py
    ```
6.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
