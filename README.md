 # Machine Learning Applied to Video Games

  A Flask web application that demonstrates Machine Learning concepts through real-world video game industry use cases, with a focus on linear regression using a real Steam
  dataset.

  ## About

  This project was developed for the Machine Learning course at **Universidad de Cundinamarca — Extension Chia**. The chosen domain is **Video Games**, exploring how ML
  algorithms are used by companies like Valve, EA, Riot Games, and Epic Games.

  **Author:** Duvan Lozano Romero — Systems Engineering and Computing

  ## Features

  - **4 ML Use Cases** with detailed explanations, code, and results
    - Sales Prediction (Random Forest)
    - Game Recommendation (KNN)
    - Cheat Detection (Isolation Forest)
    - Dynamic Difficulty & Matchmaking (KMeans)
  - **Linear Regression — Basic Concepts** page with theory and examples
  - **Linear Regression — Application** using a real Steam dataset (200,000 records)
    - Scatter plot with regression line
    - Interactive prediction form
  - **Activity 1** — Linear Regression (grade prediction) and Logistic Regression (purchase prediction)

  ## Navigation

  | Page | Route | Description |
  |------|-------|-------------|
  | Home | `/` | Introduction, types of ML, process, companies |
  | Use Cases | `/use-cases` | 4 cases with sidebar navigation |
  | Basic Concepts | `/linear-regression/concepts` | Linear regression theory |
  | Application | `/linear-regression/application` | Steam dataset, graph, prediction form |
  | Linear Regression | `/linearRegression/` | Grade prediction from study hours |
  | Logistic Regression | `/logisticRegression/` | Purchase prediction from customer data |

  ## Tech Stack

  - **Backend:** Python, Flask
  - **ML:** scikit-learn, pandas, numpy, matplotlib
  - **Frontend:** HTML, CSS (Inter font), Bootstrap 5
  - **Dataset:** Steam User Behavior Dataset (200K rows)

  ## Project Structure

  ├── app.py                        # Flask routes
  ├── SteamLinearRegression.py      # Steam model + graph generation
  ├── LinearRegrression.py          # Simple linear regression (grades)
  ├── LogisticRegressionModel.py    # Logistic regression (purchases)
  ├── dataset_steam.csv             # Steam dataset (200,000 rows)
  ├── dataset_regresion_logistica.csv
  ├── static/
  │   └── regression_plot.png       # Generated regression graph
  └── templates/
      ├── index.html                # Home page
      ├── use_cases.html            # 4 ML use cases
      ├── basic_concepts.html       # Linear regression theory
      ├── application.html          # Steam application + prediction
      ├── linearRegressionGrade2.html
      └── logisticRegression.html

  ## How to Run

  1. Clone the repository:
     ```bash
     git clone https://github.com/Niumaster69/Machinlearning.git
     cd Machinlearning

  2. Install dependencies:
  pip install flask scikit-learn pandas numpy matplotlib
  3. Run the application:
  py -m flask run
  4. Open in browser: http://127.0.0.1:5000

  Branches

  ┌───────────────────┬────────────────────────────────────────────────────┐
  │      Branch       │                      Purpose                       │
  ├───────────────────┼────────────────────────────────────────────────────┤
  │ master            │ Main branch with all merged changes                │
  ├───────────────────┼────────────────────────────────────────────────────┤
  │ activity_one      │ Initial development (linear & logistic regression) │
  ├───────────────────┼────────────────────────────────────────────────────┤
  │ use-cases         │ 4 ML gaming use cases                              │
  ├───────────────────┼────────────────────────────────────────────────────┤
  │ linear-regression │ Basic concepts + Steam dataset application         │
  └───────────────────┴────────────────────────────────────────────────────┘

  Model Results

  Steam Linear Regression:
  - Users in dataset: 11,350
  - Slope: 9.7841 (each additional game ≈ 9.78 more hours)
  - Intercept: 194.16
  - R² Score: 0.2145

  ---
  Universidad de Cundinamarca — Extension Chia — 2026
