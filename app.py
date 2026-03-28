from flask import Flask, render_template, request
import LinearRegrression
import LogisticRegressionModel
import SteamLinearRegression
import ConsoleWarsLogisticRegression
import RatingKNNClassifier

app = Flask(__name__)
#py -m flask run

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/use-cases')
def use_cases():
    return render_template('use_cases.html')

@app.route('/linear-regression/concepts')
def lr_concepts():
    return render_template('basic_concepts.html')

@app.route('/linear-regression/application', methods=["GET", "POST"])
def lr_application():
    result = None
    if request.method == "POST":
        games = float(request.form["games_owned"])
        result = round(SteamLinearRegression.predict_hours(games), 2)
    return render_template(
        "application.html",
        result=result,
        slope=round(SteamLinearRegression.slope, 4),
        intercept=round(SteamLinearRegression.intercept, 2),
        r2_score=round(SteamLinearRegression.r2, 4)
    )

@app.route('/linearRegression/', methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateResult = LinearRegrression.calculate_grade(hours)
    return render_template("linearRegressionGrade2.html", result=calculateResult)

@app.route('/logisticRegression/', methods=["GET", "POST"])
def logisticRegression():
    calculateResult = None
    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso_mensual = float(request.form["ingreso_mensual"])
        visitas_web_mes = float(request.form["visitas_web_mes"])
        tiempo_sitio_min = float(request.form["tiempo_sitio_min"])
        compras_previas = float(request.form["compras_previas"])
        descuento_usado = float(request.form["descuento_usado"])
        input_scaled = LogisticRegressionModel.scaler.transform([[edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado]])
        calculateResult = LogisticRegressionModel.logistic_model.predict(input_scaled)[0]
    return render_template("logisticRegression.html", result=calculateResult)

@app.route('/logistic-regression/concepts')
def logistic_concepts():
    return render_template('logistic_regression_concepts.html')

@app.route('/logistic-regression/application', methods=["GET", "POST"])
def logistic_application():
    result = None
    probabilities = None
    if request.method == "POST":
        genre = request.form["genre"]
        na_sales = float(request.form["na_sales"])
        eu_sales = float(request.form["eu_sales"])
        jp_sales = float(request.form["jp_sales"])
        other_sales = float(request.form["other_sales"])
        critic_score = float(request.form["critic_score"])
        user_score = float(request.form["user_score"])
        result, probabilities = ConsoleWarsLogisticRegression.predict_platform(
            genre, na_sales, eu_sales, jp_sales, other_sales, critic_score, user_score
        )
    return render_template(
        "logistic_regression_application.html",
        result=result,
        probabilities=probabilities,
        genres=ConsoleWarsLogisticRegression.all_genres,
        accuracy=ConsoleWarsLogisticRegression.accuracy,
        precision=ConsoleWarsLogisticRegression.precision,
        recall=ConsoleWarsLogisticRegression.recall,
        f1=ConsoleWarsLogisticRegression.f1,
        dataset_size=ConsoleWarsLogisticRegression.dataset_size,
        train_size=ConsoleWarsLogisticRegression.train_size,
        test_size=ConsoleWarsLogisticRegression.test_size,
    )

@app.route('/knn/concepts')
def knn_concepts():
    return render_template('knn_concepts.html')

@app.route('/knn/application', methods=["GET", "POST"])
def knn_application():
    result = None
    probabilities = None
    if request.method == "POST":
        genre = request.form["genre"]
        platform_family = request.form["platform_family"]
        na_sales = float(request.form["na_sales"])
        eu_sales = float(request.form["eu_sales"])
        jp_sales = float(request.form["jp_sales"])
        other_sales = float(request.form["other_sales"])
        critic_score = float(request.form["critic_score"])
        user_score = float(request.form["user_score"])
        result, probabilities = RatingKNNClassifier.predict_rating(
            genre, platform_family, na_sales, eu_sales, jp_sales,
            other_sales, critic_score, user_score
        )
    return render_template(
        "knn_application.html",
        result=result,
        probabilities=probabilities,
        genres=RatingKNNClassifier.all_genres,
        platform_families=RatingKNNClassifier.all_platform_families,
        accuracy=RatingKNNClassifier.accuracy,
        precision=RatingKNNClassifier.precision,
        recall=RatingKNNClassifier.recall,
        f1=RatingKNNClassifier.f1,
        dataset_size=RatingKNNClassifier.dataset_size,
        train_size=RatingKNNClassifier.train_size,
        test_size=RatingKNNClassifier.test_size,
        k_value=RatingKNNClassifier.K_VALUE,
    )
