from flask import Flask, render_template, request
import LinearRegrression
import LogisticRegressionModel
app = Flask(__name__)
#py -m flask run

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/aplicacion1')
def aplicacion1():
    return render_template('ventas.html')

@app.route('/aplicacion2')
def aplicacion2():
    return render_template('recomendacion.html')

@app.route('/linearRegression/',methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateResult = LinearRegrression.calculate_grade(hours)
        return render_template("linearRegressionGrade2.html", result = calculateResult)
    return render_template("linearRegressionGrade2.html", result = calculateResult)

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
    return render_template("logisticRegression.html", result=calculateResult)
