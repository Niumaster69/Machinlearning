from flask import Flask, render_template
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
