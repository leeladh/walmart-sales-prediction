from flask import Flask,render_template, render_template_string, url_for, redirect, request
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

le1 = load('label_encode.joblib')
model = load('m2.joblib')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def homapge():
	if request.method == "POST":
		i1 = int(request.form.get('input1'))
		i2 = float(request.form.get('input2'))
		i3 = float(request.form.get('input3'))
		i4 = float(request.form.get('input4'))
		i5 = float(request.form.get('input5'))
		i6 = int(request.form.get('input6'))
		i7 = request.form.get('input7')
		# print(type(i7))
		i8 = int(request.form.get('input8'))
		i9 = int(request.form.get('input9'))
		i10 = int(request.form.get('input10'))

		i7 = le1.transform(np.array([i7.upper()])).tolist()[0]

		inp_array = np.array([i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]).reshape(1, -1)
		result = model.predict(inp_array)[0]
		# print(result)

		return redirect(url_for('results', res=result))
	else:
		return render_template("home.html")

@app.route('/results/<res>')
def results(res):
	r = str(res)
	return render_template_string("<h1> Your result is " + r + ".</h1>")


if __name__ == "__main__":
	app.run()
