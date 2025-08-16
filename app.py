from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Đọc dữ liệu
df = pd.read_csv("Iris.csv")

# 2. Chuẩn bị dữ liệu
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# 3. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Huấn luyện mô hình K-NN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 5. Khởi tạo Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            sl = float(request.form["sepal_length"])
            sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"])
            pw = float(request.form["petal_width"])
            sample = [[sl, sw, pl, pw]]
            prediction = knn.predict(sample)[0]
        except:
            prediction = "Lỗi: vui lòng nhập đúng số!"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
