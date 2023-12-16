from flask import Flask, render_template
from Machine_Learning_And_Neural_Network_Models.Classification_Logical_Regression.Classification_Logical_Regression import logical_regression_bp
from Machine_Learning_And_Neural_Network_Models.Decision_Tree_Classification.Decision_Tree_Classification import decision_tree_bp
from Machine_Learning_And_Neural_Network_Models.Clustering_Linkage_Dendrogram.Clustering_Linkage_Dendrogram import dendrogram_bp
from Machine_Learning_And_Neural_Network_Models.Clustering_SOM_Neural_Network.Clustering_SOM_Neural_Network import som_bp
from Machine_Learning_And_Neural_Network_Models.Prediction_Linear_Regression.Prediction_Linear_Regression import linear_regression_bp
from Machine_Learning_And_Neural_Network_Models.Prediction_Neural_Network_MLPRegressor.Prediction_Neural_Network_MLPRegressor import mlp_regressor_bp

app = Flask(__name__)

# Регистрация Blueprint'ов
app.register_blueprint(logical_regression_bp, url_prefix='/Classification_Logical_Regression')
app.register_blueprint(decision_tree_bp, url_prefix='/Decision_Tree_Classification')
app.register_blueprint(dendrogram_bp, url_prefix='/Clustering_Linkage_Dendrogram')
app.register_blueprint(som_bp, url_prefix='/Clustering_SOM_Neural_Network')
app.register_blueprint(linear_regression_bp, url_prefix='/Prediction_Linear_Regression')
app.register_blueprint(mlp_regressor_bp, url_prefix='/Prediction_Neural_Network_MLPRegressor')

# Главная страница
@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="localhost", port=5000)
