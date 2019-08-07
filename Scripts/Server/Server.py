from flask import Flask, request
from Scripts.Data.AcousticData import PureAcousticData
import tensorflow as tf


class Server:
    def __init__(self,model_obj):
        self.model_obj = model_obj
        self.graph = tf.get_default_graph()
        self.app = Flask(__name__)

        @self.app.route("/recognize", methods=["POST"])
        def recognize():
            f = request.files["file"]
            f.save("test.wav")
            data_obj = PureAcousticData("test.wav")
            with self.graph.as_default():
                res = self.model_obj.predict(data_obj)
            return " ".join(res)
    
    def run(self):
        self.app.run("0.0.0.0", debug=False)

