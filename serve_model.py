from flask import Flask, request, jsonify
from object_classify import predict
from PIL import Image
import base64
from io import StringIO, BytesIO


app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK'
    })


@app.route('/', methods=['POST'])
def detect():
    data = request.json

    image_binary = base64.b64decode(data['img'])
    image_f = BytesIO()
    image_f.write(image_binary)
    image_f.seek(0)
    image = Image.open(image_f)
    label = predict(image)
    return jsonify({
        'label': label
    })
    # img_out = m.predict(image_np)['img_out']
    # vis_file = BytesIO()
    # scipy.misc.imsave(vis_file, img_out, format='png')
    # vis_file.seek(0)
    # vis_binary = vis_file.read()
    # return jsonify({
    #     'img_out': base64.b64encode(vis_binary).decode('utf-8'),
    # })


if __name__ == '__main__':
    PORT = 5900
    print('server running at '+str(PORT))
    app.run(debug=False, port=PORT, host='0.0.0.0')
