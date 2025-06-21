from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/post-example', methods=['POST'])
def handle_post():
    data = request.get_json()
    name = data.get('name', 'Guest')
    return jsonify({
        'message': f'Hello, {name}!',
        'received_data': data
    })

if __name__ == '__main__':
    app.run(debug=True)
