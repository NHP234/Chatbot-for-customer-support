from flask import Flask, render_template, request, jsonify
from main import generate_response # Import hàm từ main.py

app = Flask(__name__)

# Dòng này sẽ tải mô hình của bạn một lần khi ứng dụng khởi động
# Bạn sẽ cần điều chỉnh phần này để tích hợp với main.py
# Ví dụ: from main import get_chatbot_response, load_model
# load_model() 

# Khi main.py được import, các mô hình sẽ được tải và huấn luyện (dựa trên cấu trúc hiện tại của main.py)
# Không cần gọi load_model() riêng ở đây nếu main.py đã xử lý việc đó khi được import.

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Gọi hàm chatbot từ main.py
    bot_response = generate_response(user_message)
    print(f"DEBUG app.py - Bot response from generate_response: {bot_response}")
    print(f"DEBUG app.py - Type of bot_response: {type(bot_response)}")
        
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    # Chạy app ở chế độ debug sẽ tự động tải lại khi có thay đổi code,
    # nhưng cũng có thể gây ra việc tải/huấn luyện mô hình nhiều lần nếu cấu trúc của main.py không tối ưu cho việc này.
    # Để sản xuất, bạn nên tắt debug=True và có cơ chế tải mô hình hiệu quả hơn.
    app.run(host='0.0.0.0', port=5000, debug=True) 