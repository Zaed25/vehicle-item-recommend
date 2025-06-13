# app.py

from flask import Flask, request, jsonify
from recommender import Recommender # <-- استيراد الفئة المحدثة

# --- 1. تهيئة محرك التوصية وتطبيق فلاسك ---
# يتم إنشاء نسخة واحدة من محرك التوصية عند بدء تشغيل التطبيق
recommender_instance = Recommender(
    artifacts_path='recommendation_artifacts.pkl',
    inventory_path='inventory_final.csv'
)
app = Flask(__name__)

# --- 2. تعريف نقاط النهاية (Endpoints) الخاصة بالـ API ---

@app.route('/')
def index():
    """نقطة نهاية للتحقق من أن الخدمة تعمل"""
    return jsonify({"status": "online", "message": "Car Recommendation API is running."})

@app.route('/recommend', methods=['GET'])
def recommend_collaborative():
    """نقطة نهاية للسيارات المعروفة باستخدام الترشيح التعاوني"""
    item_id = request.args.get('item_id')
    if not item_id:
        return jsonify({"error": "item_id parameter is required"}), 400

    recommendations = recommender_instance.get_collaborative_recommendations(item_id)

    if recommendations is None:
        return jsonify({"error": f"Item '{item_id}' not found in the collaborative model"}), 404

    return jsonify({"recommendations": recommendations})

@app.route('/recommend/unseen', methods=['POST'])
def recommend_content_based():
    """نقطة نهاية للسيارات الجديدة/غير المرئية باستخدام الترشيح المعتمد على المحتوى"""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    car_features = request.get_json()
    
    # --- التحقق من صحة المدخلات للحقول الإلزامية فقط ---
    # تم الآن إزالة 'Body' من القائمة
    required_fields = ['YearOfMaking', 'Make', 'Model', 'Trim']
    missing_fields = [field for field in required_fields if field not in car_features]
    if missing_fields:
        return jsonify({"error": f"Request body is missing required fields: {missing_fields}"}), 400

    # الحصول على التوصيات باستخدام الدالة الجديدة
    # ستقوم الفئة recommender بمعالجة الحقول الاختيارية مثل السعر والقوة الحصانية
    recommendations = recommender_instance.get_content_based_recommendations(car_features)

    return jsonify({"recommendations": recommendations})

# --- ينتهي الملف هنا. سيقوم خادم Gunicorn بتشغيل الكائن 'app' ---
