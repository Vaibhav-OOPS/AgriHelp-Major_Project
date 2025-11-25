import json
import os
import sqlite3
import re
import numpy as np
from flask import Flask, render_template, request, redirect, session, flash, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
from PIL import Image
import pandas as pd
from functools import wraps
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# -------------------- Flask Config --------------------
app = Flask(__name__)
app.secret_key = "dyuiknbvcxswe678ijc6i"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "sample")
DATA_GOV_BASE_URL = os.getenv("DATA_GOV_BASE_URL", "https://api.data.gov.in/resource")
DATA_GOV_LIMIT = int(os.getenv("DATA_GOV_LIMIT", "50"))
CROP_DATASET_ID = os.getenv("CROP_DATASET_ID", "9ef84268-d588-465a-a308-a864a43d0070")
CROP_COMMODITY_FIELD = os.getenv("CROP_COMMODITY_FIELD", "commodity")
CROP_PRICE_FIELD = os.getenv("CROP_PRICE_FIELD", "modal_price")
CROP_DATE_FIELD = os.getenv("CROP_DATE_FIELD", "arrival_date")
CROP_PRICE_DIVISOR = float(os.getenv("CROP_PRICE_DIVISOR", "100"))
CROP_STATE_FILTER = os.getenv("CROP_STATE_FILTER")
CROP_DISTRICT_FILTER = os.getenv("CROP_DISTRICT_FILTER")
CROP_MARKET_FILTER = os.getenv("CROP_MARKET_FILTER")
FERTILIZER_DATASET_ID = os.getenv("FERTILIZER_DATASET_ID")
FERTILIZER_COMMODITY_FIELD = os.getenv("FERTILIZER_COMMODITY_FIELD", "fertilizer")
FERTILIZER_PRICE_FIELD = os.getenv("FERTILIZER_PRICE_FIELD", "retail_price")
FERTILIZER_DATE_FIELD = os.getenv("FERTILIZER_DATE_FIELD", "date")
FERTILIZER_PRICE_DIVISOR = float(os.getenv("FERTILIZER_PRICE_DIVISOR", "1"))
FERTILIZER_STATE_FILTER = os.getenv("FERTILIZER_STATE_FILTER")
FERTILIZER_DISTRICT_FILTER = os.getenv("FERTILIZER_DISTRICT_FILTER")
FERTILIZER_MARKET_FILTER = os.getenv("FERTILIZER_MARKET_FILTER")


# -------------------- Database Setup --------------------
DB_NAME = "users.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password TEXT NOT NULL
        )
        """)
        conn.commit()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# -------------------- Load Crop Prediction Model --------------------
model = pickle.load(open('crop_recommendation/best_model.pkl', 'rb'))
mx = pickle.load(open('crop_recommendation/minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('crop_recommendation/standscaler.pkl', 'rb'))

# Dictionaries
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

price_dict = {
    1: 34, 2: 25, 3: 82.75, 4: 68.5, 5: 30, 6: 52.5, 7: 55, 8: 105, 9: 49.5, 10: 22,
    11: 100, 12: 95.14, 13: 34.61, 14: 120, 15: 67.81, 16: 63.05, 17: 97.24, 18: 65,
    19: 82.33, 20: 100, 21: 98, 22: 17.6
}

fallback_crop_prices = {name.lower(): price_dict[idx] for idx, name in crop_dict.items()}

fertilizer_price_reference = {
    "urea": 266.0,
    "urea and dap": 520.0,
    "urea and mop": 475.0,
    "dap": 1350.0,
    "dap and mop": 1480.0,
    "mop": 920.0,
    "good npk": 1180.0
}

explanation_dict = {
    "rice": [
        "Rice thrives in high humidity and warm temperatures.",
        "Nitrogen and phosphorus levels support leaf growth and tillering.",
        "The soil pH and rainfall are ideal for paddy cultivation.",
        "Consistent water supply ensures high yield."
    ],
    "maize": [
        "Maize prefers moderate rainfall and warm climate.",
        "Balanced NPK levels enhance cob and kernel development.",
        "Good temperature and humidity support rapid growth.",
        "Soil nutrients match the maize crop’s nitrogen demand."
    ],
    "jute": [
        "Jute grows best in warm and humid conditions.",
        "High nitrogen promotes fiber quality and yield.",
        "Phosphorus and potassium improve stem strength.",
        "Adequate rainfall ensures proper retting and growth."
    ],
    "cotton": [
        "Cotton grows best in warm and dry climates.",
        "Slightly alkaline pH suits root development.",
        "Moderate rainfall prevents fungal issues.",
        "High potassium boosts fiber quality and yield."
    ],
    "coconut": [
        "Coconut palms prefer tropical climates with high humidity.",
        "Balanced NPK supports nut formation and leaf growth.",
        "Well-drained soil prevents root rot.",
        "Consistent rainfall helps steady fruiting."
    ],
    "papaya": [
        "Papaya thrives in warm, frost-free environments.",
        "High nitrogen encourages vegetative growth.",
        "Potassium and phosphorus improve fruit quality.",
        "Proper rainfall and humidity prevent diseases."
    ],
    "orange": [
        "Oranges require subtropical climate with moderate humidity.",
        "Nitrogen and potassium support fruit size and sweetness.",
        "Phosphorus aids in flowering and fruit set.",
        "Well-drained soil prevents root diseases."
    ],
    "apple": [
        "Apple grows best in temperate regions with cool winters.",
        "Adequate nitrogen and potassium enhance fruit quality.",
        "Balanced phosphorus supports flowering.",
        "Proper temperature and rainfall ensure fruit development."
    ],
    "muskmelon": [
        "Muskmelon prefers warm and sunny conditions.",
        "High potassium improves fruit sweetness and size.",
        "Nitrogen supports healthy vine growth.",
        "Moderate rainfall ensures optimal fruit formation."
    ],
    "watermelon": [
        "Watermelon thrives in warm, sunny climates.",
        "Balanced NPK supports vine and fruit growth.",
        "Soil with good drainage prevents waterlogging.",
        "Adequate rainfall ensures juicy and sweet fruits."
    ],
    "grapes": [
        "Grapes prefer warm, dry climate with good sunlight.",
        "Nitrogen promotes leaf growth, potassium improves fruit quality.",
        "Phosphorus supports flowering and fruit set.",
        "Moderate rainfall prevents fungal diseases."
    ],
    "mango": [
        "Mango trees love warm, sunny climates.",
        "Moderate nitrogen and phosphorus favor flowering.",
        "Good drainage and low humidity prevent fruit rot.",
        "Rainfall levels are ideal for fruit setting."
    ],
    "banana": [
        "Banana thrives in warm, humid environments.",
        "High nitrogen and potassium support leaf and fruit growth.",
        "Balanced pH ensures healthy root absorption.",
        "Consistent rainfall aids steady fruiting."
    ],
    "pomegranate": [
        "Pomegranate grows well in warm, dry climates.",
        "Moderate nitrogen encourages fruit set.",
        "Potassium improves fruit quality and sweetness.",
        "Low humidity reduces fungal disease risk."
    ],
    "lentil": [
        "Lentils prefer cool and moderately dry conditions.",
        "Nitrogen fixation by roots reduces fertilizer need.",
        "Phosphorus supports flowering and pod formation.",
        "Moderate rainfall prevents waterlogging and disease."
    ],
    "blackgram": [
        "Blackgram thrives in warm and moist conditions.",
        "Nitrogen and phosphorus improve flowering and pod development.",
        "Potassium enhances grain quality.",
        "Well-drained soil prevents root rot."
    ],
    "mungbean": [
        "Mungbean grows well in warm, sunny climates.",
        "Nitrogen and potassium promote pod and seed development.",
        "Moderate phosphorus supports flowering.",
        "Adequate rainfall ensures healthy plant growth."
    ],
    "mothbeans": [
        "Mothbeans tolerate hot, arid conditions.",
        "Nitrogen and potassium improve seed yield.",
        "Phosphorus supports root and flower development.",
        "Low rainfall is ideal; waterlogging must be avoided."
    ],
    "pigeonpeas": [
        "Pigeonpeas thrive in semi-arid regions.",
        "Nitrogen fixation enhances soil fertility.",
        "Phosphorus and potassium support flowering and pod formation.",
        "Moderate rainfall ensures proper fruiting."
    ],
    "kidneybeans": [
        "Kidneybeans prefer moderate temperatures and humidity.",
        "Nitrogen supports healthy leaf and pod growth.",
        "Phosphorus enhances root and flowering development.",
        "Adequate rainfall is essential for good yield."
    ],
    "chickpea": [
        "Chickpeas grow well in cool, dry climates.",
        "Nitrogen and potassium support flowering and grain filling.",
        "Phosphorus enhances root development.",
        "Low humidity reduces risk of fungal diseases."
    ],
    "coffee": [
        "Coffee grows well in moderate temperatures and high humidity.",
        "Balanced soil nutrients enhance bean flavor.",
        "Slightly acidic pH suits coffee root systems.",
        "Adequate rainfall supports flowering and fruiting cycles."
    ]
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name, email, phone, password = (
            request.form["name"],
            request.form["email"],
            request.form["phone"],
            request.form["password"],
        )
        hashed_password = generate_password_hash(password)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                (name, email, phone, hashed_password),
            )
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, password = request.form["email"], request.form["password"]
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"], session["user_name"] = user["id"], user["name"]
            flash("Login successful!", "success")
            return redirect(url_for("crop_predict"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))


import requests
from bs4 import BeautifulSoup


def fetch_price_from_data_gov(
    item_name,
    dataset_id,
    commodity_field,
    price_field,
    *,
    date_field=None,
    price_divisor=1.0,
    filters=None,
    limit=None,
):
    if not dataset_id:
        return None, None

    if not DATA_GOV_API_KEY:
        return None, None

    query_filters = dict(filters or {})
    query_filters[commodity_field] = item_name.title()

    try:
        url = f"{DATA_GOV_BASE_URL}/{dataset_id}"
        params = {
            "api-key": DATA_GOV_API_KEY,
            "format": "json",
            "limit": limit or DATA_GOV_LIMIT,
            "filters": json.dumps(query_filters),
        }
        if date_field:
            params["sort"] = f"{date_field} desc"

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        for record in data.get("records", []):
            price_value = record.get(price_field)
            if price_value is None:
                continue

            try:
                normalized = float(str(price_value).replace(",", ""))
                normalized = normalized / price_divisor
            except ValueError:
                continue

            metadata_bits = []
            for key in ("state", "district", "market"):
                val = record.get(key)
                if val:
                    metadata_bits.append(val)
            if date_field and record.get(date_field):
                metadata_bits.append(record[date_field])
            source_label = "data.gov.in"
            if metadata_bits:
                source_label += f" ({', '.join(metadata_bits)})"

            return round(normalized, 2), source_label
    except Exception as exc:
        print(f"data.gov.in fetch failed for {item_name}: {exc}")

    return None, None


def fetch_price_via_openai(item_name, entity_type):
    if not OPENAI_API_KEY:
        return None

    try:
        prompt = (
            f"You are an agriculture market assistant. "
            f"Return the most recent average retail price for the {entity_type} '{item_name}' "
            "in India, expressed in INR per kilogram (or per bag for fertilizers). "
            "Only return the numeric value without currency symbol."
        )
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        price_text = response.choices[0].message.content.strip()
        match = re.search(r"\d+(\.\d+)?", price_text)
        if match:
            return float(match.group())
    except Exception as exc:
        print(f"OpenAI price lookup failed for {item_name}: {exc}")
    return None


# ------------------ Fetch Market Price (API → fallback) ------------------
def fetch_latest_market_price(crop_name):
    crop_filters = {}
    if CROP_STATE_FILTER:
        crop_filters["state"] = CROP_STATE_FILTER
    if CROP_DISTRICT_FILTER:
        crop_filters["district"] = CROP_DISTRICT_FILTER
    if CROP_MARKET_FILTER:
        crop_filters["market"] = CROP_MARKET_FILTER

    price, label = fetch_price_from_data_gov(
        crop_name,
        CROP_DATASET_ID,
        CROP_COMMODITY_FIELD,
        CROP_PRICE_FIELD,
        date_field=CROP_DATE_FIELD,
        price_divisor=CROP_PRICE_DIVISOR,
        filters=crop_filters,
    )
    if price is not None:
        return price, label

    try:
        crop_search = crop_name.title()
        url = "https://agmarknet.gov.in/PriceAndArrivals/DatewiseCommodityReport.aspx"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if table:
            for row in table.find_all("tr"):
                cols = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cols) >= 5 and crop_search.lower() in cols[0].lower():
                    price_str = cols[4]
                    price_value = float(price_str.replace(",", "")) / 100
                    return round(price_value, 2), "Agmarknet"
    except Exception as e:
        print("Agmarknet fetch failed:", e)

    fallback = fetch_price_via_openai(crop_name, "crop")
    if fallback is not None:
        return fallback, "OpenAI"

    reference_price = fallback_crop_prices.get(crop_name.lower())
    if reference_price is not None:
        return reference_price, "reference price"

    return None, None


def fetch_fertilizer_market_price(fertilizer_name):
    fert_filters = {}
    if FERTILIZER_STATE_FILTER:
        fert_filters["state"] = FERTILIZER_STATE_FILTER
    if FERTILIZER_DISTRICT_FILTER:
        fert_filters["district"] = FERTILIZER_DISTRICT_FILTER
    if FERTILIZER_MARKET_FILTER:
        fert_filters["market"] = FERTILIZER_MARKET_FILTER

    price, label = fetch_price_from_data_gov(
        fertilizer_name,
        FERTILIZER_DATASET_ID,
        FERTILIZER_COMMODITY_FIELD,
        FERTILIZER_PRICE_FIELD,
        date_field=FERTILIZER_DATE_FIELD,
        price_divisor=FERTILIZER_PRICE_DIVISOR,
        filters=fert_filters,
    )
    if price is not None:
        return price, label

    fallback = fetch_price_via_openai(fertilizer_name, "fertilizer")
    if fallback is not None:
        return fallback, "OpenAI"

    reference_price = fertilizer_price_reference.get(fertilizer_name.lower())
    if reference_price is not None:
        return reference_price, "reference price"

    return None, None

# --------------------------------------------------------------------------


@app.route("/crop_predict", methods=["GET", "POST"])
@login_required
def crop_predict():
    result = None
    image_path = None
    explanation_text = None

    if request.method == "POST":
        try:
            result_source = None
            # ---------------- Input Values ----------------
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # ---------------- Model Prediction ----------------
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            mx_features = mx.transform(features)
            sc_features = sc.transform(mx_features)
            prediction = model.predict(sc_features)[0]

            crop_name = crop_dict[prediction]

            # ---------------- Fetch Market Price ----------------
            latest_price, price_source = fetch_latest_market_price(crop_name)
            if latest_price:
                market_price = f"₹{latest_price}"
                result_source = price_source
            else:
                # fallback from dictionary
                market_price = f"₹{price_dict[prediction]}"
                result_source = "model reference table"

            # ---------------- Image Handling ----------------
            image_folder = os.path.join("static", "images")
            valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
            for ext in valid_exts:
                possible_path = os.path.join(image_folder, f"{crop_name}{ext}")
                if os.path.exists(possible_path):
                    image_path = f"{crop_name.lower()}{ext}"
                    break

            # ---------------- Explanation Handling ----------------
            explanation_text = explanation_dict.get(crop_name, [
                "This crop suits the current soil and weather pattern.",
                "Nutrient and pH levels are within optimal range.",
                "Rainfall and temperature support healthy growth.",
                "The model identified this crop as the best economic choice."
            ])

            # ---------------- Final Result ----------------
            result = {
                "crop_name": crop_name.upper(),
                "market_price": market_price,
                "price_source": result_source
            }

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template(
        "crop_predict.html",
        result=result,
        image_path=image_path,
        explanation_text=explanation_text
    )


# -------------------- Load Fertilizer Model --------------------
with open('Fertilizer_Prediction_ML-main/rf_fertilizer_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('Fertilizer_Prediction_ML-main/soil_mapping.pkl', 'rb') as f:
    soil_mapping = pickle.load(f)

with open('Fertilizer_Prediction_ML-main/crop_mapping.pkl', 'rb') as f:
    crop_mapping = pickle.load(f)

with open('Fertilizer_Prediction_ML-main/fertilizer_mapping.pkl', 'rb') as f:
    fertilizer_reverse_mapping = pickle.load(f)

fertilizer_explanations = {
    "Urea": [
        "Urea provides high nitrogen content to boost leaf growth.",
        "It is suitable for crops with nitrogen-deficient soil.",
        "Supports overall vegetative development and yield."
    ],
    "Urea and DAP": [
        "Combination supplies both nitrogen and phosphorus.",
        "Enhances root development and early growth.",
        "Balanced nutrients improve overall crop performance."
    ],
    "Urea and MOP": [
        "Supplies nitrogen and potassium simultaneously.",
        "Promotes leaf growth and strong stem development.",
        "Ideal for crops needing potassium for fruiting."
    ],
    "DAP": [
        "DAP provides nitrogen and phosphorus for root & leaf growth.",
        "Suitable for phosphorus-deficient soils.",
        "Encourages early maturity and higher yield."
    ],
    "DAP and MOP": [
        "Supplies phosphorus and potassium for strong roots & fruits.",
        "Improves flowering and fruit quality.",
        "Balances nutrients for optimal crop health."
    ],
    "MOP": [
        "MOP supplies potassium for fruiting, flowering, and overall plant health.",
        "Strengthens stems and increases disease resistance.",
        "Ideal for potassium-deficient soils."
    ],
    "Good NPK": [
        "Balanced NPK fertilizer provides all essential nutrients.",
        "Supports growth, root development, and fruiting equally.",
        "Suitable for general purpose crop nutrition."
    ]
}

def encode_value(value, mapping):
    return mapping.get(value, -1)  # -1 if unseen

def predict_fertilizer(input_dict):
    df = pd.DataFrame({
        'Temperature': [input_dict['Temperature']],
        'Humidity': [input_dict['Humidity']],
        'Rainfall': [input_dict['Rainfall']],
        'pH': [input_dict['pH']],
        'N': [input_dict['N']],
        'P': [input_dict['P']],
        'K': [input_dict['K']],
        'Soil': [encode_value(input_dict['Soil'], soil_mapping)],
        'Crop': [encode_value(input_dict['Crop'], crop_mapping)]
    })
    pred_idx = rf_model.predict(df)[0]
    return fertilizer_reverse_mapping[pred_idx]

@app.route("/fertilizer_predict", methods=["GET", "POST"])
@login_required
def fertilizer_predict():
    result = None
    image_path = None
    explanation_text = None
    market_price = None
    price_source = None

    if request.method == "POST":
        try:
            user_input = {
                'Temperature': float(request.form['Temperature']),
                'Humidity': float(request.form['Humidity']),
                'Rainfall': float(request.form['Rainfall']),
                'pH': float(request.form['pH']),
                'N': float(request.form['N']),
                'P': float(request.form['P']),
                'K': float(request.form['K']),
                'Soil': request.form['Soil'],
                'Crop': request.form['Crop']
            }

            fertilizer = predict_fertilizer(user_input)
            result = fertilizer

            explanation_text = fertilizer_explanations.get(fertilizer, ["No explanation available."])

            live_price, price_source = fetch_fertilizer_market_price(fertilizer)
            if live_price is not None:
                market_price = f"₹{live_price}"

            # Image search
            image_folder = "fertilizer_images"
            valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
            for ext in valid_exts:
                possible_path = os.path.join(image_folder, f"{fertilizer}{ext}")
                if os.path.exists(possible_path):
                    image_path = f"{fertilizer.replace(' ', '_')}.jpeg"  # just the filename
                    break

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")

    return render_template(
        "fertilizer_predict.html",
        result=result,
        image_path=image_path,
        explanation_text=explanation_text,
        market_price=market_price,
        price_source=price_source
    )


@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/price_lookup", methods=["GET", "POST"])
@login_required
def price_lookup():
    lookup_result = None
    item_type = "crop"
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        item_type = request.form.get("item_type", "crop")

        if not query:
            flash("Please provide a crop or fertilizer name.", "warning")
        else:
            if item_type == "crop":
                price_value, source = fetch_latest_market_price(query)
            else:
                price_value, source = fetch_fertilizer_market_price(query)

            if price_value is None:
                flash("Unable to fetch real-time price right now. Please try again later.", "danger")
            else:
                lookup_result = {
                    "name": query.title(),
                    "price": f"₹{price_value}",
                    "type": item_type,
                    "source": source or "external provider"
                }

    return render_template(
        "price_lookup.html",
        lookup_result=lookup_result,
        selected_type=item_type,
        query=query
    )


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

PLANT_MODEL_PATH = "plant disease/plant_disease_model.h5"
CLASS_NAMES_PATH = "plant disease/class_names.pkl"

plant_model = load_model(PLANT_MODEL_PATH)

with open(CLASS_NAMES_PATH, "rb") as f:
    plant_classes = pickle.load(f)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# -------------------- Plant Disease Route --------------------
@app.route("/plant_predict", methods=["GET", "POST"])
@login_required
def plant_predict():
    result = None
    image_path = None

    if request.method == "POST":
        if "plant_image" not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)

        file = request.files["plant_image"]
        if file.filename == "":
            flash("No image selected", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join("static/uploads", filename)
            file.save(save_path)
            image_path = filename

            # Preprocess and predict
            img_tensor = preprocess_image(save_path)
            pred = plant_model.predict(img_tensor, verbose=0)
            class_index = np.argmax(pred)
            confidence = np.max(pred) * 100
            predicted_class = plant_classes[class_index]

            result = {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2)
            }

        else:
            flash("Invalid file type. Only png/jpg/jpeg allowed.", "danger")

    return render_template("plant_predict.html", result=result, image_path=image_path)

from functools import wraps



# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
