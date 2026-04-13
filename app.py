import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GuavaVision · Deteksi Penyakit Jambu",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.hero {
    background: linear-gradient(135deg, #0f2027 0%, #1a3a2a 50%, #0d2818 100%);
    border: 1px solid #238636;
    border-radius: 16px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(35,134,54,.3) 0%, transparent 70%);
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    color: #ffffff;
    margin: 0 0 .6rem;
}
.hero-title span { color: #3fb950; }
.hero-sub { font-size: 1.05rem; color: #8b949e; font-weight: 300; margin: 0; }

.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
}
.card-title { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #e6edf3; margin: 0 0 .4rem; }
.card-val { font-size: 2.2rem; font-weight: 600; color: #3fb950; margin: 0; line-height: 1; }
.card-desc { font-size: .82rem; color: #8b949e; margin: .3rem 0 0; }

.sec-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e6edf3;
    border-left: 4px solid #3fb950;
    padding-left: .75rem;
    margin: 1.5rem 0 1rem;
}

.conf-row { display: flex; align-items: center; gap: 10px; margin: .45rem 0; }
.conf-label { width: 130px; font-size: .83rem; color: #c9d1d9; flex-shrink: 0; }
.conf-bar-bg { flex: 1; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 4px; }
.conf-pct { width: 48px; text-align: right; font-size: .83rem; color: #8b949e; }

.info-pill {
    display: flex;
    align-items: flex-start;
    gap: .7rem;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: .5rem 0;
}
.info-pill-icon { font-size: 1.4rem; flex-shrink: 0; }
.info-pill-body .label {
    font-size: .72rem; color: #8b949e;
    text-transform: uppercase; letter-spacing: .08em; margin: 0;
}
.info-pill-body .value { font-size: .93rem; color: #e6edf3; font-weight: 500; margin: .1rem 0 0; }

.model-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: .9rem 1.2rem;
    border-radius: 8px;
    margin: .35rem 0;
    background: #0d1117;
    border: 1px solid #21262d;
}
.model-row.best { background: #0f2a1a; border: 1px solid #238636; }
.model-name { font-weight: 500; color: #e6edf3; font-size: .95rem; }
.model-badge-best {
    font-size: .68rem; padding: .15rem .5rem;
    background: #238636; color: #fff;
    border-radius: 999px; text-transform: uppercase; letter-spacing: .06em;
}

.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .55rem 1.4rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    width: 100%;
}
.stButton > button:hover { background: #2ea043 !important; }

[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #21262d !important; }
[data-baseweb="tab"] { color: #8b949e !important; }
[aria-selected="true"] { color: #3fb950 !important; border-bottom: 2px solid #3fb950 !important; }

hr { border-color: #21262d; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Konstanta ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "best_model_guava.h5"   # ← Sesuaikan jika path berbeda
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["anthracnose", "healthy", "multiple", "scorch", "yld"]
BAR_COLORS  = ["#3fb950", "#58a6ff", "#f0883e", "#d29922", "#bc8cff"]

CLASS_INFO = {
    "healthy": {
        "label": "Sehat",
        "emoji": "✅",
        "color": "#3fb950",
        "desc": "Daun jambu dalam kondisi sehat, tidak terdeteksi penyakit.",
        "action": "Pertahankan kondisi perawatan saat ini. Lakukan pemantauan rutin.",
        "severity": "Tidak ada",
    },
    "anthracnose": {
        "label": "Antraknosa",
        "emoji": "🍂",
        "color": "#f85149",
        "desc": "Penyakit jamur Colletotrichum gloeosporioides — bercak coklat/hitam pada daun dan buah.",
        "action": "Semprotkan fungisida berbasis tembaga. Pangkas bagian terinfeksi. Pastikan sirkulasi udara baik.",
        "severity": "Tinggi",
    },
    "scorch": {
        "label": "Daun Gosong",
        "emoji": "🔥",
        "color": "#d29922",
        "desc": "Kerusakan daun akibat paparan panas berlebihan, kekeringan, atau kekurangan air.",
        "action": "Tingkatkan frekuensi penyiraman. Berikan naungan sementara. Periksa drainase tanah.",
        "severity": "Sedang",
    },
    "multiple": {
        "label": "Infeksi Ganda",
        "emoji": "⚠️",
        "color": "#f85149",
        "desc": "Tanaman mengalami lebih dari satu jenis penyakit secara bersamaan.",
        "action": "Segera isolasi tanaman. Konsultasikan dengan ahli pertanian. Lakukan penanganan komprehensif.",
        "severity": "Sangat Tinggi",
    },
    "yld": {
        "label": "Yellow Leaf Disease",
        "emoji": "🟡",
        "color": "#d29922",
        "desc": "Daun menguning akibat defisiensi nutrisi, infeksi virus, atau gangguan akar.",
        "action": "Periksa pH tanah dan nutrisi. Berikan pupuk mikro. Cek kemungkinan infeksi virus.",
        "severity": "Sedang",
    },
}

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Return (model, error_str). error_str = None jika sukses."""
    if not os.path.exists(path):
        return None, f"File tidak ditemukan: `{path}`"
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

# ── Preprocessing VGG16 ────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = arr[..., ::-1]        # RGB → BGR
    arr[..., 0] -= 103.939
    arr[..., 1] -= 116.779
    arr[..., 2] -= 123.680
    return np.expand_dims(arr, axis=0)

# ── Prediksi ───────────────────────────────────────────────────────────────────
def run_prediction(model, img: Image.Image):
    x     = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")
    st.markdown("---")

    model, model_err = load_model(MODEL_PATH)

    if model_err:
        st.error(f"**Model gagal dimuat**\n\n{model_err}")
    else:
        st.success("✅ Model VGG16 berhasil dimuat")

    st.markdown("---")
    st.markdown("### 📊 Performa Model")
    st.markdown("""
    <div class="model-row best">
        <div><div class="model-name">🏆 VGG16</div></div>
        <div style="text-align:right">
            <div style="font-size:1.2rem;font-weight:700;color:#3fb950">99.3%</div>
            <span class="model-badge-best">Terpilih</span>
        </div>
    </div>
    <div class="model-row">
        <div class="model-name">MobileNetV2</div>
        <div style="font-size:1.1rem;font-weight:600;color:#58a6ff">92.4%</div>
    </div>
    <div class="model-row">
        <div class="model-name">MobileNetV3</div>
        <div style="font-size:1.1rem;font-weight:600;color:#8b949e">78.9%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Dataset")
    st.markdown("""
    <div style='font-size:.83rem;color:#8b949e;line-height:1.9'>
    • <b style='color:#c9d1d9'>Sekunder:</b> 54,608 gambar<br>
    • <b style='color:#c9d1d9'>Primer:</b> 500 gambar<br>
    • <b style='color:#c9d1d9'>Split:</b> 70 / 15 / 15 %<br>
    • <b style='color:#c9d1d9'>Kelas:</b> 5 kategori<br>
    • <b style='color:#c9d1d9'>Input:</b> 224 × 224 px
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"GuavaVision v1.0 · VGG16 · `{MODEL_PATH}`")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <p class="hero-title">Guava<span>Vision</span></p>
    <p class="hero-sub">Sistem Deteksi Penyakit Daun Jambu · Deep Learning VGG16 · Akurasi 99.3%</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, title, val, desc in [
    (c1, "Test Accuracy", "99.3%", "VGG16 — Model Terbaik"),
    (c2, "Test Loss",     "0.026", "Cross-entropy loss"),
    (c3, "Kelas",         "5",     "Kategori penyakit"),
    (c4, "Dataset",       "3,000", "Gambar setelah sampling"),
]:
    with col:
        st.markdown(f"""
        <div class="card">
            <p class="card-title">{title}</p>
            <p class="card-val">{val}</p>
            <p class="card-desc">{desc}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 Deteksi Penyakit", "📊 Analisis Model", "📖 Panduan Kelas"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — DETEKSI
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="sec-head">Upload Gambar Daun Jambu</p>', unsafe_allow_html=True)

    # Blokir jika model belum siap
    if model_err:
        st.error(
            f"❌ **Model tidak dapat dimuat.**\n\n"
            f"Pastikan file `{MODEL_PATH}` berada satu folder dengan `app.py`, "
            f"kemudian restart aplikasi."
        )
        st.stop()

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader(
            "Pilih gambar (JPG / PNG / JPEG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🔬 Analisis Sekarang", use_container_width=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:3.5rem 1rem;
                        background:#161b22;border:2px dashed #30363d;
                        border-radius:12px;color:#8b949e'>
                <div style='font-size:3rem'>🍃</div>
                <p style='margin:.5rem 0 0;font-size:.9rem'>Belum ada gambar yang diunggah</p>
                <p style='font-size:.78rem;color:#484f58'>Seret & lepas atau klik untuk memilih file</p>
            </div>
            """, unsafe_allow_html=True)
            run_btn = False

    with col_res:
        st.markdown('<p class="sec-head">Hasil Prediksi</p>', unsafe_allow_html=True)

        if uploaded and run_btn:
            with st.spinner("Menganalisis gambar..."):
                try:
                    class_name, confidence, probs = run_prediction(model, img)
                except Exception as e:
                    st.error(f"Gagal memproses gambar: {e}")
                    st.stop()

            info = CLASS_INFO[class_name]

            # Kotak hasil utama
            st.markdown(f"""
            <div style='background:#0d1117;
                        border:1px solid {info["color"]}50;
                        border-left:4px solid {info["color"]};
                        border-radius:12px;
                        padding:1.3rem 1.5rem;
                        margin-bottom:1rem'>
                <p style='font-size:.72rem;color:#8b949e;margin:0 0 .3rem;
                           text-transform:uppercase;letter-spacing:.09em'>Deteksi</p>
                <p style='font-size:1.75rem;font-family:"DM Serif Display",serif;
                           color:#e6edf3;margin:0'>
                    {info["emoji"]} {info["label"]}
                </p>
                <p style='font-size:2.3rem;font-weight:700;color:{info["color"]};margin:.2rem 0 0'>
                    {confidence * 100:.2f}%
                </p>
                <p style='font-size:.8rem;color:#8b949e;margin:.1rem 0 0'>Tingkat Kepercayaan</p>
            </div>
            """, unsafe_allow_html=True)

            # Info pills
            st.markdown(f"""
            <div class="info-pill">
                <div class="info-pill-icon">📋</div>
                <div class="info-pill-body">
                    <p class="label">Deskripsi</p>
                    <p class="value">{info["desc"]}</p>
                </div>
            </div>
            <div class="info-pill">
                <div class="info-pill-icon">💊</div>
                <div class="info-pill-body">
                    <p class="label">Tindakan yang Disarankan</p>
                    <p class="value">{info["action"]}</p>
                </div>
            </div>
            <div class="info-pill">
                <div class="info-pill-icon">🚨</div>
                <div class="info-pill-body">
                    <p class="label">Tingkat Keparahan</p>
                    <p class="value" style="color:{info['color']}">{info["severity"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown(
                '<p class="sec-head" style="font-size:1.05rem;margin-top:1.3rem">'
                'Distribusi Probabilitas</p>',
                unsafe_allow_html=True,
            )
            sorted_idx = np.argsort(probs)[::-1]
            bars_html = ""
            for rank, idx in enumerate(sorted_idx):
                pct   = float(probs[idx]) * 100
                color = BAR_COLORS[rank]
                label = CLASS_INFO[CLASS_NAMES[idx]]["label"]
                bars_html += f"""
                <div class="conf-row">
                    <span class="conf-label">{label}</span>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill"
                             style="width:{pct:.2f}%;background:{color}"></div>
                    </div>
                    <span class="conf-pct">{pct:.1f}%</span>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div style='background:#161b22;border:1px solid #21262d;border-radius:12px;
                        padding:4rem 2rem;text-align:center'>
                <div style='font-size:2.8rem'>🔬</div>
                <p style='color:#8b949e;margin:.6rem 0 0;font-size:.9rem'>
                    Upload gambar terlebih dahulu<br>
                    lalu klik <b style='color:#3fb950'>Analisis Sekarang</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — ANALISIS MODEL
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="sec-head">Perbandingan Model</p>', unsafe_allow_html=True)

    for name, acc, loss, best in [
        ("VGG16",       99.33, 0.0259, True),
        ("MobileNetV2", 92.44, 0.2440, False),
        ("MobileNetV3", 78.90, 0.6120, False),
    ]:
        cls   = "best" if best else ""
        badge = '<span class="model-badge-best">🏆 Terpilih</span>' if best else ""
        color = "#3fb950" if best else "#58a6ff" if "V2" in name else "#8b949e"
        st.markdown(f"""
        <div class="model-row {cls}">
            <div>
                <div class="model-name">{name} {badge}</div>
                <div style="font-size:.78rem;color:#8b949e;margin-top:.2rem">Test Loss: {loss}</div>
            </div>
            <div style="text-align:right">
                <div style="font-size:1.55rem;font-weight:700;color:{color}">{acc}%</div>
                <div style="font-size:.72rem;color:#8b949e">Test Accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="sec-head">Classification Report — VGG16</p>', unsafe_allow_html=True)

    report_rows = [
        ("Anthracnose", 1.0000, 0.9778, 0.9888, 90),
        ("Healthy",     0.9783, 1.0000, 0.9890, 90),
        ("Multiple",    1.0000, 0.9889, 0.9944, 90),
        ("Scorch",      1.0000, 1.0000, 1.0000, 90),
        ("YLD",         0.9890, 1.0000, 0.9945, 90),
        ("Macro Avg",   0.9935, 0.9933, 0.9933, 450),
    ]
    rows_html = ""
    for cls_name, p, r, f1, sup in report_rows:
        bold = "font-weight:600;" if cls_name == "Macro Avg" else ""
        bg   = "background:#0f2a1a;" if cls_name == "Macro Avg" else ""
        rows_html += f"""
        <tr style='{bg}'>
            <td style='{bold}color:#e6edf3;padding:.55rem .8rem'>{cls_name}</td>
            <td style='color:#3fb950;text-align:center;padding:.55rem'>{p:.4f}</td>
            <td style='color:#58a6ff;text-align:center;padding:.55rem'>{r:.4f}</td>
            <td style='color:#f0883e;text-align:center;padding:.55rem'>{f1:.4f}</td>
            <td style='color:#8b949e;text-align:center;padding:.55rem'>{sup}</td>
        </tr>"""

    st.markdown(f"""
    <div style='overflow-x:auto'>
    <table style='width:100%;border-collapse:collapse;font-size:.88rem'>
        <thead>
            <tr style='border-bottom:2px solid #21262d'>
                <th style='color:#8b949e;text-align:left;padding:.55rem .8rem;font-weight:500'>Kelas</th>
                <th style='color:#3fb950;text-align:center;padding:.55rem;font-weight:500'>Precision</th>
                <th style='color:#58a6ff;text-align:center;padding:.55rem;font-weight:500'>Recall</th>
                <th style='color:#f0883e;text-align:center;padding:.55rem;font-weight:500'>F1-Score</th>
                <th style='color:#8b949e;text-align:center;padding:.55rem;font-weight:500'>Support</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sec-head">Detail Dataset</p>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    with d1:
        dataset_bars = [
            ("Anthracnose", 99, "#f85149", "10,921"),
            ("Healthy",    100, "#3fb950", "11,000"),
            ("Multiple",   100, "#bc8cff", "11,009"),
            ("Scorch",      97, "#d29922", "10,678"),
            ("YLD",        100, "#58a6ff", "11,000"),
        ]
        bars_html2 = "".join([
            f"""<div class="conf-row">
                <span class="conf-label" style='width:110px'>{lbl}</span>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%;background:{col}"></div>
                </div>
                <span class="conf-pct">{num}</span>
            </div>"""
            for lbl, pct, col, num in dataset_bars
        ])
        st.markdown(f"""
        <div class="card">
            <p class="card-title">Dataset Sekunder (Publik)</p>
            <div style='margin-top:.8rem'>{bars_html2}</div>
        </div>""", unsafe_allow_html=True)

    with d2:
        config_rows = [
            ("Base Model",  "VGG16 (ImageNet)",  "#e6edf3"),
            ("Input Size",  "224 × 224 px",       "#e6edf3"),
            ("Optimizer",   "Adam",               "#e6edf3"),
            ("Split",       "70 / 15 / 15 %",     "#e6edf3"),
            ("Fine-tuning", "✓ Aktif",            "#3fb950"),
            ("Augmentasi",  "✓ Aktif",            "#3fb950"),
        ]
        cfg_html = "".join([
            f"""<div style='display:flex;justify-content:space-between;
                           border-bottom:1px solid #21262d;padding:.2rem 0'>
                <span style='color:#8b949e'>{k}</span>
                <span style='color:{vc};font-weight:500'>{v}</span>
            </div>"""
            for k, v, vc in config_rows
        ])
        st.markdown(f"""
        <div class="card">
            <p class="card-title">Konfigurasi Training VGG16</p>
            <div style='margin-top:.8rem;font-size:.85rem;line-height:2.1'>{cfg_html}</div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — PANDUAN KELAS
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="sec-head">Panduan Kelas Penyakit</p>', unsafe_allow_html=True)

    for key, info in CLASS_INFO.items():
        with st.expander(f"{info['emoji']}  {info['label']}  ({key})", expanded=(key == "healthy")):
            ca, cb = st.columns([1, 2])
            with ca:
                st.markdown(f"""
                <div style='background:#161b22;border:2px solid {info["color"]}40;
                            border-radius:10px;padding:1.5rem;text-align:center'>
                    <div style='font-size:3rem'>{info["emoji"]}</div>
                    <p style='color:{info["color"]};font-weight:600;font-size:1.05rem;margin:.5rem 0 0'>
                        {info["label"]}
                    </p>
                    <p style='color:#8b949e;font-size:.78rem;margin:.2rem 0 0'>
                        Kelas: <code>{key}</code>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with cb:
                st.markdown(f"""
                <div class="info-pill" style='margin:.3rem 0'>
                    <div class="info-pill-icon">📋</div>
                    <div class="info-pill-body">
                        <p class="label">Deskripsi</p>
                        <p class="value">{info["desc"]}</p>
                    </div>
                </div>
                <div class="info-pill" style='margin:.3rem 0'>
                    <div class="info-pill-icon">💊</div>
                    <div class="info-pill-body">
                        <p class="label">Penanganan</p>
                        <p class="value">{info["action"]}</p>
                    </div>
                </div>
                <div class="info-pill" style='margin:.3rem 0'>
                    <div class="info-pill-icon">🚨</div>
                    <div class="info-pill-body">
                        <p class="label">Tingkat Keparahan</p>
                        <p class="value" style="color:{info['color']}">{info["severity"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:1.5rem;border-top:1px solid #21262d'>
    <p style='color:#484f58;font-size:.8rem;margin:0'>
        GuavaVision · Deteksi Penyakit Daun Jambu · VGG16 Transfer Learning
    </p>
    <p style='color:#484f58;font-size:.75rem;margin:.3rem 0 0'>
        Dataset: Primer (500) + Sekunder (54,608 gambar) · 5 Kelas · Test Accuracy 99.33%
    </p>
</div>
""", unsafe_allow_html=True)