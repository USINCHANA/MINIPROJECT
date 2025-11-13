# 🌟 Advanced Pneumonia Detection Result Dashboard 🌟
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ✅ Load the trained model
model = load_model(r"C:\Users\sinch\Desktop\pneumonia_model.keras")
print("✅ Model loaded successfully!")

# ✅ Choose image path
img_path = r"C:\Users\sinch\Downloads\chest_xray\test\PNEUMONIA\person100_bacteria_475.jpeg"

# ✅ Preprocess image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# ✅ Predict
raw_pred = model.predict(img_array)
prediction = raw_pred[0][0]

# ✅ Set result label, color & confidence
if prediction >= 0.5:
    label = "PNEUMONIA DETECTED 😷"
    confidence = prediction * 100
    color = "red"
    advice = "⚠️ Seek medical advice immediately. Early treatment helps recovery."
else:
    label = "NORMAL LUNGS 🫁"
    confidence = (1 - prediction) * 100
    color = "green"
    advice = "✅ Your lungs look healthy! Maintain good hygiene and stay fit."

# ✅ Create figure layout
fig = plt.figure(figsize=(7, 8))
grid = plt.GridSpec(3, 1, height_ratios=[4, 0.5, 1.2], hspace=0.4)

# =======================
# 1️⃣ IMAGE DISPLAY
# =======================
ax1 = fig.add_subplot(grid[0])
ax1.imshow(image.load_img(img_path))
ax1.axis("off")

# Diagnosis banner
rect = patches.FancyBboxPatch(
    (0.02, 0.90), 0.96, 0.09, boxstyle="round,pad=0.03",
    transform=ax1.transAxes, facecolor=color, alpha=0.3
)
ax1.add_patch(rect)
ax1.text(0.5, 0.94, label, color=color, fontsize=16, fontweight='bold',
         ha='center', va='center', transform=ax1.transAxes)

# =======================
# 2️⃣ CONFIDENCE BAR
# =======================
ax2 = fig.add_subplot(grid[1])
ax2.barh(["Confidence"], [confidence], color=color, alpha=0.8)
ax2.set_xlim(0, 100)
ax2.set_xlabel("Confidence (%)")
ax2.set_title("AI Confidence Level", fontsize=12)
for i, v in enumerate([confidence]):
    ax2.text(v + 1, i, f"{v:.2f}%", color='black', va='center', fontsize=12)

# =======================
# 3️⃣ SUMMARY BOX
# =======================
ax3 = fig.add_subplot(grid[2])
ax3.axis("off")
ax3.text(0.5, 0.7, f"🧠 AI Diagnosis: {label}",
         ha='center', fontsize=13, fontweight='bold', color=color)
ax3.text(0.5, 0.45, f"📈 Confidence: {confidence:.2f}%",
         ha='center', fontsize=11, color='black')
ax3.text(0.5, 0.2, advice,
         ha='center', fontsize=10, color='darkblue', wrap=True)

plt.suptitle("Pneumonia Detection Result Dashboard", fontsize=15, fontweight='bold')
plt.show()

# ✅ Terminal summary
print("\n🩺 ====== AI DIAGNOSIS SUMMARY ======")
print(f"📂 Image Path: {img_path}")
print(f"🧠 Result: {label}")
print(f"📊 Confidence: {confidence:.2f}%")
print(f"💡 Advice: {advice}")
print("=====================================")