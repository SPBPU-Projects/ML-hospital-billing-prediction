# =========================================
# مرحله 1) وارد کردن کتابخانه‌ها + خواندن دیتاست
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# چرا این کتابخانه‌ها؟
# pandas: برای خواندن فایل CSV و کار با جدول داده‌ها (DataFrame)
# numpy: برای کارهای عددی و آرایه‌ای (بعداً برای نام ویژگی‌ها و ...) مفید است
# matplotlib: برای رسم نمودارها (EDA و نمودارهای تشخیصی مدل)

# تنظیمات چاپ دیتافریم در کنسول تا ستون‌ها کامل‌تر نمایش داده شوند
pd.set_option("display.max_columns", 200)  # تعداد ستون‌های قابل نمایش را زیاد می‌کند
pd.set_option("display.width", 200)        # عرض چاپ در کنسول را بیشتر می‌کند تا خط‌ها کمتر بشکنند

# مسیر فایل دیتاست
# r"..." یعنی رشته خام (Raw String) تا بک‌اسلش‌های ویندوز مشکل ایجاد نکنند
path = r"data/healthcare_dataset.csv"
df = pd.read_csv(path)  # فایل CSV را به DataFrame تبدیل می‌کند

# چاپ شکل داده‌ها (تعداد سطر و ستون)
# طبق خروجی شما: Shape: (55500, 15)
print("Shape:", df.shape)

# چاپ 10 ردیف اول برای بررسی سریع ساختار و مقادیر
# دفاع: برای sanity check اولیه (درست بودن ستون‌ها و نمونه داده‌ها)
print(df.head(10))

# اطلاعات کلی دیتافریم: نوع داده‌ها، تعداد مقادیر non-null، مصرف حافظه
# طبق خروجی شما: همه ستون‌ها 55500 non-null هستند (یعنی مقدار گمشده نداریم)
print("\n--- df.info() ---")
df.info()

# توصیف آماری/کیفی ستون‌ها:
# برای عددی: میانگین، انحراف معیار، مین/ماکس، چارک‌ها
# برای متنی: تعداد یکتا، پرتکرارترین مقدار و فراوانی
print("\n--- describe(include='all').T (top 20 rows) ---")
print(df.describe(include="all").T.head(20))

# بررسی مقادیر گمشده (Missing Values)
# df.isna().sum(): تعداد NaN در هر ستون
# فقط ستون‌هایی که تعدادشان >0 است چاپ می‌شود
# طبق خروجی شما: Series([], ...) یعنی هیچ missing نداریم
print("\n--- Missing values (only >0) ---")
missing = df.isna().sum().sort_values(ascending=False)
print(missing[missing > 0])

# بررسی ردیف‌های تکراری (Duplicate Rows)
# طبق خروجی شما: 534 ردیف تکراری وجود دارد
print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

# چاپ لیست ستون‌ها برای اینکه دقیق بدانیم چه ستون‌هایی داریم
print("\n--- Columns ---")
print(list(df.columns))


# =========================================
# مرحله 2) تعیین متغیر هدف (Target) و ویژگی‌ها (Features)
# =========================================

# هدف: پیش‌بینی مقدار Billing Amount => مسئله رگرسیون
TARGET = "Billing Amount"

# ستون‌هایی که حذف می‌کنیم چون:
# Name: اسم فرد معمولاً اطلاعات قابل تعمیم ندارد و نویز ایجاد می‌کند
# Doctor/Hospital: تعداد مقدار یکتا خیلی زیاد است و می‌تواند باعث overfit و عدم تعمیم شود
# (طبق describe شما: Doctor و Hospital تعداد یکتای بسیار بالا دارند)
DROP_COLS = ["Name", "Doctor", "Hospital"]

# ساخت لیست ویژگی‌ها: هر چیزی غیر از ستون‌های حذف شده و غیر از Target
FEATURES = [c for c in df.columns if c not in DROP_COLS + [TARGET]]

print("Target:", TARGET)
print("\nFeatures:")
for c in FEATURES:
    print("-", c)

# نکته دفاع:
# حذف Doctor/Hospital معمولاً برای جلوگیری از «حافظه‌سازی» مدل و بزرگ شدن بی‌دلیل تعداد ویژگی‌هاست.


# =========================================
# مرحله 3) تحلیل اولیه داده (EDA) + مهندسی ویژگی
# =========================================

# 1) تبدیل ستون‌های تاریخ از رشته (object) به datetime
# چون برای محاسبه اختلاف تاریخ باید datetime باشند
# errors="coerce": اگر تاریخ نامعتبر باشد به NaT تبدیل می‌شود
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

# گزارش تعداد تاریخ‌های خراب (NaT)
# طبق خروجی شما: 0 مورد خراب بوده => همه تاریخ‌ها سالم هستند
bad_adm = df["Date of Admission"].isna().sum()
bad_dis = df["Discharge Date"].isna().sum()
print("\n--- Date parsing ---")
print("تعداد تاریخ پذیرش نامعتبر (NaT):", bad_adm)
print("تعداد تاریخ ترخیص نامعتبر (NaT):", bad_dis)

# 2) مهندسی ویژگی: طول مدت بستری (Length_of_Stay) بر حسب روز
# تعریف: تاریخ ترخیص - تاریخ پذیرش
df["Length_of_Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

print("\n--- نمونه Length_of_Stay ---")
print(df[["Date of Admission", "Discharge Date", "Length_of_Stay"]].head(10))

print("\n--- آمار Length_of_Stay ---")
print(df["Length_of_Stay"].describe())

# بررسی منفی بودن مدت بستری
# اگر منفی باشد یعنی داده تاریخ‌ها مشکل دارد (ترخیص قبل از پذیرش)
# طبق خروجی شما: 0 مورد منفی
neg_los = (df["Length_of_Stay"] < 0).sum()
print("\nتعداد Length_of_Stay منفی:", int(neg_los))

# 3) حذف ردیف‌های تکراری
# چرا؟ چون وجود duplicate باعث می‌شود بعضی نمونه‌ها چندبار در یادگیری شمرده شوند و bias ایجاد شود
# طبق خروجی شما: 534 ردیف حذف شد و shape شد (54966, 16)
print("\n--- حذف تکراری‌ها ---")
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print("تعداد ردیف‌های تکراری حذف شده:", before - after)
print("شکل جدید داده:", df.shape)

# 4) بررسی توزیع هدف (Billing Amount)
# طبق خروجی شما: مین منفی است (حدود -2008) که منطقی نیست
print("\n--- خلاصه Billing Amount ---")
print(df["Billing Amount"].describe())
print("\nکمترین Billing Amount:", df["Billing Amount"].min())
print("بیشترین Billing Amount:", df["Billing Amount"].max())

# رسم هیستوگرام برای دیدن شکل کلی توزیع (چولگی، گستره، تراکم)
plt.figure()
plt.hist(df["Billing Amount"], bins=50)
plt.xlabel("Billing Amount")
plt.ylabel("Frequency")
plt.title("توزیع Billing Amount")
plt.tight_layout()
plt.show()

# رسم Boxplot برای مشاهده outlier و پراکندگی کلی
plt.figure()
plt.boxplot(df["Billing Amount"], vert=False)
plt.xlabel("Billing Amount")
plt.title("نمودار جعبه‌ای Billing Amount")
plt.tight_layout()
plt.show()

# 5) مقایسه میانگین Billing Amount در گروه‌ها (تحلیل گروهی)
# هدف: ببینیم آیا نوع پذیرش یا نوع بیماری روی میانگین هزینه اثر دارد یا نه

# 5.1) میانگین Billing Amount به تفکیک Admission Type
# طبق خروجی شما:
# Elective≈25612.14, Urgent≈25514.53, Emergency≈25505.33
print("\n--- میانگین Billing Amount بر اساس Admission Type ---")
group_admission = df.groupby("Admission Type")["Billing Amount"].mean().sort_values(ascending=False)
print(group_admission)

plt.figure()
group_admission.plot(kind="bar")
plt.ylabel("میانگین Billing Amount")
plt.title("میانگین Billing Amount بر اساس نوع پذیرش")
plt.tight_layout()
plt.show()

# 5.2) میانگین Billing Amount به تفکیک Medical Condition
# طبق خروجی شما:
# Obesity≈25804.36 ... Cancer≈25152.32
print("\n--- میانگین Billing Amount بر اساس Medical Condition ---")
group_condition = df.groupby("Medical Condition")["Billing Amount"].mean().sort_values(ascending=False)
print(group_condition)

plt.figure(figsize=(9, 4))
group_condition.plot(kind="bar")
plt.ylabel("میانگین Billing Amount")
plt.title("میانگین Billing Amount بر اساس وضعیت پزشکی")
plt.tight_layout()
plt.show()

# 6) بررسی رابطه Age و Billing Amount با scatter
# هدف: ببینیم آیا روندی با سن وجود دارد یا خیر
plt.figure()
plt.scatter(df["Age"], df["Billing Amount"], alpha=0.3)
plt.xlabel("Age")
plt.ylabel("Billing Amount")
plt.title("رابطه سن با Billing Amount")
plt.tight_layout()
plt.show()

# 7) به‌روزرسانی لیست FEATURES بعد از اضافه کردن Length_of_Stay
# چون الان یک ستون جدید داریم که باید به عنوان ویژگی وارد مدل شود
FEATURES = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
print("\n--- ویژگی‌های به‌روزشده ---")
for c in FEATURES:
    print("-", c)


# =========================================
# مرحله 4) پیش‌پردازش + تقسیم داده + ساخت ColumnTransformer
# =========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1) حذف مقادیر نامعتبر هدف: Billing Amount منفی
# چرا؟ هزینه منفی معمولاً خطای ثبت داده است و مدل را گمراه می‌کند
# طبق خروجی شما: 106 ردیف حذف شد و shape شد (54860, 16)
before = df.shape[0]
df = df[df[TARGET] >= 0].copy()
after = df.shape[0]
print("\nتعداد ردیف‌های Billing Amount منفی حذف شده:", before - after)
print("شکل بعد از حذف منفی‌ها:", df.shape)

# 2) حذف ستون‌های تاریخ خام
# چون از آن‌ها ویژگی معنی‌دار Length_of_Stay ساخته‌ایم و تاریخ خام را نمی‌خواهیم
DATE_COLS = ["Date of Admission", "Discharge Date"]
df = df.drop(columns=DATE_COLS)

# به‌روزرسانی FEATURES بعد از حذف تاریخ‌ها
# طبق خروجی شما: 10 ویژگی نهایی
FEATURES = [c for c in df.columns if c not in DROP_COLS + [TARGET]]
print("\nویژگی‌های نهایی:")
for c in FEATURES:
    print("-", c)

# 3) جدا کردن X (ویژگی‌ها) و y (هدف)
X = df[FEATURES]
y = df[TARGET]

# 4) شناسایی ستون‌های عددی و دسته‌ای
# عددی‌ها: نیاز به مقیاس‌بندی (StandardScaler)
# دسته‌ای‌ها: نیاز به One-Hot Encoding
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nویژگی‌های عددی:", numeric_features)
print("ویژگی‌های دسته‌ای:", categorical_features)

# 5) تعریف pipeline های پیش‌پردازش
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())  # مقیاس‌بندی ویژگی‌های عددی
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    # handle_unknown="ignore": اگر در val/test دسته جدید دید خطا ندهد
    # sparse_output=False: خروجی را dense می‌دهد (ساده‌تر برای دیدن shape)
])

# 6) ColumnTransformer: اعمال پردازش مناسب روی هر نوع ستون
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 7) تقسیم داده به Train/Validation/Test با نسبت 70/15/15
# ابتدا 15% برای تست
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# سپس از 85% باقی‌مانده، 0.1765 می‌گیریم تا تقریباً 15% کل برای validation شود
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

# طبق خروجی شما:
# Train: (38400, 10)
# Validation: (8231, 10)
# Test: (8229, 10)
print("\nاندازه تقسیم داده:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# 8) fit کردن preprocessor فقط روی train برای جلوگیری از Data Leakage
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep   = preprocessor.transform(X_val)
X_test_prep  = preprocessor.transform(X_test)

# طبق خروجی شما بعد از One-Hot:
# Train: (38400, 35)
# Validation: (8231, 35)
# Test: (8229, 35)
print("\nاندازه داده بعد از پیش‌پردازش:")
print("Train:", X_train_prep.shape)
print("Validation:", X_val_prep.shape)
print("Test:", X_test_prep.shape)


# =========================================
# مرحله 5) آموزش مدل‌ها + ارزیابی روی Validation
# =========================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# تابع ارزیابی رگرسیون
# MAE: میانگین قدر مطلق خطا (قابل فهم و به واحد پول)
# RMSE: حساس‌تر به خطاهای بزرگ (outlier)
# R2: میزان واریانس توضیح داده شده (اگر نزدیک صفر یا منفی باشد، مدل ضعیف است)
def evaluate_regression(y_true, y_pred, name="model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse_val = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "MAE": mae, "RMSE": rmse_val, "R2": r2}

# ساخت pipeline برای اینکه preprocessing و مدل با هم یکپارچه باشند
# مزیت: جلوی اشتباهات (مثل fit/transform جداگانه) را می‌گیرد و روند استاندارد می‌شود

# مدل 1: رگرسیون خطی (به عنوان baseline ساده)
linreg_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

# مدل 2: رندوم فارست برای روابط غیرخطی
# n_estimators=300: تعداد درخت‌ها
# n_jobs=-1: استفاده از همه هسته‌های CPU
rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nدر حال آموزش Linear Regression ...")
linreg_model.fit(X_train, y_train)

print("در حال آموزش Random Forest ...")
rf_model.fit(X_train, y_train)

# پیش‌بینی روی validation
lin_pred = linreg_model.predict(X_val)
rf_pred = rf_model.predict(X_val)

# محاسبه معیارها
results = []
results.append(evaluate_regression(y_val, lin_pred, name="LinearRegression"))
results.append(evaluate_regression(y_val, rf_pred, name="RandomForest(n=300)"))

results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n--- نتایج Validation ---")
print(results_df)

# طبق خروجی شما:
# RandomForest: MAE≈11742.11, RMSE≈13792.26, R2≈0.065
# LinearRegression: MAE≈12375.92, RMSE≈14268.99, R2≈-0.0007
# نتیجه: RF بهتر است چون خطا کمتر و R2 مثبت دارد

# چاپ چند نمونه از پیش‌بینی‌ها برای بررسی منطقی بودن خروجی‌ها
print("\n--- چند نمونه پیش‌بینی (Validation) ---")
sample = pd.DataFrame({
    "y_true": y_val.iloc[:10].values,
    "lin_pred": lin_pred[:10],
    "rf_pred": rf_pred[:10],
})
print(sample)


# =========================================
# مرحله 6) تنظیم ابرپارامترها + Cross-Validation
# =========================================

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge

# چرا این مرحله؟
# چون مدل‌های پیچیده مثل RF با تنظیم پارامترها ممکن است بهتر شوند
# همچنین Ridge یک baseline خطی با regularization است

# تعریف RMSE بدون warning برای نسخه‌های مختلف sklearn
try:
    from sklearn.metrics import root_mean_squared_error
    def rmse(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)
except ImportError:
    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

def evaluate_on_val(model, X_val, y_val, name="model"):
    pred = model.predict(X_val)
    return {
        "model": name,
        "MAE": mean_absolute_error(y_val, pred),
        "RMSE": rmse(y_val, pred),
        "R2": r2_score(y_val, pred)
    }

# 1) Ridge baseline
ridge_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", Ridge(random_state=42))
])

print("\nدر حال آموزش Ridge (baseline) ...")
ridge_model.fit(X_train, y_train)
ridge_val_res = evaluate_on_val(ridge_model, X_val, y_val, name="Ridge")
print("نتیجه Ridge روی Validation:", ridge_val_res)

# طبق خروجی شما: Ridge تقریباً مثل Linear است و بهتر نشده

# 2) تنظیم RF با RandomizedSearchCV
rf_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

# تعریف فضای جستجوی پارامترها
param_distributions = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.5, 0.8],
}

search = RandomizedSearchCV(
    estimator=rf_pipe,
    param_distributions=param_distributions,
    n_iter=20,  # 20 ترکیب تصادفی امتحان می‌شود
    scoring="neg_root_mean_squared_error",  # هدف کمینه کردن RMSE
    cv=5,       # 5-fold cross validation روی train
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nدر حال اجرای RandomizedSearchCV برای RandomForest (cv=5) ...")
search.fit(X_train, y_train)

print("\nبهترین پارامترهای RF:")
print(search.best_params_)

print("\nبهترین امتیاز CV (neg RMSE):", search.best_score_)
print("بهترین RMSE در CV:", -search.best_score_)

best_rf = search.best_estimator_

# ارزیابی بهترین مدل روی validation
best_rf_val_res = evaluate_on_val(best_rf, X_val, y_val, name="BestRF(RandomizedSearch)")
print("\nنتیجه بهترین RF روی Validation:", best_rf_val_res)

# مقایسه همه مدل‌ها روی validation
comparison = pd.DataFrame([
    evaluate_on_val(linreg_model, X_val, y_val, name="LinearRegression"),
    ridge_val_res,
    evaluate_on_val(rf_model, X_val, y_val, name="RF(n=300 default)"),
    best_rf_val_res
]).sort_values("RMSE")

print("\n--- مقایسه نهایی روی Validation (بعد از tuning) ---")
print(comparison)

# طبق خروجی شما: RF پیش‌فرض (n=300) بهتر از tuned بوده (RMSE کمتر)


# =========================================
# مرحله 7) آموزش نهایی + ارزیابی روی Test + نمودارهای تشخیصی
# =========================================

import os

# انتخاب مدل نهایی بر اساس validation
final_model = rf_model
final_model_name = "RandomForest(n=300 default)"

print("\n=== مدل نهایی ===")
print(final_model_name)

# آموزش مدل نهایی روی Train + Validation
# چرا؟ چون بعد از انتخاب مدل، بهتر است از داده بیشتری برای یادگیری استفاده کنیم
# ولی Test را دست نمی‌زنیم تا ارزیابی بی‌طرفانه بماند
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

print("\nدر حال آموزش مدل نهایی روی Train+Validation ...")
final_model.fit(X_trainval, y_trainval)

# پیش‌بینی روی Test
y_test_pred = final_model.predict(X_test)

# محاسبه معیارهای نهایی روی Test
final_test_results = {
    "model": final_model_name,
    "MAE": mean_absolute_error(y_test, y_test_pred),
    "RMSE": rmse(y_test, y_test_pred),
    "R2": r2_score(y_test, y_test_pred)
}

print("\n=== نتایج نهایی روی Test ===")
print(final_test_results)

# طبق خروجی شما:
# MAE≈11521.92, RMSE≈13599.11, R2≈0.08049

# ذخیره خروجی پیش‌بینی‌ها برای گزارش/تحویل
os.makedirs("outputs", exist_ok=True)
pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_test_pred})
pred_path = os.path.join("outputs", "test_predictions.csv")
pred_df.to_csv(pred_path, index=False)
print("\nفایل ذخیره شد:", pred_path)

# محاسبه residuals (خطاها)
# residual = مقدار واقعی - مقدار پیش‌بینی
residuals = y_test.values - y_test_pred

# نمودار توزیع residuals:
# اگر مدل خوب باشد residuals حول 0 متمرکز می‌شود
plt.figure()
plt.hist(residuals, bins=50)
plt.xlabel("Residual (y_true - y_pred)")
plt.ylabel("Frequency")
plt.title("توزیع residuals روی Test")
plt.tight_layout()
plt.show()

# نمودار Predicted vs True:
# اگر مدل عالی باشد نقاط نزدیک خط y=x می‌افتند
plt.figure()
plt.scatter(y_test.values, y_test_pred, alpha=0.3)
plt.xlabel("True Billing Amount")
plt.ylabel("Predicted Billing Amount")
plt.title("مقایسه مقدار واقعی و پیش‌بینی (Test)")
plt.tight_layout()
plt.show()

# چاپ چند نمونه از خروجی‌ها برای بررسی/قرار دادن در گزارش
print("\n--- چند نمونه پیش‌بینی (Test) ---")
print(pred_df.head(10))


# =========================================
# مرحله 8) اهمیت ویژگی‌ها (Feature Importance) در RandomForest
# =========================================

# هدف این بخش: تفسیرپذیری
# RandomForest می‌تواند بگوید کدام ویژگی‌ها بیشترین تاثیر را در پیش‌بینی داشته‌اند

# گرفتن نام ویژگی‌های one-hot شده
ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)

# ترکیب نام ویژگی‌های عددی و دسته‌ای (one-hot)
all_feature_names = np.concatenate([numeric_features, cat_feature_names])

# دسترسی به مدل RF داخل pipeline
rf_fitted = final_model.named_steps["model"]

# گرفتن میزان اهمیت ویژگی‌ها
importances = rf_fitted.feature_importances_
fi = pd.DataFrame({"feature": all_feature_names, "importance": importances}).sort_values("importance", ascending=False)

print("\n--- 15 ویژگی برتر از نظر اهمیت ---")
print(fi.head(15))

# طبق خروجی شما سه مورد اول:
# Room Number≈0.2205، Age≈0.1635، Length_of_Stay≈0.1397

# رسم نمودار 15 ویژگی برتر
plt.figure(figsize=(10, 5))
plt.barh(fi.head(15)["feature"][::-1], fi.head(15)["importance"][::-1])
plt.xlabel("Importance")
plt.title("15 ویژگی برتر از نظر اهمیت در RandomForest")
plt.tight_layout()
plt.show()
