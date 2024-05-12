import tkinter as tk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# X değişkenini global olarak tanımla
X = None

def train_model():
    global X  # X değişkenini global olarak tanımla
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    dataset = pd.read_csv(dataset_url, sep=' ', header=None)
    dataset.columns = [
        'existing_account_status', 'duration_month', 'credit_history', 'purpose', 'credit_amount',
        'savings_account', 'employment_duration', 'installment_rate', 'personal_status_sex',
        'other_debtors', 'present_residence_since', 'property', 'age', 'other_installment_plans',
        'housing', 'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_approval'
    ]
    
    # Kategorik sütunları One-Hot Encoding ile işleyin
    categorical_columns = ['existing_account_status', 'credit_history', 'purpose', 'savings_account',
                           'employment_duration', 'personal_status_sex', 'other_debtors', 'property',
                           'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker']
    dataset_encoded = pd.get_dummies(dataset, columns=categorical_columns)
    
    X = dataset_encoded.drop(columns=['credit_approval'])
    y = dataset_encoded['credit_approval']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns  # Model ve özellik isimlerini döndür

def predict_credit_approval(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def create_gui():
    root = tk.Tk()
    root.title("Kredi Başvuru Değerlendirme")

    def on_submit():
        global X  # X değişkenine eriş
        input_data = {
            'existing_account_status': existing_account_status_var.get(),
            'duration_month': duration_month_var.get(),
            'credit_history': credit_history_var.get(),
            'purpose': purpose_var.get(),
            'credit_amount': credit_amount_var.get(),
            'savings_account': savings_account_var.get(),
            'employment_duration': employment_duration_var.get(),
            'installment_rate': installment_rate_var.get(),
            'personal_status_sex': personal_status_sex_var.get(),
            'other_debtors': other_debtors_var.get(),
            'present_residence_since': present_residence_since_var.get(),
            'property': property_var.get(),
            'age': age_var.get(),
            'other_installment_plans': other_installment_plans_var.get(),
            'housing': housing_var.get(),
            'existing_credits': existing_credits_var.get(),
            'job': job_var.get(),
            'people_liable': people_liable_var.get(),
            'telephone': telephone_var.get(),
            'foreign_worker': foreign_worker_var.get()
        }

        # One-Hot Encoding ile işlenmiş veri seti oluştur
        input_data_series = pd.Series(input_data)
        input_data_encoded = pd.get_dummies(input_data_series).reindex(columns=X_columns, fill_value=0)  # X_columns'ı kullan

        prediction = predict_credit_approval(model, input_data_encoded)

        result_label.config(text="Kredi Alabilir" if prediction[0] == 1 else "Kredi Alamaz")

    model, X_columns = train_model()  # Modeli ve özellik isimlerini al

    # Giriş alanları
    existing_account_status_label = tk.Label(root, text="Mevcut Hesap Durumu:")
    existing_account_status_label.grid(row=0, column=0)
    existing_account_status_var = tk.StringVar(root)
    existing_account_status_var.set("A11")
    existing_account_status_entry = tk.Entry(root, textvariable=existing_account_status_var)
    existing_account_status_entry.grid(row=0, column=1)

    duration_month_label = tk.Label(root, text="Kredi Süresi (Ay):")
    duration_month_label.grid(row=1, column=0)
    duration_month_var = tk.IntVar(root)
    duration_month_entry = tk.Entry(root, textvariable=duration_month_var)
    duration_month_entry.grid(row=1, column=1)

    credit_history_label = tk.Label(root, text="Kredi Geçmişi:")
    credit_history_label.grid(row=2, column=0)
    credit_history_var = tk.StringVar(root)
    credit_history_var.set("A30")
    credit_history_entry = tk.Entry(root, textvariable=credit_history_var)
    credit_history_entry.grid(row=2, column=1)

    purpose_label = tk.Label(root, text="Kredi Amaç:")
    purpose_label.grid(row=3, column=0)
    purpose_var = tk.StringVar(root)
    purpose_var.set("A40")
    purpose_entry = tk.Entry(root, textvariable=purpose_var)
    purpose_entry.grid(row=3, column=1)

    credit_amount_label = tk.Label(root, text="Kredi Miktarı:")
    credit_amount_label.grid(row=4, column=0)
    credit_amount_var = tk.IntVar(root)
    credit_amount_entry = tk.Entry(root, textvariable=credit_amount_var)
    credit_amount_entry.grid(row=4, column=1)

    savings_account_label = tk.Label(root, text="Tasarruf Hesabı:")
    savings_account_label.grid(row=5, column=0)
    savings_account_var = tk.StringVar(root)
    savings_account_var.set("A65")
    savings_account_entry = tk.Entry(root, textvariable=savings_account_var)
    savings_account_entry.grid(row=5, column=1)

    employment_duration_label = tk.Label(root, text="İş Süresi:")
    employment_duration_label.grid(row=6, column=0)
    employment_duration_var = tk.StringVar(root)
    employment_duration_var.set("A75")
    employment_duration_entry = tk.Entry(root, textvariable=employment_duration_var)
    employment_duration_entry.grid(row=6, column=1)

    installment_rate_label = tk.Label(root, text="Taksit Oranı:")
    installment_rate_label.grid(row=7, column=0)
    installment_rate_var = tk.StringVar(root)
    installment_rate_entry = tk.Entry(root, textvariable=installment_rate_var)
    installment_rate_entry.grid(row=7, column=1)

    personal_status_sex_label = tk.Label(root, text="Kişisel Durum/Cinsiyet:")
    personal_status_sex_label.grid(row=8, column=0)
    personal_status_sex_var = tk.StringVar(root)
    personal_status_sex_var.set("A93")
    personal_status_sex_entry = tk.Entry(root, textvariable=personal_status_sex_var)
    personal_status_sex_entry.grid(row=8, column=1)

    other_debtors_label = tk.Label(root, text="Diğer Borçlu Durumu:")
    other_debtors_label.grid(row=9, column=0)
    other_debtors_var = tk.StringVar(root)
    other_debtors_var.set("A101")
    other_debtors_entry = tk.Entry(root, textvariable=other_debtors_var)
    other_debtors_entry.grid(row=9, column=1)

    present_residence_since_label = tk.Label(root, text="Mevcut Oturum Süresi (Yıl):")
    present_residence_since_label.grid(row=10, column=0)
    present_residence_since_var = tk.IntVar(root)
    present_residence_since_entry = tk.Entry(root, textvariable=present_residence_since_var)
    present_residence_since_entry.grid(row=10, column=1)

    property_label = tk.Label(root, text="Mülk Tipi:")
    property_label.grid(row=11, column=0)
    property_var = tk.StringVar(root)
    property_var.set("A121")
    property_entry = tk.Entry(root, textvariable=property_var)
    property_entry.grid(row=11, column=1)

    age_label = tk.Label(root, text="Yaş:")
    age_label.grid(row=12, column=0)
    age_var = tk.IntVar(root)
    age_entry = tk.Entry(root, textvariable=age_var)
    age_entry.grid(row=12, column=1)

    other_installment_plans_label = tk.Label(root, text="Diğer Taksit Planları:")
    other_installment_plans_label.grid(row=13, column=0)
    other_installment_plans_var = tk.StringVar(root)
    other_installment_plans_var.set("A143")
    other_installment_plans_entry = tk.Entry(root, textvariable=other_installment_plans_var)
    other_installment_plans_entry.grid(row=13, column=1)

    housing_label = tk.Label(root, text="Konut Durumu:")
    housing_label.grid(row=14, column=0)
    housing_var = tk.StringVar(root)
    housing_var.set("A151")
    housing_entry = tk.Entry(root, textvariable=housing_var)
    housing_entry.grid(row=14, column=1)

    existing_credits_label = tk.Label(root, text="Mevcut Kredi Sayısı:")
    existing_credits_label.grid(row=15, column=0)
    existing_credits_var = tk.IntVar(root)
    existing_credits_entry = tk.Entry(root, textvariable=existing_credits_var)
    existing_credits_entry.grid(row=15, column=1)

    job_label = tk.Label(root, text="Meslek Durumu:")
    job_label.grid(row=16, column=0)
    job_var = tk.StringVar(root)
    job_var.set("A171")
    job_entry = tk.Entry(root, textvariable=job_var)
    job_entry.grid(row=16, column=1)

    people_liable_label = tk.Label(root, text="Yükümlü Kişi Sayısı:")
    people_liable_label.grid(row=17, column=0)
    people_liable_var = tk.IntVar(root)
    people_liable_entry = tk.Entry(root, textvariable=people_liable_var)
    people_liable_entry.grid(row=17, column=1)

    telephone_label = tk.Label(root, text="Telefon Durumu:")
    telephone_label.grid(row=18, column=0)
    telephone_var = tk.StringVar(root)
    telephone_var.set("A191")
    telephone_entry = tk.Entry(root, textvariable=telephone_var)
    telephone_entry.grid(row=18, column=1)

    foreign_worker_label = tk.Label(root, text="Yabancı İşçi Durumu:")
    foreign_worker_label.grid(row=19, column=0)
    foreign_worker_var = tk.StringVar(root)
    foreign_worker_var.set("A201")
    foreign_worker_entry = tk.Entry(root, textvariable=foreign_worker_var)
    foreign_worker_entry.grid(row=19, column=1)

    # Gönder butonu
    submit_button = tk.Button(root, text="Gönder", command=on_submit)
    submit_button.grid(row=20, column=0, columnspan=2)

    # Sonuç alanı
    global result_label
    result_label = tk.Label(root, text="")
    result_label.grid(row=21, column=0, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
