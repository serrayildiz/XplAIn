import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

def get_dataset_from_url(url):
    try:
        dataset = pd.read_csv(url)
        return dataset
    except Exception as e:
        print("Error loading dataset from URL:", e)
        return None

def get_dataset_from_file():
    try:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        dataset = pd.read_csv(file_path)
        return dataset
    except Exception as e:
        print("Error loading dataset from file:", e)
        return None

def load_dataset():
    dataset_source = dataset_source_var.get()
    if dataset_source == "File":
        dataset = get_dataset_from_file()
    elif dataset_source == "URL":
        url = url_entry.get()
        dataset = get_dataset_from_url(url)
    else:
        dataset = None
    
    if dataset is not None:
        root.destroy()
        preprocess_and_train(dataset)

def preprocess_and_train(dataset):
    X, y = preprocess_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    explain_model(model, X_test)

def preprocess_dataset(dataset):
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    # Eğer sınıf etiketleri kategorik ise kodlayalım
    if y.dtype == 'object':
        y = pd.get_dummies(y, drop_first=True)
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def explain_model(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

def main():
    global root
    root = tk.Tk()
    root.title("Explainable AI: Dataset Selection")
    
    canvas = tk.Canvas(root, width=600, height=300)
    canvas.pack()
    
    background_image = Image.open("ai_background.jpg")
    background_image = background_image.resize((600, 300), Image.BILINEAR)
    background_img = ImageTk.PhotoImage(background_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=background_img)
    
    frame = tk.Frame(root, bg="#f0f0f0", bd=5)
    frame.place(relx=0.5, rely=0.5, relwidth=0.75, relheight=0.15, anchor="center")
    
    global dataset_source_var
    dataset_source_var = tk.StringVar()
    dataset_source_var.set("File")
    
    file_button = tk.Radiobutton(frame, text="From File", variable=dataset_source_var, value="File", bg="#f0f0f0", font=("Helvetica", 12, "bold"))
    file_button.grid(row=0, column=0, padx=10, pady=10)
    
    url_button = tk.Radiobutton(frame, text="From URL", variable=dataset_source_var, value="URL", bg="#f0f0f0", font=("Helvetica", 12, "bold"))
    url_button.grid(row=0, column=1, padx=10, pady=10)
    
    global url_entry
    url_entry = tk.Entry(frame, font=("Helvetica", 10))
    url_entry.grid(row=0, column=2, padx=10, pady=10, ipadx=50)
    
    load_button = tk.Button(root, text="Load Dataset", command=load_dataset, bg="#80c1ff", font=("Helvetica", 12, "bold"))
    load_button.place(relx=0.5, rely=0.8, relwidth=0.3, relheight=0.1, anchor="n")
    
    root.mainloop()

if __name__ == "__main__":
    main()
