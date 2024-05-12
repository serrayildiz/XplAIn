# Kredi Başvuru Değerlendirme Uygulaması

Bu uygulama, kullanıcıların kredi başvurularını değerlendirmek için bir arayüz sunar. Kullanıcılar, belirli kredi başvurusu parametrelerini girdikten sonra, uygulama bu bilgilere dayanarak başvurunun onaylanıp onaylanmayacağını tahmin eder.

## Nasıl Çalışır?

Bu uygulama, Python programlama dili kullanılarak geliştirilmiştir ve aşağıdaki bileşenleri içerir:

1. **Veri Ön İşleme ve Model Eğitimi:** Uygulama, bir `RandomForestClassifier` kullanarak kredi başvurusu verilerini değerlendirmek için bir makine öğrenimi modeli eğitir. Veri ön işleme adımları arasında eksik veri doldurma, kategorik değişkenleri kodlama ve veri kümesini eğitim ve test setlerine ayırma bulunmaktadır. Model, [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) üzerinde bulunan Almanya kredi veri seti kullanılarak eğitilmiştir.

2. **GUI (Grafiksel Kullanıcı Arayüzü):** Tkinter kütüphanesi kullanılarak basit bir grafiksel kullanıcı arayüzü oluşturulmuştur. Kullanıcılar, kredi başvurusu ile ilgili bilgileri girebilir ve ardından uygulama, bu bilgilere dayanarak kredi başvurusunun onaylanıp onaylanmayacağını tahmin eder.

## Kurulum

1. Bu projeyi klonlayın veya ZIP dosyası olarak indirin:

    ```
    git clone https://github.com/kullanici/kredi-basvuru-degerlendirme.git
    ```

2. Python 3.6 veya daha yeni bir sürümü yükleyin.

3. Gerekli Python kütüphanelerini yükleyin:

    ```
    pip install pandas scikit-learn
    ```

4. Uygulamayı çalıştırın:

    ```
    python kredi_basvuru_degerlendirme.py
    ```

## Kullanım

1. Uygulama başlatıldığında, bir kullanıcı arayüzü görüntülenir.
2. Kullanıcılar, çeşitli kredi başvurusu parametrelerini girmek için metin kutularını kullanabilir.
3. "Gönder" düğmesine tıkladıktan sonra, uygulama kredi başvurusunun onaylanıp onaylanmayacağını tahmin eder ve sonucu ekranda gösterir.

## Örnek Girdiler

Bu uygulama, kullanıcıların kredi başvurularını değerlendirmek için çeşitli parametreler sağlamasını gerektirir. İşte örnek girdi değerleri:

- **Mevcut Hesap Durumu:** A11
- **Kredi Süresi (Ay):** 24
- **Kredi Geçmişi:** A30
- **Kredi Amaç:** A40
- **Kredi Miktarı:** 4000
- **Tasarruf Hesabı:** A65
- **İş Süresi:** A75
- **Taksit Oranı:** 4
- **Kişisel Durum/Cinsiyet:** A93
- **Diğer Borçlu Durumu:** A101
- **Mevcut Oturum Süresi (Yıl):** 4
- **Mülk Tipi:** A121
- **Yaş:** 35
- **Diğer Taksit Planları:** A143
- **Konut Durumu:** A151
- **Mevcut Kredi Sayısı:** 2
- **Meslek Durumu:** A171
- **Yükümlü Kişi Sayısı:** 1
- **Telefon Durumu:** A191
- **Yabancı İşçi Durumu:** A201

Bu değerleri girdikten sonra "Gönder" düğmesine tıklayarak tahmini alabilirsiniz.

## Dikkat Edilmesi Gerekenler

- Kullanıcılar, doğru bilgileri girmelidir çünkü tahminler bu bilgilere dayanır.
- Veri setindeki kategorik değerler, belirli bir kodlama düzenine sahiptir. Kullanıcıların bu kodlama düzenini takip etmeleri önemlidir.
- Uygulamanın sunduğu tahminler, sadece bir tahmindir ve kesin sonuçlar garanti edilmez.

Bu örnek girdiler, kullanıcıların uygulamayı nasıl kullanacaklarını anlamalarına yardımcı olabilir ve doğru girdileri nasıl sağlayacaklarını gösterebilir.

