# Python 3.9 tabanlı bir imaj kullanıyoruz
FROM python:3.9-slim

# Çalışma dizinini ayarlıyoruz
WORKDIR /app

# Gerekli sistem paketlerini yüklüyoruz
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Bellek optimizasyonu için Python ayarları
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Gereksinim dosyalarını kopyalıyoruz
COPY requirements.txt .

# Python paketlerini yüklüyoruz
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Uygulama kodunu kopyalıyoruz
COPY . .

# models ve cache klasörlerini oluştur
RUN mkdir -p models cache

# Port numarasını belirtiyoruz
ENV PORT=8080

# Uygulamayı gunicorn ile çalıştırıyoruz
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "app:app"] 