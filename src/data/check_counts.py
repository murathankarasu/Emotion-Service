import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(file_path):
    # Veri setini yükle
    df = pd.read_csv(file_path)
    
    # Sınıf dağılımını hesapla
    class_counts = df['label'].value_counts()
    
    # Görselleştirme
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Sınıf Dağılımı')
    plt.xlabel('Duygu Kategorileri')
    plt.ylabel('Örnek Sayısı')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    # İstatistikleri yazdır
    print("\nSınıf Dağılımı:")
    print(class_counts)
    print("\nToplam Örnek Sayısı:", len(df))
    print("\nSınıf Sayısı:", len(class_counts))

if __name__ == "__main__":
    analyze_dataset("data/text.csv") 