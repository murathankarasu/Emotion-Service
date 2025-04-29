import pandas as pd
import random

def create_balanced_dataset(input_file, output_file, samples_per_class=1000):
    # Veri setini yükle
    df = pd.read_csv(input_file)
    
    # Her sınıftan örnek seç
    balanced_df = pd.DataFrame()
    
    for label in df['label'].unique():
        # Sınıfa ait örnekleri al
        class_samples = df[df['label'] == label]
        
        # Eğer sınıfta yeterli örnek varsa
        if len(class_samples) >= samples_per_class:
            # Rastgele örnek seç
            selected_samples = class_samples.sample(n=samples_per_class, random_state=42)
        else:
            # Yeterli örnek yoksa, mevcut örnekleri tekrarla
            selected_samples = class_samples.sample(n=samples_per_class, replace=True, random_state=42)
        
        balanced_df = pd.concat([balanced_df, selected_samples])
    
    # Veri setini karıştır
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Dengeli veri setini kaydet
    balanced_df.to_csv(output_file, index=False)
    
    # İstatistikleri yazdır
    print("\nDengeli Veri Seti İstatistikleri:")
    print(balanced_df['label'].value_counts())
    print("\nToplam Örnek Sayısı:", len(balanced_df))
    print("\nSınıf Sayısı:", len(balanced_df['label'].unique()))

if __name__ == "__main__":
    create_balanced_dataset(
        input_file="data/text.csv",
        output_file="data/balanced_text.csv",
        samples_per_class=1000
    ) 