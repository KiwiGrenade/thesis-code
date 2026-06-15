# Porównanie autorskiej i wybranych architektur głębokich sieci neuronowych w zadaniu klasyfikacji obrazów

Repozytorium zawiera kod źródłowy, notebooki oraz skrypty pomocnicze użyte do przeprowadzenia eksperymentów badawczych w pracy magisterskiej dotyczącej klasyfikacji obrazów ryb. Głównym celem projektu jest porównanie wybranych architektur głębokich sieci neuronowych z autorskimi wariantami sieci konwolucyjnej, określanymi jako **RybNet V1** oraz **RybNet V2**.

Projekt obejmuje pełny potok eksperymentalny: przygotowanie danych, ładowanie modeli, trening, walidację, testowanie, zapis metryk, benchmarkowanie modeli, generowanie raportów, tworzenie wykresów oraz archiwizację wyników.

Zbiór danych użyty w eksperymentach jest dostępny na platformie Kaggle: <https://www.kaggle.com/datasets/jakubjakw/jj-f4k-trajectory>.

---

## 1. Struktura projektu

```text
.
├── README.md
├── requirements.txt
├── models
│   ├── convnext_tiny.py
│   ├── modernstem_eca_msf_resnet50.py
│   ├── modernstem_eca_resnet50.py
│   ├── resnet50_baseline.py
│   └── swin_transformer_v2_tiny.py
├── notebooks
│   ├── plot_metrics.ipynb
│   └── train_model.ipynb
├── src
│   ├── archive.py
│   ├── benchmark.py
│   ├── callbacks.py
│   ├── data_module.py
│   ├── __init__.py
│   ├── lit_classifier.py
│   ├── losses.py
│   ├── metrics_plotting.py
│   ├── model_loader.py
│   └── reporting.py
└── util_scripts
    └── dataset_split_script.py
```

Projekt został podzielony na cztery główne części:

- `models/` — implementacje architektur klasyfikacyjnych.
- `src/` — wspólna logika treningu, danych, metryk, raportowania i archiwizacji.
- `notebooks/` — notebooki sterujące treningiem oraz analizą wyników.
- `util_scripts/` — skrypt pomocniczy związany z przygotowaniem podziału zbioru danych.
- `requirements.txt` — lista bibliotek wymaganych do uruchomienia kodu i notebooków.

---

## 2. Opis katalogu `models/`

Katalog `models/` zawiera implementacje wszystkich architektur porównywanych w eksperymentach. Każdy plik udostępnia funkcję `build_model(num_classes)`, dzięki czemu model może być dynamicznie załadowany przez wspólny mechanizm z pliku `src/model_loader.py`.

### `models/resnet50_baseline.py`

Plik zawiera bazowy model referencyjny oparty na architekturze **ResNet-50** z biblioteki `torchvision`.

Zawartość:

- import modelu `resnet50` z `torchvision.models`,
- podmiana warstwy klasyfikacyjnej `fc` na nową warstwę `nn.Linear`,
- funkcja `build_model(num_classes)`, która zwraca model dostosowany do liczby klas w zbiorze danych.

Model ten pełni rolę punktu odniesienia dla autorskich architektur RybNet.

### `models/convnext_tiny.py`

Plik zawiera implementację modelu **ConvNeXt Tiny**.

Zawartość:

- import modelu `convnext_tiny` z `torchvision.models`,
- zastąpienie ostatniej warstwy klasyfikatora nową warstwą `nn.Linear`,
- funkcja `build_model(num_classes)`.

Model ConvNeXt Tiny został użyty jako nowoczesna architektura konwolucyjna do porównania z ResNet-50 oraz autorskimi wariantami RybNet.

### `models/swin_transformer_v2_tiny.py`

Plik zawiera implementację modelu **Swin Transformer V2 Tiny**.

Zawartość:

- import modelu `swin_v2_t` z `torchvision.models`,
- zastąpienie głowicy klasyfikacyjnej `head` nową warstwą liniową,
- funkcja `build_model(num_classes)`,
- blok testowy `if __name__ == "__main__"`, który wypisuje strukturę modelu oraz liczbę parametrów.

Model ten reprezentuje podejście transformerowe w zadaniu klasyfikacji obrazów.

### `models/modernstem_eca_resnet50.py`

Plik zawiera implementację autorskiej architektury **RybNet V1**. Model bazuje na strukturze ResNet-50, ale wprowadza kilka modyfikacji względem modelu referencyjnego.

Zawartość:

- klasa `ECAModule`, implementująca mechanizm Efficient Channel Attention,
- klasa `ECABottleneck`, czyli zmodyfikowany blok bottleneck z modułem ECA,
- klasa `ModernStemECAResNet50`, będąca główną implementacją modelu RybNet V1,
- funkcja `build_model(num_classes)`,
- blok testowy sprawdzający działanie modelu na losowym tensorze wejściowym.

Najważniejsze cechy modelu:

- zastąpienie klasycznego wejściowego bloku ResNet sekwencją kilku konwolucji `3 × 3`,
- zastosowanie aktywacji `GELU`,
- dodanie mechanizmu atencji kanałowej ECA w blokach bottleneck,
- wykorzystanie połączenia globalnego uśredniania i globalnego maksimum w głowicy klasyfikacyjnej,
- zastosowanie klasyfikatora z normalizacją, dropoutem i warstwami liniowymi.

### `models/modernstem_eca_msf_resnet50.py`

Plik zawiera implementację autorskiej architektury **RybNet V2**. Jest to rozwinięcie wariantu RybNet V1 o moduł wieloskalowej fuzji cech.

Zawartość:

- klasa `ECAModule`, implementująca uwagę kanałową,
- klasa `ECABottleneck`, czyli bottleneck ResNet z modułem ECA,
- klasa `MultiScaleFusion`, odpowiadająca za połączenie reprezentacji z różnych poziomów sieci,
- klasa `ModernECAMSFResNet50`, będąca główną implementacją RybNet V2,
- funkcja `build_model(num_classes)`.

Najważniejsze cechy modelu:

- zmodyfikowany blok wejściowy oparty na konwolucjach `3 × 3`,
- zastosowanie mechanizmu ECA w blokach resztowych,
- pobieranie map cech z poziomów `c3`, `c4` i `c5`,
- projekcja map cech do wspólnej liczby kanałów za pomocą konwolucji `1 × 1`,
- interpolacja głębszych map cech do rozmiaru mapy `c3`,
- konkatenacja i fuzja wieloskalowej reprezentacji,
- głowica klasyfikacyjna oparta na połączeniu `AdaptiveAvgPool2d` i `AdaptiveMaxPool2d`.

RybNet V2 jest główną autorską architekturą badaną w projekcie.

---

## 3. Opis katalogu `src/`

Katalog `src/` zawiera wspólne moduły używane przez notebooki i modele. Odpowiada za organizację danych, konfigurację treningu, obsługę funkcji straty, logowanie metryk, raportowanie wyników i wizualizację rezultatów.

### `src/model_loader.py`

Moduł odpowiada za dynamiczne ładowanie modeli z plików źródłowych.

Zawartość:

- funkcja `load_model_factory(model_file, factory_name="build_model")`.

Działanie modułu:

- sprawdza, czy wskazany plik modelu istnieje,
- ładuje plik `.py` jako moduł Pythona,
- sprawdza, czy moduł zawiera funkcję `build_model`,
- zwraca referencję do funkcji budującej model.

Dzięki temu notebook treningowy może wybierać model przez zmianę jednej zmiennej `MODEL_NAME`, bez modyfikowania kodu treningowego.

### `src/data_module.py`

Moduł zawiera klasę `ImageFolderDataModule`, która dziedziczy po `LightningDataModule`.

Zawartość:

- przygotowanie zbiorów `train`, `val` i `test` z użyciem `torchvision.datasets.ImageFolder`,
- transformacje treningowe i ewaluacyjne,
- generowanie obiektów `DataLoader`,
- odczyt nazw klas,
- obliczanie liczności klas,
- obliczanie wag klas dla wybranych funkcji straty.

Transformacje treningowe obejmują:

- zmianę rozmiaru obrazu do `image_size × image_size`,
- losowe odbicie poziome,
- losową rotację,
- modyfikację jasności, kontrastu, nasycenia i odcienia,
- konwersję do tensora,
- normalizację zgodną ze standardowymi wartościami ImageNet.

Transformacje walidacyjne i testowe obejmują zmianę rozmiaru, konwersję do tensora i normalizację.

### `src/losses.py`

Moduł definiuje funkcje straty używane w eksperymentach.

Zawartość:

- klasa `FocalLoss`,
- klasa `ClassBalancedFocalLoss`,
- funkcja `make_loss(...)`.

Obsługiwane funkcje straty:

- `ce` — standardowa entropia krzyżowa,
- `weighted_ce` — ważona entropia krzyżowa,
- `focal` — Focal Loss,
- `cb_focal` — Class-Balanced Focal Loss.

Moduł umożliwia kontrolowane porównanie wpływu różnych funkcji straty na jakość klasyfikacji.

### `src/lit_classifier.py`

Moduł zawiera główną klasę treningową `LitImageClassifier`, dziedziczącą po `LightningModule`.

Zawartość:

- przechowywanie modelu bazowego i funkcji straty,
- definicja metryk treningowych, walidacyjnych i testowych,
- implementacja kroków `training_step`, `validation_step` i `test_step`,
- logowanie metryk dla każdego etapu,
- zbieranie predykcji i etykiet podczas testowania,
- konfiguracja optymalizatora i schedulerów.

Metryki obejmują:

- accuracy,
- precision,
- recall,
- F1-score,
- macro precision,
- macro recall,
- macro F1,
- balanced accuracy,
- macierz pomyłek dla zbioru testowego.

Obsługiwane optymalizatory:

- `AdamW`,
- `SGD`.

Obsługiwane schedulery:

- `CosineAnnealingLR`,
- `StepLR`,
- brak schedulera.

### `src/benchmark.py`

Moduł odpowiada za pomiar podstawowej złożoności i efektywności działania modeli.

Zawartość:

- funkcja `collect_model_statistics(...)`,
- funkcja `save_model_statistics(...)`.

Funkcja `collect_model_statistics(...)` zapisuje najważniejsze informacje techniczne o modelu:

- całkowitą liczbę parametrów,
- liczbę parametrów trenowalnych i nietrenowalnych,
- szacowany rozmiar modelu w MB,
- średni czas inferencji dla wybranych rozmiarów batcha,
- liczbę przetwarzanych obrazów na sekundę,
- szczytowe zużycie pamięci VRAM, jeśli model działa na GPU.

Domyślnie benchmark wykonywany jest dla batchy `1`, `16` i `32`. Przed właściwym pomiarem wykonywana jest krótka faza rozgrzewkowa, a następnie czas inferencji liczony jest na podstawie serii powtórzeń. Funkcja `save_model_statistics(...)` zapisuje wyniki do pliku CSV, który w projekcie przyjmuje nazwę `model_benchmarks.csv`.

Moduł jest wykorzystywany w notebooku `train_model.ipynb` po zakończeniu treningu i testowania modelu.

### `src/reporting.py`

Moduł odpowiada za zapis wyników treningu i testowania do plików CSV.

Zawartość:

- uporządkowane listy kolumn historii treningu i podsumowania testowego,
- funkcje pomocnicze do konwersji danych na `DataFrame`,
- funkcja `save_training_artifacts(...)`.

Generowane pliki:

- `history_by_epoch.csv`,
- `best_validation_metrics.csv`,
- `test_summary.csv`,
- `test_classification_report.csv`,
- `test_confusion_matrix.csv`,
- `test_per_class_metrics.csv`,
- `per_class_recall.csv`.

Moduł korzysta z `sklearn.metrics.classification_report` oraz `sklearn.metrics.confusion_matrix`.

### `src/metrics_plotting.py`

Moduł zawiera funkcje do wczytywania wyników eksperymentów i generowania wykresów.

Zawartość:

- funkcje wczytujące historię treningu, najlepsze wyniki walidacyjne i wyniki testowe,
- funkcje generujące wykresy jednej metryki dla wielu eksperymentów,
- funkcje porównujące metryki treningowe i walidacyjne,
- funkcje generujące wykresy słupkowe wyników testowych,
- funkcja rysująca macierz pomyłek,
- funkcje pomocnicze do bezpiecznego nazewnictwa plików i rozpoznawania folderów metryk.

Najważniejsze funkcje:

- `load_history`,
- `load_histories`,
- `load_test_summaries`,
- `load_best_validation`,
- `available_history_metrics`,
- `plot_metric`,
- `plot_train_val_pair`,
- `plot_metric_grid`,
- `plot_test_metric_bar`,
- `plot_confusion_matrix`.

Moduł jest wykorzystywany przede wszystkim przez notebook `plot_metrics.ipynb`.

### `src/archive.py`

Moduł obsługuje archiwizację wyników eksperymentów oraz rozpakowywanie danych.

Zawartość:

- `make_training_archive(output_dir, archive_path)`,
- `extract_training_archives(archives_dir, archive_names, extract_dir=None)`,
- `prepare_dataset_from_tar(dataset_tar, data_dir, skip_if_exists=True)`.

Zastosowanie:

- pakowanie folderu eksperymentu do archiwum `.tar.gz`,
- rozpakowywanie archiwów z wynikami eksperymentów,
- przygotowanie datasetu z pliku `.tar` do katalogu roboczego.

Moduł jest istotny w środowisku WCSS, gdzie dane i wyniki są przechowywane na zasobach dyskowych, a obliczenia wykonywane są w katalogu tymczasowym.

### `src/callbacks.py`

Moduł definiuje callback `EpochMetricsLogger` dla PyTorch Lightning.

Zawartość:

- klasa `EpochMetricsLogger`,
- metoda `on_validation_epoch_end`, zapisująca metryki po epoce walidacyjnej,
- metoda pomocnicza `_get`, pobierająca wartości metryk z obiektu `trainer.callback_metrics`.

Callback zapisuje metryki do pliku CSV i wypisuje najważniejsze wartości w konsoli.

### `src/__init__.py`

Plik inicjalizujący pakiet `src`. Umożliwia importowanie modułów z katalogu `src` jako części pakietu Pythona.

---

## 4. Opis katalogu `notebooks/`

Katalog `notebooks/` zawiera notebooki Jupyter sterujące wykonaniem eksperymentów i analizą wyników.

### `notebooks/train_model.ipynb`

Notebook odpowiada za pełny proces treningu pojedynczego eksperymentu.

Główne zadania notebooka:

- ustawienie ścieżek projektu, danych i archiwów,
- wybór datasetu,
- wybór modelu przez zmienną `MODEL_NAME`,
- konfiguracja hiperparametrów,
- przygotowanie danych,
- dynamiczne załadowanie modelu,
- utworzenie funkcji straty,
- utworzenie obiektu `LitImageClassifier`,
- konfiguracja callbacków i loggera,
- trening modelu w PyTorch Lightning,
- test najlepszego checkpointu,
- zapis wyników,
- wykonanie benchmarku modelu,
- wygenerowanie plików metryk,
- spakowanie całego folderu eksperymentu do archiwum `.tar.gz`.

Najważniejsze parametry ustawione w notebooku:

```python
SEED = 777
IMAGE_SIZE = 288
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER_NAME = "adamw"
SCHEDULER_NAME = "cosine"
LOSS_NAME = "ce"
FOCAL_GAMMA = 2.0
CB_BETA = 0.9999
```

Notebook zapisuje między innymi:

- `config.json`,
- `test_results.json`,
- `training_time.json`,
- `model_benchmarks.csv`,
- `history_by_epoch.csv`,
- `best_validation_metrics.csv`,
- `test_summary.csv`,
- `test_classification_report.csv`,
- `test_confusion_matrix.csv`.

Uwaga techniczna: w komórce benchmarku modelu parametr `image_size` jest wpisany jako `256`, mimo że globalna konfiguracja eksperymentu używa `IMAGE_SIZE = 288`. Dla pełnej spójności warto zmienić tę wartość na `image_size=IMAGE_SIZE`.

### `notebooks/plot_metrics.ipynb`

Notebook odpowiada za analizę wyników wielu eksperymentów i generowanie wykresów porównawczych.

Główne zadania notebooka:

- rozpakowanie archiwów z wynikami modeli,
- przypisanie czytelnych nazw eksperymentów,
- wczytanie historii treningu,
- wczytanie najlepszych wyników walidacyjnych,
- wczytanie wyników testowych,
- generowanie wykresów metryk,
- generowanie wykresów słupkowych dla zbioru testowego,
- generowanie macierzy pomyłek,
- eksport zbiorczych tabel CSV.

Modele porównywane w notebooku:

- `ConvNeXt Tiny`,
- `ResNet-50`,
- `Swin Transformer V2 Tiny`,
- `RybNet V1`,
- `RybNet V2`.

Notebook generuje między innymi:

- wykres `val_macro_f1`,
- wykresy `train` vs `val`,
- wykresy strat,
- wykresy accuracy,
- wykresy F1-score,
- wykresy recall,
- wykresy balanced accuracy,
- wykresy słupkowe metryk testowych,
- zwykłą i znormalizowaną macierz pomyłek,
- pliki `combined_history_by_epoch.csv`, `combined_best_validation_metrics.csv` i `combined_test_summary.csv`.

---

## 5. Opis katalogu `util_scripts/`

Katalog `util_scripts/` zawiera skrypt pomocniczy wykorzystywany podczas przygotowywania zbioru danych do formatu wymaganego przez moduł `ImageFolderDataModule`.

### `util_scripts/dataset_split_script.py`

Skrypt pomocniczy przeznaczony do podziału zbioru danych na podzbiory treningowy, walidacyjny i testowy.

Oczekiwany rezultat działania:

```text
dataset/
├── train
├── val
└── test
```

Taka struktura jest wymagana przez `ImageFolderDataModule`, który korzysta z `torchvision.datasets.ImageFolder`.

Zależności projektu znajdują się w pliku `requirements.txt` umieszczonym w głównym katalogu repozytorium.

---

## 6. Przepływ pracy eksperymentalnej

Typowy przebieg eksperymentu wygląda następująco:

1. Przygotowanie danych w strukturze `train/val/test`.
2. Umieszczenie archiwum datasetu w katalogu danych środowiska obliczeniowego.
3. Wybór modelu w notebooku `train_model.ipynb` przez ustawienie zmiennej `MODEL_NAME`.
4. Ustawienie hiperparametrów, między innymi `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LOSS_NAME` i `SEED`.
5. Rozpakowanie datasetu do katalogu roboczego.
6. Utworzenie obiektu `ImageFolderDataModule`.
7. Dynamiczne załadowanie modelu przez `load_model_factory`.
8. Utworzenie funkcji straty przez `make_loss`.
9. Utworzenie klasyfikatora `LitImageClassifier`.
10. Trening modelu z użyciem PyTorch Lightning.
11. Zapis najlepszego checkpointu według `val_macro_f1`.
12. Testowanie najlepszego checkpointu.
13. Zapis metryk i raportów.
14. Spakowanie wyników eksperymentu do archiwum `.tar.gz`.
15. Wczytanie archiwów w `plot_metrics.ipynb` i przygotowanie wykresów porównawczych.

---

## 7. Dane wejściowe

Zbiór danych jest dostępny pod adresem: <https://www.kaggle.com/datasets/jakubjakw/jj-f4k-trajectory>.

Projekt zakłada użycie danych obrazowych w strukturze zgodnej z `torchvision.datasets.ImageFolder`:

```text
data/
├── train
│   ├── class_1
│   ├── class_2
│   └── ...
├── val
│   ├── class_1
│   ├── class_2
│   └── ...
└── test
    ├── class_1
    ├── class_2
    └── ...
```

W notebooku treningowym używany jest dataset:

```python
DATASET_FILE_NAME = "F4K_trajectory_V1"
```

Archiwum datasetu jest oczekiwane w lokalizacji:

```python
DATASET_TAR = PD_PATH / "datasets" / f"{DATASET_FILE_NAME}.tar"
```

---

## 8. Wyniki generowane przez eksperyment

Dla każdego eksperymentu tworzony jest osobny katalog wynikowy. Nazwa sesji jest budowana automatycznie na podstawie modelu, funkcji straty, liczby epok, rozmiaru obrazu i ziarna losowego, np.:

```text
modernstem_eca_resnet50_ce_50_288_seed_777
```

W katalogu eksperymentu zapisywane są między innymi:

```text
output_dir/
├── checkpoints
│   ├── best-...
│   └── last.ckpt
├── logs
│   └── metrics.csv
├── metrics
│   ├── best_validation_metrics.csv
│   ├── history_by_epoch.csv
│   ├── model_benchmarks.csv
│   ├── test_classification_report.csv
│   ├── test_confusion_matrix.csv
│   ├── test_results.json
│   ├── test_summary.csv
│   └── training_time.json
└── config.json
```

Następnie cały katalog eksperymentu jest pakowany do archiwum:

```text
SESSION_NAME.tar.gz
```

---

## 9. Najważniejsze technologie

Projekt wykorzystuje następujące biblioteki i narzędzia:

- Python,
- PyTorch,
- Torchvision,
- PyTorch Lightning,
- Torchmetrics,
- Scikit-learn,
- Pandas,
- Matplotlib,
- Jupyter Notebook.

Eksperymenty są przygotowane do uruchamiania w środowisku obliczeniowym z dostępem do GPU. Notebook treningowy automatycznie wybiera `gpu`, jeżeli `torch.cuda.is_available()` zwraca wartość prawdziwą.

---

## 10. Instalacja środowiska

Poniższe kroki pozwalają przygotować środowisko lokalne lub środowisko obliczeniowe z dostępem do GPU.

1. Sklonować repozytorium:

```bash
git clone https://github.com/KiwiGrenade/thesis-code.git
cd thesis-code
```

2. Utworzyć i aktywować środowisko wirtualne:

```bash
python -m venv .venv
source .venv/bin/activate
```

W systemie Windows:

```bash
.venv\Scripts\activate
```

3. Zainstalować zależności z pliku `requirements.txt` znajdującego się w katalogu głównym projektu:

```bash
pip install -r requirements.txt
```

W przypadku pracy na GPU należy upewnić się, że zainstalowana wersja pakietów `torch` i `torchvision` jest zgodna z wersją CUDA dostępną w środowisku uruchomieniowym.

4. Pobrać zbiór danych z Kaggle:

<https://www.kaggle.com/datasets/jakubjakw/jj-f4k-trajectory>

Po pobraniu archiwum należy umieścić je w katalogu:

```text
datasets/F4K_trajectory_V1.tar
```

Docelowa struktura powinna wyglądać następująco:

```text
thesis-code/
├── datasets
│   └── F4K_trajectory_V1.tar
├── models
├── notebooks
├── src
├── util_scripts
├── README.md
└── requirements.txt
```

5. Uruchomić środowisko Jupyter:

```bash
jupyter lab
```

---

## 11. Uruchomienie treningu modelu

Trening pojedynczego modelu realizowany jest w notebooku `notebooks/train_model.ipynb`.

1. Otworzyć notebook:

```text
notebooks/train_model.ipynb
```

2. Dostosować ścieżki do lokalizacji projektu i danych. W środowisku lokalnym można ustawić je na katalog repozytorium:

```python
PROJ_DIR = Path("/sciezka/do/thesis-code")
PD_PATH = PROJ_DIR
TMP_DIR = PROJ_DIR / "tmp"
```

W środowisku WCSS można pozostawić konfigurację opartą na zmiennych `PDDIRS` i `TMPDIR`, jeżeli wskazują one właściwe katalogi robocze.

3. Wybrać zbiór danych i model:

```python
DATASET_FILE_NAME = "F4K_trajectory_V1"
MODEL_NAME = "modernstem_eca_msf_resnet50"
```

Dostępne wartości `MODEL_NAME` odpowiadają nazwom plików w katalogu `models/`, bez rozszerzenia `.py`:

```text
resnet50_baseline
convnext_tiny
swin_transformer_v2_tiny
modernstem_eca_resnet50
modernstem_eca_msf_resnet50
```

4. W razie potrzeby zmienić hiperparametry eksperymentu:

```python
SEED = 777
IMAGE_SIZE = 288
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER_NAME = "adamw"
SCHEDULER_NAME = "cosine"
LOSS_NAME = "ce"
```

5. Uruchomić wszystkie komórki notebooka.

Po zakończeniu treningu notebook zapisze checkpointy, historię uczenia, metryki testowe, raport klasyfikacji, macierz pomyłek, benchmark modelu oraz archiwum eksperymentu `.tar.gz`.

---

## 12. Generowanie wykresów i wyników zbiorczych

Wykresy oraz tabele porównawcze generowane są w notebooku `notebooks/plot_metrics.ipynb`.

1. Upewnić się, że archiwa `.tar.gz` z wynikami eksperymentów znajdują się w katalogu wskazanym przez `ARCHIVES_DIR`.

2. Otworzyć notebook:

```text
notebooks/plot_metrics.ipynb
```

3. Uzupełnić listę archiwów wynikowych:

```python
archive_names = [
    "convnext_tiny_ce_50_288_seed_42.tar.gz",
    "resnet50_baseline_ce_50_288_seed_42.tar.gz",
    "swin_transformer_v2_tiny_ce_50_288_seed_42.tar.gz",
    "modernstem_eca_resnet50_ce_50_288_seed_42.tar.gz",
    "modernstem_eca_msf_resnet50_ce_50_288_seed_42.tar.gz",
]
```

4. Ustawić czytelne nazwy eksperymentów:

```python
EXPERIMENT_NAMES = [
    "ConvNeXt Tiny",
    "ResNet-50",
    "Swin Transformer V2 Tiny",
    "RybNet V1",
    "RybNet V2",
]
```

5. Uruchomić wszystkie komórki notebooka.

Notebook wygeneruje wykresy przebiegu uczenia, porównania metryk treningowych i walidacyjnych, wykresy słupkowe wyników testowych, macierze pomyłek oraz zbiorcze pliki CSV:

```text
combined_history_by_epoch.csv
combined_best_validation_metrics.csv
combined_test_summary.csv
```

---

## 13. Uwagi metodologiczne

- Wszystkie modele są ładowane przez wspólny interfejs `build_model(num_classes)`, co ułatwia porównywanie architektur w tych samych warunkach.
- Ten sam `ImageFolderDataModule` odpowiada za przygotowanie danych dla wszystkich modeli.
- Metryką monitorowaną podczas wyboru najlepszego checkpointu jest `val_macro_f1`.
- Wyniki testowe są zapisywane dopiero po załadowaniu najlepszego checkpointu.
- Projekt umożliwia porównywanie różnych funkcji straty, ale końcowa konfiguracja w notebooku używa standardowej entropii krzyżowej `ce`.
- Wariant RybNet V2 rozszerza RybNet V1 o moduł wieloskalowej fuzji cech, co stanowi główną różnicę konstrukcyjną między autorskimi architekturami.

---

## 14. Elementy wymagające ewentualnego uporządkowania

W projekcie warto rozważyć drobne poprawki porządkowe:

1. Ujednolicenie parametru `image_size` w wywołaniu benchmarku modelu w `train_model.ipynb` przez zastąpienie wartości stałej `256` zmienną `IMAGE_SIZE`.
2. Dostosowanie domyślnych ścieżek w notebookach do trybu uruchamiania poza środowiskiem WCSS, jeżeli projekt ma być często odtwarzany lokalnie.

---

## 15. Autor

Projekt został przygotowany na potrzeby pracy magisterskiej dotyczącej porównania autorskiej i wybranych architektur głębokich sieci neuronowych w zadaniu klasyfikacji obrazów.
