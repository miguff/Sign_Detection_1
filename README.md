# Sign_Detection_1

Itt a VGGA16-os modellt alakítottam át úgy, hogy az felismerje az én általam kiválasztott képeket és 'bounding boxokat' rajzol köré.

Ahhoz, hogy működjön egy megadott fájl elrendezésben kell betölteni az adott fileokat:
    
├── dataset(Ebben lesznek benne a Kaggle kép adatbázisa)
│   ├── annotations (Ebben a Bounding boxokat tartalmazó csv file, amit a xm_to_csv.py-al készítünk el a sok xml file-ból
│   └── images (Ezek maguk a képek)
├── output (Ide fog bekerülni, hogy miket ad ki a program)
│   ├── detector.h5 (Ez maga modell ami megfogja mondani, hogy melyik micsoda)
│   ├── lb.pickle (Ebben elvannak mente a label nevek binarizálva, ez kell majd, hogy megtudjuk jósolni, hogy minek nézi a gép
│   └── test_paths.txt (Ebben pedig a teszt képeknek az elérési útja van benne)
├── Config.py (Ez a konfiguráviós fájl, amiben benne van, hogy mit honnan olvasson ki)
├── predict.py (Ezzel fogja megjósolni egy képen, hogy mi is van rajta a képen)
├── train.py (Ezzel trenírozzuk, hogy felismerje a képeket)
├── vieo_test.py (Ezzel tudjuk megnézni, hogy videóban, hogyan működik)
└── xm_to_csv.py (Ezzel alakítjuk át az xml fileokat amik a bounding box koordinátákat tartalmazzák csv file-á
