from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Date de antrenament — poți genera cu LLM o singură dată
training_data = [
    # SEARCH (20 exemple)
    ("găsește facturile din martie", "search"),
    ("caută contractele cu TechSoft", "search"),
    ("arată documentele din 2024", "search"),
    ("caută toate facturile restante", "search"),
    ("găsește contractul cu furnizorul ABC", "search"),
    ("arată rapoartele financiare din ultimul trimestru", "search"),
    ("caută documentele care conțin termenul audit", "search"),
    ("găsește facturile emise către clientul X", "search"),
    ("listează contractele active", "search"),
    ("caută toate documentele PDF", "search"),
    ("găsește facturile cu scadența depășită", "search"),
    ("arată documentele modificate săptămâna trecută", "search"),
    ("caută rapoartele de vânzări din ianuarie", "search"),
    ("găsește contractele semnate în 2023", "search"),
    ("filtrează facturile după furnizor", "search"),
    ("unde găsesc raportul anual?", "search"),
    ("caută documentele asociate proiectului Apollo", "search"),
    ("arată toate facturile cu valoare peste 10000 de lei", "search"),
    ("găsește fișierele încărcate ieri", "search"),
    ("recuperează contractele care expiră luna aceasta", "search"),

    # EXTRACT (20 exemple)
    ("extrage datele din factură", "extract"),
    ("scoate sumele din contract", "extract"),
    ("parsează PDF-ul atașat", "extract"),
    ("extrage numele furnizorului din factură", "extract"),
    ("identifică data emiterii documentului", "extract"),
    ("extrage valoarea totală și TVA-ul", "extract"),
    ("citește numărul facturii din document", "extract"),
    ("extrage clauzele principale din contract", "extract"),
    ("identifică datele de contact din PDF", "extract"),
    ("transformă tabelul din document în JSON", "extract"),
    ("extrage toate datele calendaristice din raport", "extract"),
    ("citește numele și adresele clienților din fișier", "extract"),
    ("extrage lista produselor și prețurile lor", "extract"),
    ("identifică termenii de plată din contract", "extract"),
    ("convertește tabelul din PDF într-un CSV", "extract"),
    ("extrage semnăturile și datele de semnare", "extract"),
    ("preia toate câmpurile din formularul atașat", "extract"),
    ("identifică IBAN-ul și codul fiscal din factură", "extract"),
    ("extrage indicatorii financiari din raportul anual", "extract"),
    ("transformă conținutul documentului într-o structură JSON", "extract"),

    # ANALYZE (20 exemple)
    ("compară vânzările Q1 vs Q2", "analyze"),
    ("analizează trendurile", "analyze"),
    ("identifică tendințele veniturilor din ultimul an", "analyze"),
    ("compară cheltuielile între departamente", "analyze"),
    ("determină care sunt cei mai profitabili clienți", "analyze"),
    ("analizează evoluția costurilor lunare", "analyze"),
    ("identifică anomaliile din datele financiare", "analyze"),
    ("calculează rata de creștere a vânzărilor", "analyze"),
    ("compară performanța financiară din 2023 și 2024", "analyze"),
    ("determină care furnizori au cele mai mari costuri", "analyze"),
    ("analizează distribuția facturilor pe categorii", "analyze"),
    ("identifică variațiile neobișnuite ale cheltuielilor", "analyze"),
    ("evaluează evoluția marjei de profit", "analyze"),
    ("calculează media valorilor facturilor", "analyze"),
    ("analizează relația dintre venituri și cheltuieli", "analyze"),
    ("identifică lunile cu cele mai mari vânzări", "analyze"),
    ("compară bugetul planificat cu cheltuielile reale", "analyze"),
    ("determină procentul facturilor plătite la timp", "analyze"),
    ("detectează valorile aberante din setul de date", "analyze"),
    ("estimează evoluția veniturilor pe baza datelor istorice", "analyze"),

    # REPORT (20 exemple)
    ("generează raport lunar", "report"),
    ("creează overview pentru Q1", "report"),
    ("generează un raport financiar anual", "report"),
    ("creează un rezumat al vânzărilor pe regiuni", "report"),
    ("pregătește raportul cheltuielilor pentru luna trecută", "report"),
    ("realizează un raport privind facturile restante", "report"),
    ("generează un raport de performanță pentru departamentul de vânzări", "report"),
    ("creează un dashboard cu principalii indicatori financiari", "report"),
    ("pregătește un raport comparativ pentru ultimele două trimestre", "report"),
    ("sintetizează rezultatele financiare într-un raport", "report"),
    ("generează un raport cu furnizorii și plățile efectuate", "report"),
    ("creează o prezentare a evoluției veniturilor", "report"),
    ("pregătește raportul anual pentru conducere", "report"),
    ("generează un sumar al contractelor active și expirate", "report"),
    ("creează un raport privind bugetul și cheltuielile", "report"),
    ("exportă un raport al facturilor în format Excel", "report"),
    ("redactează un raport cu indicatorii de performanță", "report"),
    ("creează un rezumat executiv pentru trimestrul curent", "report"),
    ("generează un raport cu evoluția lunară a profitului", "report"),
    ("pregătește raportul final pentru proiectul Apollo", "report"),
]

# Separăm textele și label-urile
texts = [t[0] for t in training_data]
labels = [t[1] for t in training_data]

# Creăm pipeline-ul
classifier = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

# Antrenăm
classifier.fit(texts, labels)

# Salvăm pentru producție
joblib.dump(classifier, 'intent_classifier.joblib')

# 💡 TIP: Generează 50-100 exemple per clasă cu LLM,
# apoi antrenezi classifier-ul o singură dată.
