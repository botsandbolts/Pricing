"""Concise public-price labels in all 36 supported languages.

These strings are presentation only; core model input tokens stay canonical.
No machine translation service is used at runtime.
"""
from pricing.i18n import locale, LOCALES

_KEYS = ("ask", "bid", "explain", "why", "gap", "engine")
_ROWS = r"""en-GB|Seller asking|Wholesale auction|Explain this price|Why this price?|Indicative price gap|Model details
en-US|Seller asking|Wholesale auction|Explain this price|Why this price?|Indicative price gap|Model details
de|Angebotspreis|Großhandelsauktion|Preis erklären|Warum dieser Preis?|Unverbindliche Preisdifferenz|Modelldetails
fr|Prix demandé|Vente aux enchères pro|Expliquer ce prix|Pourquoi ce prix ?|Écart de prix indicatif|Détails du modèle
ru|Цена объявления|Оптовый аукцион|Объяснить цену|Почему такая цена?|Ориентировочная разница цен|О модели
uk|Ціна оголошення|Оптовий аукціон|Пояснити ціну|Чому така ціна?|Орієнтовна різниця цін|Деталі моделі
bg|Обявена цена|Търг на едро|Обяснение на цената|Защо тази цена?|Ориентировъчна разлика|Данни за модела
hr|Tražena cijena|Veleprodajna dražba|Objasni cijenu|Zašto ta cijena?|Okvirna razlika cijena|Detalji modela
cs|Nabídková cena|Velkoobchodní aukce|Vysvětlit cenu|Proč tato cena?|Orientační rozdíl cen|Detaily modelu
da|Udbudspris|Engrosauktion|Forklar prisen|Hvorfor denne pris?|Vejledende prisforskel|Modeldetaljer
nl|Vraagprijs|Groothandelsveiling|Leg de prijs uit|Waarom deze prijs?|Indicatief prijsverschil|Modeldetails
et|Kuulutuse hind|Hulgimüügioksjon|Selgita hinda|Miks selline hind?|Hinnanguline hinnavahe|Mudeli üksikasjad
fi|Pyyntihinta|Tukkuhuutokauppa|Selitä hinta|Miksi tämä hinta?|Suuntaa-antava hintaero|Mallin tiedot
el|Ζητούμενη τιμή|Δημοπρασία χονδρικής|Εξήγηση τιμής|Γιατί αυτή η τιμή;|Ενδεικτική διαφορά τιμής|Στοιχεία μοντέλου
hu|Hirdetési ár|Nagykereskedelmi aukció|Az ár magyarázata|Miért ennyi?|Tájékoztató árkülönbség|Modelladatok
ga|Praghas iarrtha|Ceant mórdhíola|Mínigh an praghas|Cén fáth an praghas seo?|Bearna praghais tháscach|Sonraí an mhúnla
it|Prezzo richiesto|Asta all'ingrosso|Spiega il prezzo|Perché questo prezzo?|Divario di prezzo indicativo|Dettagli del modello
lv|Sludinājuma cena|Vairumtirdzniecības izsole|Izskaidrot cenu|Kāpēc šāda cena?|Aptuvenā cenu starpība|Modeļa informācija
lt|Skelbimo kaina|Didmeninis aukcionas|Paaiškinti kainą|Kodėl tokia kaina?|Orientacinis kainų skirtumas|Modelio informacija
mt|Prezz mitlub|Irkant bl-ingrossa|Spjega l-prezz|Għaliex dan il-prezz?|Differenza indikattiva|Dettalji tal-mudell
pl|Cena ofertowa|Aukcja hurtowa|Wyjaśnij cenę|Skąd ta cena?|Orientacyjna różnica cen|Szczegóły modelu
pt|Preço pedido|Leilão grossista|Explicar o preço|Porquê este preço?|Diferença de preço indicativa|Detalhes do modelo
ro|Preț cerut|Licitație angro|Explică prețul|De ce acest preț?|Diferență de preț orientativă|Detalii despre model
sk|Ponuková cena|Veľkoobchodná aukcia|Vysvetliť cenu|Prečo táto cena?|Orientačný cenový rozdiel|Detaily modelu
sl|Oglasna cena|Veleprodajna dražba|Pojasni ceno|Zakaj ta cena?|Okvirna razlika v ceni|Podrobnosti modela
es|Precio anunciado|Subasta mayorista|Explicar este precio|¿Por qué este precio?|Diferencia orientativa|Detalles del modelo
sv|Annonspris|Grossistauktion|Förklara priset|Varför detta pris?|Ungefärlig prisskillnad|Modellinformation
tr|İlan fiyatı|Toptan açık artırma|Fiyatı açıkla|Neden bu fiyat?|Tahmini fiyat farkı|Model ayrıntıları
nb|Annonsepris|Engrosauksjon|Forklar prisen|Hvorfor denne prisen?|Veiledende prisforskjell|Modelldetaljer
sr-Latn|Oglasna cena|Veleprodajna aukcija|Objasni cenu|Zašto ova cena?|Okvirna razlika cena|Detalji modela
sr-Cyrl|Оглашена цена|Велепродајна аукција|Објасни цену|Зашто ова цена?|Оквирна разлика цена|Детаљи модела
sq|Çmimi i shpallur|Ankand me shumicë|Shpjego çmimin|Pse ky çmim?|Diferencë orientuese|Detajet e modelit
bs|Oglasna cijena|Veleprodajna aukcija|Objasni cijenu|Zašto ova cijena?|Okvirna razlika cijena|Detalji modela
mk|Огласена цена|Аукција на големо|Објасни ја цената|Зошто оваа цена?|Приближна разлика|Детали за моделот
is|Auglýst verð|Heildsöluuppboð|Útskýrðu verðið|Hvers vegna þetta verð?|Áætlaður verðmunur|Upplýsingar um líkanið
ca|Preu anunciat|Subhasta majorista|Explica el preu|Per què aquest preu?|Diferència orientativa|Detalls del model"""
COPY = {}
for row in _ROWS.splitlines():
    language, *words = row.split("|")
    if len(words) != len(_KEYS) or not all(words):
        raise ValueError(f"Incomplete pricing copy: {language}")
    COPY[language] = dict(zip(_KEYS, words))
if set(COPY) != set(LOCALES):
    raise ValueError("Pricing-language coverage must match the 36 supported locales")

def short(key: str) -> str:
    return COPY[locale()][key]
