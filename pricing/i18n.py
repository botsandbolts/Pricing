"""Static, privacy-preserving interface localisation for Car Price Studio.

The selected language affects *presentation only*. Model features, enums, widget
keys, market IDs and scientific targets must remain canonical and untranslated.
English remains the explicit fallback while additional research panels are
migrated. Browser language is read locally; no IP geolocation, translation
service or tracking cookies.
"""
from __future__ import annotations

import streamlit as st
from pricing.i18n_extra_core import EXTRA_LOCALES, CORE as EXTRA_CORE, PANELS as EXTRA_PANELS, COUNTRIES as EXTRA_COUNTRIES, VOICE as EXTRA_VOICE

# All 24 official EU languages; English is offered in both UK and US variants.
LOCALES = {
    "bg": "🇧🇬 Български", "hr": "🇭🇷 Hrvatski",
    "cs": "🇨🇿 Čeština", "da": "🇩🇰 Dansk",
    "nl": "🇳🇱 Nederlands", "en-GB": "🇬🇧 English (UK)",
    "en-US": "🇺🇸 English (US)", "et": "🇪🇪 Eesti",
    "fi": "🇫🇮 Suomi", "fr": "🇫🇷 Français",
    "de": "🇩🇪 Deutsch", "el": "🇬🇷 Ελληνικά",
    "hu": "🇭🇺 Magyar", "ga": "🇮🇪 Gaeilge",
    "it": "🇮🇹 Italiano", "lv": "🇱🇻 Latviešu",
    "lt": "🇱🇹 Lietuvių", "mt": "🇲🇹 Malti",
    "pl": "🇵🇱 Polski", "pt": "🇵🇹 Português",
    "ro": "🇷🇴 Română", "sk": "🇸🇰 Slovenčina",
    "sl": "🇸🇮 Slovenščina", "es": "🇪🇸 Español",
    "sv": "🇸🇪 Svenska",
}
LOCALES.update(EXTRA_LOCALES)


KEYS = (
    "market_title", "market_intro", "open_us", "open_eu",
    "form", "country", "make", "model", "year", "month",
    "mileage", "power", "fuel", "transmission", "calculate", "panels",
)

# Core calculator and market-gateway copy, stored locally and never translated
# by a third-party API. Further research-text translation can be added here.
_ROWS = r"""
bg|Два континента. Един гараж.|Независими оценки на автомобили. Без регистрация и реклами.|Отвори студиото за САЩ|Отвори европейското студио|Разкажете ни за автомобила|Държава|Марка|Модел|Година на първа регистрация|Месец на първа регистрация|Пробег (км)|Мощност (к.с.)|Гориво|Скоростна кутия|Изчисли обявната цена|Раздели на европейското студио
hr|Dva kontinenta. Jedna garaža.|Neovisne procjene cijena automobila. Bez registracije i oglasa.|Otvori američki studio|Otvori europski studio|Recite nam o svom automobilu|Država|Marka|Model|Godina prve registracije|Mjesec prve registracije|Kilometraža (km)|Snaga (KS)|Gorivo|Mjenjač|Izračunaj oglašenu cijenu|Odjeljci europskog studija
cs|Dva kontinenty. Jedna garáž.|Nezávislé odhady cen aut. Bez registrace a reklam.|Otevřít americké studio|Otevřít evropské studio|Popište své auto|Země|Značka|Model|Rok první registrace|Měsíc první registrace|Nájezd (km)|Výkon (k)|Palivo|Převodovka|Vypočítat nabídkovou cenu|Sekce evropského studia
da|To kontinenter. Én garage.|Uafhængige bilprisvurderinger. Ingen registrering eller reklamer.|Åbn USA-studiet|Åbn Europa-studiet|Fortæl om din bil|Land|Mærke|Model|Første registreringsår|Første registreringsmåned|Kilometerstand (km)|Effekt (hk)|Brændstof|Gearkasse|Beregn annonceret pris|Europæiske studiopaneler
nl|Twee continenten. Eén garage.|Onafhankelijke autowaarderingen. Zonder registratie of advertenties.|Open de VS-studio|Open de Europese studio|Vertel ons over je auto|Land|Merk|Model|Jaar eerste toelating|Maand eerste toelating|Kilometerstand (km)|Vermogen (pk)|Brandstof|Transmissie|Bereken de vraagprijs|Panelen van de Europese studio
et|Kaks mandrit. Üks garaaž.|Sõltumatud autohinna hinnangud. Registreerimise ja reklaamideta.|Ava USA stuudio|Ava Euroopa stuudio|Räägi oma autost|Riik|Mark|Mudel|Esmase registreerimise aasta|Esmase registreerimise kuu|Läbisõit (km)|Võimsus (hj)|Kütus|Käigukast|Arvuta kuulutuse hind|Euroopa stuudio jaotised
fi|Kaksi mannerta. Yksi autotalli.|Riippumattomat autojen hinta-arviot. Ei rekisteröitymistä tai mainoksia.|Avaa Yhdysvaltain studio|Avaa Euroopan studio|Kerro autostasi|Maa|Merkki|Malli|Ensirekisteröintivuosi|Ensirekisteröintikuukausi|Ajokilometrit (km)|Teho (hv)|Polttoaine|Vaihteisto|Laske pyyntihinta|Euroopan studion osiot
fr|Deux continents. Un garage.|Estimations automobiles indépendantes. Sans inscription ni publicité.|Ouvrir le studio américain|Ouvrir le studio européen|Parlez-nous de votre voiture|Pays|Marque|Modèle|Année de première immatriculation|Mois de première immatriculation|Kilométrage (km)|Puissance (ch)|Carburant|Transmission|Estimer le prix demandé|Sections du studio européen
de|Zwei Kontinente. Eine Garage.|Unabhängige Fahrzeugbewertungen. Ohne Anmeldung oder Werbung.|US-Studio öffnen|Europa-Studio öffnen|Erzähl uns von deinem Auto|Land|Marke|Modell|Jahr der Erstzulassung|Monat der Erstzulassung|Kilometerstand (km)|Leistung (PS)|Kraftstoff|Getriebe|Angebotspreis berechnen|Bereiche des Europa-Studios
el|Δύο ήπειροι. Ένα γκαράζ.|Ανεξάρτητες εκτιμήσεις τιμών αυτοκινήτων. Χωρίς εγγραφή ή διαφημίσεις.|Άνοιγμα στούντιο ΗΠΑ|Άνοιγμα ευρωπαϊκού στούντιο|Πείτε μας για το αυτοκίνητό σας|Χώρα|Μάρκα|Μοντέλο|Έτος πρώτης ταξινόμησης|Μήνας πρώτης ταξινόμησης|Χιλιόμετρα (χλμ.)|Ισχύς (ίπποι)|Καύσιμο|Κιβώτιο ταχυτήτων|Υπολογισμός τιμής αγγελίας|Ενότητες ευρωπαϊκού στούντιο
hu|Két kontinens. Egy garázs.|Független autóár-becslések. Regisztráció és hirdetések nélkül.|Amerikai stúdió megnyitása|Európai stúdió megnyitása|Meséljen az autójáról|Ország|Márka|Modell|Első forgalomba helyezés éve|Első forgalomba helyezés hónapja|Kilométeróra-állás (km)|Teljesítmény (LE)|Üzemanyag|Sebességváltó|Hirdetési ár kiszámítása|Az európai stúdió részei
ga|Dhá mhór-roinn. Garáiste amháin.|Meastacháin neamhspleácha ar phraghsanna carranna. Gan chlárú ná fógraí.|Oscail stiúideo SAM|Oscail stiúideo na hEorpa|Inis dúinn faoi do charr|Tír|Déantús|Samhail|Bliain an chéad chlárúcháin|Mí an chéad chlárúcháin|Míleáiste (km)|Cumhacht (hp)|Breosla|Tarchur|Ríomh an praghas iarrtha|Painéil stiúideo na hEorpa
it|Due continenti. Un solo garage.|Valutazioni auto indipendenti. Senza registrazione né pubblicità.|Apri lo studio USA|Apri lo studio europeo|Parlaci della tua auto|Paese|Marca|Modello|Anno di prima immatricolazione|Mese di prima immatricolazione|Chilometraggio (km)|Potenza (CV)|Carburante|Cambio|Calcola il prezzo richiesto|Sezioni dello studio europeo
lv|Divi kontinenti. Viena garāža.|Neatkarīgi auto cenu novērtējumi. Bez reģistrācijas un reklāmām.|Atvērt ASV studiju|Atvērt Eiropas studiju|Pastāstiet par savu auto|Valsts|Marka|Modelis|Pirmās reģistrācijas gads|Pirmās reģistrācijas mēnesis|Nobraukums (km)|Jauda (ZS)|Degviela|Ātrumkārba|Aprēķināt sludinājuma cenu|Eiropas studijas sadaļas
lt|Du žemynai. Vienas garažas.|Nepriklausomi automobilių kainų vertinimai. Be registracijos ir reklamų.|Atidaryti JAV studiją|Atidaryti Europos studiją|Papasakokite apie automobilį|Šalis|Markė|Modelis|Pirmos registracijos metai|Pirmos registracijos mėnuo|Rida (km)|Galia (AG)|Degalai|Pavarų dėžė|Apskaičiuoti skelbimo kainą|Europos studijos skyriai
mt|Żewġ kontinenti. Garaxx wieħed.|Stimi indipendenti tal-prezzijiet tal-karozzi. Bla reġistrazzjoni jew reklami.|Iftaħ l-istudjo tal-Istati Uniti|Iftaħ l-istudjo Ewropew|Għidilna dwar il-karozza tiegħek|Pajjiż|Marka|Mudell|Sena tal-ewwel reġistrazzjoni|Xahar tal-ewwel reġistrazzjoni|Kilometraġġ (km)|Qawwa (hp)|Fjuwil|Trażmissjoni|Ikkalkula l-prezz mitlub|Sezzjonijiet tal-istudjo Ewropew
pl|Dwa kontynenty. Jeden garaż.|Niezależne wyceny samochodów. Bez rejestracji i reklam.|Otwórz studio USA|Otwórz studio europejskie|Opowiedz nam o swoim samochodzie|Kraj|Marka|Model|Rok pierwszej rejestracji|Miesiąc pierwszej rejestracji|Przebieg (km)|Moc (KM)|Paliwo|Skrzynia biegów|Oblicz cenę ofertową|Panele studia europejskiego
pt|Dois continentes. Uma garagem.|Avaliações independentes de automóveis. Sem registo nem publicidade.|Abrir estúdio dos EUA|Abrir estúdio europeu|Fale-nos do seu carro|País|Marca|Modelo|Ano da primeira matrícula|Mês da primeira matrícula|Quilometragem (km)|Potência (cv)|Combustível|Transmissão|Calcular o preço anunciado|Painéis do estúdio europeu
ro|Două continente. Un singur garaj.|Estimări independente ale prețurilor auto. Fără cont sau reclame.|Deschide studioul SUA|Deschide studioul european|Spune-ne despre mașina ta|Țară|Marcă|Model|Anul primei înmatriculări|Luna primei înmatriculări|Kilometraj (km)|Putere (CP)|Combustibil|Transmisie|Calculează prețul cerut|Panourile studioului european
sk|Dva kontinenty. Jedna garáž.|Nezávislé odhady cien áut. Bez registrácie a reklám.|Otvoriť štúdio USA|Otvoriť európske štúdio|Povedzte nám o svojom aute|Krajina|Značka|Model|Rok prvej registrácie|Mesiac prvej registrácie|Najazdené kilometre (km)|Výkon (k)|Palivo|Prevodovka|Vypočítať ponukovú cenu|Panely európskeho štúdia
sl|Dve celini. Ena garaža.|Neodvisne ocene cen avtomobilov. Brez registracije in oglasov.|Odpri studio ZDA|Odpri evropski studio|Povejte nam o svojem avtomobilu|Država|Znamka|Model|Leto prve registracije|Mesec prve registracije|Kilometrina (km)|Moč (KM)|Gorivo|Menjalnik|Izračunaj oglaševano ceno|Razdelki evropskega studia
es|Dos continentes. Un garaje.|Valoraciones independientes de coches. Sin registro ni anuncios.|Abrir estudio de EE. UU.|Abrir estudio europeo|Háblanos de tu coche|País|Marca|Modelo|Año de primera matriculación|Mes de primera matriculación|Kilometraje (km)|Potencia (CV)|Combustible|Transmisión|Calcular precio anunciado|Secciones del estudio europeo
sv|Två kontinenter. Ett garage.|Oberoende bilvärderingar. Utan registrering eller annonser.|Öppna USA-studion|Öppna Europastudion|Berätta om din bil|Land|Märke|Modell|Första registreringsår|Första registreringsmånad|Miltal (km)|Effekt (hk)|Bränsle|Växellåda|Beräkna annonspris|Paneler i Europastudion
""".strip()

ENGLISH = dict(zip(KEYS, (
    "Two continents. One garage.",
    "Independent car-price estimates. No registration. No ads.",
    "Open the US Studio", "Open the European Studio",
    "Tell us about your car", "Country", "Make", "Model",
    "First-registration year", "First-registration month", "Mileage (km)",
    "Power (hp)", "Fuel", "Transmission", "Calculate asking price",
    "European Studio panels",
)))
CATALOGS = {"en-GB": ENGLISH, "en-US": {
    **ENGLISH, "year": "First-registration year", "month": "First-registration month",
}}
for _line in _ROWS.splitlines():
    _parts = _line.split("|")
    if len(_parts) != len(KEYS) + 1:
        raise ValueError("Invalid localisation row: " + _parts[0])
    _locale, *_values = _parts
    CATALOGS[_locale] = dict(zip(KEYS, _values))
CATALOGS.update(EXTRA_CORE)
if set(CATALOGS) != set(LOCALES):
    raise ValueError("Locale catalog is incomplete")


def browser_locale(value: str | None) -> str:
    """Map navigator.language into one of our explicit supported locales."""
    code = str(value or "").strip().replace("_", "-").lower()
    language, *subtags = code.split("-")
    if language == "en":
        return "en-US" if "us" in subtags else "en-GB"
    if language == "sr":
        # Browser BCP-47 usually defaults to Serbian Cyrillic without a script.
        return "sr-Latn" if "latn" in subtags else "sr-Cyrl"
    if language in ("nb", "nn", "no"):
        # Nynorsk is not among the requested new variants.
        return "nb"
    return language if language in LOCALES else "en-GB"


def locale() -> str:
    """Return explicit user choice, local browser storage or browser default."""
    value = st.session_state.get("cps_locale", "en-GB")
    return value if value in LOCALES else "en-GB"


def tr(key: str, **kwargs) -> str:
    """Localised UI copy; untranslated research copy explicitly stays English."""
    result = CATALOGS.get(locale(), ENGLISH).get(key, ENGLISH.get(key, key))
    return result.format(**kwargs) if kwargs else result


# Distinct national/editorial voice instead of mechanical literal translation.
# These are *jokes and calls to action*, not claims of improved model accuracy.
VOICE = {
    "bg": ("Съседът има мнение. Данните имат аргументи.", "Проверете оценката и ни кажете какво не сме улучили."),
    "hr": ("Svatko zna cijenu. A što kažu podaci?", "Isprobajte procjenu i recite nam gdje griješimo."),
    "cs": ("Každý má názor. Data mají poslední slovo?", "Vyzkoušejte odhad a napište nám, co nesedí."),
    "da": ("Mindre mavefornemmelse. Mere dokumentation.", "Test prisen på din bil, og fortæl os, hvad vi skal forbedre."),
    "nl": ("Je zwager heeft een mening. Wij hebben data.", "Test de vraagprijs. Is het raak of zitten we ernaast?"),
    "en-GB": ("Your mate has a theory. The data has questions.", "Give your car a spin through the model — then tell us where it misses."),
    "en-US": ("Your buddy knows the market. So does everyone at the cookout.", "Test the estimate, kick the tires, and tell us what we missed."),
    "et": ("Arvamusi jagub. Aga mida näitavad andmed?", "Proovi hinnangut ja ütle, mida võiks parandada."),
    "fi": ("Mutu-tuntuma ei ole hinnoittelumalli.", "Kokeile arviota ja kerro, mikä meni pieleen."),
    "fr": ("Votre beau-frère a un avis. Nous avons des données.", "Testez le prix et dites-nous franchement où nous nous trompons."),
    "de": ("Der Schwager schätzt. Wir rechnen nach.", "Testen Sie die Bewertung – und sagen Sie uns, wo wir danebenliegen."),
    "el": ("Όλοι ξέρουν την τιμή. Τι λένε όμως τα δεδομένα;", "Δοκιμάστε την εκτίμηση και πείτε μας τι να βελτιώσουμε."),
    "hu": ("A sógor tudja az árat. Mi megkérdezzük az adatokat.", "Próbálja ki, és írja meg, hol tévedünk."),
    "ga": ("Tá tuairim ag cách. Cad a deir na sonraí?", "Bain triail as agus inis dúinn cad atá le feabhsú."),
    "it": ("Il cognato ha un prezzo. I dati ne hanno un altro.", "Prova la stima: promossi o bocciati? Dicci la verità!"),
    "lv": ("Viedokļu netrūkst. Ko saka dati?", "Izmēģini novērtējumu un pastāsti, ko uzlabot."),
    "lt": ("Nuomonių netrūksta. O ką sako duomenys?", "Išbandyk vertinimą ir parašyk, ką galime patobulinti."),
    "mt": ("Kulħadd għandu opinjoni. Xi tgħid id-data?", "Ipprova l-istima u għidilna x'nistgħu ntejbu."),
    "pl": ("Szwagier zna cenę. My sprawdzamy dane.", "Przetestuj wycenę i napisz bez ogródek, gdzie się mylimy."),
    "pt": ("O teu primo sabe o preço. Os dados também têm opinião.", "Experimenta a avaliação e diz-nos onde falhámos."),
    "ro": ("Cumnatul știe prețul. Noi întrebăm datele.", "Testează estimarea și spune-ne ce nu merge."),
    "sk": ("Švagor pozná cenu. My sa pýtame dát.", "Otestujte odhad a povedzte nám, kde sa mýlime."),
    "sl": ("Vsak ima svojo ceno. Kaj pravijo podatki?", "Preizkusite oceno in nam povejte, kaj izboljšati."),
    "es": ("Tu cuñado pone el precio. Nosotros consultamos los datos.", "Pruébalo y dinos sin pelos en la lengua dónde fallamos."),
    "sv": ("Alla har en åsikt. Vad säger siffrorna?", "Testa värderingen och berätta vad vi borde förbättra."),
}
VOICE.update(EXTRA_VOICE)



def format_eur(amount: float) -> str:
    """Present only an EUR amount in local typography; never convert it.

    Display separators are linguistic conventions, not model/currency changes.
    The quote retains full numeric precision in session state and exports.
    """
    number = format(float(amount), ",.0f")
    lang = locale()
    if lang in ("en-GB", "en-US", "ga", "mt"):
        return "€" + number
    if lang in ("de", "nl", "es", "it", "pt", "sl", "el", "tr", "ca", "is", "sr-Latn", "sr-Cyrl", "bs", "mk"):
        return number.replace(",", ".") + " €"
    # French and most Central/Northern/Eastern European publication styles.
    return number.replace(",", "\u202f") + " €"


def editorial_voice() -> tuple[str, str]:
    return VOICE[locale()]




# Friendly country names for the eight currently supported European *models*.
# Language selection never changes model-market coverage or country ISO codes.
_COUNTRY_KEYS = ("de", "fr", "nl", "be", "at", "es", "it", "pl")
_COUNTRY_ROWS = r"""
bg|Германия|Франция|Нидерландия|Белгия|Австрия|Испания|Италия|Полша
hr|Njemačka|Francuska|Nizozemska|Belgija|Austrija|Španjolska|Italija|Poljska
cs|Německo|Francie|Nizozemsko|Belgie|Rakousko|Španělsko|Itálie|Polsko
da|Tyskland|Frankrig|Nederlandene|Belgien|Østrig|Spanien|Italien|Polen
nl|Duitsland|Frankrijk|Nederland|België|Oostenrijk|Spanje|Italië|Polen
et|Saksamaa|Prantsusmaa|Madalmaad|Belgia|Austria|Hispaania|Itaalia|Poola
fi|Saksa|Ranska|Alankomaat|Belgia|Itävalta|Espanja|Italia|Puola
fr|Allemagne|France|Pays-Bas|Belgique|Autriche|Espagne|Italie|Pologne
de|Deutschland|Frankreich|Niederlande|Belgien|Österreich|Spanien|Italien|Polen
el|Γερμανία|Γαλλία|Κάτω Χώρες|Βέλγιο|Αυστρία|Ισπανία|Ιταλία|Πολωνία
hu|Németország|Franciaország|Hollandia|Belgium|Ausztria|Spanyolország|Olaszország|Lengyelország
ga|An Ghearmáin|An Fhrainc|An Ísiltír|An Bheilg|An Ostair|An Spáinn|An Iodáil|An Pholainn
it|Germania|Francia|Paesi Bassi|Belgio|Austria|Spagna|Italia|Polonia
lv|Vācija|Francija|Nīderlande|Beļģija|Austrija|Spānija|Itālija|Polija
lt|Vokietija|Prancūzija|Nyderlandai|Belgija|Austrija|Ispanija|Italija|Lenkija
mt|Il-Ġermanja|Franza|In-Netherlands|Il-Belġju|L-Awstrija|Spanja|L-Italja|Il-Polonja
pl|Niemcy|Francja|Holandia|Belgia|Austria|Hiszpania|Włochy|Polska
pt|Alemanha|França|Países Baixos|Bélgica|Áustria|Espanha|Itália|Polónia
ro|Germania|Franța|Țările de Jos|Belgia|Austria|Spania|Italia|Polonia
sk|Nemecko|Francúzsko|Holandsko|Belgicko|Rakúsko|Španielsko|Taliansko|Poľsko
sl|Nemčija|Francija|Nizozemska|Belgija|Avstrija|Španija|Italija|Poljska
es|Alemania|Francia|Países Bajos|Bélgica|Austria|España|Italia|Polonia
sv|Tyskland|Frankrike|Nederländerna|Belgien|Österrike|Spanien|Italien|Polen
""".strip()
COUNTRY_CATALOGS = {
    "en-GB": dict(zip(_COUNTRY_KEYS, (
        "Germany", "France", "Netherlands", "Belgium", "Austria",
        "Spain", "Italy", "Poland",
    ))),
    "en-US": dict(zip(_COUNTRY_KEYS, (
        "Germany", "France", "Netherlands", "Belgium", "Austria",
        "Spain", "Italy", "Poland",
    ))),
}
for _row in _COUNTRY_ROWS.splitlines():
    _fields = _row.split("|")
    if len(_fields) != len(_COUNTRY_KEYS) + 1:
        raise ValueError("Invalid country localization: " + _fields[0])
    _code, *_names = _fields
    COUNTRY_CATALOGS[_code] = dict(zip(_COUNTRY_KEYS, _names))
COUNTRY_CATALOGS.update(EXTRA_COUNTRIES)
if set(COUNTRY_CATALOGS) != set(LOCALES):
    raise ValueError("Country localization is incomplete")


def localized_country_for(locale_code: str, country_code: str) -> str:
    """Pure formatter: pins the locale for Streamlit widget serialization."""
    key = str(country_code).strip().lower()
    return COUNTRY_CATALOGS.get(locale_code, COUNTRY_CATALOGS["en-GB"]).get(key, country_code)


def localized_country(country_code: str) -> str:
    return localized_country_for(locale(), country_code)


# Navigation labels are translated at display time; underlying route IDs remain
# unchanged. The entries are the order of the original NAV menus.
_PANEL_KEYS = ("price", "inspector", "timeline", "indices", "evaluation", "models",
               "science", "detective", "outliers", "intelligence", "merchants", "garage")
_PANEL_ROWS = r"""
bg|Оцени автомобила|Проверка на обяви|Стойност във времето|Ценови индекси|Оценка|Лаборатория на модели|Науката|Детектив на сделки|Лаборатория за аномалии|Пазарни анализи|Търговци|Избери автокъща
hr|Procijeni auto|Pregled oglasa|Vrijednost kroz vrijeme|Indeksi cijena|Evaluacija|Laboratorij modela|Znanost|Detektiv ponuda|Laboratorij odstupanja|Analiza tržišta|Trgovci|Odaberi autosalon
cs|Ocenit auto|Kontrola inzerátů|Hodnota v čase|Cenové indexy|Vyhodnocení|Laboratoř modelů|Věda|Detektiv nabídek|Laboratoř odlehlých hodnot|Tržní analýza|Prodejci|Vyberte autosalon
da|Vurdér min bil|Annonceinspektør|Værdi over tid|Prisindeks|Evaluering|Modellaboratorium|Videnskaben|Tilbudsdetektiv|Afvigelseslaboratorium|Markedsindsigt|Forhandlere|Vælg forhandler
nl|Waardeer mijn auto|Advertentiecheck|Waarde door de tijd|Prijsindices|Evaluatie|Modellenlab|De wetenschap|Deal Detective|Uitschieterslab|Marktinzicht|Handelaars|Kies je garage
et|Hinda autot|Kuulutuste kontroll|Väärtus ajas|Hinnaindeksid|Hindamine|Mudelite labor|Teadus|Tehingute detektiiv|Erindite labor|Turuanalüüs|Müüjad|Vali esindus
fi|Arvioi autoni|Ilmoitusten tarkistus|Arvo ajan mittaan|Hintaindeksit|Arviointi|Mallilaboratorio|Tiede|Tarjousetsivä|Poikkeamien laboratorio|Markkinatieto|Myyjät|Valitse autoliike
fr|Estimer ma voiture|Analyse des annonces|Valeur dans le temps|Indices de prix|Évaluation|Labo des modèles|La science|Détective des offres|Labo des anomalies|Analyse du marché|Professionnels|Choisir un garage
de|Auto bewerten|Inserate prüfen|Wert im Zeitverlauf|Preisindizes|Evaluierung|Modelllabor|Die Wissenschaft|Deal-Detektiv|Ausreißerlabor|Marktanalyse|Händleranalyse|Werkstatt finden
el|Εκτίμηση αυτοκινήτου|Έλεγχος αγγελιών|Αξία στον χρόνο|Δείκτες τιμών|Αξιολόγηση|Εργαστήριο μοντέλων|Η επιστήμη|Έλεγχος προσφορών|Εργαστήριο αποκλίσεων|Ανάλυση αγοράς|Έμποροι|Επιλογή συνεργείου
hu|Autóm értéke|Hirdetések vizsgálata|Érték időben|Árindexek|Értékelés|Modell-labor|A tudomány|Ajánlatvizsgáló|Kiugró értékek|Piaci elemzés|Kereskedők|Válassz szalont
ga|Luach mo chairr|Scrúdú fógraí|Luach le himeacht ama|Innéacsanna praghsanna|Meastóireacht|Saotharlann na samhlacha|An eolaíocht|Bleachtaire margaí|Saotharlann asluite|Eolas margaidh|Déileálaithe|Roghnaigh garáiste
it|Valuta la mia auto|Controlla gli annunci|Valore nel tempo|Indici dei prezzi|Valutazione|Laboratorio modelli|La scienza|Detective delle offerte|Laboratorio anomalie|Analisi di mercato|Rivenditori|Scegli un autosalone
lv|Novērtē auto|Sludinājumu pārbaude|Vērtība laika gaitā|Cenu indeksi|Novērtējums|Modeļu laboratorija|Zinātne|Darījumu detektīvs|Anomāliju laboratorija|Tirgus analīze|Tirgotāji|Izvēlies autosalonu
lt|Įvertink automobilį|Skelbimų patikra|Vertė laikui bėgant|Kainų indeksai|Vertinimas|Modelių laboratorija|Mokslas|Sandorių detektyvas|Išskirčių laboratorija|Rinkos analizė|Pardavėjai|Pasirink saloną
mt|Ivvaluta l-karozza|Spezzjoni tar-reklami|Valur maż-żmien|Indiċijiet tal-prezzijiet|Evalwazzjoni|Laboratorju tal-mudelli|Ix-xjenza|Investigatur tal-offerti|Laboratorju tal-anomaliji|Analiżi tas-suq|Negozjanti|Agħżel garaxx
pl|Wyceń mój samochód|Sprawdź ogłoszenia|Wartość w czasie|Indeksy cen|Ocena modeli|Laboratorium modeli|Nauka|Detektyw okazji|Laboratorium anomalii|Analiza rynku|Handlarze|Wybierz komis
pt|Avaliar o meu carro|Analisar anúncios|Valor ao longo do tempo|Índices de preços|Avaliação|Laboratório de modelos|A ciência|Detetive de negócios|Laboratório de anomalias|Análise de mercado|Comerciantes|Escolher stand
ro|Evaluează mașina|Verifică anunțuri|Valoare în timp|Indici de preț|Evaluare|Laborator de modele|Știința|Detectiv de oferte|Laborator de anomalii|Analiza pieței|Dealeri|Alege un dealer
sk|Ohodnoť auto|Kontrola inzerátov|Hodnota v čase|Cenové indexy|Hodnotenie|Laboratórium modelov|Veda|Detektív ponúk|Laboratórium odchýlok|Analýza trhu|Predajcovia|Vyber autobazár
sl|Oceni moj avto|Pregled oglasov|Vrednost skozi čas|Cenovni indeksi|Vrednotenje|Laboratorij modelov|Znanost|Detektiv ponudb|Laboratorij odstopanj|Tržna analiza|Trgovci|Izberi salon
es|Valora mi coche|Revisar anuncios|Valor a lo largo del tiempo|Índices de precios|Evaluación|Laboratorio de modelos|La ciencia|Detective de ofertas|Laboratorio de anomalías|Análisis de mercado|Concesionarios|Elige concesionario
sv|Värdera min bil|Granska annonser|Värde över tid|Prisindex|Utvärdering|Modellabb|Vetenskapen|Affärsdetektiven|Avvikelselabb|Marknadsanalys|Handlare|Välj bilhandlare
""".strip()
PANEL_CATALOGS = {
    "en-GB": dict(zip(_PANEL_KEYS, (
        "Value my car", "Listing Inspector", "Value over time", "Price indices",
        "Evaluation", "Model lab", "The science", "Deal Detective",
        "Outlier lab", "Market intelligence", "Dealer view", "Pick your garage",
    ))),
    "en-US": dict(zip(_PANEL_KEYS, (
        "Price my car", "Listing Inspector", "Value over time", "Price indexes",
        "Evaluation", "Model lab", "The science", "Deal Detective",
        "Outlier lab", "Market intelligence", "Dealer view", "Pick your dealership",
    ))),
}
for _row in _PANEL_ROWS.splitlines():
    _fields = _row.split("|")
    if len(_fields) != len(_PANEL_KEYS) + 1:
        raise ValueError("Invalid panel localization: " + _fields[0])
    _code, *_names = _fields
    PANEL_CATALOGS[_code] = dict(zip(_PANEL_KEYS, _names))
PANEL_CATALOGS.update(EXTRA_PANELS)
if set(PANEL_CATALOGS) != set(LOCALES):
    raise ValueError("Panel localization is incomplete")

_PANEL_ROUTE_KEYS = {
    "🚗 Price my car": "price", "🔎 Listing Inspector": "inspector",
    "📉 Value over time": "timeline", "📅 Value over time": "timeline",
    "📈 Price Indices": "indices", "📊 Evaluation": "evaluation",
    "🧪 Model lab": "models", "🔬 The science": "science",
    "🧠 The science": "science", "🕵️ Deal Detective": "detective",
    "🧹 Outlier lab": "outliers", "🌍 Market intelligence": "intelligence",
    "🏪 Merchant lens": "merchants", "🚦 Pick your garage": "garage",
    "↔ Two-market garage": "garage",
}


def localized_panel_for(locale_code: str, route: str) -> str:
    """Pure formatter: preserve stable widget labels across reruns."""
    key = _PANEL_ROUTE_KEYS.get(route)
    if not key:
        return route
    return route.split(" ", 1)[0] + " " + PANEL_CATALOGS.get(locale_code,PANEL_CATALOGS["en-GB"])[key]


def localized_panel(route: str) -> str:
    return localized_panel_for(locale(),route)


# This script runs in the same origin as the Streamlit page, not a sandboxed
# iframe. Only a BCP-47 locale from our static allowlist is ever persisted.
_BROWSER_STORAGE_JS = """
export default function({data, setStateValue}) {
  const storageKey = "car-price-studio.language.v1";
  let stored = "";
  try {
    if (data.write && data.allowed.includes(data.write)) {
      localStorage.setItem(storageKey, data.write);
    }
    const candidate = localStorage.getItem(storageKey) || "";
    stored = data.allowed.includes(candidate) ? candidate : "";
  } catch (err) {
    // A privacy-hardened browser may disable localStorage; session choice works.
  }
  setStateValue("stored", stored);
}
"""


def _on_locale_change() -> None:
    new = st.session_state.get("cps_locale_select", "en-GB")
    if new in LOCALES:
        st.session_state["cps_locale"] = new
        st.session_state["cps_locale_manual"] = new
        # Shareable language link. No analytics, no IP-based locale inference.
        st.query_params["lang"] = new


def render_language_selector() -> str:
    """Browser locale on first visit; manual choice persisted in localStorage.

    Priority: manual choice this session > explicit link > localStorage >
    navigator.language > en-GB. Market and model training inputs stay untouched.
    Browser storage is local to the user's device and is not analytics consent.
    """
    requested = st.query_params.get("lang")
    browser = browser_locale(getattr(st.context, "locale", None))
    if not st.session_state.get("_cps_locale_initialised"):
        st.session_state["cps_locale"] = requested if requested in LOCALES else browser
        st.session_state["_cps_locale_initialised"] = True
    elif (requested in LOCALES and
          requested != st.session_state.get("_cps_lang_url") and
          not st.session_state.get("cps_locale_manual")):
        st.session_state["cps_locale"] = requested
    st.session_state["_cps_lang_url"] = requested

    # V2 has first-party DOM privileges and can actually access localStorage.
    # It reports a saved locale, causing one rerun at most when it changes.
    bridge = st.components.v2.component("cps_browser_language_storage",
                                         js=_BROWSER_STORAGE_JS)
    result = bridge(
        data={"write": st.session_state.get("cps_locale_manual", ""),
              "allowed": list(LOCALES)},
        key="cps_browser_language_storage",
        on_stored_change=lambda: None,
    )
    stored = getattr(result, "stored", None)
    if (stored in LOCALES and not st.session_state.get("cps_locale_manual")
            and requested not in LOCALES):
        st.session_state["cps_locale"] = stored
    current = locale()
    if st.session_state.get("cps_locale_select") != current:
        st.session_state["cps_locale_select"] = current
    _, right = st.columns([2.4, 1], gap="small")
    from pricing.i18n_story import story
    with right:
        st.selectbox(
            "🌐 "+story("picker"),
            tuple(LOCALES),
            key="cps_locale_select",
            format_func=lambda code: LOCALES[code],
            on_change=_on_locale_change,
            persist_state="session",
        )
    return locale()
