"""Static, privacy-preserving interface localisation for Car Price Studio.

The selected language affects *presentation only*. Model features, enums, widget
keys, market IDs and scientific targets must remain canonical and untranslated.
English remains the explicit fallback while additional research panels are
migrated. No network translation, language inference or tracking cookies.
"""
from __future__ import annotations

import streamlit as st

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
if set(CATALOGS) != set(LOCALES):
    raise ValueError("Locale catalog is incomplete")


def locale() -> str:
    """Current session locale with safe fallback; no cookie or IP inference."""
    value = st.session_state.get("cps_locale", "en-GB")
    return value if value in LOCALES else "en-GB"


def tr(key: str, **kwargs) -> str:
    """Localised UI copy; missing deeper scientific copy stays English."""
    result = CATALOGS.get(locale(), ENGLISH).get(key, ENGLISH.get(key, key))
    return result.format(**kwargs) if kwargs else result


def _on_locale_change() -> None:
    new = st.session_state.get("cps_locale_select", "en-GB")
    if new in LOCALES:
        st.session_state["cps_locale"] = new
        st.query_params["lang"] = new


def render_language_selector() -> str:
    """Visible on every route, independent from country and pricing market.

    A URL query parameter makes language shareable across reloads, without
    adding a persistence or tracking cookie. Selection never changes vehicle
    feature units, model choice or existing widget keys.
    """
    requested = st.query_params.get("lang")
    if requested in LOCALES and requested != st.session_state.get("_cps_lang_url"):
        st.session_state["cps_locale"] = requested
    st.session_state["_cps_lang_url"] = requested
    st.session_state.setdefault("cps_locale", "en-GB")
    current = locale()
    if st.session_state.get("cps_locale_select") != current:
        st.session_state["cps_locale_select"] = current
    _, right = st.columns([2.4, 1], gap="small")
    with right:
        st.selectbox(
            "🌐 Language / Sprache / Langue",
            tuple(LOCALES),
            key="cps_locale_select",
            format_func=lambda code: LOCALES[code],
            on_change=_on_locale_change,
            persist_state="session",
        )
    return locale()
