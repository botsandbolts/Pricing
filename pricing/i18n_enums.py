"""Localise categorical choices at presentation time only.

The returned UI value stays the original canonical training/serving token.
"""
from pricing.i18n import locale

KEYS=["petrol","diesel","electric","hybrid","automatic","manual","unknown","other","yes","no","gas","sedan","hatchback","wagon","van","convertible"]
ROWS=r"""en-GB|Petrol|Diesel|Electric|Hybrid|Automatic|Manual|Unknown|Other|Yes|No|Petrol|Saloon|Hatchback|Estate|Van|Convertible
en-US|Gasoline|Diesel|Electric|Hybrid|Automatic|Manual|Unknown|Other|Yes|No|Gas|Sedan|Hatchback|Wagon|Van|Convertible
nl|Benzine|Diesel|Elektrisch|Hybride|Automaat|Handgeschakeld|Onbekend|Anders|Ja|Nee|Benzine|Sedan|Hatchback|Stationwagen|Bestelwagen|Cabrio
fr|Essence|Diesel|Électrique|Hybride|Automatique|Manuelle|Inconnu|Autre|Oui|Non|Essence|Berline|Compacte|Break|Fourgonnette|Cabriolet
de|Benzin|Diesel|Elektro|Hybrid|Automatik|Schaltgetriebe|Unbekannt|Andere|Ja|Nein|Benzin|Limousine|Kompaktwagen|Kombi|Transporter|Cabrio
es|Gasolina|Diésel|Eléctrico|Híbrido|Automático|Manual|Desconocido|Otro|Sí|No|Gasolina|Berlina|Compacto|Familiar|Furgoneta|Descapotable
it|Benzina|Diesel|Elettrico|Ibrido|Automatico|Manuale|Sconosciuto|Altro|Sì|No|Benzina|Berlina|Due volumi|Station wagon|Furgone|Cabriolet
pl|Benzyna|Diesel|Elektryczny|Hybrydowy|Automatyczna|Manualna|Nieznane|Inne|Tak|Nie|Benzyna|Sedan|Hatchback|Kombi|Furgon|Kabriolet
pt|Gasolina|Gasóleo|Elétrico|Híbrido|Automática|Manual|Desconhecido|Outro|Sim|Não|Gasolina|Berlina|Utilitário|Carrinha|Furgão|Descapotável
bg|Бензин|Дизел|Електрически|Хибрид|Автоматична|Ръчна|Неизвестно|Друго|Да|Не|Бензин|Седан|Хечбек|Комби|Ван|Кабриолет
hr|Benzin|Dizel|Električni|Hibridni|Automatski|Ručni|Nepoznato|Ostalo|Da|Ne|Benzin|Limuzina|Hatchback|Karavan|Kombi|Kabriolet
cs|Benzín|Nafta|Elektro|Hybrid|Automatická|Manuální|Neznámé|Jiné|Ano|Ne|Benzín|Sedan|Hatchback|Kombi|Dodávka|Kabriolet
da|Benzin|Diesel|Elektrisk|Hybrid|Automatgear|Manuelt gear|Ukendt|Andet|Ja|Nej|Benzin|Sedan|Hatchback|Stationcar|Varevogn|Cabriolet
et|Bensiin|Diisel|Elektriline|Hübriid|Automaat|Manuaal|Teadmata|Muu|Jah|Ei|Bensiin|Sedaan|Luukpära|Universaal|Kaubik|Kabriolett
fi|Bensiini|Diesel|Sähkö|Hybridi|Automaatti|Manuaali|Tuntematon|Muu|Kyllä|Ei|Bensiini|Sedan|Viistoperä|Farmari|Pakettiauto|Avoauto
el|Βενζίνη|Ντίζελ|Ηλεκτρικό|Υβριδικό|Αυτόματο|Χειροκίνητο|Άγνωστο|Άλλο|Ναι|Όχι|Βενζίνη|Σεντάν|Χάτσμπακ|Στέισον|Βαν|Κάμπριο
hu|Benzin|Dízel|Elektromos|Hibrid|Automata|Kézi|Ismeretlen|Egyéb|Igen|Nem|Benzin|Szedán|Ötajtós|Kombi|Furgon|Kabrió
ga|Peitreal|Díosal|Leictreach|Hibrideach|Uathoibríoch|Lámhleabhar|Anaithnid|Eile|Tá|Níl|Peitreal|Sedan|Hatchback|Eastát|Veain|Inchomhshóite
lv|Benzīns|Dīzelis|Elektrisks|Hibrīds|Automātiskā|Mehāniskā|Nezināms|Cits|Jā|Nē|Benzīns|Sedans|Hečbeks|Universālis|Furgons|Kabriolets
lt|Benzinas|Dyzelinas|Elektrinis|Hibridinis|Automatinė|Mechaninė|Nežinoma|Kita|Taip|Ne|Benzinas|Sedanas|Hečbekas|Universalas|Furgonas|Kabrioletas
mt|Petrol|Diżil|Elettriku|Ibridu|Awtomatiku|Manwali|Mhux magħruf|Ieħor|Iva|Le|Petrol|Sedan|Hatchback|Station wagon|Vann|Konvertibbli
ro|Benzină|Motorină|Electric|Hibrid|Automată|Manuală|Necunoscut|Altul|Da|Nu|Benzină|Sedan|Hatchback|Break|Furgonetă|Decapotabilă
sk|Benzín|Nafta|Elektrický|Hybridný|Automatická|Manuálna|Neznáme|Iné|Áno|Nie|Benzín|Sedan|Hatchback|Kombi|Dodávka|Kabriolet
sl|Bencin|Dizel|Električni|Hibridni|Samodejni|Ročni|Neznano|Drugo|Da|Ne|Bencin|Limuzina|Kombilimuzina|Karavan|Kombi|Kabriolet
sv|Bensin|Diesel|El|Hybrid|Automat|Manuell|Okänt|Annat|Ja|Nej|Bensin|Sedan|Halvkombi|Kombi|Skåpbil|Cabriolet"""
ENUMS={}
for line in ROWS.splitlines():
    code,*names=line.split("|")
    if len(names)!=len(KEYS) or not all(names):
        raise ValueError("Incomplete enum localization: "+code)
    ENUMS[code]=dict(zip(KEYS,names))

def option_label_for(locale_code, raw):
    value=str(raw).strip()
    if value=="__missing__":
        return "—"
    return ENUMS.get(locale_code, ENUMS["en-GB"]).get(value.lower(),value)


def option_label(raw):
    return option_label_for(locale(),raw)
