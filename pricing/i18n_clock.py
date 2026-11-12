"""Localized public date-control labels.

The selected code is canonical and independent from translated strings.
"""
from pricing.i18n import locale, LOCALES
KEYS=("today","history","monthend","proxy")
_ROWS=r"""en-GB|Today|Past date|Month end|Allow a hypothetical auction proxy for newer cars
en-US|Today|Past date|End of month|Allow a hypothetical auction proxy for newer cars
de|Heute|Früheres Datum|Monatsende|Hypothetischen Auktionsvergleich für jüngere Fahrzeuge erlauben
fr|Aujourd’hui|Date passée|Fin du mois|Autoriser un équivalent d’enchère hypothétique pour les voitures récentes
ru|Сегодня|Прошлая дата|Конец месяца|Разрешить условный аукционный аналог для новых автомобилей
uk|Сьогодні|Минула дата|Кінець місяця|Дозволити умовний аукціонний аналог для новіших авто
bg|Днес|Минала дата|Края на месеца|Разреши условен аукционен аналог за по-нови коли
hr|Danas|Prošli datum|Kraj mjeseca|Dopusti hipotetsku aukcijsku usporedbu za novija vozila
cs|Dnes|Minulé datum|Konec měsíce|Povolit hypotetické aukční srovnání pro novější auta
da|I dag|Tidligere dato|Månedens udgang|Tillad hypotetisk auktionssammenligning for nyere biler
nl|Vandaag|Eerdere datum|Maandeinde|Sta hypothetische veilingvergelijking voor nieuwere auto's toe
et|Täna|Varasem kuupäev|Kuu lõpp|Luba uuematele autodele hüpoteetiline oksjonivõrdlus
fi|Tänään|Aiempi päivä|Kuukauden loppu|Salli hypoteettinen huutokauppavertailu uudemmille autoille
el|Σήμερα|Παλαιότερη ημερομηνία|Τέλος μήνα|Να επιτρέπεται υποθετικό ανάλογο δημοπρασίας για νεότερα οχήματα
hu|Ma|Korábbi dátum|Hónap vége|Feltételezett aukciós összehasonlítás engedélyezése újabb autóknál
ga|Inniu|Dáta roimhe seo|Deireadh na míosa|Ceadaigh samhail cheant hipitéiseach do ghluaisteáin níos nuaí
it|Oggi|Data passata|Fine mese|Consenti confronto d'asta ipotetico per auto più recenti
lv|Šodien|Iepriekšējs datums|Mēneša beigas|Atļaut hipotētisku izsoles salīdzinājumu jaunākiem auto
lt|Šiandien|Ankstesnė data|Mėnesio pabaiga|Leisti hipotetinį aukciono palyginimą naujesniems automobiliams
mt|Illum|Data tal-passat|Tmiem ix-xahar|Ippermetti tqabbil ipotetiku tal-irkant għal karozzi aktar ġodda
pl|Dzisiaj|Wcześniejsza data|Koniec miesiąca|Pozwól na hipotetyczny odpowiednik aukcyjny nowszych aut
pt|Hoje|Data anterior|Fim do mês|Permitir comparação hipotética de leilão para carros mais recentes
ro|Astăzi|Dată anterioară|Sfârșit de lună|Permite comparație ipotetică cu licitații pentru mașini mai noi
sk|Dnes|Starší dátum|Koniec mesiaca|Povoliť hypotetické aukčné porovnanie pre novšie autá
sl|Danes|Prejšnji datum|Konec meseca|Dovoli hipotetično primerjavo z dražbo za novejše avte
es|Hoy|Fecha anterior|Fin de mes|Permitir comparación hipotética con subastas de coches más nuevos
sv|I dag|Tidigare datum|Månadsslut|Tillåt hypotetisk auktionsjämförelse för nyare bilar
tr|Bugün|Geçmiş tarih|Ay sonu|Daha yeni araçlar için varsayımsal açık artırma karşılaştırmasına izin ver
nb|I dag|Tidligere dato|Månedsslutt|Tillat hypotetisk auksjonssammenligning for nyere biler
sr-Latn|Danas|Raniji datum|Kraj meseca|Dozvoli hipotetičko aukcijsko poređenje za novija vozila
sr-Cyrl|Данас|Ранији датум|Крај месеца|Дозволи хипотетичко аукцијско поређење за новија возила
sq|Sot|Datë e kaluar|Fund muaji|Lejo krahasim hipotetik me ankand për makina më të reja
bs|Danas|Raniji datum|Kraj mjeseca|Dozvoli hipotetičko aukcijsko poređenje za novija auta
mk|Денес|Претходен датум|Крај на месецот|Дозволи хипотетичка аукциска споредба за понови возила
is|Í dag|Fyrri dagsetning|Mánaðamót|Leyfa ímyndaðan uppboðssamanburð fyrir nýrri bíla
ca|Avui|Data passada|Final de mes|Permet comparació hipotètica de subhasta per a cotxes més nous"""
COPY={}
for line in _ROWS.splitlines():
    code,*names=line.split("|")
    if len(names)!=len(KEYS) or not all(names):
        raise ValueError("Incomplete date-control locale: "+code)
    COPY[code]=dict(zip(KEYS,names))
if set(COPY)!=set(LOCALES):
    raise ValueError("Date-control language coverage mismatch")

def clock_text(key: str) -> str:
    return COPY[locale()][key]
