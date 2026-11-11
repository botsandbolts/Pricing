"""Curated high-priority non-EU European language variants for Car Price Studio.

Presentation-only translations. The platform's underlying market coverage,
model features, consent semantics, and legal source documents do not change.
"""
EXTRA_LOCALES={
    "uk": "🇺🇦 Українська",
    "ru": "🇷🇺 Русский",
    "tr": "🇹🇷 Türkçe",
    "nb": "🇳🇴 Norsk bokmål",
    "sr-Latn": "🇷🇸 Srpski (latinica)",
    "sr-Cyrl": "🇷🇸 Српски (ћирилица)",
    "sq": "🇦🇱 Shqip",
    "bs": "🇧🇦 Bosanski",
    "mk": "🇲🇰 Македонски",
    "is": "🇮🇸 Íslenska",
    "ca": "🇦🇩 Català"
}

def _parse(rows,keys):
    out={}
    for row in rows.splitlines():
        code,*values=row.split("|")
        if len(values)!=len(keys) or any(not x.strip() for x in values):
            raise ValueError("Incomplete extended translation: "+code)
        out[code]=dict(zip(keys,values))
    if set(out)!=set(EXTRA_LOCALES):
        raise ValueError("High-priority locale coverage mismatch")
    return out

CORE_KEYS=["market_title","market_intro","open_us","open_eu","form","country","make","model","year","month","mileage","power","fuel","transmission","calculate","panels"]
CORE_ROWS=r"""uk|Два континенти. Один гараж.|Незалежна оцінка авто. Без реєстрації та реклами.|Відкрити студію США|Відкрити європейську студію|Розкажіть про своє авто|Країна|Марка|Модель|Рік першої реєстрації|Місяць першої реєстрації|Пробіг (км)|Потужність (к. с.)|Пальне|Коробка передач|Розрахувати ціну в оголошенні|Розділи європейської студії
ru|Два континента. Один гараж.|Независимая оценка автомобилей. Без регистрации и рекламы.|Открыть студию США|Открыть европейскую студию|Расскажите об автомобиле|Страна|Марка|Модель|Год первой регистрации|Месяц первой регистрации|Пробег (км)|Мощность (л. с.)|Топливо|Коробка передач|Рассчитать цену объявления|Разделы европейской студии
tr|İki kıta. Tek garaj.|Bağımsız araç fiyat tahminleri. Üyelik ve reklam yok.|ABD stüdyosunu aç|Avrupa stüdyosunu aç|Aracınızı anlatın|Ülke|Marka|Model|İlk tescil yılı|İlk tescil ayı|Kilometre (km)|Güç (bg)|Yakıt|Şanzıman|İlan fiyatını hesapla|Avrupa stüdyosu bölümleri
nb|To kontinenter. Én garasje.|Uavhengige bilprisvurderinger. Ingen registrering eller reklame.|Åpne USA-studioet|Åpne Europa-studioet|Fortell om bilen din|Land|Merke|Modell|Førstegangsregistrert år|Førstegangsregistrert måned|Kilometerstand (km)|Effekt (hk)|Drivstoff|Girkasse|Beregn annonsepris|Paneler i Europastudioet
sr-Latn|Dva kontinenta. Jedna garaža.|Nezavisna procena cene automobila. Bez registracije i reklama.|Otvori studio za SAD|Otvori evropski studio|Ispričajte nam o automobilu|Država|Marka|Model|Godina prve registracije|Mesec prve registracije|Kilometraža (km)|Snaga (KS)|Gorivo|Menjač|Izračunaj oglašenu cenu|Paneli evropskog studija
sr-Cyrl|Два континента. Једна гаража.|Независна процена цене аутомобила. Без регистрације и реклама.|Отвори студио за САД|Отвори европски студио|Испричајте нам о аутомобилу|Држава|Марка|Модел|Година прве регистрације|Месец прве регистрације|Километража (км)|Снага (КС)|Гориво|Мењач|Израчунај оглашену цену|Панели европског студија
sq|Dy kontinente. Një garazh.|Vlerësime të pavarura të çmimeve të makinave. Pa regjistrim dhe reklama.|Hap studion e SHBA-së|Hap studion evropiane|Na tregoni për makinën|Shteti|Marka|Modeli|Viti i regjistrimit të parë|Muaji i regjistrimit të parë|Kilometrazhi (km)|Fuqia (kf)|Karburanti|Kambio|Llogarit çmimin e shpalljes|Panelet e studios evropiane
bs|Dva kontinenta. Jedna garaža.|Nezavisne procjene cijena automobila. Bez registracije i reklama.|Otvori studio za SAD|Otvori evropski studio|Recite nam o automobilu|Država|Marka|Model|Godina prve registracije|Mjesec prve registracije|Kilometraža (km)|Snaga (KS)|Gorivo|Mjenjač|Izračunaj oglašenu cijenu|Paneli evropskog studija
mk|Два континента. Една гаража.|Независни проценки на цените на автомобилите. Без регистрација и реклами.|Отвори го студиото за САД|Отвори го европското студио|Кажете ни за автомобилот|Земја|Марка|Модел|Година на прва регистрација|Месец на прва регистрација|Километража (км)|Моќност (КС)|Гориво|Менувач|Пресметај огласена цена|Панели на европското студио
is|Tvær heimsálfur. Einn bílskúr.|Óháð verðmat á bílum. Engin skráning og engar auglýsingar.|Opna bandarísku útgáfuna|Opna evrópsku útgáfuna|Segðu okkur frá bílnum|Land|Tegund|Gerð|Ár fyrstu skráningar|Mánuður fyrstu skráningar|Akstur (km)|Afl (hö)|Eldsneyti|Gírkassi|Reikna auglýst verð|Hlutar evrópsku útgáfunnar
ca|Dos continents. Un garatge.|Valoracions independents de cotxes. Sense registre ni anuncis.|Obre l'estudi dels EUA|Obre l'estudi europeu|Explica'ns com és el teu cotxe|País|Marca|Model|Any de primera matriculació|Mes de primera matriculació|Quilometratge (km)|Potència (CV)|Combustible|Canvi|Calcula el preu anunciat|Seccions de l'estudi europeu"""
CORE=_parse(CORE_ROWS,CORE_KEYS)

PANEL_KEYS=["price","inspector","timeline","indices","evaluation","models","science","detective","outliers","intelligence","merchants","garage"]
PANEL_ROWS=r"""uk|Оцінити авто|Перевірка оголошень|Вартість із часом|Індекси цін|Оцінювання|Лабораторія моделей|Наука|Детектив угод|Лабораторія аномалій|Аналітика ринку|Продавці|Оберіть автосалон
ru|Оценить авто|Проверка объявлений|Стоимость со временем|Индексы цен|Оценка|Лаборатория моделей|Наука|Детектив сделок|Лаборатория аномалий|Аналитика рынка|Дилеры|Выбор автосалона
tr|Aracımın değerini hesapla|İlanları incele|Zaman içindeki değer|Fiyat endeksleri|Model değerlendirmesi|Model laboratuvarı|Bilimsel yöntem|Fırsat dedektifi|Aykırı değerler|Piyasa analizi|Satıcılar|Galeri seç
nb|Verdsett bilen min|Annonsekontroll|Verdi over tid|Prisindekser|Evaluering|Modellaboratorium|Forskningen|Tilbudsdetektiv|Avvikslaboratorium|Markedsinnsikt|Forhandlere|Velg bilforhandler
sr-Latn|Proceni auto|Provera oglasa|Vrednost kroz vreme|Indeksi cena|Evaluacija|Laboratorija modela|Nauka|Detektiv ponuda|Laboratorija odstupanja|Analiza tržišta|Prodavci|Izaberi auto-plac
sr-Cyrl|Процени ауто|Провера огласа|Вредност кроз време|Индекси цена|Евалуација|Лабораторија модела|Наука|Детектив понуда|Лабораторија одступања|Анализа тржишта|Продавци|Изабери ауто-плац
sq|Vlerëso makinën|Kontrollo shpalljet|Vlera me kalimin e kohës|Indekset e çmimeve|Vlerësimi|Laboratori i modeleve|Shkenca|Zbuluesi i ofertave|Laboratori i anomalive|Analiza e tregut|Tregtarët|Zgjidh sallonin
bs|Procijeni auto|Provjera oglasa|Vrijednost kroz vrijeme|Indeksi cijena|Evaluacija|Laboratorija modela|Nauka|Detektiv ponuda|Laboratorija odstupanja|Analiza tržišta|Trgovci|Odaberi auto-kuću
mk|Процени автомобил|Проверка на огласи|Вредност низ времето|Индекси на цени|Евалуација|Лабораторија за модели|Науката|Детектив за понуди|Лабораторија за отстапувања|Анализа на пазарот|Продавачи|Избери автосалон
is|Meta bílinn minn|Skoða auglýsingar|Verðmæti með tímanum|Verðvísitölur|Mat á líkönum|Líkanasmiðja|Vísindin|Tilboðsleit|Frávikagreining|Markaðsgreining|Söluaðilar|Velja bílasölu
ca|Valora el meu cotxe|Revisa anuncis|Valor al llarg del temps|Índexs de preus|Avaluació|Laboratori de models|La ciència|Detectiu d'ofertes|Laboratori d'anomalies|Anàlisi del mercat|Venedors|Tria concessionari"""
PANELS=_parse(PANEL_ROWS,PANEL_KEYS)

COUNTRY_KEYS=["de","fr","nl","be","at","es","it","pl"]
COUNTRY_ROWS=r"""uk|Німеччина|Франція|Нідерланди|Бельгія|Австрія|Іспанія|Італія|Польща
ru|Германия|Франция|Нидерланды|Бельгия|Австрия|Испания|Италия|Польша
tr|Almanya|Fransa|Hollanda|Belçika|Avusturya|İspanya|İtalya|Polonya
nb|Tyskland|Frankrike|Nederland|Belgia|Østerrike|Spania|Italia|Polen
sr-Latn|Nemačka|Francuska|Holandija|Belgija|Austrija|Španija|Italija|Poljska
sr-Cyrl|Немачка|Француска|Холандија|Белгија|Аустрија|Шпанија|Италија|Пољска
sq|Gjermania|Franca|Holanda|Belgjika|Austria|Spanja|Italia|Polonia
bs|Njemačka|Francuska|Nizozemska|Belgija|Austrija|Španija|Italija|Poljska
mk|Германија|Франција|Холандија|Белгија|Австрија|Шпанија|Италија|Полска
is|Þýskaland|Frakkland|Holland|Belgía|Austurríki|Spánn|Ítalía|Pólland
ca|Alemanya|França|Països Baixos|Bèlgica|Àustria|Espanya|Itàlia|Polònia"""
COUNTRIES=_parse(COUNTRY_ROWS,COUNTRY_KEYS)

VOICE_KEYS=["headline","invitation"]
VOICE_ROWS=r"""uk|Кум уже назвав ціну. А що кажуть дані?|Перевірте оцінку й розкажіть, де ми помилилися.
ru|Знакомый уже назвал цену. А что говорят данные?|Проверьте оценку и расскажите, где мы промахнулись.
tr|Galerici bir fiyat söyler. Veriler ne diyor?|Fiyatı deneyin, tutturamadığımız yeri bize söyleyin.
nb|Svoger har et prisforslag. Vi har data.|Prøv beregningen og si fra når vi bommer.
sr-Latn|Kum zna cenu. Hajde da pitamo i podatke.|Proverite procenu i recite gde smo omanuli.
sr-Cyrl|Кум зна цену. Хајде да питамо и податке.|Проверите процену и реците где смо оманули.
sq|Shoku ka një çmim. Të dhënat kanë një mendim tjetër.|Provoje vlerësimin dhe na thuaj ku gabuam.
bs|Komšija zna cijenu. Mi pitamo podatke.|Isprobajte procjenu i recite gdje smo promašili.
mk|Комшијата има цена. А што велат податоците?|Пробајте ја проценката и кажете каде грешиме.
is|Mágurinn veit verðið. Við spurðum líka gögnin.|Prófaðu matið og segðu okkur hvar við skjátlumst.
ca|El cunyat ja té preu. Nosaltres consultem les dades.|Prova la valoració i digues-nos on fallem."""
VOICE={code:(v["headline"],v["invitation"]) for code,v in _parse(VOICE_ROWS,VOICE_KEYS).items()}

A11Y_KEYS=["reading","larger","contrast","skip","help"]
A11Y_ROWS=r"""uk|Налаштування читання|Збільшити текст|Вищий контраст|До інструментів|Можна також збільшити сторінку в браузері.
ru|Настройки чтения|Крупнее текст|Выше контраст|К инструментам|Также можно увеличить страницу в браузере.
tr|Okuma seçenekleri|Büyük yazı|Yüksek kontrast|Araçlara geç|Tarayıcınızdan da yakınlaştırabilirsiniz.
nb|Leseinnstillinger|Større tekst|Høyere kontrast|Gå til verktøy|Du kan også zoome i nettleseren.
sr-Latn|Podešavanja čitanja|Veći tekst|Jači kontrast|Idi na alate|Stranicu možete uvećati u pregledaču.
sr-Cyrl|Подешавања читања|Већи текст|Јачи контраст|Иди на алате|Страницу можете увећати у прегледачу.
sq|Opsionet e leximit|Tekst më i madh|Kontrast më i lartë|Te mjetet|Mund ta zmadhoni faqen edhe në shfletues.
bs|Postavke čitanja|Veći tekst|Jači kontrast|Idi na alate|Stranicu možete uvećati i u pregledniku.
mk|Опции за читање|Поголем текст|Поголем контраст|Оди до алатките|Можете да ја зголемите страницата во прелистувачот.
is|Lestrarstillingar|Stærri texti|Meiri birtuskil|Fara í verkfæri|Þú getur líka stækkað síðuna í vafranum.
ca|Opcions de lectura|Text més gran|Més contrast|Ves a les eines|També pots ampliar la pàgina des del navegador."""
A11Y=_parse(A11Y_ROWS,A11Y_KEYS)
