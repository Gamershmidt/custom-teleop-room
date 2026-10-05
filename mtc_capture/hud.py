"""Status panel for the headset: rendered on the Mac as a PNG (so the font, incl. Cyrillic, is under our
control, no web fonts needed in the headset) and shown head-locked below the centre of view."""

import io

from PIL import Image, ImageDraw, ImageFont

FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
COLOR = dict(grey=(90, 96, 105), blue=(46, 110, 214), yellow=(214, 160, 10), green=(26, 150, 70), red=(200, 30, 35),
             orange=(220, 120, 20))
SIZE = (1024, 256)   # aspect 4:1, shown 0.12 m tall at 1 m

T = {
    "ru": dict(
        uncal=("НЕ ОТКАЛИБРОВАНО", "Встаньте на домашнюю точку лицом вдоль площадки",
               "Сведите руки перед лицом и сожмите пальцы на 1 с"),
        refused=("КАЛИБРОВКА ОТКЛОНЕНА", "Глаза на {eye:.2f} м, ожидается {exp:.2f} м (пол смещён на {off:+.2f} м?)",
                 "Встаньте прямо или исправьте высоту пола в настройках границы"),
        ret=("ВЕРНИТЕСЬ НА СТАРТ", "{where}", "Встаньте на серый круг лицом к синей цели"),
        arming=("ПРИГОТОВЬТЕСЬ… {n}", "{where}", "Стойте спокойно"),
        face=("ПОВЕРНИТЕСЬ К ЦЕЛИ", "{where}", "Смотрите на синий круг, отсчёт начнётся сам"),
        rec=("● ЗАПИСЬ {s:.0f} с", "{where}", "Идите к синей цели, берегите руки"),
        rec_dirty=("● ЗАПИСЬ {s:.0f} с — КАСАНИЕ", "{where}", "Задето: {what}"),
        hands=("● ЗАПИСЬ {s:.0f} с", "{where}", "Руки не видны камерам! Держите их перед собой"),
        safe=("ГОТОВО: БЕЗОПАСНО", "{where}", "Вернитесь на домашнюю точку"),
        unsafe=("ГОТОВО — БЫЛО КАСАНИЕ", "{where}", "Вернитесь на домашнюю точку, дубль повторится"),
        aborted=("ПРЕРВАНО", "{where}", "Вернитесь на домашнюю точку"),
        finished=("ВСЁ ЗАПИСАНО", "Все выбранные сегменты готовы", "Можно снять очки"),
        where="маршрут {route} · сегмент {seg}/{nseg} · дубль {take}/{takes}"),
    "en": dict(
        uncal=("NOT CALIBRATED", "Stand on the home spot facing along the floor",
               "Pinch both hands together in front of your face for 1 s"),
        refused=("CALIBRATION REFUSED", "Eyes at {eye:.2f} m, expected {exp:.2f} m (floor off by {off:+.2f} m?)",
                 "Stand upright or fix the floor height in the boundary setup"),
        ret=("GO BACK TO THE START", "{where}", "Stand on the grey pad facing the blue goal"),
        arming=("GET READY… {n}", "{where}", "Stand still"),
        face=("FACE THE GOAL", "{where}", "Look at the blue pad; the countdown starts by itself"),
        rec=("● RECORDING {s:.0f} s", "{where}", "Walk to the blue goal, protect the hands"),
        rec_dirty=("● RECORDING {s:.0f} s — CONTACT", "{where}", "Touched: {what}"),
        hands=("● RECORDING {s:.0f} s", "{where}", "Hands not visible! Keep them in front of you"),
        safe=("DONE: SAFE", "{where}", "Walk back to the home spot"),
        unsafe=("DONE — THERE WAS CONTACT", "{where}", "Walk back; the take will be repeated"),
        aborted=("ABORTED", "{where}", "Walk back to the home spot"),
        finished=("ALL DONE", "Every selected segment is recorded", "You can take the headset off"),
        where="route {route} · segment {seg}/{nseg} · take {take}/{takes}"),
}


def _font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


_F = {}


def font(size):
    if size not in _F:
        _F[size] = _font(size)
    return _F[size]


def render_lines(title, line2, line3, color="grey"):
    """Free-form panel (same look as render)."""
    T["_free"] = dict(_x=(title, line2, line3), where="")
    try:
        return render("_x", color, "_free")
    finally:
        T.pop("_free", None)


def render(key, color, lang="ru", **fmt):
    """key: one of T[lang]; color: COLOR name; fmt: values for the templates. -> PNG bytes."""
    tt = T.get(lang, T["en"])
    fmt.setdefault("where", tt["where"].format(**{k: fmt.get(k, "?") for k in ("route", "seg", "nseg", "take", "takes")}))
    title, l2, l3 = (x.format(**fmt) for x in tt[key]) if lang != "_free" else tt[key]
    img = Image.new("RGB", SIZE, (24, 26, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 22, SIZE[1]], fill=COLOR[color])            # status colour bar
    d.rectangle([0, 0, SIZE[0] - 1, SIZE[1] - 1], outline=COLOR[color], width=6)
    width = SIZE[0] - 48 - 24

    def fit(text, size):   # shrink until the line fits the panel
        while size > 18 and d.textlength(text, font=font(size)) > width:
            size -= 2
        return font(size)

    d.text((48, 18), title, font=fit(title, 78), fill=COLOR[color] if color != "grey" else (230, 230, 230))
    d.text((48, 120), l2, font=fit(l2, 40), fill=(215, 215, 215))
    d.text((48, 176), l3, font=fit(l3, 44), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
