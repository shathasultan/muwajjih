"""Deterministic synthetic dataset generator for the triage classifier.

Usage (from the project root):

    python -m train.generate_dataset

Writes data/train.csv, data/val.csv, data/test.csv.

WHY IT IS BUILT THIS WAY
------------------------
The obvious design -- one list of complete sentences per intent -- produces a
model that scores 100% on its own test split and falls over on the first real
message. The reason is that each intent's sentences share a template skeleton,
so the classifier learns the skeleton instead of the vocabulary: it recognises
"وش حالة طلبي {order}؟" rather than learning that طلبي and حالة mean someone is
asking about an order.

So messages here are COMPOSED from three independent parts:

    [opener] + core + [closer]

`core` carries the discriminative content and is drawn from a per-intent pool.
Openers and closers are drawn from pools SHARED BY EVERY INTENT -- a polite
greeting, a thank-you, a "please help me" -- precisely so that they carry no
label information. A model that tries to key on them gets no signal, which
forces it onto the content words. That is the entire point, and it is what
makes the held-out probes in tests/behavioural/test_model_quality.py (written
by hand, never generated) a meaningful test rather than a formality.

Each core pool also deliberately spans registers: Modern Standard Arabic and
Gulf dialect, long sentences and three-word fragments, with and without the
polite framing real customers use.

Everything is driven by a fixed seed, so re-running this script always produces
byte-identical output -- required for reproducible training, and for CI to
re-verify the pipeline without committing a data file.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Slot vocabulary. Separate from the phrases so adding a product or a city
# never requires touching sentence structure.
# ---------------------------------------------------------------------------

PRODUCTS = [
    "الجوال",
    "الساعة الذكية",
    "السماعات اللاسلكية",
    "الشاحن السريع",
    "الحقيبة",
    "الكفرات",
    "شاشة اللابتوب",
    "لوحة المفاتيح",
    "الطابعة",
    "المكنسة الكهربائية",
    "المكيف",
    "الثلاجة الصغيرة",
    "الخلاط",
    "السخان",
    "المروحة",
    "جهاز البخار",
    "الميكروويف",
    "الغسالة",
    "الراوتر",
    "الكاميرا",
]

CITIES = ["الرياض", "جدة", "الدمام", "مكة", "المدينة", "أبها", "تبوك", "حائل", "الخبر", "بريدة"]

# Shared by every intent ON PURPOSE -- see the module docstring. These must
# carry zero label information.
OPENERS = [
    "",
    "",
    "",
    "",  # most messages have no opener at all
    "السلام عليكم،",
    "مساء الخير،",
    "صباح الخير،",
    "هلا،",
    "مرحبا،",
    "لو سمحتم،",
    "عفوا،",
    "أرجو المساعدة،",
    "بعد إذنكم،",
]

CLOSERS = [
    "",
    "",
    "",
    "",
    "وشكرا لكم.",
    "أرجو الرد بأسرع وقت.",
    "بانتظار ردكم.",
    "جزاكم الله خير.",
    "ولكم جزيل الشكر.",
    "أتمنى تفيدوني.",
    "شاكر لكم.",
]


def order_no(rng: random.Random) -> str:
    return f"ORD-{rng.randint(10000, 99999)}"


def days(rng: random.Random) -> int:
    return rng.randint(2, 21)


def money(rng: random.Random) -> int:
    return rng.choice([49, 99, 129, 199, 249, 349, 499, 799, 1200, 1999])


# ---------------------------------------------------------------------------
# Core phrase pools. One per intent. These carry ALL the discriminative
# vocabulary, so each pool is written to cover the same idea several different
# ways rather than the same way several times.
# ---------------------------------------------------------------------------


def complaint_cores(rng: random.Random) -> list[str]:
    p, o, d = rng.choice(PRODUCTS), order_no(rng), days(rng)
    return [
        f"وصلني {p} تالف ومكسور من الصندوق",
        f"{p} اللي استلمته مكسور تماما ومب شغال",
        f"{p} خربان ووقف عن الشغل بعد {d} يوم بس",
        f"الخدمة سيئة جدا وانتظرت {d} يوم بدون اي رد",
        "خدمتكم سيئة وتعاملكم غير مقبول ابدا",
        f"تجربتي معكم سيئة، الطلب {o} وصل ناقص",
        f"استلمت الطلب {o} وفيه نقص، طلبت قطعتين ووصلتني وحدة",
        f"{p} مختلف تماما عن الصورة المعروضة، هذا غش",
        f"الطلب {o} وصل بعبوة مفتوحة ومكسورة من الخارج",
        f"اتصلت اكثر من مرة بخصوص {p} التالف ولا احد يرد علي",
        f"{p} فيه عيب مصنعي واضح وما يستاهل المبلغ اللي دفعته",
        "انا زعلان ومستاء من مستوى الخدمة عندكم",
        f"{p} طلع مستعمل مو جديد، هذا شي غير مقبول",
        f"للأسف {p} جودته رديئة جدا ومخيبة للامل",
        f"عندي شكوى على الطلب {o}، وصل متأخر وتالف",
        f"دفعت {money(rng)} ريال وحصلت على {p} خربان",
        "ما توقعت منكم هالمستوى، تجربة محبطة",
        f"{p} صوته مزعج وحرارته عالية، اشك انه معيب",
        f"تأخر توصيل الطلب {o} اكثر من {d} يوم بدون اي تحديث",
        f"{p} ما ضبط معي نهائيا وضيعت وقتي وفلوسي",
    ]


def price_inquiry_cores(rng: random.Random) -> list[str]:
    p, amt, c = rng.choice(PRODUCTS), money(rng), rng.choice(CITIES)
    return [
        f"كم سعر {p} الحين؟",
        f"ابي اعرف سعر {p} مع الشحن لـ{c}",
        f"وش تكلفة {p} شامل الضريبة؟",
        f"هل فيه خصم على {p} لو اخذت قطعتين؟",
        f"كم الخصم على {p} في العروض الحالية؟",
        f"عندكم تقسيط على {p}؟ وكم القسط الشهري؟",
        "هل يوجد تقسيط وكم تكون الدفعة الاولى؟",
        f"وش الفرق بالسعر بين {p} العادي والاصدار الجديد؟",
        "هل السعر المعروض شامل الضريبة والشحن؟",
        f"شفت {p} بـ{amt} ريال عند غيركم، تنافسون السعر؟",
        f"كم يكلفني {p} مع التوصيل لـ{c}؟",
        f"هل سعر {p} ثابت ولا ينزل بالعروض الموسمية؟",
        f"وش ارخص خيار عندكم يقارب {p} بالمواصفات؟",
        f"لو اخذت {p} مع الضمان الممتد كم يصير الاجمالي؟",
        f"كم تكلفة الشحن لـ{c} لو طلبت {p}؟",
        f"هل الاسعار تختلف بين {c} و{rng.choice(CITIES)}؟",
        f"ميزانيتي {amt} ريال، وش تنصحوني فيه؟",
        f"كم فرق السعر بين المقاس الصغير والكبير من {p}؟",
        "هل فيه كوبون خصم استخدمه عند الدفع؟",
        f"ابغى قائمة اسعار {p} بكل الالوان",
    ]


def support_request_cores(rng: random.Random) -> list[str]:
    p, n = rng.choice(PRODUCTS), rng.randint(2, 9)
    return [
        f"كيف اربط {p} بالواي فاي؟ جربت {n} مرات وما ضبط",
        f"ما عرفت اربط {p} بالبلوتوث، وش الخطوة الاولى؟",
        f"{p} يعطيني رسالة خطأ غريبة، وش الحل؟",
        "التطبيق يطلع لي رسالة خطأ كل مرة احاول اسجل دخول",
        f"وين القى دليل الاستخدام لـ{p}؟",
        f"هل {p} يحتاج تحديث برنامج؟ وكيف اسويه؟",
        f"ما اعرف اشغل خاصية التحكم الصوتي في {p}، ممكن مساعدة؟",
        f"البطارية في {p} تفرغ بعد {n} ساعات، فيه اعداد اضبطه؟",
        f"كيف اسوي ريست لـ{p} بدون ما افقد اعداداتي؟",
        f"وش طريقة تنظيف {p} بدون ما اخرب اي جزء؟",
        f"هل اقدر استخدم {p} مع اكثر من جهاز بنفس الوقت؟",
        f"محتاج مساعدة في اعداد {p} لاول مرة",
        f"كيف اغير اللغة في {p}؟",
        f"وش خطوات تركيب {p} بشكل صحيح؟",
        "نسيت كلمة المرور، كيف استعيد حسابي؟",
        f"كيف اوصل {p} بالجوال؟ ما القى الخيار",
        f"فيه طريقة اسرع لشحن {p}؟",
        f"{p} ما يشتغل الا لما اعيد تشغيله، كيف احل المشكلة؟",
        "ممكن ترشدوني خطوة بخطوة للاعداد؟",
        f"وش مواصفات الشاحن المناسب لـ{p}؟",
    ]


def praise_cores(rng: random.Random) -> list[str]:
    p, c, d = rng.choice(PRODUCTS), rng.choice(CITIES), days(rng)
    return [
        f"{p} فوق التوقعات، جودة ممتازة وسعر معقول",
        f"والله خدمتكم ممتازة و{p} نظيف وبجودة عالية، تسلمون",
        f"التوصيل لـ{c} كان اسرع من المتوقع والتغليف احترافي",
        "شكرا لكم على سرعة التوصيل، تعاملكم راقي",
        "خدمة العملاء ساعدوني بسرعة، تجربة رائعة",
        f"اشتريت {p} واستخدمته اكثر من {d} يوم، ما ندمت ابدا",
        f"موقعكم سهل جدا والدفع بدون اي مشاكل، وصلني {p} بسرعة",
        f"{p} بالضبط زي الوصف، ما فيه اي مبالغة",
        f"جربت {p} وانصح فيه بقوة، يستاهل كل ريال",
        f"وصلني الطلب {order_no(rng)} قبل الموعد وبحالة ممتازة",
        f"{p} صار من افضل مشترياتي هالسنة",
        f"تعاملكم راقي والتغليف ممتاز، {p} وصل سليم لـ{c}",
        "ابغى اشكركم على المجهود الرائع، استمروا",
        "تجربة شراء ممتازة من البداية للنهاية، شكرا",
        f"{p} جودته عالية والسعر مناسب، راضي تماما",
        "فريقكم محترف وسريع في الرد، شكرا لكم",
        f"احسن متجر تعاملت معه، و{p} روعة",
        "خدمة ما شاء الله عليها، ما قصرتم ابدا",
        f"{p} يستاهل التجربة، جودة ممتازة وتوصيل سريع",
        "ممتن لكم على حسن التعامل والاهتمام",
    ]


def order_status_cores(rng: random.Random) -> list[str]:
    o, d, c = order_no(rng), days(rng), rng.choice(CITIES)
    return [
        f"وش حالة طلبي {o}؟ من {d} يوم ما تغير الوضع",
        f"وين شحنتي؟ رقم الطلب {o} وما في اي تحديث",
        f"طلبت من {d} يوم ولسه مكتوب قيد التجهيز، الطلب {o}",
        f"هل طلبي {o} خرج للشحن؟",
        f"متى يوصلني الطلب {o}؟ محتاجه قبل نهاية الاسبوع",
        f"ابي اتابع شحنة الطلب {o}، ما توصلني رسائل تحديث",
        f"لسه ما وصلني الطلب مع انه مكتوب يوصل قبل {d} ايام",
        f"وين وصل طلبي؟ صار له {d} يوم بالطريق",
        f"ابغى رقم التتبع للطلب {o}",
        f"متى تصل شحنتي لـ{c}؟",
        f"الطلب {o} متوقف عند نفس الحالة من فترة",
        "هل تم شحن طلبي ولا لسه؟",
        f"ابي اعرف موعد التسليم المتوقع للطلب {o}",
        f"شركة الشحن ما تواصلت معي، الطلب {o}",
        f"طلبي متأخر {d} ايام عن الموعد، وين وصل؟",
        f"ممكن تحديث عن حالة الشحنة رقم {o}؟",
        "هل الطلب خرج من المستودع؟",
        f"ابغى اعرف مكان شحنتي الحين، الطلب {o}",
        f"كم باقي على وصول الطلب {o}؟",
        "ما وصلتني اي رسالة عن الشحن للحين",
    ]


def return_refund_cores(rng: random.Random) -> list[str]:
    p, o, d = rng.choice(PRODUCTS), order_no(rng), days(rng)
    return [
        f"ابي ارجع {p} لانه ما يناسبني، وش خطوات الاسترجاع؟",
        "ابي ابدل المقاس لان اللي وصلني صغير علي",
        f"اريد استرداد المبلغ كامل لان {p} ما ناسبني",
        f"طلبت استرجاع الطلب {o} من {d} يوم وما نزل المبلغ",
        f"كيف استبدل {p} بمقاس ثاني؟",
        f"هل اقدر ارجع {p} بعد ما فتحت العلبة؟",
        f"متى يرجع لي المبلغ بعد الموافقة على استرجاع {o}؟",
        f"ابغى الغي الطلب {o} واسترد فلوسي",
        "وش سياسة الاسترجاع عندكم خلال كم يوم؟",
        f"ابي استبدل {p} بمنتج ثاني، ممكن؟",
        f"المبلغ ما رجع لحسابي بعد ارجاع {p}",
        f"كيف اطلب استرجاع للطلب {o}؟",
        "ابغى استرداد نقدي مو رصيد في المحفظة",
        f"{p} ما عجبني، ابي ارجعه واخذ فلوسي",
        "هل الاسترجاع مجاني ولا علي رسوم شحن؟",
        f"ابي الغي طلبي {o} قبل ما يشحن",
        "وصلني مقاس غلط، ابي استبدله بالمقاس الصحيح",
        "كم ياخذ وقت استرداد المبلغ للبطاقة؟",
        f"ابغى ارجع نص الطلب {o} واحتفظ بالباقي",
        f"طلب ارجاع: {p}، السبب انه ما طابق الوصف",
    ]


INTENTS: dict[str, object] = {
    "complaint": complaint_cores,
    "price_inquiry": price_inquiry_cores,
    "support_request": support_request_cores,
    "praise": praise_cores,
    "order_status": order_status_cores,
    "return_refund": return_refund_cores,
}

# Deterministic label order -- the contract every layer downstream agrees on.
LABELS: list[str] = sorted(INTENTS)

PER_LABEL = 500  # unique examples per intent before splitting


def compose(rng: random.Random, core: str) -> str:
    """Wrap a core phrase in label-free framing."""
    opener = rng.choice(OPENERS)
    closer = rng.choice(CLOSERS)
    return " ".join(part for part in (opener, core, closer) if part).strip()


def generate_examples(rng: random.Random) -> list[tuple[str, str]]:
    examples: set[tuple[str, str]] = set()
    shortfalls: dict[str, int] = {}

    for label, core_fn in INTENTS.items():
        attempts = 0
        count = 0
        # Generate until PER_LABEL unique pairs, with a bounded attempt budget
        # so a small pool cannot loop forever once combinations are exhausted.
        while count < PER_LABEL and attempts < PER_LABEL * 40:
            cores = core_fn(rng)  # type: ignore[operator]
            text = compose(rng, rng.choice(cores))
            attempts += 1
            if (text, label) not in examples:
                examples.add((text, label))
                count += 1
        if count < PER_LABEL:
            shortfalls[label] = count

    if shortfalls:
        # Silence here would be the real bug: a label quietly producing a
        # fraction of the requested examples skews the class balance, and a
        # headline accuracy over an imbalanced split is misleading.
        print("\n[WARNING] label(s) exhausted their unique combinations:")
        for label, count in sorted(shortfalls.items()):
            print(f"  {label}: {count}/{PER_LABEL} ({count / PER_LABEL:.0%})")
        print("  -> add core phrases or slot values in train/generate_dataset.py\n")

    return sorted(examples)  # sorted for a fully deterministic file order


def split(
    examples: list[tuple[str, str]], rng: random.Random
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    by_label: dict[str, list[tuple[str, str]]] = {label: [] for label in LABELS}
    for text, label in examples:
        by_label[label].append((text, label))

    train, val, test = [], [], []
    for label in LABELS:
        rows = by_label[label]
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train += rows[:n_train]
        val += rows[n_train : n_train + n_val]
        test += rows[n_train + n_val :]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def main() -> None:
    rng = random.Random(SEED)
    examples = generate_examples(rng)
    train, val, test = split(examples, rng)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DATA_DIR / "train.csv", train)
    write_csv(DATA_DIR / "val.csv", val)
    write_csv(DATA_DIR / "test.csv", test)

    print(f"labels: {LABELS}")
    print(f"total examples: {len(examples)}")
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
