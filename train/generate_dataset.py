"""Deterministic synthetic dataset generator for the intent classifier.

Produces Arabic customer-message / intent-label pairs by combining sentence
templates with slot values (product names, order numbers, cities, amounts).
Everything is driven by a fixed seed, so re-running this script always
produces byte-identical output -- required for reproducible training and
for CI to re-verify the pipeline without committing a large data file.

Usage (from the project root):

    python -m train.generate_dataset

Writes:
    data/train.csv
    data/val.csv
    data/test.csv
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Vocabulary used to fill template slots. Kept separate from the templates so
# adding a product or a city does not require touching sentence structure.
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
]

CITIES = ["الرياض", "جدة", "الدمام", "مكة", "المدينة", "أبها", "تبوك", "حائل"]

NAMES = ["أحمد", "سارة", "محمد", "نورة", "فهد", "ريم", "خالد", "منى", "عبدالله", "لمياء"]


def order_number(rng: random.Random) -> str:
    return f"ORD-{rng.randint(10000, 99999)}"


def amount(rng: random.Random) -> int:
    return rng.choice([49, 99, 129, 199, 249, 349, 499, 799, 1200, 1999])


def days(rng: random.Random) -> int:
    return rng.randint(2, 21)


# ---------------------------------------------------------------------------
# One template group per intent. Each template is a callable so it can pull
# whatever slots it needs from the shared rng.
# ---------------------------------------------------------------------------


def complaint_templates(rng: random.Random) -> str:
    product = rng.choice(PRODUCTS)
    order = order_number(rng)
    d = days(rng)
    templates = [
        f"وصلني {product} وهو تالف من الصندوق، الطلب {order}",
        f"{product} توقف عن العمل بعد {d} يوم بس من الاستخدام، هذا غير مقبول",
        f"استلمت طلبي {order} وفيه نقص، طلبت قطعتين ووصلتني وحدة بس",
        f"{product} اللي طلبته مختلف تماماً عن الصورة المعروضة بالموقع",
        f"تأخر توصيل الطلب {order} أكثر من {d} يوم بدون أي تحديث",
        f"صوت {product} صار مزعج جداً بعد فترة قصيرة، أشك إنه قطعة مستعملة",
        f"الطلب {order} وصل بعبوة مفتوحة ومكسورة من الخارج",
        f"اتصلت أكثر من مرة بخصوص {product} التالف ولا أحد يرد علي",
    ]
    return rng.choice(templates)


def price_inquiry_templates(rng: random.Random) -> str:
    product = rng.choice(PRODUCTS)
    amt = amount(rng)
    templates = [
        f"كم سعر {product} الحين؟ شفت عرض بس ما أعرف إذا لسه شغال",
        f"هل فيه خصم على {product} لو اشتريت قطعتين؟",
        f"وش الفرق بالسعر بين {product} العادي والإصدار الجديد؟",
        "هل السعر المعروض شامل الضريبة والشحن؟",
        f"عندكم تقسيط على {product}؟ وكم يكون القسط الشهري تقريباً؟",
        f"لو رجعت {product} خلال أسبوع هل أقدر أسترجع كامل المبلغ؟",
        f"هل الأسعار تختلف بين {rng.choice(CITIES)} و{rng.choice(CITIES)}؟",
        f"شفت {product} بـ{amt} ريال عند غيركم، تقدرون تنافسون السعر؟",
        f"كم يكلفني {product} مع التوصيل لـ{rng.choice(CITIES)}؟",
        f"هل سعر {product} ثابت ولا ينزل بالعروض الموسمية؟",
        f"وش أرخص خيار عندكم يقارب {product} بالمواصفات؟",
        f"لو أخذت {product} مع الضمان الممتد، كم يصير الإجمالي؟",
    ]
    return rng.choice(templates)


def support_request_templates(rng: random.Random) -> str:
    product = rng.choice(PRODUCTS)
    n = rng.randint(2, 9)
    templates = [
        f"كيف أربط {product} بالواي فاي؟ جربت {n} مرات ومب راضي يتوصل",
        f"{product} يعطيني رسالة خطأ غريبة، وش الحل؟",
        f"وين أقدر ألقى دليل الاستخدام لـ {product}؟",
        f"هل {product} يحتاج تحديث برنامج؟ وكيف أسويه؟",
        f"ما أعرف أشغل خاصية التحكم الصوتي في {product}، ممكن مساعدة؟",
        f"البطارية في {product} تفرغ بعد {n} ساعات بس، هل فيه إعداد أضبطه؟",
        f"كيف أسوي ريست لـ {product} بدون ما أفقد إعداداتي؟",
        f"حاولت أقرن {product} بجوالي {n} مرات وما ضبطت، وش الخطوات الصحيحة؟",
        f"{product} ما يشتغل إلا لما أعيد تشغيله، كيف أحل هالمشكلة؟",
        f"وش طريقة تنظيف {product} بدون ما أخرب أي جزء فيه؟",
        f"هل أقدر أستخدم {product} مع أكثر من جهاز بنفس الوقت؟",
    ]
    return rng.choice(templates)


def praise_templates(rng: random.Random) -> str:
    product = rng.choice(PRODUCTS)
    city = rng.choice(CITIES)
    order = order_number(rng)
    d = days(rng)
    templates = [
        f"{product} فوق التوقعات، جودة ممتازة وسعر معقول",
        f"التوصيل لـ{city} كان أسرع من المتوقع والتغليف احترافي، شكراً لكم",
        f"خدمة العملاء ساعدوني بسرعة لما تواصلت بخصوص {product}، تجربة رائعة",
        f"اشتريت {product} من عندكم واستخدمته أكثر من {d} يوم، ما ندمت أبداً",
        f"موقعكم سهل جداً والدفع كان بدون أي مشاكل، وصلني {product} بسرعة",
        f"{product} بالضبط زي الوصف، ما فيه أي مبالغة بالإعلان",
        f"جربت {product} وأنصح فيه بقوة، يستاهل كل ريال",
        f"وصلني الطلب {order} قبل الموعد وبحالة ممتازة، شكراً",
        f"{product} صار من أفضل مشترياتي هالسنة، جودته ثابتة بعد {d} يوم",
        f"تعاملكم راقي والتغليف ممتاز، {product} وصل سليم لـ{city}",
    ]
    return rng.choice(templates)


def order_status_templates(rng: random.Random) -> str:
    order = order_number(rng)
    d = days(rng)
    templates = [
        f"وش حالة طلبي {order}؟ من {d} يوم ما تغير الوضع",
        f"طلبت من {d} يوم ولسه مكتوب قيد التجهيز، الطلب {order}",
        f"هل طلبي {order} خرج للشحن؟",
        f"متى يوصلني الطلب {order}؟ محتاجه قبل نهاية الأسبوع",
        f"أبي أتابع شحنة الطلب {order}، ما توصلني رسائل تحديث",
    ]
    return rng.choice(templates)


def return_refund_templates(rng: random.Random) -> str:
    product = rng.choice(PRODUCTS)
    order = order_number(rng)
    d = days(rng)
    templates = [
        f"أبي أرجع {product} لأنه ما يناسبني، وش خطوات الاسترجاع؟",
        f"طلبت استرجاع الطلب {order} من {d} يوم وما نزل المبلغ لحسابي",
        f"كيف أستبدل {product} بمقاس ثاني؟",
        f"هل أقدر أرجع {product} بعد ما فتحت العلبة؟",
        f"متى يرجع لي المبلغ بعد ما وافقتوا على استرجاع الطلب {order}؟",
    ]
    return rng.choice(templates)


INTENTS: dict[str, object] = {
    "complaint": complaint_templates,
    "price_inquiry": price_inquiry_templates,
    "support_request": support_request_templates,
    "praise": praise_templates,
    "order_status": order_status_templates,
    "return_refund": return_refund_templates,
}

# Deterministic label order -- this is the contract every layer downstream
# (training, the API's enum, the frontend, if any) agrees on.
LABELS: list[str] = sorted(INTENTS)

PER_LABEL = 300  # total examples per intent before splitting


def generate_examples(rng: random.Random) -> list[tuple[str, str]]:
    examples: set[tuple[str, str]] = set()
    shortfalls: dict[str, int] = {}

    for label, template_fn in INTENTS.items():
        attempts = 0
        count = 0
        # Generate until we have PER_LABEL unique (text, label) pairs, with a
        # bounded number of attempts so a small template pool can't loop
        # forever once its combinations are exhausted.
        while count < PER_LABEL and attempts < PER_LABEL * 20:
            text = template_fn(rng)  # type: ignore[operator]
            attempts += 1
            if (text, label) not in examples:
                examples.add((text, label))
                count += 1
        if count < PER_LABEL:
            shortfalls[label] = count

    if shortfalls:
        # Silence here would be the real bug: a label quietly producing a
        # fraction of the requested examples skews the class balance, and a
        # headline accuracy computed over an imbalanced split is misleading.
        # Fix by adding templates or slot values for the named labels.
        print("\n[WARNING] label(s) exhausted their unique template combinations:")
        for label, count in sorted(shortfalls.items()):
            print(f"  {label}: {count}/{PER_LABEL} ({count / PER_LABEL:.0%})")
        print("  -> add templates or slot values in train/generate_dataset.py\n")

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
