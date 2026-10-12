#!/usr/bin/env python3

"""Maryna V1: E8 organizacija deterministicke pretrage celih kombinacija."""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULTS = (
    "/data/loto7_4698_k80.csv",
    "/data/loto7_4698_k80_loto_2971.csv",
    "/data/loto7_4698_k80_loto_plus_1727.csv",
)

WINDOW, STARTS = 96, 12
ALPHAS = (.01, .1, 1.)

j = np.arange(8)
MODES = np.sqrt(2 / 8) * np.cos(
    np.pi * np.arange(1, 8)[:, None]
    * (j[None, :] + .5) / 8
)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(numbers) != 7
                or numbers != sorted(numbers)
                or len(set(numbers)) != 7
                or not all(1 <= n <= 39 for n in numbers)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "kombinacija 7/39."
                )

            rows.append(numbers)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    # Osam praznih razmaka, zbir je tacno 32.
    return np.diff(edges, axis=1) - 1.


def states(rows):
    return gaps(rows) @ MODES.T / 5.


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def support(history):
    layout = gaps(history) / 5.
    tree = cKDTree(layout)

    nearest = tree.query(
        layout, k=2, p=1
    )[0][:, 1]

    return tree, max(
        float(np.quantile(nearest, .75)),
        1e-12
    )


def supported(rows, tree, radius):
    distance = tree.query(
        gaps(rows) / 5., k=1, p=1
    )[0]

    return distance <= radius + 1e-10


def nearest_d8(points):
    rounded = np.rint(points).astype(np.int64)
    residual = points - rounded

    odd = rounded.sum(axis=1) % 2 != 0
    indices = np.flatnonzero(odd)

    if len(indices):
        coordinate = np.argmax(
            np.abs(residual[indices]), axis=1
        )
        direction = np.where(
            residual[indices, coordinate] > 0,
            1,
            -1
        )
        rounded[indices, coordinate] += direction

    return rounded.astype(float)


def nearest_e8(points):
    points = np.atleast_2d(
        np.asarray(points, dtype=float)
    )

    integer = nearest_d8(points)
    half = nearest_d8(points - .5) + .5

    di = np.sum((points - integer) ** 2, axis=1)
    dh = np.sum((points - half) ** 2, axis=1)

    return np.where(
        (dh < di)[:, None],
        half,
        integer
    )


def e8_codes(points):
    # Celobrojni zapis udvostrucenih koordinata centra.
    return np.rint(
        2 * nearest_e8(points)
    ).astype(np.int64)


def features(values, reference, previous):
    delta = np.atleast_2d(values) - reference

    memory = (
        delta[:, :, None]
        * previous[None, None, :]
    ).reshape(len(delta), 49)

    # Direktan skalarni skor cele kombinacije.
    # Nema izlaznog proseka polozaja ili razmaka.
    return np.column_stack((
        delta,
        delta * delta,
        memory
    ))


def prepare(history):
    if len(history) < WINDOW + 2:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW + 2} prethodnih redova."
        )

    history = history[-(WINDOW + 2):]
    values = states(history)
    changes = np.diff(values, axis=0)

    unit = np.maximum(
        np.sqrt(np.mean(changes * changes, axis=0)),
        1e-8
    )

    gram = np.zeros((63, 63))
    rhs = np.zeros(63)

    for t in range(2, len(history)):
        previous = changes[t - 2] / unit

        actual = features(
            values[t:t + 1] / unit,
            values[t - 1] / unit,
            previous
        )

        other = features(
            states(replacements(history[t])) / unit,
            values[t - 1] / unit,
            previous
        )

        contrast = actual - other

        gram += (
            contrast.T @ contrast / len(contrast)
        )
        rhs += contrast.sum(axis=0) / len(contrast)

    gram /= WINDOW
    rhs /= WINDOW

    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )

    return (
        gram / scale[:, None] / scale[None, :],
        rhs / scale,
        scale,
        values[-1] / unit,
        changes[-1] / unit,
        unit,
        history[2:]
    )


def fit(prepared, alpha):
    gram, rhs, scale = prepared[:3]

    coef = np.linalg.solve(
        gram + alpha * np.eye(63),
        rhs
    ) / scale

    return (
        coef,
        prepared[3],
        prepared[4],
        prepared[5]
    )


def scores(rows, model):
    coef, reference, previous, unit = model

    result = features(
        states(rows) / unit,
        reference,
        previous
    ) @ coef

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor cele kombinacije."
        )

    return result


def embedding(rows, model):
    _, reference, _, unit = model
    delta = states(rows) / unit - reference

    # Sedam nezavisnih koordinata i izvedena aktivnost.
    # Ovo je 7D skup u 8D prostoru.
    # Osma koordinata nije nova nezavisna Loto dimenzija.
    activity = (
        np.linalg.norm(delta, axis=1) / np.sqrt(7.)
    )

    return np.column_stack((delta, activity))


def select_starts(history, model, use_e8=True):
    seeds = np.unique(history, axis=0)
    values = scores(seeds, model)

    ranked = np.argsort(
        -values, kind="stable"
    )

    if not use_e8:
        return (
            seeds[ranked[:STARTS]],
            len(seeds),
            0.
        )

    points = embedding(seeds, model)

    if len(points) > 1:
        d = np.linalg.norm(
            points[:, None, :] - points[None, :, :],
            axis=2
        )

        positive = d[
            np.triu_indices(len(points), 1)
        ]
        positive = positive[positive > 1e-12]

        size = (
            .5 * float(np.median(positive))
            if len(positive) else 1.
        )
    else:
        size = 1.

    size = max(size, 1e-12)
    codes = e8_codes(points / size)

    selected = []
    seen = set()

    for index in ranked:
        cell = tuple(map(int, codes[index]))

        if cell not in seen:
            seen.add(cell)
            selected.append(int(index))

    # Stvarni istorijski predstavnici razlicitih oblasti.
    # E8 centri se ne dekodiraju u Loto brojeve.
    return (
        seeds[selected[:STARTS]],
        len(seen),
        size
    )


def search(history, model, use_e8=True):
    current, cells, cell_size = select_starts(
        history, model, use_e8
    )
    current = current.copy()

    tree, radius = support(history)
    iterations = 0

    while True:
        iterations += 1

        options = np.vstack([
            np.vstack((r, replacements(r)))
            for r in current
        ])

        raw = scores(options, model)
        raw[~supported(options, tree, radius)] = -np.inf

        values = raw.reshape(len(current), 225)
        chosen = np.argmax(values, axis=1)

        better = (
            values[np.arange(len(current)), chosen]
            > values[:, 0] + 1e-10
        )

        if not np.any(better):
            best = int(np.argmax(values[:, 0]))

            return (
                tuple(map(int, current[best])),
                float(values[best, 0]),
                iterations,
                cells,
                cell_size
            )

        blocks = options.reshape(
            len(current), 225, 7
        )

        current[better] = blocks[
            np.arange(len(current))[better],
            chosen[better]
        ]


def predict(prepared, alpha, use_e8=True):
    return search(
        prepared[6],
        fit(prepared, alpha),
        use_e8
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, values):
    print(
        f"{label}: prosek={np.mean(values):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(values, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows) - test_count
    begin = split - validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW + 2
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    validation = [[] for _ in ALPHAS]

    for t in range(begin, split):
        p = prepare(rows[:t])

        for j, alpha in enumerate(ALPHAS):
            combo = predict(p, alpha)[0]
            validation[j].append(
                hits(combo, rows[t])
            )

        if (t - begin + 1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t - begin + 1}/{validation_count}",
                flush=True
            )

    means = [np.mean(v) for v in validation]
    alpha = ALPHAS[int(np.argmax(means))]

    print(
        f"Validacija: alpha={alpha:g}; "
        f"prosek={max(means):.4f}",
        flush=True
    )

    actual, previous, plain = [], [], []
    different = 0

    # Alpha ostaje fiksna tokom testa.
    # Ucenje, oblasti i podrska koriste samo prethodne redove.
    for t in range(split, len(rows)):
        p = prepare(rows[:t])
        model = fit(p, alpha)

        a = search(p[6], model, True)[0]
        b = search(p[6], model, False)[0]

        actual.append(hits(a, rows[t]))
        plain.append(hits(b, rows[t]))
        previous.append(hits(rows[t - 1], rows[t]))
        different += a != b

        if (t - split + 1) % 16 == 0:
            print(
                f"Test: {t - split + 1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split + 1}..{len(rows)}")

    summarize("MARYNA V1 E8 PRETRAGA", actual)
    summarize(
        "Isti nauceni skor bez E8 organizacije",
        plain
    )
    summarize(
        "Kontrola: prethodno izvuceno kolo",
        previous
    )

    print(
        f"E8 promenio predlog u "
        f"{different}/{test_count} test kola."
    )

    p = prepare(rows)
    model = fit(p, alpha)

    combo, score, iterations, cells, size = search(
        p[6], model, True
    )

    repeated = any(
        np.array_equal(combo, row)
        for row in rows
    )
    _, radius = support(p[6])

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows) + 1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; "
        f"iteracija={iterations}; "
        f"vec postoji u istoriji={repeated}"
    )
    print(
        f"Zauzetih E8 oblasti={cells}; "
        f"velicina celije={size:.6f}; "
        f"empirijski L1 poluprecnik={radius:.6f}"
    )
    print(
        "Proveren lokalni maksimum dozvoljenih zamena; "
        "globalni maksimum nije potvrden."
    )
    print(
        "E8 organizuje polazista. Njegova teorema "
        "optimalnog pakovanja ne prenosi se na Loto."
    )


def self_test():
    points = np.array([
        [0.] * 8,
        [.5] * 8,
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [.2, -.7, 1.1, 2.3, -1.4, .8, -.2, .6]
    ])

    centers = nearest_e8(points)
    codes = np.rint(2 * centers).astype(int)

    assert np.all(
        codes.sum(axis=1) % 4 == 0
    )
    assert all(
        np.all(row % 2 == row[0] % 2)
        for row in codes
    )

    # Iscrpna provera lokalnih suseda obe D8 grane.
    offsets = np.asarray(list(
        itertools.product((-1, 0, 1), repeat=8)
    ))

    for point, center in zip(points, centers):
        best = float(
            np.sum((point - center) ** 2)
        )

        for shift in (0., .5):
            candidates = (
                np.rint(point - shift).astype(int)
                + offsets
            )
            candidates = candidates[
                candidates.sum(axis=1) % 2 == 0
            ] + shift

            assert (
                np.min(
                    np.sum(
                        (candidates - point) ** 2,
                        axis=1
                    )
                )
                >= best - 1e-12
            )

    assert np.array_equal(
        nearest_e8(centers), centers
    )

    base = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39]
    ])

    assert np.all(
        gaps(base).sum(axis=1) == 32
    )

    restored = (
        gaps(base[0])
        + 5 * (states(base) - states(base[0])) @ MODES
    )

    assert np.allclose(
        np.cumsum(
            restored[:, :-1] + 1, axis=1
        ),
        base
    )

    history = np.tile(base, (30, 1))
    p = prepare(history)
    model = fit(p, .1)

    starts, _, size = select_starts(
        p[6], model, True
    )
    cells = e8_codes(
        embedding(starts, model) / size
    )

    assert (
        len(np.unique(cells, axis=0))
        == len(starts)
    )

    result = predict(p, .1)
    assert result == predict(
        prepare(history), .1
    )

    tree, radius = support(p[6])
    r = replacements(result[0])
    allowed = supported(r, tree, radius)

    assert (
        scores(r[allowed], model).max(
            initial=-np.inf
        )
        <= result[1] + 1e-9
    )

    changed = history.copy()
    changed[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), .1)
        == predict(prepare(changed[:-1]), .1)
    )

    print(
        "Self-test OK: najblizi E8 centri, dimenzije, "
        "oblasti, rekonstrukcija i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=1; prosek=1.5208
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
MARYNA V1 E8 PRETRAGA: prosek=0.9688; pogodaka 0..7=[17, 33, 13, 1, 0, 0, 0, 0]
Isti nauceni skor bez E8 organizacije: prosek=0.9531; pogodaka 0..7=[18, 33, 11, 2, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
E8 promenio predlog u 4/64 test kola.
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 1,x,3,y,5,z,13
Skor=1.1922556044; iteracija=19; vec postoji u istoriji=False
Zauzetih E8 oblasti=74; velicina celije=1.277239; empirijski L1 poluprecnik=2.800000
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
E8 organizuje polazista. Njegova teorema optimalnog pakovanja ne prenosi se na Loto.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=1; prosek=1.3125
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
MARYNA V1 E8 PRETRAGA: prosek=1.2344; pogodaka 0..7=[16, 23, 20, 4, 1, 0, 0, 0]
Isti nauceni skor bez E8 organizacije: prosek=1.2656; pogodaka 0..7=[16, 23, 18, 6, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
E8 promenio predlog u 3/64 test kola.
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,x,23,y,25,z,38
Skor=1.2002168312; iteracija=16; vec postoji u istoriji=False
Zauzetih E8 oblasti=80; velicina celije=1.290789; empirijski L1 poluprecnik=2.800000
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
E8 organizuje polazista. Njegova teorema optimalnog pakovanja ne prenosi se na Loto.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: alpha=0.1; prosek=1.3125
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
MARYNA V1 E8 PRETRAGA: prosek=1.1250; pogodaka 0..7=[14, 32, 14, 4, 0, 0, 0, 0]
Isti nauceni skor bez E8 organizacije: prosek=1.1719; pogodaka 0..7=[13, 31, 16, 4, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
E8 promenio predlog u 2/64 test kola.
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,x,13,y,15,z,37
Skor=1.9051163527; iteracija=7; vec postoji u istoriji=False
Zauzetih E8 oblasti=78; velicina celije=1.240989; empirijski L1 poluprecnik=3.200000
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
E8 organizuje polazista. Njegova teorema optimalnog pakovanja ne prenosi se na Loto.
"""



"""
Maryna Viazovska: The Woman Who Found Perfection in Eight Dimensions

Geometrija prostora celih kombinacija i konstrukciju funkcije koja ih rangira. 
To bi bile nove matematičke primene njenih rezultata.
Viazovska je dokazala optimalnost pakovanja kugli pomoću rešetke E₈ u osam dimenzija. 
Za Loto postoji zanimljiva veza: svaku sortiranu kombinaciju mogu tačno predstaviti pomoću osam praznih razmaka — pre prvog broja, između brojeva i posle poslednjeg. Njihov zbir je 32. Zato imam osam koordinata, ali sedam nezavisnih dimenzija. Taj uslov moram sačuvati; samo nazivanje modela „E₈” ne bi bilo dovoljno.

Prva ideja: geometrijski organizovana pretraga distribucije prelaza.
- Svaka cela kombinacija dobija svoje mesto prema razmacima.
- Geometrijski odvojena polazišta sprečavaju da pretraga stalno obrađuje gotovo iste konfiguracije.
- Skor se uči iz prethodnih stvarnih prelaza.
- Geometrija raspoređuje polazišta pretrage, a ne brojeve unutar predloga. Ne nameće jednake razmake.

Druga ideja: zajednički opis u prostoru razmaka i Fourierovom prostoru. 
U kasnijem zajedničkom radu Viazovska i saradnici koriste interpolaciju funkcije i njene Fourierove transformacije za dokaz optimalnosti energije. Rad o univerzalnoj optimalnosti
Istovremeno ocenjivati lokalne razmake i strukturu cele kombinacije, pa učiti njihov zajednički uticaj na sledeći prelaz. Predlog bi se birao direktnim rangiranjem celih kombinacija, bez pretvaranja prosečnih koordinata u sedam brojeva.
Razradio sam geometrijsku pretragu. Ona ima najjasniju konkretnu korist: proveriti da li drugačiji raspored polazišta pronalazi bolje ocenjene kombinacije od dosadašnje pretrage sa istorijskim polazištima.

Bez ikakvog random postupka, E₈ bira geometrijski odvojena polazišta pretrage; skor cele kombinacije uči se iz stvarnih prelaza. Proverio sam sva tri CSV fajla i uporedio pretragu sa istim skorom bez E₈ organizacije.

Prošle su provere najbližih E₈ centara i preslikavanja: osam praznih razmaka ima zbir 32, pa ostaje sedam nezavisnih koordinata. 
E₈ centri se ne pretvaraju u Loto brojeve — iz svake izabrane oblasti uzima se stvarno istorijsko polazište, a zatim pretražuju validne kombinacije.
Bez random postupka. E₈ organizuje polazišta pretrage; ne određuje raspodelu Loto brojeva.

CSV	        Sledeća kombinacija	E₈ — prosek na 64 kola	Isti skor bez E₈	Kontrola: prethodno izvučeno kolo
Objedinjeni	1,x,3,y,5,z,13	    0,9688	                0,9531	            1,2188
Loto	    1,x,23,y,25,z,38	1,2344	                1,2656	            1,2969
Loto Plus	1,x,13,y,15,z,37	1,1250	                1,1719	            1,2969

Ovaj test nije pokazao stabilnu korist E₈ organizacije. Proverio sam geometrijsko preslikavanje, najbliži E₈ centri, hronologija, ponovljivost i lokalni maksimum dozvoljenih zamena.
"""
 
