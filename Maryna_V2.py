#!/usr/bin/env python3

"""Maryna V2: zajednicki opis razmaka i Fourierovog prostora."""

import argparse
import csv
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

GROUPS = {
    "joint": np.arange(97),
    "gaps": np.arange(39),
    "fourier": np.arange(39, 65),
}

LABELS = {
    "joint": "MARYNA V2 ZAJEDNICKI OPIS",
    "gaps": "Kontrola: samo razmaci",
    "fourier": "Kontrola: samo Fourierov opis",
}


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

    return np.diff(edges, axis=1) - 1.


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


def features(g, reference, previous):
    delta = (
        np.atleast_2d(g) - reference
    ) / 5.

    memory = previous / 5.

    real_space = np.column_stack((
        delta,
        np.abs(delta),
        delta * delta,
        delta[:, :-1] * delta[:, 1:],
        delta * memory
    ))

    transformed = np.fft.fft(
        delta, axis=1, norm="ortho"
    )
    prior = np.fft.fft(
        memory, norm="ortho"
    )

    f = transformed[:, 1:5]
    magnitude = np.abs(f)
    alignment = f * prior[None, 1:5].conj()

    # Bispektralni clanovi cuvaju odnose faza.
    b1 = (
        transformed[:, 1] ** 2
        * transformed[:, 2].conj()
    )
    b2 = (
        transformed[:, 1]
        * transformed[:, 2]
        * transformed[:, 3].conj()
    )

    spectral = np.column_stack((
        f.real,
        f[:, :3].imag,
        magnitude * magnitude,
        magnitude,
        alignment.real,
        alignment[:, :3].imag,
        b1.real,
        b1.imag,
        b2.real,
        b2.imag
    ))

    coupling = (
        np.abs(delta)[:, :, None]
        * magnitude[:, None, :]
    ).reshape(len(delta), 32)

    # Direktan skalarni skor CELE kombinacije.
    # Fourierov opis se ne prosecuje i ne dekodira
    # u ciljne brojeve.
    return np.column_stack((
        real_space, spectral, coupling
    ))


def prepare(history):
    if len(history) < WINDOW + 2:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW + 2} prethodnih redova."
        )

    history = history[-(WINDOW + 2):]
    g = gaps(history)

    gram = np.zeros((97, 97))
    rhs = np.zeros(97)

    for t in range(2, len(history)):
        previous = g[t - 1] - g[t - 2]

        actual = features(
            g[t:t + 1],
            g[t - 1],
            previous
        )

        other = features(
            gaps(replacements(history[t])),
            g[t - 1],
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
        g[-1],
        g[-1] - g[-2],
        history[2:]
    )


def fit(prepared, alpha, group="joint"):
    gram, rhs, scale = prepared[:3]
    chosen = GROUPS[group]

    system = (
        gram[np.ix_(chosen, chosen)]
        + alpha * np.eye(len(chosen))
    )

    coefficient = np.zeros(97)
    coefficient[chosen] = (
        np.linalg.solve(system, rhs[chosen])
        / scale[chosen]
    )

    return (
        coefficient,
        prepared[3],
        prepared[4]
    )


def scores(rows, model):
    coefficient, reference, previous = model

    result = features(
        gaps(rows), reference, previous
    ) @ coefficient

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor zajednickog opisa."
        )

    return result


def search(history, model):
    seeds = np.unique(history, axis=0)

    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )
    current = seeds[order[:STARTS]]

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
                iterations
            )

        blocks = options.reshape(
            len(current), 225, 7
        )

        current[better] = blocks[
            np.arange(len(current))[better],
            chosen[better]
        ]


def predict(prepared, alpha, group="joint"):
    return search(
        prepared[5],
        fit(prepared, alpha, group)
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

    validation = {
        group: [[] for _ in ALPHAS]
        for group in GROUPS
    }

    for t in range(begin, split):
        p = prepare(rows[:t])

        for group in GROUPS:
            for j, alpha in enumerate(ALPHAS):
                combo = predict(
                    p, alpha, group
                )[0]

                validation[group][j].append(
                    hits(combo, rows[t])
                )

        if (t - begin + 1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t - begin + 1}/{validation_count}",
                flush=True
            )

    selected = {}

    for group in GROUPS:
        means = [
            np.mean(v)
            for v in validation[group]
        ]
        best = int(np.argmax(means))
        selected[group] = ALPHAS[best]

        print(
            f"{LABELS[group]}: validacija "
            f"alpha={selected[group]:g}; "
            f"prosek={means[best]:.4f}",
            flush=True
        )

    results = {
        group: [] for group in GROUPS
    }
    previous_scores = []

    # Svaki model ima sopstvenu alpha,
    # izabranu pre testa i zatim zamrznutu.
    # Svaka predikcija koristi samo prethodne redove.
    for t in range(split, len(rows)):
        p = prepare(rows[:t])

        for group in GROUPS:
            combo = predict(
                p, selected[group], group
            )[0]

            results[group].append(
                hits(combo, rows[t])
            )

        previous_scores.append(
            hits(rows[t - 1], rows[t])
        )

        if (t - split + 1) % 16 == 0:
            print(
                f"Test: {t - split + 1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split + 1}..{len(rows)}")

    for group in GROUPS:
        summarize(
            LABELS[group], results[group]
        )

    summarize(
        "Kontrola: prethodno izvuceno kolo",
        previous_scores
    )

    p = prepare(rows)
    combo, score, iterations = predict(
        p, selected["joint"], "joint"
    )

    repeated = any(
        np.array_equal(combo, row)
        for row in rows
    )
    _, radius = support(p[5])

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
        "Empirijska podrska: "
        f"L1 poluprecnik={radius:.6f}; kvantil=.75"
    )
    print(
        "Proveren lokalni maksimum dozvoljenih zamena; "
        "globalni maksimum nije potvrden."
    )
    print(
        "Dualni opis je adaptacija ideje; "
        "nije Viazovskina interpolaciona formula."
    )


def self_test():
    base = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39]
    ])

    g = gaps(base)
    f = np.fft.fft(
        g, axis=1, norm="ortho"
    )

    assert np.allclose(
        np.fft.ifft(
            f, axis=1, norm="ortho"
        ).real,
        g
    )

    assert np.allclose(
        np.sum(g * g, axis=1),
        np.sum(np.abs(f) ** 2, axis=1)
    )

    rotated = np.fft.fft(
        np.roll(g, 1, axis=1),
        axis=1,
        norm="ortho"
    )
    phase = np.exp(
        -2j * np.pi * np.arange(8) / 8
    )

    assert np.allclose(
        rotated, f * phase
    )
    assert np.allclose(
        np.cumsum(g[:, :-1] + 1, axis=1),
        base
    )
    assert features(
        g, g[0], g[1] - g[0]
    ).shape == (4, 97)

    history = np.tile(base, (30, 1))
    p = prepare(history)

    for group in GROUPS:
        model = fit(p, .1, group)
        coef = model[0] * p[2]
        ids = GROUPS[group]

        assert np.allclose(
            (
                p[0][np.ix_(ids, ids)]
                + .1 * np.eye(len(ids))
            ) @ coef[ids],
            p[1][ids],
            atol=1e-8
        )

        a = predict(p, .1, group)

        assert a == predict(
            prepare(history), .1, group
        )

        tree, radius = support(p[5])
        r = replacements(a[0])
        allowed = supported(r, tree, radius)

        assert (
            scores(r[allowed], model).max(
                initial=-np.inf
            )
            <= a[1] + 1e-9
        )

    changed = history.copy()
    changed[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), .1)
        == predict(prepare(changed[:-1]), .1)
    )

    print(
        "Self-test OK: Fourierova inverzija, norma, "
        "faze, ucenje i hronologija."
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
MARYNA V2 ZAJEDNICKI OPIS: validacija alpha=1; prosek=1.3542
Kontrola: samo razmaci: validacija alpha=0.1; prosek=1.3958
Kontrola: samo Fourierov opis: validacija alpha=0.1; prosek=1.3542
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
MARYNA V2 ZAJEDNICKI OPIS: prosek=1.1719; pogodaka 0..7=[14, 29, 18, 2, 1, 0, 0, 0]
Kontrola: samo razmaci: prosek=0.9219; pogodaka 0..7=[23, 25, 14, 2, 0, 0, 0, 0]
Kontrola: samo Fourierov opis: prosek=1.1250; pogodaka 0..7=[14, 32, 15, 2, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 26,x,28,y,30,z,39
Skor=1.0291947245; iteracija=17; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Dualni opis je adaptacija ideje; nije Viazovskina interpolaciona formula.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
MARYNA V2 ZAJEDNICKI OPIS: validacija alpha=1; prosek=1.4167
Kontrola: samo razmaci: validacija alpha=1; prosek=1.2292
Kontrola: samo Fourierov opis: validacija alpha=1; prosek=1.1458
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
MARYNA V2 ZAJEDNICKI OPIS: prosek=1.2656; pogodaka 0..7=[13, 28, 16, 7, 0, 0, 0, 0]
Kontrola: samo razmaci: prosek=1.2188; pogodaka 0..7=[16, 25, 16, 7, 0, 0, 0, 0]
Kontrola: samo Fourierov opis: prosek=1.2656; pogodaka 0..7=[15, 23, 20, 6, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,x,4,y,10,z,39
Skor=0.9967613358; iteracija=21; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Dualni opis je adaptacija ideje; nije Viazovskina interpolaciona formula.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
MARYNA V2 ZAJEDNICKI OPIS: validacija alpha=0.1; prosek=1.3333
Kontrola: samo razmaci: validacija alpha=1; prosek=1.2292
Kontrola: samo Fourierov opis: validacija alpha=1; prosek=1.4167
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
MARYNA V2 ZAJEDNICKI OPIS: prosek=1.2500; pogodaka 0..7=[12, 28, 21, 2, 1, 0, 0, 0]
Kontrola: samo razmaci: prosek=1.3125; pogodaka 0..7=[10, 27, 24, 3, 0, 0, 0, 0]
Kontrola: samo Fourierov opis: prosek=1.3594; pogodaka 0..7=[13, 22, 23, 5, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 21,x,23,y,25,z,37
Skor=1.8539827616; iteracija=20; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=3.200000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Dualni opis je adaptacija ideje; nije Viazovskina interpolaciona formula.
"""



"""
Maryna Viazovska: The Woman Who Found Perfection in Eight Dimensions

Maryna_V2.py za Druga ideja: zajednički opis u prostoru razmaka i Fourierovom prostoru.

Druga ideja: zajednički opis u prostoru razmaka i Fourierovom prostoru. 
U kasnijem zajedničkom radu Viazovska i saradnici koriste interpolaciju funkcije i njene Fourierove transformacije za dokaz optimalnosti energije. Rad o univerzalnoj optimalnosti
Istovremeno ocenjivati lokalne razmake i strukturu cele kombinacije, pa učiti njihov zajednički uticaj na sledeći prelaz. Predlog bi se birao direktnim rangiranjem celih kombinacija, bez pretvaranja prosečnih koordinata u sedam brojeva.
Razradio sam geometrijsku pretragu. Ona ima najjasniju konkretnu korist: proveriti da li drugačiji raspored polazišta pronalazi bolje ocenjene kombinacije od dosadašnje pretrage sa istorijskim polazištima.

Zajedno ocenjuje razmake i njihov Fourierov opis. 
V2 koristi amplitude, faze i njihove sprege u Fourierovom opisu, zajedno sa lokalnim razmacima. Prošle su provere Fourierove inverzije, očuvanja norme, faza i hronologije. Testiran zajednički model sa modelima koji koriste samo razmake ili samo Fourierov opis.

CSV	        Sledeća kombinacija	    V2 — prosek na 64 kola	Samo razmaci	Samo Fourier	Kontrola: prethodno izvučeno kolo
Objedinjeni	26,x,28,y,30,z,39	    1,1719	                0,9219	        1,1250	        1,2188
Loto	    1,x,4,y,10,z,39	        1,2656	                1,2188	        1,2656	        1,2969
Loto Plus	21,x,23,y,25,z,37	    1,2500	                1,3125	        1,3594	        1,2969

Zajednički opis nije pokazao doslednu prednost. Provereni su Fourierova inverzija, očuvanje norme, faze, hronologija, ponovljivost i lokalni maksimum dozvoljenih zamena.
"""
