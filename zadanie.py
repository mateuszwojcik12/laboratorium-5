#!/usr/bin/python3

import logging
import pickle
import re
import sqlite3

import mwparserfromhell as mwp
from sklearn.feature_extraction.text import TfidfVectorizer

# TU(10a): Wkleić wyrazy z https://pl.wikipedia.org/wiki/Wikipedia:Stopwords
# TU(11a): Dopisać wyrazy specyficzne dla haseł Wikipedii.
STOP_WORDS = """a, aby, ach, acz, aczkolwiek, aj, albo, ale, ależ, ani, aż, 
bardziej, bardzo, bo, bowiem, by, byli, bynajmniej, być, był, była, było, 
były, będzie, będą, cali, cała, cały, ci, cię, ciebie, co, cokolwiek, coś, czasami, czasem, czemu, czy, czyli, 
daleko, dla, dlaczego, dlatego, do, dobrze, dokąd, dość, dużo, dwa, dwaj, dwie, dwoje, dziś, dzisiaj, gdy, gdyby, gdyż, 
gdzie, gdziekolwiek, gdzieś, i, ich, ile, im, inna, inne, inny, innych, iż, ja, ją, jak, jaka, jakaś, jakby, jaki, jakichś, 
jakie, jakiś, jakiż, jakkolwiek, jako, jakoś, je, jeden, jedna, jedno, jednak, jednakże, jego, jej, jemu, jest, jestem, 
jeszcze, jeśli, jeżeli, już, ją, każdy, kiedy, kilka, kimś, kto, ktokolwiek, ktoś, która, które, którego, której, który, 
których, którym, którzy, ku, lat, lecz, lub, ma, mają, mało, mam, mi, mimo, między, mną, mnie, mogą, moi, moim, moja, moje, może, 
możliwe, można, mój, mu, musi, my, na, nad, nam, nami, nas, nasi, nasz, nasza, nasze, naszego, naszych, natomiast, natychmiast, nawet, 
nią, nic, nich, nie, niech, niego, niej, niemu, nigdy, nim, nimi, niż, no, o, obok, od, około, on, ona, one, oni, ono, oraz, oto, owszem, pan, pana, 
pani, po, pod, podczas, pomimo, ponad, ponieważ, powinien, powinna, powinni, powinno, poza, prawie, przecież, przed, przede, przedtem, przez, przy, roku, 
również, sama, są, się, skąd, sobie, sobą, sposób, swoje, ta, tak, taka, taki, takie, także, tam, te, tego, tej, temu, ten, teraz, 
też, to, tobą, tobie, toteż, trzeba, tu, tutaj, twoi, twoim, twoja, twoje, twym, twój, ty, tych, tylko, tym, u, w, wam, wami, was, wasz, 
wasza, wasze, we, według, wiele, wielu, więc, więcej, wszyscy, wszystkich, wszystkie, wszystkim, wszystko, wtedy, wy, właśnie, z, za, zapewne, 
zawsze, ze, zł, znowu, znów, został, żaden, żadna, żadne, żadnych, że, żeby
""".split(',')
STOP_WORDS = [w.strip() for w in STOP_WORDS if w.strip()]

# TU(11b): Nie likwidować liczb.
NONLETTERS_RE = re.compile(
    r'[0-9’“„”«»…–—!"#$%&\'()*+,\-./:;?@\[\\\]^_`{|}~<=>]')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info('Wczytywanie danych. To potrwa do dwóch minut.')
    connection = sqlite3.connect('artykuly.sqlite3')
    titles = []
    texts = []
    for row in connection.execute('SELECT title, text FROM Articles'):
        titles.append(row[0])
        # TU(11c): zmienić keep_template_params na True.
        text = mwp.parse(row[1]).strip_code(keep_template_params=False)
        text = NONLETTERS_RE.sub(' ', text)
        texts.append(text)
    logging.info('Tworzenie modelu. To potrwa do pół minuty.')
    # TU(10b): Zamienić na TfidfVectorizer.
    vectorizer = TfidfVectorizer(
        analyzer='word',
        min_df=3
    # TU(10a): Dopisać stop_words=STOP_WORDS.
    # TU(11d): Zmienić analyzer na 'char_wb',
    # dopisać ngram_range=(5, 5)
    # i max_features=40_000.
    )
    X = vectorizer.fit_transform(texts)
    logging.info(
            'Model gotowy. Korzysta z %d cech.',
            len(vectorizer.get_feature_names()))
    if type(vectorizer) is TfidfVectorizer:
        idfs = []
        for feature, idf in zip(
                vectorizer.get_feature_names(),
                vectorizer.idf_):
            idfs.append((idf, feature))
        logging.info(
            'Najczęstsze wyrazy: %s.',
            ', '.join(feature for idf, feature in sorted(idfs)[:20]))
    with open('baseline.pickle', 'wb') as file:
        pickle.dump(titles, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(X, file, pickle.HIGHEST_PROTOCOL)
    logging.info('Model zapisany.')


if __name__ == '__main__':
    main()
