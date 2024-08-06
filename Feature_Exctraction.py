import csv
import re
import urllib.parse
from collections import Counter
import furl as furl
import numpy as np
import pandas as pd
import enchant
import tldextract
from tld import get_tld
import sys
from random import choice
import string

# Read the original dataset
df = pd.read_csv(r'urls.csv')

# Initialize lists to store features
count = Counter()
count_a = []
count_b = []
count_c = []
count_d = []
count_e = []
count_f = []
count_g = []
count_h = []
count_i = []
count_j = []
count_k = []
count_l = []
count_m = []
count_n = []
count_o = []
count_p = []
count_q = []
count_r = []
count_s = []
count_t = []
count_u = []
count_v = []
count_w = []
count_x = []
count_y = []
count_z = []
count_protocol = []
count_www = []
url_protocol = []
count_letter = []
count_slash = []
count_dots = []
count_underscore = []
count_and = []
count_semicolon = []
count_at = []
count_dollar = []
count_percent = []
count_equal = []
count_0 = []
count_1 = []
count_2 = []
count_3 = []
count_4 = []
count_5 = []
count_6 = []
count_7 = []
count_8 = []
count_9 = []
Url = []
Type = []
Label = []
ip_dns = []
url_len = []
TLD = []
dot_url = []
first_dir_len = []
count_domain_dots = []
count_hyphen_Domain = []

count_http = []
count_https = []

lower_case_url = []
upper_case_url = []
English_words_count = []
avg_english_word_len = []

con = 0
i = 0

for con, (url, type_value, label_value) in enumerate(zip(df['URLs'], df['Type'], df['Label'])):
    con += 1
    count.update(url)
    i += 1
    Url.append(url)
    Type.append(type_value)
    Label.append(label_value)
    df_parsed = urllib.parse.urlparse(url)
    
    count_a.append(url.count('a')), count_b.append(url.count('b')), count_c.append(url.count('c'))
    count_d.append(url.count('d')), count_e.append(url.count('e')), count_f.append(url.count('f'))
    count_g.append(url.count('g')), count_h.append(url.count('h')), count_i.append(url.count('i'))
    count_j.append(url.count('j')), count_k.append(url.count('k')), count_l.append(url.count('l'))
    count_m.append(url.count('m')), count_n.append(url.count('n')), count_o.append(url.count('o'))
    count_p.append(url.count('p')), count_q.append(url.count('q')), count_r.append(url.count('r'))
    count_s.append(url.count('s')), count_t.append(url.count('t')), count_u.append(url.count('u'))
    count_v.append(url.count('v')), count_w.append(url.count('w')), count_x.append(url.count('x'))
    count_y.append(url.count('y')), count_z.append(url.count('z'))

    count_0.append(url.count('0')), count_1.append(url.count('1'))
    count_2.append(url.count('2')), count_3.append(url.count('3'))
    count_4.append(url.count('4')), count_5.append(url.count('5'))
    count_6.append(url.count('6')), count_7.append(url.count('7'))
    count_8.append(url.count('8')), count_9.append(url.count('9'))

    count_protocol.append(df_parsed.scheme.count('http'))
    count_www.append(url.count('www'))

    count_underscore.append(url.count('_'))
    count_at.append(url.count('@'))
    count_dollar.append(url.count('$'))
    count_percent.append(url.count('%'))
    count_equal.append(url.count('='))

    # LENGTH FEATURES
    url_len.append(len(url))
    count_domain_dots.append(df_parsed.netloc.count('.'))
    dot_url.append(df_parsed.netloc.count('.'))
    first_dir_len.append(len(df_parsed.path))

    bbb = df_parsed.netloc.split('.')
    TLD.append(len(bbb[-1]) if len(bbb) > 1 else 0)

    count_hyphen_Domain.append(df_parsed.netloc.count('-'))
    count_semicolon.append(url.count(';'))
    count_and.append(url.count('&'))
    count_http.append(url.count('http'))
    count_https.append(url.count('https'))

    digit = letter = lower_letter = upper_letter = 0
    for ch in url:
        if ch.isdigit():
            digit += 1
        elif ch.isalpha():
            letter += 1
            if ch.islower():
                lower_letter += 1
            elif ch.isupper():
                upper_letter += 1
    count_letter.append(letter)
    lower_case_url.append(lower_letter)
    upper_case_url.append(upper_letter)

    d = enchant.Dict("en_US")
    alphabet_regular_expression = re.compile(r"[^a-zA-Z]")
    string_without_non_alphabet = re.sub(alphabet_regular_expression, " ", url)
    en_words_lst = [word for word in filter(None, string_without_non_alphabet.split(' ')) if d.check(word)]

    English_words_count.append(len(en_words_lst))
    avg_english_word_len.append(sum(len(word) for word in en_words_lst) / len(en_words_lst) if en_words_lst else 0)

    my_hostname = df_parsed.netloc
    my_hostname_match = re.match(r"([0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|(\d{1,3}\.){3}\d{1,3}", my_hostname)
    ip_dns.append(1 if my_hostname_match else 0)

# Create a DataFrame with all the calculated features
df1 = pd.DataFrame({
    'URLs': Url,
    'Type': Type,
    'Label': Label,
    'ip_DNS': ip_dns,
    'url_len': url_len,
    'Dots Count': dot_url,
    'Dots_domain': count_domain_dots,
    'hyphen_domain': count_hyphen_Domain,
    'Count_&': count_and,
    'First Dir Len': first_dir_len,
    'TLD': TLD,
    'Count_;': count_semicolon,
    'Count_http': count_http,
    'alphabets_url': count_letter,
    'lower_case_letter': lower_case_url,
    'upper_case_letter': upper_case_url,
    'avg_english_words': avg_english_word_len,
    'Count_a': count_a,
    'Count_b': count_b,
    'Count_c': count_c,
    'Count_d': count_d,
    'Count_e': count_e,
    'Count_f': count_f,
    'Count_g': count_g,
    'Count_h': count_h,
    'Count_i': count_i,
    'Count_j': count_j,
    'Count_k': count_k,
    'Count_l': count_l,
    'Count_m': count_m,
    'Count_o': count_o,
    'Count_p': count_p,
    'Count_q': count_q,
    'Count_r': count_r,
    'Count_s': count_s,
    'Count_t': count_t,
    'Count_u': count_u,
    'Count_v': count_v,
    'Count_w': count_w,
    'Count_x': count_x,
    'Count_y': count_y,
    'Count_z': count_z,
    'Count_0': count_0,
    'Count_1': count_1,
    'Count_2': count_2,
    'Count_3': count_3,
    'Count_4': count_4,
    'Count_5': count_5,
    'Count_6': count_6,
    'Count_7': count_7,
    'Count_8': count_8,
    'Count_9': count_9,
    'Count_www': count_www,
    'Count_equal': count_equal,
    'Count_underscore': count_underscore,
    'Count_at': count_at,
    'Count_dollar': count_dollar,
    'Count_percent': count_percent
})

#  Featrue Extracted new CSV 
df1.to_csv('Extracted_Features.csv', index=False)
print(df1)
