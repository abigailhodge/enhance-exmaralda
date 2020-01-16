import os
import re
import csv
import spacy
import argparse
from operator import itemgetter
from spacy.tokenizer import Tokenizer

# TODO: Separate some of this stuff out into different files


# writes all noun chunks found in the given directory to a csv file
# includes information about pos tag distribution and metadata
def process_noun_chunk(dirpath):
    pos_dicts = []
    pos_fieldnames = ['noun_chunk', 'length', 'tt_ratio', 'filename', 'speaker_type', 'speaker_lang', 'speaker_gender',
                  'register', 'mode', 'starting_point', 'ending_point',
                  'AJ0', 'AJC', 'AJS', 'AT0', 'AV0', 'AVP', 'AVQ', 'CJC', 'CJS',
                  'CJT', 'CRD', 'DPS', 'DT0', 'DTQ', 'EX0', 'ITJ', 'NN0', 'NN1',
                  'NN2', 'NP0', 'ORD', 'PNI', 'PNP', 'PNQ', 'PNX', 'POS', 'PRF',
                  'PRP', 'PUL', 'PUN', 'PUQ', 'PUR', 'TO0', 'UNC', 'VBB', 'VBD',
                  'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI', 'VDN',
                  'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB',
                  'VVD', 'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0'
                    ]
    for filepath in os.listdir(dirpath):
        with open(dirpath+filepath, 'r', encoding='utf-8') as f:
            lines=f.readlines()
        norms, lemmas, cus, pos_tags = extract_norm_cu(lines)
        noun_chunk_spans = get_noun_chunks(norms)
        pos_dicts += pos_per_noun_chunk(noun_chunk_spans, pos_tags, filepath)
    write_to_csv(pos_dicts, pos_fieldnames, "/Users/abigailhodge/PycharmProjects/improve-exmaralda/test.csv")


# writes all instances of zero article nouns in the given directory to a csv file
# Each line includes the noun, filename, time, context, and metadata
def process_zero_arts(dirpath):
    zero_article_dicts = []
    zero_article_fieldnames = ['filepath', 'time', 'cu', 'zero_article_noun',
                               'speaker_type', 'speaker_lang',
                               'speaker_gender', 'mode', 'register']
    for filepath in os.listdir(dirpath):
        with open(dirpath+filepath, 'r', encoding='utf-8') as f:
            lines=f.readlines()
        norms, lemmas, cus, pos_tags = extract_norm_cu(lines)
        zero_article_dicts += get_zero_articles(norms, cus, pos_tags, filepath)
    write_to_csv(zero_article_dicts, zero_article_fieldnames, '/Users/abigailhodge/PycharmProjects/improve-exmaralda/arts.csv')


# creates two csvs, one with word count, unique word count, and type-token ratio of all cus in a given directory
# and one with the same information for each file in the directory
# also includes metadata information
def process_counts(dirpath):
    cu_count_dicts = []
    file_count_dicts = []
    cu_count_fieldnames = ['filepath', 'start_time', 'end_time', 'word_count', 'unique_word_count', 'tt_ratio',
                        'speaker_type', 'speaker_lang','speaker_gender', 'mode', 'register']
    file_count_fieldnames = ['filepath', 'word_count', 'unique_word_count', 'tt_ratio',
                        'speaker_type', 'speaker_lang','speaker_gender', 'mode', 'register']
    for filepath in os.listdir(dirpath):
        with open(dirpath+filepath, 'r', encoding='utf-8') as f:
            lines=f.readlines()
        norms, lemmas, cus, pos_tags = extract_norm_cu(lines)
        cus_with_times = create_times(lemmas, cus)
        file_count_dicts.append(get_file_counts(filepath, lemmas))
        cu_count_dicts += get_counts(filepath, cus_with_times)
    write_to_csv(cu_count_dicts, cu_count_fieldnames, '/Users/abigailhodge/PycharmProjects/improve-exmaralda/cu_counts.csv')
    write_to_csv(file_count_dicts, file_count_fieldnames, '/Users/abigailhodge/PycharmProjects/improve-exmaralda/file_counts.csv')


# extracts normalized tokens, lemmas, pos tags, and cus from ExMARALDA lines
# first step of data processing for pretty much anything you want to do
def extract_norm_cu(lines):
    norms = []
    cus = []
    pos_tags = []
    lemmas = []
    in_cu_tier = False
    in_norm_tier = False
    in_lemma_tier = False
    in_pos_tier = False
    for line in lines:
        if in_cu_tier:
            info = re.match("<event start=\"T([0-9]+)\" end=\"T([0-9]+)\"", line)
            if info:
                start, end = info.groups()
                cus.append([int(start), int(end)])
            else:
                in_cu_tier = False
        if in_norm_tier:
            info = re.match("<event start=\"T([0-9]+)\" end=\"T([0-9]+)\">(.*)<", line)
            if info:
                groups = info.groups()
                norms.append([int(groups[0]), int(groups[1]), groups[2].strip()])
            else:
                in_norm_tier = False
        if in_lemma_tier:
            info = re.match("<event start=\"T([0-9]+)\" end=\"T([0-9]+)\">(.*)<", line)
            if info:
                groups = info.groups()
                lemmas.append([int(groups[0]), int(groups[1]), groups[2].strip()])
            else:
                in_norm_tier = False
        if in_pos_tier:
            info = re.match("<event start=\"T([0-9]+)\" end=\"T([0-9]+)\">(.*)<", line)
            if info:
                groups = info.groups()
                pos_tags.append([int(groups[0]), int(groups[1]), groups[2]])
            else:
                in_pos_tier = False

        if re.search("category=\"cu\"", line):
            info = re.search("<event start=\"T([0-9]+)\" end=\"T([0-9]+)\"", line)
            if info:
                start, end = info.groups()
                cus.append([int(start), int(end)])
            in_cu_tier = True
        if re.search("category=\"norm\"", line):
            info = re.search("<event start=\"T([0-9]+)\" end=\"T([0-9]+)\">(.*)<", line)
            if info:
                groups = info.groups()
                norms.append([int(groups[0]), int(groups[1]), groups[2].strip()])
            in_norm_tier = True
        if re.search("category=\"lemma\"", line):
            info = re.search("<event start=\"T([0-9]+)\" end=\"T([0-9]+)\">(.*)<", line)
            if info:
                groups = info.groups()
                lemmas.append([int(groups[0]), int(groups[1]), groups[2].strip()])
            in_lemma_tier = True
        if re.search("category=\"pos_lang\"", line):
            info = re.search("<event start=\"T([0-9]+)\" end=\"T([0-9]+)\">(.*)<", line)
            if info:
                groups = info.groups()
                pos_tags.append([int(groups[0]), int(groups[1]), groups[2]])
            in_pos_tier = True
    return norms, lemmas, cus, pos_tags


# finds all noun chunks in a list of norms
# uses the spacy noun chunker
def get_noun_chunks(norms):
    nlp = spacy.load('en')
    nlp.tokenizer = Tokenizer(nlp.vocab)
    norm_len = len(norms)
    just_words = [norm[2] for norm in norms]
    full_text = ' '.join(just_words)
    doc = nlp(full_text)
    noun_chunk_spans = []
    big_noun_chunks = {np
                      for nc in doc.noun_chunks
                      for np in [
                        nc,
                        doc[
                          nc.root.left_edge.i
                          :nc.root.right_edge.i+1]]}
    for noun_chunk in big_noun_chunks:
        starting_num = noun_chunk.start
        if starting_num >= norm_len:
            starting_idx = norms[-1][1]
        else:
            starting_idx = norms[noun_chunk.start][0]
        ending_num = noun_chunk.end - 1
        if ending_num >= norm_len:
            ending_idx = norms[-1][1]
        else:
            ending_idx = norms[ending_num][1]
        text = noun_chunk.text
        unique_words = []
        word_count = 0
        for word in text.split():
            word_count += 1
            if word not in unique_words:
                unique_words.append(word)
        min_noun_chunk = [starting_idx, ending_idx, text, word_count, len(unique_words) / word_count]
        if min_noun_chunk not in noun_chunk_spans:
            noun_chunk_spans.append(min_noun_chunk)
    return noun_chunk_spans


# for each noun chunk, determines the pos tags in the same span
def pos_per_noun_chunk(noun_chunk_spans, pos_tags, filepath):
    lines = []
    sorted_noun_chunks = sorted(noun_chunk_spans, key=itemgetter(0))
    for noun_chunk in sorted_noun_chunks:
        pos_dict = {'noun_chunk': noun_chunk[2], 'length': noun_chunk[3], 'tt_ratio': noun_chunk[4],
                    'filename': filepath, 'starting_point': noun_chunk[0], 'ending_point': noun_chunk[1],
                    'AJ0': 0, 'AJC': 0, 'AJS': 0, 'AT0': 0, 'AV0': 0, 'AVP': 0, 'AVQ': 0, 'CJC': 0, 'CJS': 0,
                    'CJT': 0, 'CRD': 0, 'DPS': 0, 'DT0': 0, 'DTQ': 0, 'EX0': 0, 'ITJ': 0, 'NN0': 0, 'NN1': 0,
                    'NN2': 0, 'NP0': 0, 'ORD': 0, 'PNI': 0, 'PNP': 0, 'PNQ': 0, 'PNX': 0, 'POS': 0, 'PRF': 0,
                    'PRP': 0, 'PUL': 0, 'PUN': 0, 'PUQ': 0, 'PUR': 0, 'TO0': 0, 'UNC': 0, 'VBB': 0, 'VBD': 0,
                    'VBG': 0, 'VBI': 0, 'VBN': 0, 'VBZ': 0, 'VDB': 0, 'VDD': 0, 'VDG': 0, 'VDI': 0, 'VDN': 0,
                    'VDZ': 0, 'VHB': 0, 'VHD': 0, 'VHG': 0, 'VHI': 0, 'VHN': 0, 'VHZ': 0, 'VM0': 0, 'VVB': 0,
                    'VVD': 0, 'VVG': 0, 'VVI': 0, 'VVN': 0, 'VVZ': 0, 'XX0': 0, 'ZZ0': 0}
        if "USbi" in filepath:
            pos_dict['speaker_type'] = 'heritage'
        else:
            pos_dict['speaker_type'] = 'monolingual'
        if "G" in filepath:
            pos_dict['speaker_lang'] = 'Greek'
        elif "T" in filepath:
            pos_dict['speaker_lang'] = 'Turkish'
        elif "R" in filepath:
            pos_dict['speaker_lang'] = 'Russian'
        elif "D" in filepath:
            pos_dict['speaker_lang'] = 'German'
        else:
            pos_dict['speaker_lang'] = 'English'
        if 'M' in filepath:
            pos_dict['speaker_gender'] = 'Male'
        else:
            pos_dict['speaker_gender'] = 'Female'
        if '_f' in filepath:
            pos_dict['register'] = 'formal'
        else:
            pos_dict['register'] = 'informal'
        if 'w' in filepath:
            pos_dict['mode'] = 'written'
        else:
            pos_dict['mode'] = 'spoken'
        for pos in pos_tags:
            if noun_chunk[0] <= pos[0] < noun_chunk[1] and pos[2] in pos_dict:
                pos_dict[pos[2]] = pos_dict[pos[2]] + 1
        lines.append(pos_dict)
    return lines


# finds all nouns with no article in the file
# file is represented as a list of norms, cus, and pos tags
def get_zero_articles(norms, cus, pos_tags, filepath):
    zero_articles = []
    speaker_type, speaker_lang, speaker_gender, mode, register = extract_metadata(filepath)
    for cu in cus:
        in_det = False
        in_noun = False
        token_list = []
        len_pos = len(pos_tags)
        for i in range(len(norms)):
            norm = norms[i]
            if len_pos > i:
                pos = pos_tags[i]
                if cu[0] > norm[0]:
                    continue
                elif cu[0] <= norm[0] < cu[1] and cu[0] < norm[1] <= cu[1]:
                    token_list.append(norm[2])
                    if pos[2] in ("DTQ", "DPS", "DT0", "AT0", "CRD", "POS"):
                        in_det = True
                    elif pos[2] in ("NN1", "NN2", "NN0") and norm[2] not in ("groceries", "time", "control", "fault",
                                                                             "case", "order", "babe", "behalf"):
                        if in_det:
                            in_noun = True
                        elif in_noun:
                            pass
                        else:
                            zero_article_dict = {'filepath': filepath, 'time': norm[0], 'cu': ' '.join(token_list), 'zero_article_noun': norm[2],
                                                 'speaker_type': speaker_type, 'speaker_lang': speaker_lang,
                                                 'speaker_gender': speaker_gender, 'mode': mode, 'register': register}
                            zero_articles.append(zero_article_dict)
                        in_det = False
                    else:
                        in_noun = False
                else:
                    break
    return zero_articles


# gets # words, # unique words, and type-token ratio for each cu in the given cu list
def get_counts(filepath, cus):
    count_dicts = []
    speaker_type, speaker_lang, speaker_gender, mode, register = extract_metadata(filepath)
    for cu in cus:
        start_time = cu[0]
        end_time = cu[1]
        word_count = cu[2]
        unique_word_count = cu[3]
        tt_ratio = cu[4]
        count_dict = {'filepath': filepath, 'start_time': start_time, 'end_time': end_time, 'word_count': word_count,
                      'unique_word_count': unique_word_count, 'tt_ratio': tt_ratio, 'speaker_type': speaker_type,
                      'speaker_lang': speaker_lang,  'speaker_gender': speaker_gender, 'mode': mode, 'register': register}
        count_dicts.append(count_dict)
    return count_dicts


# gets # words, # unique words, and type-token ratio for the entire file
# file is represented as a list of norms (filepath is there to provide metadata)
def get_file_counts(filepath, norms):
    speaker_type, speaker_lang, speaker_gender, mode, register = extract_metadata(filepath)
    word_count, unique_word_count, tt_ratio = get_tt_ratio(norms)
    return {'filepath': filepath, 'word_count': word_count, 'unique_word_count': unique_word_count,
            'tt_ratio': tt_ratio, 'speaker_type': speaker_type, 'speaker_lang': speaker_lang,
            'speaker_gender': speaker_gender, 'mode': mode, 'register': register}


# Extracts speaker type, language, gender, register, and mode from a given filepath
def extract_metadata(filepath):
    if "USbi" in filepath:
        speaker_type = 'heritage'
    else:
        speaker_type = 'monolingual'
    if "G" in filepath:
        speaker_lang = 'Greek'
    elif "T" in filepath:
        speaker_lang = 'Turkish'
    elif "R" in filepath:
        speaker_lang = 'Russian'
    elif "D" in filepath:
        speaker_lang = 'German'
    else:
        speaker_lang = 'English'
    if 'M' in filepath:
        speaker_gender = 'Male'
    else:
        speaker_gender = 'Female'
    if '_f' in filepath:
        register = 'formal'
    else:
        register = 'informal'
    if 'w' in filepath:
        mode = 'written'
    else:
        mode = 'spoken'
    return speaker_type, speaker_lang, speaker_gender, register, mode


# annotates cus with word count, unique word count, and type-token ratio
def create_times(norms, cus):
    enhanced_cus = []
    for cu in cus:
        unique_words = []
        cu_word_count = 0
        for norm in norms:
            if cu[0] <= norm[0] < cu[1] and cu[0] < norm[1] <= cu[1]:
                info = re.search("[^.!?(),;:\"]+", norm[2])
                if info and norm[2] != "äh":
                    word = norm[2]
                    cu_word_count += 1
                    if word not in unique_words:
                        unique_words.append(word)
        cu.append(cu_word_count)
        cu.append(len(unique_words))
        if cu_word_count == 0:
            cu.append("n/a")
        else:
            cu.append('%.3f' % (len(unique_words) / cu_word_count))
        enhanced_cus.append(cu)
    return enhanced_cus


# finds number of words, number of unique words, and type-token ratio for whole file
# TODO: Abstract this and the method above to remove duplicate code
def get_tt_ratio(norms):
    unique_words = []
    word_count = 0
    for norm in norms:
        info = re.search("[^.!?(),;:\"]+", norm[2])
        if info and norm[2] != "äh":
            word = norm[2]
            word_count += 1
            if word not in unique_words:
                unique_words.append(word)
    if word_count == 0:
        tt_ratio = '0'
    else:
        tt_ratio = '%.3f' % (len(unique_words) / word_count)
    return word_count, len(unique_words), tt_ratio


# writes a list of dicts to a csv with the given fieldnames.
# fields of the dicts and fieldnames should match
def write_to_csv(dicts, fieldnames, filepath):
    csv_file = open(filepath, 'w+')
    with csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for noun_chunk in dicts:
            writer.writerow(noun_chunk)


# adds tiers necessary for referent introduction coding to all ExMARALDA files in given directory
def add_ref_tiers(dirpath):
    for filepath in os.listdir(dirpath):
        with open(dirpath+filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        write_lines = lines[:-2]
        write_lines.append("</tier><tier id=\"TIE100\" speaker=\"norm\" category=\"referent\" type=\"a\" "
                           "display-name=\"norm [referent]\">\n")
        write_lines.append("</tier><tier id=\"TIE101\" speaker=\"norm\" category=\"r-type\" type=\"a\" "
                           "display-name=\"norm [r-type]\">\n")
        write_lines.append("</tier><tier id=\"TIE102\" speaker=\"norm\" category=\"conj_referent\" type=\"a\" "
                           "display-name=\"norm [conj_referent]\">\n")
        write_lines.append("</tier></basic-body>\n")
        write_lines.append("</basic-transcription>\n")
        with open(dirpath + filepath, 'w', encoding='utf-8') as f:
            f.writelines(write_lines)

# this was for back when we were representing tt-ratio/word count/etc as additional tiers in ExMARALDA
# rather than as CUs
def get_write_lines(lines, cus, tt_ratio, word_count):
    write_lines = lines[:-2]
    write_lines.append("</tier><tier id=\"TIE99\" speaker=\"norm\" category=\"cu_word_count\" type=\"a\" display-name=\""
                       "norm [cu_word_count]\">")
    for cu in cus:
        write_lines.append("<event start=\"T%d\" end=\"T%d\">%d</event>\n" % (cu[0], cu[1], cu[2]))
    write_lines.append("</tier><tier id=\"TIE98\" speaker=\"norm\" category=\"cu_unique_word_count\" type=\"a\" display-name=\""
                       "norm [cu_unique_word_count]\">")
    for cu in cus:
        write_lines.append("<event start=\"T%d\" end=\"T%d\">%d</event>\n" % (cu[0], cu[1], cu[3]))
    write_lines.append("</tier><tier id=\"TIE97\" speaker=\"norm\" category=\"cu_tt_ratio\" type=\"a\" display-name=\""
                       "norm [cu_tt_ratio]\">")
    for cu in cus:
        write_lines.append("<event start=\"T%d\" end=\"T%d\">%.5s</event>\n" % (cu[0], cu[1], cu[4]))
    write_lines.append("</tier><tier id=\"TIE96\" speaker=\"norm\" category=\"file_tt_ratio\" type=\"a\" display-name=\""
                       "norm [file_tt_ratio]\">")
    write_lines.append("<event start=\"T0\" end=\"T1\">%.3f</event>\n" % tt_ratio)
    write_lines.append("</tier><tier id=\"TIE95\" speaker=\"norm\" category=\"file_word_count\" type=\"a\" display-name=\""
        "norm [file_word_count]\">")
    write_lines.append("<event start=\"T0\" end=\"T1\">%d</event>\n" % word_count)
    write_lines.append("</tier></basic-body>\n")
    write_lines.append("</basic-transcription>\n")
    return write_lines


def parse_command_line():
    description = ''
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('in_dir', metavar='in_dir', type=str,
                        help='Filepath of directory to enhance')
    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    process_counts(args.in_dir)

