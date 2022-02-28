import nltk.data
import spacy
import copy
import re

MONEY = '#MONEY#'
NUMBER = '#NUMBER#'
ACRONYM = '#ACRONYM#'
ACAPITAL = '#ACAPITAL#'
TIME = '#TIME#'
HTTP = '#HTTP#'
TDOT = '#TDOT#'
LDQ = '#LDQ#'
RDQ = '#RDQ#'
LSQ = '#LSQ#'
RSQ = '#RSQ#'
DASH = '#DASH#'
LDASH = '#LDASH#'

# tokens used by UNILM

UNILM_RSQ = "-rq-"
UNILM_LSQ = "-lq-"
UNILM_RRB = "-rrb-"
UNILM_LRB = "-lrb-"
UNILM_RCB = "-rcb-"
UNILM_LCB = "-lcb-"
UNILM_RSB = "-rsb-"
UNILM_LSB = "-lsb-"
UNILM_HTTP = "-http-"
UNILM_LDQ = "-bq-"
UNILM_RDQ = "-eq-"

DDSHORT = ['e.g.', 'a.m.', 'p.m.', 'i.e.', 'm.d.']
DDSHORT_TOK = ['#EG#', '#AM#', '#PM#', '#IE#', '#MD#']
NUMBER_POT = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


def pre_quote_tokenize(paragraph=None, token='_quote_token'):
    """
        Quotes often contain multiple sentences, these sentences should not be split imo
    """
    rexp = r'[\s\S]*?([\"][A-Za-z]([\s\S]*?)([^\s][\"]))'
    moved_quotes = []
    match = re.match(rexp, paragraph)
    while match is not None:
        moved_quotes.append(match.group(1))
        paragraph = paragraph[:match.span(1)[0]] + token + paragraph[match.span(1)[1]:]
        match = re.match(rexp, paragraph)
    return paragraph, moved_quotes

def pre_dialogue_tokenize(string=None, token='_conversation_token. '):
    """
        Dialogues between people usually are written with many smaller sentences e.g.
        Kate: Hi. I was wondering something. How many apples do you sell?
        Vender: Hello Kate...
        all dialogues should be turned into tokens
    """
    # assert string is not None
    # rexp = r'[\s\S]*?(([A-Z][a-z]+\s)*[A-Z][a-z]+[:][\s\S]*?)(([A-Z][a-z]+\s)*[A-Z][a-z]+[:])'
    rexp = r'[\s\S]*?(([A-Z][a-z]+\s)*([A-Z]{2,}|([A-Z][a-z]+))[:][\s\S]*?)(([A-Z][a-z]+\s)*([A-Z]{2,}|([A-Z][a-z]+))[:])'
    moved_dialogues = []
    match = re.match(rexp, string)
    while match is not None:
        moved_dialogues.append(match.group(1))
        span = match.span(1)
        string = string[:span[0]] + token + string[span[1]:]
        match = re.match(rexp, string)
    return string, moved_dialogues

def post_dialogue_tokenize(sentences=None, moved_dialogues=None, token='_conversation_token.'):
    assert isinstance(sentences, list)
    """
        Put the line-breaking words back into the sentences after splitting with nltk-sentence-splitter
    """
    new_sentences = []
    for sentence in sentences:
        match_spans = [m.span() for m in re.finditer(token, sentence)]
        for i in range(len(match_spans) -1, -1, -1):
            span = match_spans[i]
            word = moved_dialogues.pop(i)
            sentence = sentence[:span[0]] + word + " " + sentence[span[1]:]
        new_sentences.append(sentence)
    return new_sentences

def post_quote_tokenize(sentences=None, moved_quotes=None, token='_quote_token'):
    assert isinstance(sentences, list)
    """
        Turn the quote tokens back into text after splitting with nltk-sentence-splitter
    """
    new_sentences = []
    for sentence in sentences:
        match_spans = [m.span() for m in re.finditer(token, sentence)]
        for i in range(len(match_spans) -1, -1, -1):
            span = match_spans[i]
            word = moved_quotes.pop(i)
            sentence = sentence[:span[0]] + word + " " + sentence[span[1]:]
        new_sentences.append(sentence)
    return new_sentences

# string, moved_dialogues = pre_dialogue_tokenize()

# nltk.data.load('tokenizers/punkt/english.pickle')
# nltk_tokenizer = nltk.tokenize
#
# sentences = nltk_tokenizer.sent_tokenize(string)
# new_sentences = post_dialogue_tokenize(sentences=sentences, moved_dialogues=moved_dialogues, token='_conversation_token.')
# print(new_sentences)

def pre_sentence_tokenize(string=None, token='no_line_break'):
    """
        Move line-breaking words that should not break the line and replace them with a token
        before utilizing the nltk-sentence tokenizer.
    """
    assert string is not None
    regular_ex = [r'[A-Z][a-z]*[.]',
                  r'[A-Za-z][.]([A-Za-z][.])+',
                  r'[Nn]o[.] \d+',
                  r'[Mm]r[.]',
                  r'[a-z]+[.]"',
                  r'et al[.]',
                  r'Gov[.]',
                  r'CMD[.]',
                  r'SGT[.]',
                  r'Sgt[.]']
    shielded_words = ['January.',
                      'February.',
                      'Mars.',
                      'April.',
                      'May.',
                      'June.',
                      'July.',
                      'August.',
                      'September.',
                      'Oktober.',
                      'November.',
                      'December.']

    iterators = [re.finditer(reg, string) for reg in regular_ex]
    tuples = [(item.group(), item.span()) for iterator in iterators for item in iterator if item.group() not in shielded_words]
    tuples.sort(key=lambda x: (x[1][0], x[1][1]))
    if not tuples:
        return string, []
    # remove words found that overlap each other
    save_tuples = [tuples[0]]
    i = 1
    span_start = tuples[0][1][0]
    span_end = tuples[0][1][1]
    while i < len(tuples):
        new_span_start = tuples[i][1][0]
        new_span_end = tuples[i][1][1]
        if new_span_start > span_end:
            save_tuples.append(tuples[i])
            span_start = tuples[i][1][0]
            span_end = tuples[i][1][1]
        elif span_start == new_span_start and span_end < new_span_end:
            save_tuples.pop(-1)
            save_tuples.append(tuples[i])
            span_end = tuples[i][1][1]
        i += 1

    # put token in place of line-breaking words
    moved_words = []
    for i in range(len(save_tuples) -1, -1, -1):
        word, span = save_tuples[i]
        string = string[:span[0]] + token + string[span[1]:]
        moved_words.insert(0, word)
    return string, moved_words

# s = "ATLANTA, Georgia -- Going back to work after my wife had our first child was an emotional roller coaster.\n\n\n\nThe author says that being \"Mr. Mom\" is appealing, but putting the idea into practice is harder than it looks."
# string, moved_words = pre_sentence_tokenize(s)
# print(string)
# print(moved_words)

def post_sentence_tokenize(sentences=None, moved_words=None, token='no_line_break'):
    """
        Put the line-breaking words back into the sentences after splitting with nltk-sentence-splitter
    """
    new_sentences = []
    for sentence in sentences:
        match_spans = [m.span() for m in re.finditer(token, sentence)]
        for i in range(len(match_spans) -1, -1, -1):
            span = match_spans[i]
            word = moved_words.pop(i)
            sentence = sentence[:span[0]] + word + sentence[span[1]:]
        new_sentences.append(sentence)
    return new_sentences


def wrapped_nltk_sentence_split(nltk_tokenizer=None, paragraph=None, verbose=False, use_dialogue=False, use_quote=False):
    assert paragraph
    if not nltk_tokenizer:
        nltk.data.load('tokenizers/punkt/english.pickle')
        nltk_tokenizer = nltk.tokenize

    # change some shortenings with "." to words without dots
    ccwwd = change_capital_words_with_dot()
    _, paragraph = ccwwd.use(None, paragraph)
    if use_quote:
        paragraph, moved_quotes = pre_quote_tokenize(paragraph)
        if verbose:
            print(paragraph)
    if use_dialogue:
        if verbose:
            print(paragraph)
        paragraph, moved_dialogues = pre_dialogue_tokenize(paragraph)

    paragraph, moved_words = pre_sentence_tokenize(string=paragraph)
    if verbose:
        print(paragraph)
    sentences = nltk_tokenizer.sent_tokenize(paragraph)
    if verbose:
        print(sentences)
    sentences = post_sentence_tokenize(sentences=sentences, moved_words=moved_words)
    if verbose:
        print(sentences)
    if use_dialogue:
        sentences = post_dialogue_tokenize(sentences=sentences, moved_dialogues=moved_dialogues)
        if verbose:
            print(paragraph)
    if use_quote:
        sentences = post_quote_tokenize(sentences=sentences, moved_quotes=moved_quotes)
        if verbose:
            print(paragraph)

    if verbose:
        print(sentences)
        print("end of wrapped_nltk_sentence_split")
    return sentences

class TokenizeRule():

    def __init__(self):
        self.feature_money = MONEY
        self.feature_acronym = ACRONYM
        self.LDQ = LDQ
        self.RDQ = RDQ
        self.LSQ = LSQ
        self.RSQ = RSQ
        self.DASH = DASH
        self.LDASH = LDASH
        self.NUMBER = NUMBER
        self.TIME = TIME
        self._HTTP = "http"

        self.UNILM_map = {UNILM_LSQ : "'",
                          UNILM_RSQ : "'",
                          UNILM_RRB : ")",
                          UNILM_LRB : "(",
                          UNILM_RCB : "}",
                          UNILM_LCB : "{",
                          UNILM_RSB : "]",
                          UNILM_LSB : "[",
                          UNILM_HTTP : HTTP,
                          UNILM_LDQ : '"',
                          UNILM_RDQ : '"',
                         }

        self.shortwords = ['mr.', 'dr.', 'sr.', 'jr.', 'etc.', '...']
    def __str__(self):
        return self._name


class FeatureHolder():
    def __init__(self):

        self.raw_string = None
        self.with_feature = None
        self.without_feature = None

        self.feature_ddshort_list = DDSHORT
        self.feature_ddshort_tok_list = DDSHORT_TOK

        self.acronym_token = "ACRONYM"
        self.acapital_token = "ACAPITAL"

        # default meta-characters
        self.default_dash = "-"
        self.default_ldash = "—"
        self.default_ldq = '"'
        self.default_rdq = '"'
        self.default_lsq = "'"
        self.default_rsq = "'"
        self.default_lrb = "("
        self.default_rrb = ")"
        self.default_lsb = "["
        self.default_rsb = "]"
        self.default_lcb = "{"
        self.default_rcb = "}"

        # features are words that are exchanged for tokens in the text
        self.feature_money = MONEY
        self.feature_time = TIME
        self.feature_acronym = ACRONYM
        self.feature_number = NUMBER
        self.feature_acapital = ACAPITAL
        self.feature_http = HTTP
        self.feature_tdot = TDOT
        self.feature_ldq = LDQ
        self.feature_rdq = RDQ
        self.feature_lsq = LSQ
        self.feature_rsq = RSQ
        self.feature_dash = DASH
        self.feature_ldash = LDASH
        self.feature_rrb = "#RRB#"
        self.feature_lrb = "#LRB#"
        self.feature_rsb = "#RSB#"
        self.feature_lsb = "#LSB#"
        self.feature_rcb = "#RCB#"
        self.feature_lcb = "#LCB#"
        self.twoworddot_replacement = "#TWOWORDDOT#"

        self.default_to_feature_map =  {self.feature_dash: self.default_dash,
                                        self.feature_ldash: self.default_ldash,
                                        self.feature_ldq: self.default_ldq,
                                        self.feature_rdq: self.default_rdq,
                                        self.feature_rsq: self.default_rsq,
                                        self.feature_lsq: self.default_lsq,
                                        self.feature_rrb: self.default_rrb,
                                        self.feature_lrb: self.default_lrb,
                                        self.feature_rsb: self.default_rsb,
                                        self.feature_lsb: self.default_lsb,
                                        self.feature_rcb: self.default_rcb,
                                        self.feature_lcb: self.default_lcb,
                                        }

        self.three_word_dot = {r'[Ee]tc'}

        self.twoworddot_regexp = {r'[Mm][Rr][.]',
                                  r'[Dd][Rr][.]',
                                  r'[Jj][Rr][.]',
                                  r'[Cc][Oo][.]',
                                  r'[Mm][Aa][.]',
                                  r'[Bb][Cc][.]',
                                  r'[A][Dd][.]',
                                  r'[Mm][Aa][.]',
                                  r'[Aa][.][Mm]',
                                  r'[Pp][.][Mm]'}

        self.short_words = {'mr.': 'MRDOT',
                            'dr.': 'DRDOT',
                            'sr.': 'SRDOT',
                            'jr.': 'JRDOT',
                            'etc.': 'ETCDOT',
                            '...': 'DOTDOTDOT',
                            'e.g.': '#EG#',
                            'a.m.': '#AM#',
                            'p.m.': '#PM#',
                            'i.e.': '#IE#',
                            'm.d.': '#MD#',
                            }

        self.short_words_rev = {self.short_words[key]: key for key in self.short_words.keys()}

        self.feature_tag_pnumber_dict = NUMBER_POT

        # feature_list used for acapital not to overwrite other features during pre-processing
        self.feature_list = [self.feature_money,
                             self.feature_time,
                             self.feature_acronym,
                             self.feature_number,
                             self.feature_acapital,
                             self.feature_http,
                             self.feature_tdot,
                             self.feature_ldq,
                             self.feature_rdq,
                             self.feature_lsq,
                             self.feature_rsq,
                             self.feature_dash,
                             self.feature_ldash,
                             self.twoworddot_replacement,
                             ]



        # these features are kept as tags during training
        self.feature_tag_acronym = []
        self.feature_tag_pnumber = []
        self.feature_tag_capital = []
        self.feature_tag_acapital = []
        self.feature_tag_answer = []

        # these features are turned back into words after pre-processing
        self.feature_money_list = []
        self.feature_time_list = []
        self.feature_acronym_list = []
        self.feature_number_list = []
        self.feature_acapital_list = []
        # start using other word for non-tokens
        self.separate_twoworddot_list = []


    def set_raw_string(self, raw_string=None):
        self.raw_string = raw_string

    def reset(self):
        self.feature_tag_acronym = []
        self.feature_tag_pnumber = []
        self.feature_tag_capital = []
        self.feature_tag_acapital = []
        self.feature_tag_answer = []

        self.feature_money_list = []
        self.feature_time_list = []
        self.feature_acronym_list = []
        self.feature_number_list = []
        self.feature_acapital_list = []

        self.separate_twoworddot_list = []

        self.raw_string = None
        self.with_feature = None
        self.without_feature = None

    def bundle_lists(self, s):
        self.with_feature = s.split(" ")
        self.without_feature = copy.deepcopy(self.with_feature)

        self.feature_tag_acronym = [0] * len(self.with_feature)
        self.feature_tag_pnumber = [0] * len(self.with_feature)
        self.feature_tag_capital = [0] * len(self.with_feature)
        self.feature_tag_acapital = [0] * len(self.with_feature)

        for i, word in enumerate(self.with_feature):
            # Money
            if word == self.feature_money:
                self.without_feature[i] = self.feature_money_list.pop(0)

            # Time
            elif word == self.feature_time:
                self.without_feature[i] = self.feature_time_list.pop(0)

            # Acronyms
            elif word == self.feature_acronym:
                self.without_feature[i] = self.feature_acronym_list.pop(0)
                self.feature_tag_acronym[i] = 1

            # Numbers
            elif word == self.feature_number:
                self.without_feature[i] = self.feature_number_list.pop(0)
                # keep . because it represents decimals, remove "," convention.
                word = self.without_feature[i]
                word = re.sub(r'[,]', '', word)
                word = re.sub(r'(\d+)\'(\d+)', r'\1.\2', word)
                if len(word) > 1 and word[0:2] == '0.':
                    self.without_feature[i] = word[1:]
                else:
                    self.without_feature[i] = word
                # if we have an error, we have a bad character in the word
                try:
                    length = len(str(round(float(word))))
                except:
                    print("Bad character in feature numbers")
                    return False
                if length in self.feature_tag_pnumber_dict:
                    self.feature_tag_pnumber[i] = self.feature_tag_pnumber_dict[length]
                else:
                    self.feature_tag_pnumber[i] = len(self.feature_tag_pnumber_dict) + 1

            # All capital words
            elif word == self.feature_acapital:
                self.without_feature[i] = self.feature_acapital_list.pop(0)
                self.feature_tag_acapital[i] = 1

            # Fix other features that are just masks
            elif word in self.default_to_feature_map:
                self.without_feature[i] = self.default_to_feature_map[ word ]

            elif word in self.short_words_rev:
                self.without_feature[i] = self.short_words_rev[ word ]

            elif word == self.twoworddot_replacement:
                self.without_feature[i] = self.separate_twoworddot_list.pop(0)

            # Capital first letter
            elif len(word) > 0:
                if word[0].isupper() and not word in self.feature_list and len(word) > 1:
                    self.feature_tag_capital[i] = 1

        return True

    def before_spacy(self):
        """
            Exchange words that will be split 'wrongly' by Spacy by tokens
            after Spacy tokenization we will move exchange them back.
        """
        assert len(self.feature_acronym_list) == 0

        for i, word in enumerate(self.without_feature):
            match = re.match(r'([a-zA-Z]+[.]([a-zA-Z]+[.])+)', word)
            if match:
                self.feature_acronym_list.append(word)
                self.without_feature[i] = self.acronym_token
            elif word.isupper() and word not in self.feature_list and len(word) > 3:
                self.feature_acapital_list.append(word)
                self.without_feature[i] = self.acapital_token

    def after_spacy(self):
        for i, word in enumerate(self.without_feature):
            if word == self.acronym_token:
                self.without_feature[i] = self.feature_acronym_list.pop(0)
            elif word == self.acapital_token:
                self.without_feature[i] = self.feature_acapital_list.pop(0)

    def get_tokens(self):
        return self.without_feature

    def get_features(self):
        return {'tag_acronym': self.feature_tag_acronym,
                'tag_pnumber': self.feature_tag_pnumber,
                'tag_capital': self.feature_tag_capital,
                'tag_acapital': self.feature_tag_acapital,}

    def print(self):
        print("with features: "+"\n"+" ".join(self.with_feature)+"\n")
        print("without features: "+"\n"+" ".join(self.without_feature))

class PreTokenizer():
    def __init__(self, rules=None):
        self.rules = rules if rules else []
        self.rules.sort(key=lambda x: x._order)
        self.f_holder = FeatureHolder()

    def tokenize(self, string, step_wise=False, bundle=True):
        self.f_holder.reset()
        self.f_holder.set_raw_string(string)
        for rule in self.rules:
            self.f_holder, string = rule.use(self.f_holder, string)
            assert self.f_holder is not None
            assert string is not None
            if step_wise:
                print(rule._name)
                print(string)

        if not bundle:
            self.f_holder.without_feature = string.split(" ")
            return copy.deepcopy(self.f_holder)

        bundle_worked = self.f_holder.bundle_lists(string)
        if not bundle_worked:
            print("bundling did not work for some reason")

    def get_tokens(self):
        return self.f_holder.get_tokens()

    def get_fholder(self):
        return copy.deepcopy(self.f_holder)

    def get_tokens_as_string(self):
        return " ".join(self.f_holder.get_tokens())
    # def pretokenize(self, sentence, step_wise=False, bundle=False):
    #     sentence_list = listify(sentence)
    #     feature_holders = []
    #
    #     for string in sentence_list:
    #         self.f_holder.set_raw_string(raw_string=string)
    #         for rule in self.rules:
    #             self.f_holder, string = rule.use(self.f_holder, string)
    #             assert self.f_holder is not None
    #             assert string is not None
    #             if step_wise:
    #                 print(rule._name)
    #                 print(string)
    #
    #         if not bundle:
    #             return self.f_holder, string
    #
    #         passed = self.f_holder.bundle_lists(string)
    #         if not passed:
    #             return None
    #         feature_holders.append( copy.deepcopy(self.f_holder) )
    #         self.f_holder.reset()
    #
    #     return feature_holders

def make_new_pretokenizer():
    return PreTokenizer([
                        pad_whitespace_rule(),
                        remove_words_inside_clamps(),
                        classless_feature_rule(),
                        change_capital_words_with_dot(),
                        remove_bad_characters(),
                        remove_unicode_characters(),
                        remove_simple_misstakes(),
                        three_dot_rule(),
                        UNILM_to_standard(),
                        http_rule(),
                        whitespace_special_character_rule(),
                        comma_rule(),
                        split_sentence_end_dots(),
                        brackets_rule(),
                        slash_rule(),
                        feature_time_rule(),
                        normalize_dash_rule(),
                        feature_tag_acapital_rule(),
                        feature_common_shortening_rule(),
                        end_accent_rule(),
                        feature_tag_acronym_rule(),
                        space_dot_rule(),
                        pre_quote_rule(),
                        quote_rule(),
                        currency_rule(),
                        pre_number_rule(),
                        feature_tag_number_rule(),
                        clean_whitespace_rule(),
                        spacy_special_word_rules(),
                        two_word_dot_rule(),
                        split_colon_rule(),
                        ])

def single_whitespace(s):
    return re.sub(r'\s+', r" ", s)


### NEW STRUCTURE:

# Solving apostrophes
# still need to convert odd apostrophes to a default version
# SOLVED

# Solving single and double parenthesis
# solved them but it's possible that for single quotes it will catch words that end with ' that is not a quote
# SOLVED

# Solving accents and diacritics https://nlp.stanford.edu/IR-book/html/htmledition/accents-and-diacritics-1.html

# Solving acronyms and capital letters
# Here I would like to separate out the important shortenings that shouldn't be put in a category

# Solving currencies

# Solving dates and time


# one rule to convert UNILM pprocessing to my pprocessing tokens



class pad_whitespace_rule(TokenizeRule):
    _order = 0
    _name = "pad_whitespace_rule"
    _desription = "pad string with white-space, beginning and end for all rules to work"

    def use(self, f_holder, s):
        return f_holder, " " + s + " "

class classless_feature_rule(TokenizeRule):
    _order = 1
    _name = "classless_feature_rule"
    _description = "some categories I don't have time to fix, I will replace them with a single word instead"
    def use(self, f_holder, s):
        s = re.sub(r"\d+[°]\s?\d+[′]\s?\d+([.]\d+)?[″]\s?[NWSE]", r"coordinate", s)
        return f_holder, s

class remove_words_inside_clamps(TokenizeRule):
    _order = 3
    _name = "remove_words_inside_clamps"
    _desription = "turn [ and ] into ( )"
    def use(self, f_holder, s):
        s = re.sub(r"\[(.*?)\]", r"(\1)", s)
        s = re.sub(r"\[", r"(", s)
        s = re.sub(r"\]", r")", s)
        return f_holder, s

class whitespace_special_character_rule(TokenizeRule):
    _order = 4
    _name = "whitespace_special_character_rule"
    _desription = "put whitespace between between all %, ;, ?, !, #, = "

    def use(self, f_holder, s):
        # turn !.. ?.. into ! ... and ? ... etc
        s = re.sub(r"([!?])[.][.]", r" \1 ... ", s)
        s = re.sub(r"([%]|[;]|[¡]|[#]|[=]|[>]|[<]|[~]|[_]|[§])", r" \1 ", s)
        s = re.sub(r"(?!([?]|[!])\")([?]|[!])", r" \1 ", s)
        s = re.sub(r"([?]|[!])([\"])", r" \1\2", s)
        return f_holder, s

class brackets_rule(TokenizeRule):
    _order = 5
    _name = "brackets_rule"
    _description = "put whitespace between all '[', ']', '(', ')', '{', '}'"

    def use(self, f_holder, s):
        return f_holder, single_whitespace(re.sub(r'([\[\]\(\)\{\}])', r' \1 ', s))

class normalize_dash_rule(TokenizeRule):
    _name = "normalize_dash_rule"
    _order = 6
    _description = "there are multiple versions of dashes, turn them all into tokens"

    def use(self, f_holder, s):
        s = re.sub(r'[—|–|−|—]|--', ' ' + self.LDASH + ' ', s)
        s = re.sub(r'-', ' ' + self.DASH + ' ', s)
        return f_holder, s



class pre_quote_rule(TokenizeRule):
    _order = 7
    _name = "pre_quote_rule"
    _description = """unify all versions of quotes: `` to ", '' to " etc."""
    def use(self, f_holder, s):
        s = re.sub(r"(``)|('')|[“]|[”]", r'"', s)
        s = re.sub(r"[`’‘´]", r"'", s)
        # single quotes between special characters and capital letters
        s = re.sub(r"([.]\')([A-Z])", r"\1 \2", s)
        return f_holder, s

class remove_unicode_characters(TokenizeRule):
    _order = 8
    _name = "remove_unicode_characters"
    _description = "remove specified unicode characters"
    def use(self, f_holder, s):
        encoded_string = s.encode("ascii", "ignore")
        s = encoded_string.decode()
        s = s.replace(u'\u030d', '')
        s = s.replace(u'\u00c2', '')
        s = s.replace(u'\u00bb', '')
        s = s.replace(u'\u00e2', '')
        s = s.replace(u'\u20ac', '')
        s = s.replace(u'\u00b2', '')
        return f_holder, s

class comma_rule(TokenizeRule):
    _order = 9
    _name = "comma_rule"
    _description = "split ., and , attached to a letter"

    def use(self, f_holder, s):
        s = re.sub(r',([^\s\d]+)', r' , \1', s)
        s = re.sub(r',([^\s\d]+)', r' , \1', s)
        s = re.sub(r',([^\s\d]+)', r' , \1', s)
        s = re.sub(r',([^\s\d]+)', r' , \1', s)
        s = re.sub(r'([^\s]+), ', r'\1 , ', s)
        s = re.sub(r'[.][,]', r". ,", s)
        # also split , that is attached to numbers and other characters
        s = re.sub(r'(\d)[,]([^\d\s]+[\S]*) ', r"\1 , \2 ", s)
        return f_holder, s


class feature_time_rule(TokenizeRule):
    _order = 10
    _name = "feature_time_rule"
    _desription = "classify time and split : after finding time"

    def use(self, f_holder, s):
        # split dot from end of numbers
        s = re.sub(r"([\d]+)[.] ", r"\1 . ", s)
        s = s.split(" ")
        for i, word in enumerate(s):
            match = re.match(r"(\d+[:]\d+)(.*)", word)
            if match:
                if len(match.groups()) > 1:
                    s[i] = self.TIME + " " + match.groups()[-1]
                    f_holder.feature_time_list.append("".join(list(match.groups()[:-1])))
                else:
                    s[i] = self.TIME + match.group
                    f_holder.feature_time_list.append(word)
        s = " ".join(s)
        s = re.sub(r":", " : ", s)
        return f_holder, s

class change_capital_words_with_dot(TokenizeRule):
    _order = 11
    _name = "change_capital_words_with_dot"
    _description = "some words e.g. CMD. should be turned into Cmd."
    def use(self, f_holder, s):
        # im is split by spacy to i m, don't want this
        s = s.replace(r" im ", r" em ")
        s = s.replace(r"[Nn]/[Aa]", "notavailable")
        s = re.sub(r"\s[Cc][Aa][.]\s", r" circa ", s)
        s = re.sub(r"\s[Cc][Aa] [.]", r" circa ", s)
        s = re.sub(r"(\s|[\(,])[Ed][Dd](\s|[\),])", r"\1 editor \2", s)
        s = re.sub(r"\s[Ee][Dd] [.]", r" editor ", s)
        s = re.sub(r"(\s|[\(,])[Cc][Ff](\s|[\),])", r"\1 conferatur \2", s)
        s = re.sub(r"\s[Cc][Ff] [.]", r" conferatur ", s)
        s = re.sub(r"\s*([Dd])[Rr][.]([\S]|\s)", r" \1octor \2", s)
        s = re.sub(r"Sc[.]D[.]", "Doctor of Science", s)
        s = re.sub(r"D[.]Sc[.]", "Doctor of Science", s)
        s = re.sub(r"[SD][.][SD][.]", "Doctor of Science", s)
        s = re.sub(r"([Ss])[r][.]", r"\1enior", s)
        s = re.sub(r" theses(\s|^)", " thesises ", s)
        s = s.replace("CMD.", "Cmd.")
        s = s.replace("SGT.", "Sgt.")
        s = re.sub(r"(\d|\s)[Ff][Mm](\s|[.])", r"\1 frequency modulation \2", s)
        s = re.sub(r"(\d|\s)[Aa][Mm](\s|[.])", r"\1 amplitude modulation \2", s)
        s = re.sub(r"([(\[]|\s)ca[.] ", r"\1 circa ", s)
        s = re.sub(r"([Jj])[Rr][\\]*\'", r"\1unior  '", s)
        s = re.sub(r" ([Jj])([Rr])[.]", r" \1unior ", s)
        s = re.sub(r" ([Jj])([Rr]) [.] ", r" \1unior ", s)
        s = re.sub(r" ([Jj])([Rr]) ", r" \1unior ", s)
        s = re.sub(r" ([Aa])([Dd])([.]|\s|^)", r" \1fter death of christ ", s)
        s = re.sub(r" ([Aa])([Dd])([,\)-]) ", r" \1fter death of christ \3 ", s)
        s = re.sub(r" ([Aa])([Dd])([,\)-])([,\)-]) ", r" \1fter death of christ \3 \4 ", s)
        s = re.sub(r" ([Bb])([Cc])([.]|\s|^)\s*", r" \1efore christ ", s)
        s = re.sub(r" ([Bb])([Cc])([,\)-]) ", r" \1efore christ \3 ", s)
        s = re.sub(r" ([Bb])([Cc])([,\)-])([,\)-]) ", r" \1efore christ \2 \3 ", s)
        return f_holder, s

class split_sentence_end_dots(TokenizeRule):
    _order = 12
    _name = "split sentence end dots"
    _desription = "words that end with dot should have their dot split from them"
    def use(self, f_holder, s):
        s = re.sub(r"([a-z]+)([.])(\s|^)", r"\1 \2 \3", s)
        return f_holder, s

class remove_bad_characters(TokenizeRule):
    _order = 13
    _name = "remove_bad_characters"
    _desription = "some characters we don't want to handle, and they don't work with regular expressions"

    def use(self, f_holder, s):
        # characters replace by space
        bad_characters = ['♠', '⁄', '+', '→', '×', '*', '±', '½', '…', '°', '⟨', '⟨', '◌', 'ʰ', '⟩', 'Ã', '©', '®']
        for bad in bad_characters:
            s = s.replace(bad, ' ')
        # characters just removed
        s = s.replace('^', '')
        # some characters look similar but are different, choose one common
        s = re.sub(r"[‚,]", r",", s)
        # normalize multiple prints of the same character
        s = re.sub(r'["]["]+', r'"', s)
        # change № to #
        s = re.sub(r'[№]', r' # ', s)
        # "' shouldn't be between each other
        # s = re.sub(r"\"'", r'"', s)
        return f_holder, s

class slash_rule(TokenizeRule):
    _order = 14
    _name = "slash_rule"
    _desription = "put whitespace between /"
    def use(self, f_holder, s):
        new_s = []
        for i, word in enumerate(s.split(" ")):
            # don't separate mixes of numbers and capital letters (often concepts or names/products)
            # this is to render problems with acapital_rule
            # also check that there are no lower-case characters

            # old code
            # if re.match(r'(?=.*[0-9])(?=.*[A-Z])([A-Z0-9]+)[/]([A-Z]+|[0-9]+)[^a-z]', word):
            word = re.sub(r"([/])", r" \1 ", word)
            new_s.append(word)
            # if re.match(r'([A-Z-Z0-9]+)[/](?![A-Z0-9]+[a-z]+)([A-Z0-9]+)', word):
            #     new_s.append(word)
            # else:

        s = " ".join(new_s)
        s = single_whitespace(s)
        return f_holder, s


class remove_simple_misstakes(TokenizeRule):
    _order = 17
    _name = "remove_simple_misstakes"
    _desription = "found some writings that might be mistakes or a condition that I don't understand, remove them"
    def use(self, f_holder, s):
        # remove : from .: -- it doesn't have any use
        s = re.sub(r"[.][:]", r". ", s)
        return f_holder, s


class UNILM_to_standard(TokenizeRule):
    _order = 25
    _name = "UNILM_to_standard"
    _description = "this rule turns the tokens UNILM use to the standard of this module"

    def use(self, f_holder, s):
        s = s.split(" ")
        for i, word in enumerate(s):
            for unilm_token in self.UNILM_map.keys():
                if word == unilm_token:
                    s[i] = self.UNILM_map[unilm_token]
        return f_holder, " ".join(s)


# class unwanted_dotwords(TokenizeRule):
#     _order = 30
#     _name = "unwanted_dotwords"
#     _description = "mask common words ending with dot such that sentence splitting works properly"
#
#     def use(self, f_holder, s, encode=True):
#         if encode:
#             for w in self.shortwords:
#                 tgt = w.replace(".", "DOT").upper()
#                 s = s.replace(w, tgt)
#             return s
#         else:
#             for w in self.shortwords:
#                 src = w.replace(".", "DOT").upper()
#                 s = s.replace(src, w)
#             return f_holder, s

class split_colon_rule(TokenizeRule):
    _order = 60
    _name = "split_colon_rule"
    _desription = "split colons that are not between numbers"

    def use(self, f_holder, s):
        # split dot from end of numbers
        s = re.sub(r"([^\d]):([^\d])", r"\1 : \2 ", s)
        return f_holder, s

class three_dot_rule(TokenizeRule):
    _order = 61
    _name = 'three dot rule'
    _description = 'space triple dots'
    def use(self, f_holder, s):
        s = re.sub(r"(\s[A-Za-z]+)[.][.]", r"\1 ...", s)
        # turn sequences of more than 3 dots to 3 dots
        s = re.sub(r"[.][.][.]+", r' ... ', s)
        # put space between dot and comma
        s = re.sub(r"[.][,]", r". ,", s)
        # move triple dots from end of word
        s = re.sub(r"(\S)[.][.][.] ", r"\1 ... ", s)
        return f_holder, s

class end_accent_rule(TokenizeRule):
    _order = 62
    _name = "end_accent_rule"
    _desription = "split the accent at end of any word, split accent from s, nt, re"

    def use(self, f_holder, s):
        s = s.split(" ")
        for i, word in enumerate(s):
            match = re.match(r"(.+?)'(([Ss])|([Ss])[,.])$", word)
            if match:
                s[i] = " ".join([match.group(1), "'", match.group(2)])
            # else:
            #     s[i] = re.sub(r"(.*?)'", r"\1 ' ", s[i])
        s = " ".join(s)
        return f_holder, s

class two_word_dot_rule(TokenizeRule):
    _order = 65
    _name = "two_word_dot_words"
    _description = "all words that concist of two words ending with a dot should be not be split"
    _need_separate_list = True

    def use(self, f_holder, s):
        s = s.split(" ")
        for i, word in enumerate(s):
            for regexp in f_holder.twoworddot_regexp:
                if re.match(regexp, word):
                    # move word to separate list
                    f_holder.separate_twoworddot_list.append(word)
                    s[i] = f_holder.twoworddot_replacement
        return f_holder, " ".join(s)


class feature_common_shortening_rule(TokenizeRule):
    _order = 81
    _name = "common_shortenings"
    _description = "remove the dots from common shortening of words"

    def use(self, f_holder, s):
        s = s.split(" ")
        for i, word in enumerate(s):
            for j, ddshort in enumerate(f_holder.feature_ddshort_list):
                if word == ddshort:
                    s[i] = f_holder.feature_ddshort_tok_list[j]
        s = " ".join(s)
        return f_holder, s

class currency_rule(TokenizeRule):
    # putting currency here to capture currenies as ALL CAPITAL
    _order = 85
    _name = "currency_rule"
    _desription = "turn $100, or £100 into # 100"
    def use(self, f_holder, s):
        s = re.sub(r' [rR][sS][.](\d)', r' # \1', s) # Rs. rupees (indian)
        s = re.sub(r' [rR][sS][.] (\d)', r' # \1', s) # Rs. rupees (indian)
        s = re.sub(r' [rR][sS] (\d)', r' # \1', s) # Rs. rupees (indian)
        s = re.sub(r'[¢]', r' cents ', s)
        s = re.sub(r'[₹]', r'', s) # remove yan(?)
        s = re.sub(r'([$£€¥])[$£€¥]+', r'\1', s) # turn multiples of curency signs into single curency sign
        s = re.sub(r'[$£€¥](\d|([.]\d))', r" # \1", s)
        s = re.sub(r'[$](US)', r" # \1", s)
        s = re.sub(r'[$](T)(\d)', r" # \1 \2", s)
        s = re.sub(r'[$](NZ)(\d|([.]\d))', r" # \1 \2", s)
        return f_holder, s

class http_rule(TokenizeRule):
    _order = 150
    _name = "http rule"
    _description = "turn all links into tokens"

    def use(self, f_holder, s):
        s = re.sub(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)\s', self._HTTP + " ", s)
        s = re.sub(r'\swww[.][a-zA-Z0-9]+[.][A-Za-z]+\s', " " + self._HTTP + " ", s)
        s = re.sub(r'\s[A-Za-z0-9]+[.]((org)|(com))([.]{0,1})\s', " " + self._HTTP + " ", s)
        return f_holder, s

class feature_tag_acronym_rule(TokenizeRule):
    _order = 250
    _name = "feature_tag_acronym_rule"
    _description = "turn abreviations with dots into tokenized"

    def use(self, f_holder, s):
        # separate double dots wherever they are, not touching tripple dots
        s = re.sub(r"([^.\s])[.][.]([^.\s])", r"\1. .\2", s)
        s = s.split(" ")
        for i, word in enumerate(s):
            match = re.match(r'([a-zA-Z][.]([a-zA-Z][.])+)', word)
            match2 = re.match(r'[A-Z][.][A-Z][a-z]+([.]){0,1}', word)
            word = word.replace(".", "").upper()
            if match or match2:
                f_holder.feature_acronym_list.append(word)
                s[i] = self.feature_acronym
        return f_holder, " ".join(s)

class simple_quote_rule(TokenizeRule):
    _order = 399

    # TODO fix so that it doesn't conflict with inches
    def use(self, f_holder, s):
        s = s.split(" ")
        result = []
        inside_quote = False
        for i, word in enumerate(s):
            if re.match(r'\"[\S]+\"', word):
                result += [self.LDQ, word[1:-1], self.RDQ]
            elif re.match(r'([\S]*)\"(.*)', word):
                match = re.match(r'([\S]*)\"(.*)', word)
                if match.group(1):
                    result += [match.group(1)]
                if inside_quote:
                    result += [self.RDQ]
                    inside_quote = False
                else:
                    result += [self.LDQ]
                    inside_quote = True
                if match.group(2):
                    result += [match.group(2)]
            else:
                result += [word]
        return f_holder, " ".join(result)


class quote_rule(TokenizeRule):
    _order = 62
    _name = 'quote_rule'
    _description = "turn double and inner (single-quotes) into tokens. Single quotes can begin with ` or '."


    def use(self, f_holder, s):
        verbose = True
        result = []
        # move dot after "
        s = re.sub(r'(["]|[\'])[.]', r"\1 .", s)
        s = s.split(" ")
        inside_quotes = []
        inside_simple_quotes = []
        for i, word in enumerate(s):
            # print(" ".join(result))
            # print("inside_simple_quotes {}".format(inside_simple_quotes))
            # print("inside_quotes {}".format(inside_quotes))
            # match beginning "
            if re.match(r'\"([\S]+)\'[s]\"', word):
                m = re.match(r'\"([\S]+)\'[s]\"', word)
                result += [self.LDQ, m.group(1), "'", "s", self.RDQ]
            elif re.match(r"\'([A-Za-z]+)\'[s]\'", word):
                m = re.match(r"\'([A-Za-z]+)\'[s]\'", word)
                result += [self.LSQ, m.group(1), "'", "s", self.RSQ]
            elif re.match(r'\"([\S]+)\"([\S]*)', word):
                m = re.match(r'\"([\S]+)\"([\S]*)', word)
                result += [self.LDQ, m.group(1), self.RDQ]
                if m.group(2):
                    result += [m.group(2)]
            elif re.match(r"\"'([A-Za-z]+)'", word):
                match = re.match(r"\"'([A-Za-z]+)'", word)
                result += [self.LDQ, self.LSQ, match.group(1), self.RSQ]
                inside_quotes.append(len(inside_quotes) + 1)
                print(result)
            elif re.match(r'"', word):
                inside_quotes.append(len(inside_quotes) + 1)
                result += [self.LDQ] + [word[1:]] if not len(word) == 1 else [self.LDQ]
            # try to match double ' inside quote
            elif re.match(r'(\')([\S]+)(\')([\S]*\")', word):
                match = re.match(r'\'([\S]+[^\s]*)\'([\S]*)(\")', word)
                result += [self.LSQ, match.group(1), self.RSQ, match.group(2)]
                if inside_quotes:
                    result += [self.RDQ]
                    inside_quotes.pop(-1)
                else:
                    result += ['"']
                    print("found inches after closed single quotes")
            elif re.match(r'.*"$', word):
                # print("matched ending double quote")
                match = re.match(r'([\S^\']+)([^\s]*\')([\S]*)\"', word)
                if match and match.group(2):
                    # print('matched group 2')
                    if inside_simple_quotes:
                        result += [match.group(1), self.RSQ, match.group(3)]
                        inside_simple_quotes.pop(-1)
                    else:
                        result += [match.group(1), "'", match.group(3)]
                        print("found single quote without a starting single quotation")
                    if inside_quotes:
                        result += [self.RDQ]
                        inside_quotes.pop(-1)
                    else:
                        result += ['"']
                        print("probably found inches")
                elif not inside_quotes:
                    # print("not matched group 2")
                    if re.match(r'\d+"$', word):
                        print("found inches")
                        result += [word[:-1]] + ['inches']
                    else:
                        result += [word[:-1]] + ['"']
                else:
                    inside_quotes.pop(-1)
                    result += [word[:-1]] + [self.RDQ]
            elif re.match(r'([,])(\')([\S]+)', word):
                print(word)
                match = re.match(r'([,])(\')([\S]+)', word)
                if not inside_simple_quotes:
                    result += [match.group(1), self.LSQ, match.group(3)]
                    inside_simple_quotes.append(len(inside_simple_quotes) + 1)
                else:
                    print("found a starting quote right after , but already in a single quote expression")
                    result += [match.group(1), "'", match.group(3)]
            elif re.match(r'([\S]*)(\")([\.\,]*)', word):
                if not inside_quotes:
                    # if this happens, there is a typographic error in the text -- or we have caught a " that represents inches
                    print("using simple quote rule")
                    return simple_quote_rule().use(f_holder, " ".join(s))
                else:
                    match = re.match(r'([\S]*)(\")([\.\,]*)', word)
                    result += [match.group(1), self.RDQ]
                    # catch '.' after double quote
                    if match.group(3):
                        result += [match.group(3)]
                    inside_quotes.pop(-1)
            elif re.match(r"[\'`][\S]+\'", word):
                result += [self.LSQ, word[1:-1], self.RSQ]
            elif re.match(r"^[\'`][\S]", word):
                inside_simple_quotes.append(len(inside_simple_quotes) + 1)
                result += [self.LSQ, word[1:]]
            elif re.match(r"([\S]+)(\')([\,\.]+)", word):
                match = re.match(r"([\S]+)(\')([\,\.]+)", word)
                if inside_simple_quotes:
                    inside_simple_quotes.pop(-1)
                    result += [match.group(1), self.RSQ, match.group(3)]
                else:
                    result += [match.group(1), ' ', match.group(2), ' ', match.group(3)]
            elif re.match(r"([\S]+)(\')$", word):
                match = re.match(r"([\S]+)(\')$", word)
                if inside_simple_quotes:
                    inside_simple_quotes.pop(-1)
                    match2 = re.match(r"([A-Za-z]+)\'([s])", match.group(1))
                    if match2:
                        result += [match2.group(1), "'", match2.group(2), self.RSQ]
                    else:
                        result += [match.group(1), self.RSQ]
                else:
                    print("found single quote without a starting single quotation")
                    result += [match.group(1), "'"]
            else:
                result += [word]

        # if the quotes are not closed, re-factor the changes
        i = 0
        while i < len(result):
            if result[i] == self.LSQ:
                found_end = False
                for j in range(i+1, len(result)):
                    if result[j] == self.RSQ:
                        found_end = True
                        i = j
                if not found_end:
                    result[i] = ''
                    result[i+1] = "'" + result[i+1]
            i += 1
        result = " ".join(result)
        result = re.sub(r"([\S])' ", r"\1 ' ", result)
        result = re.sub(r" '([\S])", r" ' \1", result)
        return f_holder, result



class space_dot_rule(TokenizeRule):
    # put this rule here to split " from dots so that they can be detected by quote rules
    _order = 410
    _name = 'space dot rule'
    _description = 'put a space between word and ".", ",", "!" or "?"'

    def use(self, f_holder, s):
        s = re.sub(r"(\s[A-Za-z]+)[.][.]", r"\1 ...", s)
        # turn sequences of more than 3 dots to 3 dots
        s = re.sub(r"[.][.][.]+", r' ... ', s)
        # put space between dot and comma
        s = re.sub(r"[.][,]", r". ,", s)
        # move triple dots from end of word
        s = re.sub(r"(\S)[.][.][.] ", r"\1 ... ", s)
        s = re.sub(r"[.][.][.]", " DOTDOTDOT ", s)
        # split dots that are not between digits
        s = re.sub(r" ([^\d\.\s]+)[.]([^\d\.\s]+) ", r" \1 . \2 ", s)
        # split dots at end of sentence or words
        s = re.sub(r"([\s][^\s\.][^\s\.]+)([.])\s", r" \1 \2 ", s)
        # split dots from small letter characters
        s = re.sub(r"([\s][^A-Z\.\s])([.])\s", r" \1 \2 ", s)
        # split dots after %
        s = re.sub(r'([%])[.]', r"\1 .", s)
        # split dots from mixes of numbers and letters,
        # TODO this will fail in other mixes of letters and numbers
        s = re.sub(r'([A-Za-z]+)[.]([0-9]+)', r" \1 . \2 ", s)
        s = re.sub(r'([0-9]+)[.]([A-Za-z]+)', r" \1 . \2 ", s)
        return f_holder, s

class pre_number_rule(TokenizeRule):
    _order = 415
    _name = 'pre_number_rule'
    _description = 'split the following from numbers: %, th, :, -, km'

    def use(self, f_holder, s):
        # large numbers somehow get's caught in a viscious loop
        match = re.findall(r" (\d+) ", s)
        for m in match:
            if len(m) > 6:
                s = s.replace(m, m[0:6])
        match = re.findall(r" (\d+[.])", s)
        for m in match:
            if len(m) > 6:
                s = s.replace(m, m[0:6])
        # catch numbers that are at end of sentences
        s = re.sub(r"[\s]([\S]*\d+)+[.] ", r" \1 . ", s)
        s = re.sub(r"[\s](\d+|\d+[,.]\d+)(th|%|[:]|[+]|cm|dm|m)[\s]", r" \1 \2 ", s)
        # split degrees
        s = re.sub(r"[\s](\d+|\d+[,.]\d+)(°)", r" \1 ", s)
        # remove ′ from numbers
        s = re.sub(r"[\s](\d+|\d+[,.]\d+)(′)[\s]", r" \1 ", s)
        # unique case 1.x, turn x into 0
        s = re.sub(r"[\s](\d)[.][x] ", r" \1.0 ", s)
        # remove from end of numbers?
        s = re.sub(r"\s(\d+)[?]", r" \1 ", s)
        # # add space between # and number
        s = re.sub(r"\s[#](\d+) ", r" # \1 ", s)
        # # separate m, g, . at end of number
        s = s.split(" ")
        for i, word in enumerate(s):
            s[i] = re.sub(r"(\d+[\.]*\d*)([KkMmGg]|(mm))", r"\1 \2", word)
            s[i] = re.sub(r"(\d+)(([PpFfAa][Mm])|((ft)|(in)|[s]))", r"\1 \2", s[i])
            s[i] = re.sub(r"(\d)([A-Z]+)", r"\1 \2", s[i])
        return f_holder, " ".join(s)

class feature_tag_acapital_rule(TokenizeRule):
    _order = 420
    _name = "feature_tag_acapital_rule"
    _desription = "mark words which consists of only capital letter"

    def use(self, f_holder, s):
        s = s.split(" ")
        insert_later = []
        for i, word in enumerate(s):
            word = word.strip()
            match = re.match(r'([a-zA-Z]+[.]([a-zA-Z+]+[.])+)', word)
            if match and i < len(s) - 1:
                if re.match(r"[A-Z][a-z]+", s[i+1]):
                    # next word is start of new sentence
                    word = word.replace(".", "").upper()
                    insert_later.append(i+1+len(insert_later))
            if word.isupper() and word not in f_holder.feature_list and len(word) > 1:
                if not word[0] == '#' and not word[-1] == '#':
                    # if word[-1] == '.' and len(word) >= 3:
                    #     s[i] = f_holder.feature_acapital + " ."
                    #     f_holder.feature_acapital_list.append(word[:-1])
                    # else:
                    s[i] = f_holder.feature_acapital
                    f_holder.feature_acapital_list.append(word)
        for idx in insert_later:
            s.insert(idx, ".")
        return f_holder, " ".join(s)

class feature_tag_number_rule(TokenizeRule):
    _order = 999
    _name = 'feature_tag_number_rule'
    _description = 'will catch any string starting with a number with , or . in it'

    def use(self, f_holder, s):
        s = s.split(" ")
        for i, word in enumerate(s):
            # TODO recognize numbers that have more than one dot and categorize them atm just subtract extra dots and numbers
            match = re.match(r"(\d+[.]\d+)([.]\d+)+", word)
            if match:
                word = match.groups()[0]
            # regular expression checks that the numbers don't end with a alphabet character
            no_alphabet_match = re.match(r'(?!(((\d+)(([,.]*)([\d]+))*)[A-Za-z]+))((\d+)(([,.]*)([\d]+))*)', word)
            no_slash_match = re.match(r'(?!(\d+[/]\d+))', word)
            if no_alphabet_match and no_slash_match:
                s[i] = self.NUMBER
                f_holder.feature_number_list.append(word)
        return f_holder, " ".join(s)

class clean_whitespace_rule(TokenizeRule):
    _order = 9998
    _name = 'clean_whitespace_rule'
    _description = 'remove redundant whitespaces'

    def use(self, f_holder, s):
        return f_holder, single_whitespace(s).strip(" ")


class spacy_special_word_rules(TokenizeRule):
    _order = 9999
    _name = 'spacy_special_word_rules'
    _description = 'spacy splits some words, do that to'

    def use(self, f_holder, s):
        map = {
               ' Ive ': " I have ",
               ' wed ': " wedded ",
               ' id ': ' identification ',
               # quick fix, this is a name
               ' Arent ': ' Are not ',
               ' arent ': ' are not ',
               " ca.": 'circa',
               " ca .": 'circa',
               "I'd": 'I d',
               "it'd": "it d",
               "he'd": 'he d',
               "He'd": "He d",
               "she'd": 'she d',
               "who'd": "who d",
               "we'd": "we d",
               "We'd": "We d",
               "Wed": "We d",
               "there'd": "there d",
               "There'd": "There d",
               "That'd": "That d",
               "that'd": "that d",
               "Who'll": "Who will",
               "who'll": "who will",
               "Where'd": "Where did",
               "where'd": "where did",
               "might've": "might have",
               "didnt": "did not",
               "Doesn't": "Does not",
               "Should've": "Should have",
               "should've": "should have",
               "wasn't": "was not",
               "cannot": "can not",
               "hasn't": "has not",
               "I'll": "I will",
               "I've": "I have",
               "ain't": "is not",
               "Ain't": "Is not",
               "Wasn't": "Was not",
               "y'all": "you all",
               " Id ": " Identifier ",
               "Sr . ": "Senior ",
               "sr . ": "senior ",
               "c'mon": "come on",
               }
        s = re.sub(r"([Hh]as)n\'t", r"\1 not", s)
        s = re.sub(r"([Ss]hould)n\'t", r"\1 not", s)
        s = re.sub(r"([Cc]ould)n\'t", r"\1 not", s)
        s = re.sub(r"([Yy]ou)\'ll", r" \1 will ", s)
        s = re.sub(r"([Tt]hat)\'ll", r" \1 will ", s)
        s = re.sub(r"([Tt]here)\'ll", r" \1 will ", s)
        s = re.sub(r"([Ii]t)\'ll", r" \1 will ", s)
        s = re.sub(r"([Yy]ou)\'d", r" \1 d ", s)
        s = re.sub(r" ([Tt])heses ", r" \1hesises ", s)
        s = re.sub(r"([Aa]re)n't", r"\1 not", s)
        s = re.sub(r"([Ii]s)n't", r"\1 not", s)
        s = re.sub(r"([Mm]ust)n't", r"\1 not", s)
        s = re.sub(r"([Dd]o)n't", r"\1 not", s)
        s = re.sub(r"([Dd]id)n't", r"\1 not", s)
        s = re.sub(r"([Ss]h)e'll", r"\1e will", s)
        s = re.sub(r"([Hh])e'll", r"\1e will", s)
        s = re.sub(r"([Ww])on't", r"\1ill not", s)
        s = re.sub(r"([Tt]hey)'re", r"\1 are", s)
        s = re.sub(r"([Tt]hey)'ve", r"\1 have", s)
        s = re.sub(r"([Tt]hey)'ll", r"\1 will", s)
        s = re.sub(r"([Dd]oes)nt", r"\1 not", s)
        s = re.sub(r"([Ww]e)'ve", r"\1 have", s)
        s = re.sub(r"([Ii])'m", r"\1 am", s)
        s = re.sub(r" ([Ii])m ", r" \1 am ", s)
        s = re.sub(r"([Tt]hey)'d", r"\1 d", s)
        s = re.sub(r"([Gg]ot)ta", r"\1 to", s)
        s = re.sub(r"([Ww]e)\'re", r"\1 are", s)
        s = re.sub(r"([Ww]e)\'ll", r"\1 will", s)
        s = re.sub(r"([Ww]ere)n\'t", r"\1 not", s)
        s = re.sub(r"([Dd]oes)n\'t", r"\1 not", s)
        s = re.sub(r"([Cc]an)t", r"\1 not", s)
        s = re.sub(r"([Cc]an)\'t", r"\1 not", s)
        s = re.sub(r"([Cc]ould)\'ve", r"\1 have", s)
        s = re.sub(r"([Yy]ou)\'ve", r"\1 have", s)
        s = re.sub(r"([Yy]ou)\'re", r"\1 are", s)
        s = re.sub(r"([Gg])onna", r"\1oing to", s)
        s = re.sub(r"([Nn]eed)n\'t", r"\1 not", s)
        s = re.sub(r"([Hh]ad)n\'t", r"\1 not", s)
        s = re.sub(r"([Ww]ould)\'ve", r"\1 have", s)
        s = re.sub(r"([Ww]ould)n\'t", r"\1 not", s)
        s = re.sub(r"([Ww]ho)\'ve", r"\1 have", s)
        s = re.sub(r"([Hh]ave)n\'t", r"\1 not", s)
        for key, value in map.items():
            s = s.replace(key, value)
        return f_holder, s

