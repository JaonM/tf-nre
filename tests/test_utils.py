from tf_nre.utils import *


def test_parse_text():
    text = "The <e1>child</e1> was carefully wrapped and bound into the <e2>cradle</e2> by means of a cord."
    e1_pos, e2_pos, clean_text = parse_text(text)
    assert e1_pos == 1
    assert e2_pos == 9
    print(clean_text)
    assert clean_text == "The child was carefully wrapped and bound into the cradle by means of a cord."


def test_parse_raw_text():
    line = '12	"Their <e1>composer</e1> has sunk into <e2>oblivion</e2>."'
    text = parse_raw_text(line)
    assert text == 'Their <e1>composer</e1> has sunk into <e2>oblivion</e2>.'
    line = '8002	"The <e1>company</e1> fabricates plastic <e2>chairs</e2>."'
    text = parse_raw_text(line)
    assert text == 'The <e1>company</e1> fabricates plastic <e2>chairs</e2>.'


def test_split_token_punctuation():
    token = "('word')"
    rexp = re.compile(WORD_PUNC_RE)
    tokens = split_token_punctuation(rexp, token)
    assert tokens == ['(', "'", 'word', "'", ')']

    token = "<e1>word</e1>."
    tokens = split_token_punctuation(rexp, token)
    assert tokens == ['<e1>word</e1>', '.']

    token = "<e1>word</e1>'s"
    tokens = split_token_punctuation(rexp, token)
    assert tokens == ["<e1>word</e1>'s"]

    token = "word"
    tokens = split_token_punctuation(rexp, token)
    assert tokens == ["word"]


def test_text2tokens():
    text = "As many as 18 products in your home come from this <e1>guy</e1>'s <e2>company</e2>."
    tokens = text2tokens(text)
    assert tokens == ["As", "many", "as", "18", "products", "in", "your", "home", "come", "from", "this",
                      "<e1>guy</e1>", "'s", "<e2>company</e2>", "."]
