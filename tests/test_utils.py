from tf_nre.utils import *


def test_parse_text():
    text = "The <e1>child</e1> was carefully wrapped and bound into the <e2>cradle</e2> by means of a cord."
    e1_pos, e2_pos, clean_text = parse_text(text)
    assert e1_pos == [1]
    assert e2_pos == [9]
    print(clean_text)
    assert clean_text == "the child was carefully wrapped and bound into the cradle by means of a cord ."

    text = "The <e1>child was</e1> carefully wrapped and bound into the <e2>cradle by</e2> means of a cord."
    e1_pos, e2_pos, clean_text = parse_text(text)
    assert e1_pos == [1, 2]
    assert e2_pos == [9, 10]
    print(clean_text)
    assert clean_text == "the child was carefully wrapped and bound into the cradle by means of a cord ."


def test_parse_raw_text():
    line = '12	"Their <e1>composer</e1> has sunk into <e2>oblivion</e2>."'
    text = parse_raw_text(line)
    assert text == 'Their <e1>composer</e1> has sunk into <e2>oblivion</e2>.'
    line = '8002	"The <e1>company</e1> fabricates plastic <e2>chairs</e2>."'
    text = parse_raw_text(line)
    assert text == 'The <e1>company</e1> fabricates plastic <e2>chairs</e2>.'


def test_split_token_punctuation():
    token = "('word')"
    tokens = split_token_punctuation(token)
    assert tokens == ['(', "'", 'word', "'", ')']

    token = "<e1>word</e1>."
    tokens = split_token_punctuation(token)
    assert tokens == ['<e1>word</e1>', '.']

    token = "<e1>word</e1>'s"
    tokens = split_token_punctuation(token)
    assert tokens == ["<e1>word</e1>'s"]

    token = "word"
    tokens = split_token_punctuation(token)
    assert tokens == ["word"]

    token = "US$11.508"
    tokens = split_token_punctuation(token)
    assert tokens == ["US", "$", "[NUM]"]

    token = "$11.00."
    tokens = split_token_punctuation(token)
    assert tokens == ["$", "[NUM]", "."]

    token = "200"
    tokens = split_token_punctuation(token)
    assert tokens == ["[NUM]"]

    token = "200."
    tokens = split_token_punctuation(token)
    assert tokens == ["[NUM]", '.']

    token = "11.00."
    tokens = split_token_punctuation(token)
    assert tokens == ["[NUM]", "."]

    token = 'v8'
    tokens = split_token_punctuation(token)
    assert tokens == ['v', '[NUM]']


def test_text2tokens():
    text = "As many as 18 products in your home come from this <e1>guy</e1>'s <e2>company</e2>."
    tokens = text2tokens(text)
    assert tokens == ["As", "many", "as", "[NUM]", "products", "in", "your", "home", "come", "from", "this",
                      "<e1>guy</e1>", "'s", "<e2>company</e2>", "."]
