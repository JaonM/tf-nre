from tf_nre.utils import parse_text


def test_parse_text():
    text = "The <e1>child</e1> was carefully wrapped and bound into the <e2>cradle</e2> by means of a cord."
    e1_pos, e2_pos, clean_text = parse_text(text)
    assert e1_pos == 1
    assert e2_pos == 9
    print(clean_text)
    assert clean_text == "The child was carefully wrapped and bound into the cradle by means of a cord."
