"""
module for testing the functions clean_text() and default_text_pipeline()
"""

import re
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_text, default_text_pipeline

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_text() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "text": [
                "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
                "The cast played Shakespeare.<br /><br />Shakespeare lost.",
                "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
                "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
                "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
                "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
                "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
                "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
                123,
                np.nan,
                "NULL",
            ]
        }
    )
    return df


def test_clean_default(df_text: pd.DataFrame) -> None:
    df_clean = clean_text(df_text, "text")
    df_check = df_text.copy()
    df_check["text"] = [
        "zzzzz imdb would allow one word reviews mine would",
        "cast played shakespeare shakespeare lost",
        "simon desert simon del desierto film directed luis bunuel",
        "spoilers think seen film bad acting script effects etc",
        "cannes video essay",
        "recap thread rottentomatoes excellent panel hosted erikdavis filmfatale nyc ashcrossan",
        "gameofthrones season rotten tomatometer deserve",
        "come join share thoughts week episode",
        "",
        np.nan,
        np.nan,
    ]
    assert df_check.equals(df_clean)


def test_clean_custom(df_text: pd.DataFrame) -> None:
    pipeline: List[Dict[str, Any]] = [
        {"operator": "lowercase"},
        {"operator": "remove_html"},
        {
            "operator": "replace_bracketed",
            "parameters": {"brackets": "square", "value": "**spoilers**"},
        },
        {
            "operator": "replace_bracketed",
            "parameters": {"brackets": "curly", "value": "in every aspect"},
        },
        {"operator": "remove_whitespace"},
    ]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'zzzzz!' if imdb would allow one-word reviews, that's what mine would be.",
        "the cast played shakespeare.shakespeare lost.",
        "simon of the desert (simón del desierto) is a 1965 film directed by luis buñuel.",
        "**spoilers** i don't think i've seen a film this bad before in every aspect",
        "cannes 1968: a video essay",
        "recap thread for @rottentomatoes excellent panel, hosted by @erikdavis with @filmfatale_nyc and @ashcrossan.",
        "#gameofthrones: season 8 is #rotten at 54% on the #tomatometer. but does it deserve to be?",
        "come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeurl",
        "123",
        np.nan,
        "null",
    ]
    assert df_check.equals(df_clean)


def test_clean_user(df_text: pd.DataFrame) -> None:
    def swapcase(text: str) -> str:
        return str(text).swapcase()

    def replace_z(text: str, value: str) -> str:
        return re.sub(r"[zZ]", value, text)

    pipeline: List[Dict[str, Any]] = [
        {"operator": "fillna", "parameters": {"value": ""}},
        {"operator": swapcase},
        {"operator": replace_z, "parameters": {"value": "#"}},
        {"operator": "remove_digits"},
    ]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'#####!' iF imdB WOULD ALLOW ONE-WORD REVIEWS, THAT'S WHAT MINE WOULD BE.",
        "tHE CAST PLAYED sHAKESPEARE.<BR /><BR />sHAKESPEARE LOST.",
        "sIMON OF THE dESERT (sIMÓN DEL DESIERTO) IS A  FILM DIRECTED BY lUIS bUÑUEL.",
        "[spoilers]\ni DON'T THINK i'VE SEEN A FILM THIS BAD BEFORE {ACTING, SCRIPT, EFFECTS (!), ETC...}",
        "<A HREF='/FESTIVALS/CANNES--A-VIDEO-ESSAY'>cANNES :\ta VIDEO ESSAY</A>",
        "rECAP THREAD FOR @rOTTENtOMATOES EXCELLENT PANEL, HOSTED BY @eRIKdAVIS WITH @fILMfATALE_nyc AND @aSHcROSSAN.",
        "#gAMEoFtHRONES: sEASON  IS #rOTTEN AT % ON THE #tOMATOMETER.  bUT DOES IT DESERVE TO BE?",
        "cOME JOIN AND SHARE YOUR THOUGHTS ON THIS WEEK'S EPISODE: HTTPS://TWITTER.COM/I/SPACES/FAKEurl",
        "",
        "",
        "",
    ]
    assert df_check.equals(df_clean)


def test_clean_fillna(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "fillna"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        np.nan,
    ]
    pipeline_value = [{"operator": "fillna", "parameters": {"value": "<NAN>"}}]
    df_clean_value = clean_text(df_text, "text", pipeline=pipeline_value)
    df_check_value = df_text.copy()
    df_check_value["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        "<NAN>",
        "<NAN>",
    ]
    assert df_check.equals(df_clean)
    assert df_check_value.equals(df_clean_value)


def test_clean_lowercase(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "lowercase"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'zzzzz!' if imdb would allow one-word reviews, that's what mine would be.",
        "the cast played shakespeare.<br /><br />shakespeare lost.",
        "simon of the desert (simón del desierto) is a 1965 film directed by luis buñuel.",
        "[spoilers]\ni don't think i've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>cannes 1968:\ta video essay</a>",
        "recap thread for @rottentomatoes excellent panel, hosted by @erikdavis with @filmfatale_nyc and @ashcrossan.",
        "#gameofthrones: season 8 is #rotten at 54% on the #tomatometer.  but does it deserve to be?",
        "come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeurl",
        "123",
        np.nan,
        "null",
    ]
    assert df_check.equals(df_clean)


def test_clean_sentence_case(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "sentence_case"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'zzzzz!' if imdb would allow one-word reviews, that's what mine would be.",
        "The cast played shakespeare.<br /><br />shakespeare lost.",
        "Simon of the desert (simón del desierto) is a 1965 film directed by luis buñuel.",
        "[spoilers]\ni don't think i've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>cannes 1968:\ta video essay</a>",
        "Recap thread for @rottentomatoes excellent panel, hosted by @erikdavis with @filmfatale_nyc and @ashcrossan.",
        "#gameofthrones: season 8 is #rotten at 54% on the #tomatometer.  but does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeurl",
        "123",
        np.nan,
        "Null",
    ]
    assert df_check.equals(df_clean)


def test_clean_title_case(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "title_case"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'Zzzzz!' If Imdb Would Allow One-Word Reviews, That'S What Mine Would Be.",
        "The Cast Played Shakespeare.<Br /><Br />Shakespeare Lost.",
        "Simon Of The Desert (Simón Del Desierto) Is A 1965 Film Directed By Luis Buñuel.",
        "[Spoilers]\nI Don'T Think I'Ve Seen A Film This Bad Before {Acting, Script, Effects (!), Etc...}",
        "<A Href='/Festivals/Cannes-1968-A-Video-Essay'>Cannes 1968:\tA Video Essay</A>",
        "Recap Thread For @Rottentomatoes Excellent Panel, Hosted By @Erikdavis With @Filmfatale_Nyc And @Ashcrossan.",
        "#Gameofthrones: Season 8 Is #Rotten At 54% On The #Tomatometer.  But Does It Deserve To Be?",
        "Come Join And Share Your Thoughts On This Week'S Episode: Https://Twitter.Com/I/Spaces/1Fakeurl",
        "123",
        np.nan,
        "Null",
    ]
    assert df_check.equals(df_clean)


def test_clean_uppercase(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "uppercase"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' IF IMDB WOULD ALLOW ONE-WORD REVIEWS, THAT'S WHAT MINE WOULD BE.",
        "THE CAST PLAYED SHAKESPEARE.<BR /><BR />SHAKESPEARE LOST.",
        "SIMON OF THE DESERT (SIMÓN DEL DESIERTO) IS A 1965 FILM DIRECTED BY LUIS BUÑUEL.",
        "[SPOILERS]\nI DON'T THINK I'VE SEEN A FILM THIS BAD BEFORE {ACTING, SCRIPT, EFFECTS (!), ETC...}",
        "<A HREF='/FESTIVALS/CANNES-1968-A-VIDEO-ESSAY'>CANNES 1968:\tA VIDEO ESSAY</A>",
        "RECAP THREAD FOR @ROTTENTOMATOES EXCELLENT PANEL, HOSTED BY @ERIKDAVIS WITH @FILMFATALE_NYC AND @ASHCROSSAN.",
        "#GAMEOFTHRONES: SEASON 8 IS #ROTTEN AT 54% ON THE #TOMATOMETER.  BUT DOES IT DESERVE TO BE?",
        "COME JOIN AND SHARE YOUR THOUGHTS ON THIS WEEK'S EPISODE: HTTPS://TWITTER.COM/I/SPACES/1FAKEURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_clean_remove_accents(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "remove_accents"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simon del desierto) is a 1965 film directed by Luis Bunuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_clean_remove_bracketed(df_text: pd.DataFrame) -> None:
    pipeline_all = [
        {
            "operator": "remove_bracketed",
            "parameters": {"brackets": {"angle", "curly", "round", "square"}},
        }
    ]
    df_clean_all = clean_text(df_text, "text", pipeline=pipeline_all)
    df_check_all = df_text.copy()
    df_check_all["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.Shakespeare lost.",
        "Simon of the Desert  is a 1965 film directed by Luis Buñuel.",
        "\nI don't think I've seen a film this bad before ",
        "Cannes 1968:\tA video essay",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_all_excl = [
        {
            "operator": "remove_bracketed",
            "parameters": {"brackets": {"angle", "curly", "round", "square"}, "inclusive": False},
        }
    ]
    df_clean_all_excl = clean_text(df_text, "text", pipeline=pipeline_all_excl)
    df_check_all_excl = df_text.copy()
    df_check_all_excl["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<><>Shakespeare lost.",
        "Simon of the Desert () is a 1965 film directed by Luis Buñuel.",
        "[]\nI don't think I've seen a film this bad before {}",
        "<>Cannes 1968:\tA video essay<>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_square = [{"operator": "remove_bracketed", "parameters": {"brackets": "square"}}]
    df_clean_square = clean_text(df_text, "text", pipeline=pipeline_square)
    df_check_square = df_text.copy()
    df_check_square["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check_all.equals(df_clean_all)
    assert df_check_all_excl.equals(df_clean_all_excl)
    assert df_check_square.equals(df_clean_square)


def test_clean_remove_digits(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "remove_digits"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a  film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes--a-video-essay'>Cannes :\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season  is #Rotten at % on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/fakeURL",
        "",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_clean_remove_html(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "remove_html"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "Cannes 1968:\tA video essay",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_clean_remove_prefixed(df_text: pd.DataFrame) -> None:
    pipeline_hashtag = [{"operator": "remove_prefixed", "parameters": {"prefix": "#"}}]
    df_clean_hashtag = clean_text(df_text, "text", pipeline=pipeline_hashtag)
    df_check_hashtag = df_text.copy()
    df_check_hashtag["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        " Season 8 is  at 54% on the   But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_hashtag_mention = [
        {"operator": "remove_prefixed", "parameters": {"prefix": {"#", "@"}}}
    ]
    df_clean_hashtag_mention = clean_text(df_text, "text", pipeline=pipeline_hashtag_mention)
    df_check_hashtag_mention = df_text.copy()
    df_check_hashtag_mention["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for  excellent panel, hosted by  with  and ",
        " Season 8 is  at 54% on the   But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check_hashtag.equals(df_clean_hashtag)
    assert df_check_hashtag_mention.equals(df_clean_hashtag_mention)


def test_clean_remove_punctuation(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "remove_punctuation"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        " ZZZZZ   If IMDb would allow one word reviews  that s what mine would be ",
        "The cast played Shakespeare  br    br   Shakespeare lost ",
        "Simon of the Desert  Simón del desierto  is a 1965 film directed by Luis Buñuel ",
        " SPOILERS \nI don t think I ve seen a film this bad before  acting  script  effects      etc    ",
        " a href   festivals cannes 1968 a video essay  Cannes 1968 \tA video essay  a ",
        "Recap thread for  RottenTomatoes excellent panel  hosted by  ErikDavis with  FilmFatale NYC and  AshCrossan ",
        " GameOfThrones  Season 8 is  Rotten at 54  on the  Tomatometer   But does it deserve to be ",
        "Come join and share your thoughts on this week s episode  https   twitter com i spaces 1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_clean_remove_stopwords(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "remove_stopwords"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' IMDb would allow one-word reviews, that's mine would be.",
        "cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon Desert (Simón del desierto) 1965 film directed Luis Buñuel.",
        "[SPOILERS] think I've seen film bad {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968: video essay</a>",
        "Recap thread @RottenTomatoes excellent panel, hosted @ErikDavis @FilmFatale_NYC @AshCrossan.",
        "#GameOfThrones: Season 8 #Rotten 54% #Tomatometer. deserve be?",
        "Come join share thoughts week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_custom = [
        {"operator": "remove_stopwords", "parameters": {"stopwords": {"imdb", "film"}}}
    ]
    df_clean_custom = clean_text(df_text, "text", pipeline=pipeline_custom)
    df_check_custom = df_text.copy()
    df_check_custom["text"] = [
        "'ZZZZZ!' If would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 directed by Luis Buñuel.",
        "[SPOILERS] I don't think I've seen a this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968: A video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer. But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)
    assert df_check_custom.equals(df_clean_custom)


def test_clean_remove_urls(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "remove_urls"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: ",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_clean_remove_whitespace(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "remove_whitespace"}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS] I don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968: A video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer. But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_clean_replace_bracketed(df_text: pd.DataFrame) -> None:
    pipeline_all = [
        {
            "operator": "replace_bracketed",
            "parameters": {
                "brackets": {"angle", "curly", "round", "square"},
                "value": "<REDACTED>",
            },
        }
    ]
    df_clean_all = clean_text(df_text, "text", pipeline=pipeline_all)
    df_check_all = df_text.copy()
    df_check_all["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<REDACTED><REDACTED>Shakespeare lost.",
        "Simon of the Desert <REDACTED> is a 1965 film directed by Luis Buñuel.",
        "<REDACTED>\nI don't think I've seen a film this bad before <REDACTED>",
        "<REDACTED>Cannes 1968:\tA video essay<REDACTED>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_square = [
        {
            "operator": "replace_bracketed",
            "parameters": {"brackets": "square", "value": "**SPOILERS**"},
        }
    ]
    df_clean_square = clean_text(df_text, "text", pipeline=pipeline_square)
    df_check_square = df_text.copy()
    df_check_square["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "**SPOILERS**\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_square_excl = [
        {
            "operator": "replace_bracketed",
            "parameters": {"brackets": "square", "value": "SPOILER WARNING", "inclusive": False},
        }
    ]
    df_clean_square_excl = clean_text(df_text, "text", pipeline=pipeline_square_excl)
    df_check_square_excl = df_text.copy()
    df_check_square_excl["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILER WARNING]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check_all.equals(df_clean_all)
    assert df_check_square.equals(df_clean_square)
    assert df_check_square_excl.equals(df_clean_square_excl)


def test_clean_replace_digits(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "replace_digits", "parameters": {"value": "X"}}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a X film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-X-a-video-essay'>Cannes X:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season X is #Rotten at X% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "X",
        np.nan,
        "NULL",
    ]
    pipeline_no_block = [
        {"operator": "replace_digits", "parameters": {"value": "X", "block": False}}
    ]
    df_clean_no_block = clean_text(df_text, "text", pipeline=pipeline_no_block)
    df_check_no_block = df_text.copy()
    df_check_no_block["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a X film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-X-a-video-essay'>Cannes X:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season X is #Rotten at X% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/XfakeURL",
        "X",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)
    assert df_check_no_block.equals(df_clean_no_block)


def test_clean_replace_prefixed(df_text: pd.DataFrame) -> None:
    pipeline_hashtag = [
        {"operator": "replace_prefixed", "parameters": {"prefix": "#", "value": "<HASHTAG>"}}
    ]
    df_clean_hashtag = clean_text(df_text, "text", pipeline=pipeline_hashtag)
    df_check_hashtag = df_text.copy()
    df_check_hashtag["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "<HASHTAG> Season 8 is <HASHTAG> at 54% on the <HASHTAG>  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_hashtag_mention = [
        {"operator": "replace_prefixed", "parameters": {"prefix": {"#", "@"}, "value": "<TAG>"}}
    ]
    df_clean_hashtag_mention = clean_text(df_text, "text", pipeline=pipeline_hashtag_mention)
    df_check_hashtag_mention = df_text.copy()
    df_check_hashtag_mention["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for <TAG> excellent panel, hosted by <TAG> with <TAG> and <TAG>",
        "<TAG> Season 8 is <TAG> at 54% on the <TAG>  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check_hashtag.equals(df_clean_hashtag)
    assert df_check_hashtag_mention.equals(df_clean_hashtag_mention)


def test_clean_replace_punctuation(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "replace_punctuation", "parameters": {"value": "*"}}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "*ZZZZZ** If IMDb would allow one*word reviews* that*s what mine would be*",
        "The cast played Shakespeare**br ***br **Shakespeare lost*",
        "Simon of the Desert *Simón del desierto* is a 1965 film directed by Luis Buñuel*",
        "*SPOILERS*\nI don*t think I*ve seen a film this bad before *acting* script* effects **** etc****",
        "*a href***festivals*cannes*1968*a*video*essay**Cannes 1968*\tA video essay**a*",
        "Recap thread for *RottenTomatoes excellent panel* hosted by *ErikDavis with *FilmFatale*NYC and *AshCrossan*",
        "*GameOfThrones* Season 8 is *Rotten at 54* on the *Tomatometer*  But does it deserve to be*",
        "Come join and share your thoughts on this week*s episode* https***twitter*com*i*spaces*1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_clean_replace_stopwords(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "replace_stopwords", "parameters": {"value": "<S>"}}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' <S> IMDb would allow one-word reviews, that's <S> mine would be.",
        "<S> cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon <S> <S> Desert (Simón del desierto) <S> <S> 1965 film directed <S> Luis Buñuel.",
        "[SPOILERS] <S> <S> think I've seen <S> film <S> bad <S> {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968: <S> video essay</a>",
        "Recap thread <S> @RottenTomatoes excellent panel, hosted <S> @ErikDavis <S> @FilmFatale_NYC <S> @AshCrossan.",
        "#GameOfThrones: Season 8 <S> #Rotten <S> 54% <S> <S> #Tomatometer. <S> <S> <S> deserve <S> be?",
        "Come join <S> share <S> thoughts <S> <S> week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_custom = [
        {
            "operator": "replace_stopwords",
            "parameters": {"stopwords": {"imdb", "film"}, "value": "<S>"},
        }
    ]
    df_clean_custom = clean_text(df_text, "text", pipeline=pipeline_custom)
    df_check_custom = df_text.copy()
    df_check_custom["text"] = [
        "'ZZZZZ!' If <S> would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 <S> directed by Luis Buñuel.",
        "[SPOILERS] I don't think I've seen a <S> this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968: A video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer. But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)
    assert df_check_custom.equals(df_clean_custom)


def test_clean_replace_text(df_text: pd.DataFrame) -> None:
    pipeline = [
        {"operator": "replace_text", "parameters": {"value": {"imdb": "Netflix", "film": "movie"}}}
    ]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If Netflix would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 movie directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a movie this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    pipeline_no_block = [
        {
            "operator": "replace_text",
            "parameters": {"value": {"imdb": "Netflix", "film": "movie"}, "block": False},
        }
    ]
    df_clean_no_block = clean_text(df_text, "text", pipeline=pipeline_no_block)
    df_check_no_block = df_text.copy()
    df_check_no_block["text"] = [
        "'ZZZZZ!' If Netflix would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 movie directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a movie this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @movieFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: https://twitter.com/i/spaces/1fakeURL",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)
    assert df_check_no_block.equals(df_clean_no_block)


def test_clean_replace_urls(df_text: pd.DataFrame) -> None:
    pipeline = [{"operator": "replace_urls", "parameters": {"value": "<URL>"}}]
    df_clean = clean_text(df_text, "text", pipeline=pipeline)
    df_check = df_text.copy()
    df_check["text"] = [
        "'ZZZZZ!' If IMDb would allow one-word reviews, that's what mine would be.",
        "The cast played Shakespeare.<br /><br />Shakespeare lost.",
        "Simon of the Desert (Simón del desierto) is a 1965 film directed by Luis Buñuel.",
        "[SPOILERS]\nI don't think I've seen a film this bad before {acting, script, effects (!), etc...}",
        "<a href='/festivals/cannes-1968-a-video-essay'>Cannes 1968:\tA video essay</a>",
        "Recap thread for @RottenTomatoes excellent panel, hosted by @ErikDavis with @FilmFatale_NYC and @AshCrossan.",
        "#GameOfThrones: Season 8 is #Rotten at 54% on the #Tomatometer.  But does it deserve to be?",
        "Come join and share your thoughts on this week's episode: <URL>",
        "123",
        np.nan,
        "NULL",
    ]
    assert df_check.equals(df_clean)


def test_default_text_pipeline() -> None:
    pipeline_default = [
        {"operator": "fillna"},
        {"operator": "lowercase"},
        {"operator": "remove_digits"},
        {"operator": "remove_html"},
        {"operator": "remove_urls"},
        {"operator": "remove_punctuation"},
        {"operator": "remove_accents"},
        {"operator": "remove_stopwords", "parameters": {"stopwords": None}},
        {"operator": "remove_whitespace"},
    ]
    pipeline_check = default_text_pipeline()
    assert pipeline_check == pipeline_default
