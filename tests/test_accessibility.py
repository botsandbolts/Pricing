"""Reading preferences, semantic feedback and default-palette regressions."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def contrast(a,b):
    def luminance(h):
        rgb=[int(h[i:i+2],16)/255 for i in (1,3,5)]
        rgb=[c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in rgb]
        return sum(c*w for c,w in zip(rgb,(.2126,.7152,.0722)))
    x,y=sorted((luminance(a),luminance(b)))
    return (y+.05)/(x+.05)


@pytest.mark.parametrize('ink,paper',[('#52695f','#faf8f1'),('#52695f','#edf2e6'),('#52695f','#fcf0e5'),('#52695f','#eaf1e4'),('#ffffff','#207b65'),('#fffdf5','#203e36'),('#203e36','#fffdf8')])
def test_shared_text_palette_meets_aa_contrast(ink,paper):
    assert contrast(ink,paper)>=4.5


def test_controls_and_keyboard_ring_have_non_text_contrast():
    assert contrast('#75877d','#fffefb')>=3
    assert contrast('#203e36','#faf8f1')>=3


def test_reading_settings_survive_reruns_and_status_is_escaped():
    at=AppTest.from_string("from pricing.accessibility import install,reading_options,announce\ninstall()\nreading_options()\nannounce('<not markup>')").run()
    assert not at.exception
    at.checkbox(key='a11y_larger_text').check().run()
    at.checkbox(key='a11y_high_contrast').check().run()
    at.run()
    assert not at.exception
    assert at.checkbox(key='a11y_larger_text').value
    assert at.checkbox(key='a11y_high_contrast').value
    html='\n'.join(m.value for m in at.markdown)
    assert 'font-size:112.5%' in html and 'role="status"' in html
    assert '&lt;not markup&gt;' in html and 'Skip to panels and tools' in html
