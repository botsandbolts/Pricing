"""Shared release/UI regression check using stable widget keys, not display copy."""
from pathlib import Path

def _healthy(app,phase):
    assert not app.exception,f'{phase}: '+ '; '.join(str(e.message) for e in app.exception)

def rating_controls(app,count):
    controls=[p for p in app.get('pills') if str(p.key).startswith('taste_rating_')]
    assert len(controls)==count,f'Expected {count} rating pills; found {len(controls)}'
    assert len({p.key for p in controls})==count,'Rating keys must be unique'
    return controls

def exercise_taste_ui(root,profile_count,ratings):
    from streamlit.testing.v1 import AppTest
    assert profile_count in (10,25) and len(ratings)==profile_count
    app=AppTest.from_string(f"from pathlib import Path\nfrom pricing.taste_garage_ui import render\nrender(Path({str(Path(root))!r}))").run(timeout=60)
    _healthy(app,'Initial render')
    assert any('Drive Chemistry' in m.value for m in app.markdown),'Branded introduction must be visible'
    assert any(e.label=='How Drive Chemistry picks your cars' for e in app.expander),'Plain-language guide must be available'
    buttons=[b for b in app.button if b.key==f'taste_choose_{profile_count}']
    assert len(buttons)==1,f'Missing test selector taste_choose_{profile_count}'
    buttons[0].click().run(timeout=60);_healthy(app,'Choose test')
    for control,value in zip(rating_controls(app,profile_count),ratings):control.set_value(value)
    app.run(timeout=60);_healthy(app,'Set ratings')
    assert [p.value for p in rating_controls(app,profile_count)]==list(ratings),'Ratings did not persist'
    submit=[b for b in app.button if b.key=='taste_train']
    assert len(submit)==1 and not submit[0].disabled,'Completed valid ratings must enable shortlist'
    submit[0].click().run(timeout=120);_healthy(app,'Build shortlist')
    assert 'taste_result' in app.session_state,'Submit did not produce a shortlist'
    _,result=app.session_state['taste_result']
    assert len(result['positions'])==10 and len(set(result['positions']))==10,'Expected ten distinct recommendations'
    assert result['direct_judgments']==profile_count,'Wrong test was trained'
    assert any('Your 10' in m.value for m in app.markdown),'Shortlist must be visible'
    return {'profile_count':profile_count,'rating_control':'pills','ratings_verified':profile_count,'recommendations':10,'test_selector_key':f'taste_choose_{profile_count}'}
