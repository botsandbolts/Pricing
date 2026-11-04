"""Conservative removal of explicit advertised money from description inputs."""
import re
MONEY=re.compile(r'(?i)(?:\$\s*\d[\d,]*(?:\.\d{1,2})?\s*[kK]?|\b\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|dollars)\b|\b(?:asking(?:\s+price)?|sale\s+price|price|priced\s+at)\s*[:=]?\s*\$?\s*\d[\d,]*(?:\.\d{1,2})?\s*[kK]?\b)')
def scrub(text):
    return MONEY.sub(' [money removed] ',str(text or ''))
