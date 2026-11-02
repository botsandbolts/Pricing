# Accessibility & browser test report

**Test date: 2 November 2026 · Free tools: Playwright, axe-core and pytest.**

Car Price Studio has tested accessibility features and measured responsive behaviour. This report is evidence for the stated checks, not an accessibility certificate or a promise that every browser, device or workflow works perfectly.

## Measured results

| Check | Result | Scope |
|---|---|---|
| Responsive layout | **255 / 255 configurations passed** | 85 each in Chromium, Firefox and WebKit. Locally served production app: calculator entry, 10-car and 25-car rating forms, larger text with higher contrast, and 200% base text |
| Automated accessibility | **0 detected WCAG-tagged violations in 15 scans** | axe-core inside the main app content; some colour-contrast checks remain incomplete and need manual assessment |
| Keyboard and reading controls | **Passed** | Separate shared-UI fixture: skip-link focus, keyboard rating selection, unique rating labels, reading preferences, reduced motion and forced colours |
| Accessibility regression tests | **9 passed** | Palette contrast calculations and reading-preference persistence |
| Rating interaction | **Passed** | Production app rating selection in Chromium, Firefox and WebKit contexts; the separate fixture suite also exercised keyboard selection |

The 17 tested CSS viewport sizes are: 320×568, 360×800, 375×667, 390×844, 412×915, 568×320, 667×375, 768×1024, 820×1180, 1024×768, 1280×720, 1366×768, 1440×900, 1536×864, 1920×1080, 2560×1440 and 3840×2160. Each ran in five UI states in each of three browser engines. Layout checks measure document overflow and whether form/button/radio bounds extend beyond the viewport. They do not prove every element is unobscured, every chart readable or every action functional.

A toolbar minimum width clipped framework controls at narrow widths with 200% base text. The CSS minimum was corrected and the configurations rerun successfully in Chromium, Firefox and WebKit. Increasing the CSS base font is a text-reflow test, not a real browser-zoom test.

## Browser coverage

| Browser | Status |
|---|---|
| Chrome | The public app loaded and exposed labelled fields, skip links and panel controls in Chrome's accessibility tree. This was a structural observation, not a full live-site audit. |
| Chromium 153 | Production app UI completed the automated layout and axe checks locally. This is browser-engine evidence, not testing every released Google Chrome version. |
| Firefox 153 | **85 / 85 layout checks passed; 5 axe scans with no detected violations.** Content processes initially crashed under the host restrictions. This isolated, local-only runner required Firefox sandbox compatibility settings; normal end-user configurations were not exhaustively tested. |
| Safari / WebKit | **WebKit 26.5: 85 / 85 layout checks, 5 axe scans with no detected violations.** Missing host libraries were supplied in the temporary test runtime. This is Safari-engine coverage; actual Apple Safari on macOS and iOS remains unverified. |
| Internet Explorer | No Windows/IE session was available. No IE compatibility claim is made; **not verified**. |

## Remaining work and limitations

- Resolve axe's incomplete contrast checks, including content it cannot assess automatically.
- Conduct real screen-reader testing with NVDA, VoiceOver and TalkBack, actual browser zoom, and physical phones/tablets.
- Complete actual Safari/iOS and any required Windows browser testing, and Firefox on normal end-user configurations.
- Audit generated estimates, completed recommendation lists, charts, dialogs, errors and every other panel. These are outside the 255-case matrix.
- The production audit used the installed real catalogue but did not execute model inference or generate recommendations. The separate keyboard suite uses illustrative data and stubbed prices.
- Automated checks cannot establish WCAG conformance. There has been no independent human accessibility certification.

## Reproduce the checks

Use Python dependencies from the project plus Playwright and axe-core from npm. Install a Playwright browser, then run:

```bash
python -m pytest tests/test_accessibility.py -q
node tests/accessibility_browser.cjs
AUDIT_APP=app.py node tests/responsive_audit.cjs
```

Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE` when using a separately installed Chromium executable. `AUDIT_ENGINE=firefox` selects Firefox; `AUDIT_ENGINE=webkit` selects WebKit for the matrix. On restricted hosts only, `AUDIT_FIREFOX_LOCAL_HOST=1` applies the isolated local-runner settings used in this report. This option is for the local test process, never a change users need to make to their browsers. The browser suite is also runnable on the lightweight shared-UI fixture by omitting `AUDIT_APP`. Network access outside the local test server is blocked during the matrix. Raw results include passed rules, unresolved checks and each viewport's geometry.

If you encounter an access problem, contact the author using the LinkedIn link in the footer and include the panel, browser, device and steps involved. Avoid sending personal vehicle details or private advert text.
