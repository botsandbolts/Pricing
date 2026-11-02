/* Run after installing playwright and axe-core: node tests/accessibility_browser.cjs */
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
(async () => {
  const server = spawn('python', ['-m','streamlit','run','tests/accessibility_fixture.py','--server.port','8520','--server.headless','true'], {cwd:root, stdio:'ignore'});
  let browser;
  try {
    browser = await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined, args:['--no-zygote','--single-process']});
    const page = await browser.newPage({viewport:{width:1440,height:1050}});
    await page.route('**/*', route => route.request().url().startsWith('http://localhost:8520') ? route.continue() : route.abort());
    // Wait for the local server without visiting any third-party service.
    for (let attempt=0; attempt<30; attempt++) {
      try { const response=await page.request.get('http://localhost:8520'); if(response.ok()) break; }
      catch (err) { if(attempt===29) throw err; }
      await new Promise(resolve=>setTimeout(resolve,100));
    }
    await page.goto('http://localhost:8520');
    await page.getByText('Choose a panel',{exact:true}).waitFor();
    await page.addScriptTag({path:require.resolve('axe-core')});
    async function audit() {
      const violations=await page.evaluate(async()=> (await axe.run(document.querySelector('[data-testid="stMain"]'),{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa','wcag22aa']}})).violations);
      assert.deepEqual(violations.map(v=>v.id),[],JSON.stringify(violations));
    }
    await audit();
    await page.locator('.garage-skip-link').focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.evaluate(()=>document.activeElement.id),'studio-workspace');
    await page.getByRole('radio',{name:'Pick your garage',exact:true}).focus();
    await page.keyboard.press('Space');
    await page.locator('.garage-car-art').nth(9).waitFor();
    const names=await page.getByRole('radiogroup').filter({has:page.getByRole('radio',{name:/Love it/})}).evaluateAll(groups=>groups.map(group=>group.getAttribute('aria-label')||document.getElementById(group.getAttribute('aria-labelledby')).textContent));
    assert.equal(new Set(names).size,10);
    await page.getByRole('radio',{name:/Love it/}).first().focus();
    await page.keyboard.press('Space');
    assert.equal(await page.getByRole('radio',{name:/Love it/}).first().getAttribute('aria-checked'),'true');
    await audit();
    async function noOverflow() {
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    }
    for(const width of [640,320]) { await page.setViewportSize({width,height:900}); await noOverflow(); }
    await page.getByText('Reading options',{exact:true}).click();
    await page.getByText('Larger text',{exact:true}).click();
    await page.waitForFunction(()=>getComputedStyle(document.documentElement).fontSize==='18px');
    await page.getByText('Higher contrast',{exact:true}).click();
    await page.waitForFunction(()=>getComputedStyle(document.querySelector('[data-testid="stAppViewContainer"]')).backgroundColor==='rgb(255, 253, 245)');
    await noOverflow(); await audit();
    await page.emulateMedia({reducedMotion:'reduce'});
    assert.equal(await page.getByRole('button',{name:/Choose Speed Date/}).evaluate(el=>getComputedStyle(el).transitionDuration),'0s');
    await page.emulateMedia({forcedColors:'active'});
    assert.equal(await page.getByRole('radio',{name:/Love it/}).first().evaluate(el=>getComputedStyle(el).outlineStyle),'solid');
    console.log('PASS: keyboard, unique labels, narrow reflow, reading preferences, motion, forced colors and axe scans.');
  } finally { if(browser) await browser.close(); server.kill(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
