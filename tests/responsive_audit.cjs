/* Free reproducible UI audit. npm install --no-save playwright axe-core
 * PLAYWRIGHT_CHROMIUM_EXECUTABLE=/path/to/chrome node tests/responsive_audit.cjs
 * Fixture shares production UI; model responses are deliberately stubbed. */
const {chromium,firefox,webkit}=require('playwright');
const engine=process.env.AUDIT_ENGINE||'chromium';
const {spawn}=require('node:child_process');
const fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const app=process.env.AUDIT_APP||'tests/accessibility_fixture.py';
const suffix=app==='app.py'?'-app':'';
if(!['chromium','firefox','webkit'].includes(engine))throw new Error('Use chromium, firefox or webkit');
const sizes=[[320,568],[360,800],[375,667],[390,844],[412,915],[568,320],[667,375],[768,1024],[820,1180],[1024,768],[1280,720],[1366,768],[1440,900],[1536,864],[1920,1080],[2560,1440],[3840,2160]];
(async()=>{
 const server=spawn('python',['-m','streamlit','run',app,'--server.port','8523','--server.headless','true','--browser.gatherUsageStats','false'],{cwd:root,stdio:'ignore'});
 let browser;const report={date:'2026-11-02',sourceNote:'Base commit plus the working-tree UI changes delivered with this report.',source:require('node:child_process').execSync('git rev-parse HEAD',{cwd:root}).toString().trim(),scope:app==='app.py'?'Locally served production app, real installed catalogue; initial calculator and 10/25 rating forms. Model execution and completed recommendations not tested.':'Production shared UI on local fixture; illustrative car data and stubbed price responses. Not a full deployed-site or model-functionality audit.',tools:{playwright:require('playwright/package.json').version,axe:require('axe-core/package.json').version},screens:[],scans:[],checks:[]};
 try{
 const launchOptions=engine==='chromium'?{executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE||undefined}:{};
 // This opt-in affects only the isolated local test process; network routes below block external sites.
 if(engine==='firefox'&&process.env.AUDIT_FIREFOX_LOCAL_HOST==='1'){
   launchOptions.firefoxUserPrefs={'security.sandbox.content.level':0,'security.sandbox.gpu.level':0};
   launchOptions.env={...process.env,MOZ_DISABLE_CONTENT_SANDBOX:'1',MOZ_DISABLE_RDD_SANDBOX:'1',MOZ_DISABLE_GMP_SANDBOX:'1',MOZ_DISABLE_GPU_SANDBOX:'1'};
   report.hostNotes='Firefox compatibility settings applied only to this isolated local test process.';
 }
 browser=await ({chromium,firefox,webkit}[engine]).launch(launchOptions); report.browser=browser.version();
 const page=await browser.newPage({viewport:{width:1440,height:900},hasTouch:engine!=='firefox'});
 await page.route('**/*',r=>r.request().url().startsWith('http://localhost:8523')?r.continue():r.abort());
 for(let i=0;i<80;i++){try{const r=await page.request.get('http://localhost:8523');if(r.ok())break}catch(e){}await new Promise(r=>setTimeout(r,100))}
 await page.goto('http://localhost:8523');await page.getByText('Choose a panel',{exact:true}).waitFor();
 async function scan(state){await page.addScriptTag({path:require.resolve('axe-core')});const result=await page.evaluate(async()=>await axe.run(document.querySelector('[data-testid="stMain"]'),{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa','wcag22aa']}})); report.scans.push({state,violations:result.violations,incomplete:result.incomplete,passedRules:result.passes.map(x=>x.id)});}
 async function matrix(state){for(const [width,height] of sizes){await page.setViewportSize({width,height});const geometry=await page.evaluate(()=>({viewport:innerWidth,document:document.documentElement.scrollWidth,overflow:[...document.querySelectorAll('button,input,textarea,[role="radio"]')].filter(e=>{const b=e.getBoundingClientRect();return b.width>0&&(b.left< -1||b.right>innerWidth+1)}).map(e=>({text:e.getAttribute('aria-label')||e.textContent?.slice(0,80),rect:e.getBoundingClientRect().toJSON()}))}));report.screens.push({state,width,height,...geometry,pass:geometry.document<=geometry.viewport&&geometry.overflow.length===0});}}
 await scan(app==='app.py'?'calculator entry':'calculator fixture');await matrix(app==='app.py'?'calculator entry':'calculator fixture');
 await page.setViewportSize({width:390,height:844});await page.getByRole('radio',{name:'Pick your garage',exact:true}).click();await page.locator('.garage-car-art').nth(9).waitFor();
 await page.getByRole('radio',{name:/Love it/}).first().click();report.checks.push({name:engine==='firefox'?'Pointer rating selection':'Rating selection in touch-capable context',pass:await page.getByRole('radio',{name:/Love it/}).first().getAttribute('aria-checked')==='true'});
 await scan('10-car garage');await matrix('10-car garage');
 await page.getByRole('button',{name:/Choose The Long Drive/}).click();await page.locator('.garage-car-art').nth(24).waitFor();await scan('25-car garage');await matrix('25-car garage');
 await page.getByText('Reading options',{exact:true}).click();await page.getByText('Larger text',{exact:true}).click();await page.getByText('Higher contrast',{exact:true}).click();await page.waitForFunction(()=>getComputedStyle(document.documentElement).fontSize==='18px');await scan('25-car garage: larger text and higher contrast');await matrix('25-car garage: larger text and higher contrast');
 // Double the CSS base font to exercise text reflow; this is not browser zoom.
 await page.addStyleTag({content:'html{font-size:200%!important}'});await matrix('25-car garage: 200% base text');await scan('25-car garage: 200% base text');
 }catch(error){report.error=error.message;throw error;}finally{if(browser)await browser.close();server.kill();fs.mkdirSync(path.join(root,'reports'),{recursive:true});fs.writeFileSync(path.join(root,'reports/accessibility-responsive-'+engine+suffix+'.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({screens:report.screens.length,failed:report.screens.filter(x=>!x.pass),axe:report.scans.map(x=>({state:x.state,violations:x.violations.map(y=>y.id),incomplete:x.incomplete.map(y=>y.id)})),checks:report.checks},null,2));if(report.error||report.screens.some(x=>!x.pass)||report.scans.some(x=>x.violations.length)||report.checks.some(x=>!x.pass))process.exitCode=1;}
})().catch(e=>{console.error(e);process.exitCode=1});
