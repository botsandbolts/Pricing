import fs from 'node:fs';
import assert from 'node:assert/strict';
import {clean,wordpieces,sparseText,textScore} from '../dist/engine.mjs';
const model=JSON.parse(fs.readFileSync(new URL('../dist/pricing-model.json',import.meta.url)));
const fixtures=JSON.parse(fs.readFileSync(new URL('../dist/parity-fixtures.json',import.meta.url)));
for(const f of fixtures){assert.deepEqual(wordpieces(f.text,model.vocab),f.ids,'Browser WordPiece tokens must match Python');const p=model.models.tfidf.intercept+textScore(sparseText(f.text,model),model.models.tfidf.coef);assert(Math.abs(p-f.tfidf_log)<2e-5,'Browser TF-IDF log prediction must match sklearn');}
assert(!clean('Asking $15,000. Payment 200 monthly.').includes('15,000'));
assert(clean('95000 miles, 2015 model.').includes('95000'));
assert(model.meta.max_characters>=1000);assert(model.meta.test_rows===1634);
const long=fixtures[0].text.repeat(5).slice(0,6500),ids=wordpieces(clean(long),model.vocab);assert(ids.length>254,'Long input must require multiple chunks');
for(const m of Object.values(model.models))assert(m.coef.every(Number.isFinite));
console.log(`Passed: ${fixtures.length} tokenizer and TF-IDF parity fixtures; money redaction; long-context coverage; model integrity.`);
