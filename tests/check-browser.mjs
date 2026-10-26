import fs from 'node:fs';import assert from 'node:assert/strict';import{predictRow,encodeFields}from'../dist/engine.mjs';
const model=JSON.parse(fs.readFileSync(new URL('../dist/baseline-model.json',import.meta.url),'utf8'));
const fixture=JSON.parse(fs.readFileSync(new URL('../dist/baseline-fixtures.json',import.meta.url),'utf8'));
let error=0;for(let i=0;i<fixture.rows.length;i++)error=Math.max(error,Math.abs(predictRow(fixture.rows[i],model)-fixture.expected[i]));assert.ok(error<1e-6,`Price parity error ${error}`);
const unknown=encodeFields({manufacturer:'unseen-brand',model:'unseen-model',year:2015,odometer:95000},model);assert.equal(unknown[model.features.indexOf('manufacturer')],-1);assert.ok(Number.isFinite(predictRow(unknown,model)));
console.log(`Passed: ${fixture.rows.length} native/browser parity rows, maximum USD difference ${error}; unseen categories handled.`);
