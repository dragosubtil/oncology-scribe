const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const {runInNewContext} = require('node:vm');
const source = readFileSync(resolve(__dirname,'../src/clinical_asr/assets/source-order.js'),'utf8');
const harness = () => runInNewContext(source,{window:{}});

test('unchanged source event preserves the running requests delivery order',()=>{
    const factory=harness(), start=factory('note',true), change=factory('note');
    const first=start(null,0,'No fever.','extractive','llama3.1:latest');
    const duplicate=change(null,0,'No fever.','extractive','llama3.1:latest');
    assert.equal(duplicate[1],first[1]);
});
test('a changed source invalidates the preceding delivery order',()=>{
    const factory=harness(), change=factory('note');
    const first=change(null,0,'No fever.');
    assert.ok(change(null,0,'Fever.')[1] > first[1]);
});
test('retry with identical inputs still has a new delivery order',()=>{
    const start=harness()('note',true);
    assert.ok(start(null,0,'same')[1] < start(null,0,'same')[1]);
});
test('unchanged audio objects compare by content rather than object identity',()=>{
    const change=harness()('asr');
    const first=change(null,0,{path:'a.wav',size:20},'en');
    assert.equal(change(null,0,{path:'a.wav',size:20},'en')[1],first[1]);
    assert.ok(change(null,0,{path:'b.wav',size:20},'en')[1] > first[1]);
});
test('task scopes keep their input snapshots separate',()=>{
    const factory=harness(), asr=factory('asr'), note=factory('note');
    const first=asr(null,0,'same');
    const second=note(null,0,'same');
    assert.notEqual(first[1],second[1]);
    assert.equal(asr(null,0,'same')[1],first[1]);
});
