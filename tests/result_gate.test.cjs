const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const {runInNewContext} = require('node:vm');
const source = readFileSync(resolve(__dirname,'../src/clinical_asr/assets/result-gate.js'),'utf8');

function harness(scope, order, corrections=false) {
    const window = {__oncologySourceOrders:{[scope]:order}};
    const document = {createElement() {return {
        set innerHTML(value) {this.value=value;},
        get content() {const value=this.value; return {querySelector() {
            if(typeof value !== 'string' || !value.includes('data-task-clock')) return null;
            const attr = name => value.match(new RegExp(`data-${name}="([^"]*)"`))?.[1];
            return {dataset:{eventOrder:attr('event-order'),taskKind:attr('task-kind'),status:attr('status')}};
        }};},
    };}};
    const factory = runInNewContext(source,{window,document});
    return factory(scope,corrections ? [0,1,3,4,5,7,8,9] : [0,1,2],corrections ? 2 : -1);
}
const stamp=(scope,order,status='completed')=>`<div data-task-clock data-task-kind="${scope}" data-event-order="${order}" data-status="${status}"></div>`;
const skips=values=>values.every(x=>x && x.__type__==='update' && !('value' in x));

test('delayed note is rejected after its source changes during postprocessing',()=>{
    const gate=harness('note',2);
    assert.equal(skips(gate(['old note',{raw_transcript:'old'},stamp('note',1)])),true);
});
test('current note is delivered without changing source evidence',()=>{
    const gate=harness('note',2), note={raw_transcript:'current'};
    const values=gate(['preview',note,stamp('note',2)]);
    assert.equal(values[0],'preview'); assert.equal(values[1],note);
});
test('one task cannot authorize another tasks output',()=>{
    assert.equal(skips(harness('asr',3)(['old',{},stamp('note',3)])),true);
});
test('failed request cannot republish retained mirror values',()=>{
    const result=harness('note',3)(['retained old note',{},stamp('note',3,'failed')]);
    assert.equal(skips(result.slice(0,-1)),true); assert.equal(result.at(-1),stamp('note',3,'failed'));
});
test('missing or malformed stamps fail closed',()=>{
    for(const value of [null,'',{},'<div data-task-clock></div>']) {
        assert.equal(skips(harness('note',3)(['old',{},value])),true);
    }
});
test('eligible correction choices are reconstructed only for current source',()=>{
    const report={proposals:[{id:'e0',before:'Trastuzumap',after:'Trastuzumab',protected:false},
        {id:'e1',before:'5',after:'10',protected:true}]};
    const result=harness('corrections',4,true)(['markup',report,report,[], '',null,null,false,null,stamp('corrections',4)]);
    assert.equal(result[2].choices.length,1); assert.equal(result[2].choices[0][1],'e0');
    assert.equal(result[2].value.length,0);
});

test('review and highlight frames require their current server receipt',()=>{
    const factory = runInNewContext(source,{window:{__oncologySourceOrders:{asr:7,corrections:8}},document:{}});
    const highlight = factory('asr',[0,1,3]);
    const report = {workflow:{kind:'asr',event_order:7}};
    assert.equal(highlight(['markup',report,report,null])[1],report);
    const stale = {workflow:{kind:'asr',event_order:6}};
    assert.equal(skips(highlight(['old',stale,stale,null])),true);
    const review = factory('corrections',[0,1]);
    const reviewed = {workflow:{kind:'corrections',event_order:8}};
    assert.equal(review(['reviewed',reviewed,reviewed])[1],reviewed);
    assert.equal(skips(review(['old',report,report])),true);
});
