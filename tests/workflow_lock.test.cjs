const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');
const {runInNewContext} = require('node:vm');
const source = fs.readFileSync(path.join(__dirname,'../src/clinical_asr/assets/workflow-lock.js'),'utf8');

function harness() {
    let changed;
    const controlled = extra => {
        const attributes=new Map();
        return {disabled:false,inert:false,getAttribute:key => attributes.get(key) ?? null,
            setAttribute:(key,value) => attributes.set(key,value),
            removeAttribute:key => attributes.delete(key),...extra};
    };
    const button = controlled({matches:() => true, querySelectorAll:() => []});
    const field = {disabled:false};
    const group = controlled({matches:() => false, querySelectorAll:() => [field]});
    const groups = [button,group];
    const window = {addEventListener() {}};
    runInNewContext(source,{window,document:{body:{},querySelectorAll:() => groups},
        MutationObserver:class {constructor(fn){changed=fn;} observe(){} disconnect(){}}});
    return {button,field,group,groups,changed:() => changed(),clock:window.__oncologyWorkflowClock};
}

test('busy state makes native root and nested field container inert', () => {
    const h=harness();
    h.clock('asr','queued');
    assert.equal(h.button.inert,true);
    assert.equal(h.button.getAttribute('aria-disabled'),'true');
    assert.equal(h.group.inert,true);
});
test('late server enable is rejected while the task is processing', () => {
    const h=harness();h.clock('asr','processing');
    h.button.disabled=false;h.group.inert=false;h.changed();
    assert.equal(h.button.inert,true);assert.equal(h.group.inert,true);
});
test('one completed task cannot unlock another active task', () => {
    const h=harness();h.clock('asr','processing');h.clock('note','queued');
    h.clock('asr','completed');assert.equal(h.button.inert,true);
    h.clock('note','failed');assert.equal(h.button.inert,false);
});
test('terminal state restores previous disabled and inert values', () => {
    const h=harness();h.field.disabled=true;
    h.clock('asr','processing');h.clock('asr','idle');
    assert.equal(h.button.disabled,false);assert.equal(h.field.disabled,true);
    assert.equal(h.group.inert,false);
});
test('controls mounted during a task also become inert and disabled', () => {
    const h=harness();h.clock('asr','processing');
    const attributes=new Map();
    const newButton={disabled:false,inert:false,matches:() => true,querySelectorAll:() => [],
        getAttribute:key=>attributes.get(key)??null,setAttribute:(key,value)=>attributes.set(key,value),
        removeAttribute:key=>attributes.delete(key)};
    h.groups.push(newButton);h.changed();
    assert.equal(newButton.inert,true);assert.equal(newButton.getAttribute('aria-disabled'),'true');
});

test('server enables after its lock; completion must not restore a stale native disabled value', () => {
    const h=harness();h.button.disabled=true;
    h.clock('medical','queued');
    h.button.disabled=false;h.changed();
    assert.equal(h.button.inert,true);
    h.clock('medical','completed');
    assert.equal(h.button.disabled,false);
    assert.equal(h.button.inert,false);
    assert.equal(h.button.getAttribute('aria-disabled'),null);
});

test('native availability remains owned by server updates while busy', () => {
    const h=harness();h.clock('asr','processing');
    h.button.disabled=true;h.changed();
    h.clock('asr','completed');
    assert.equal(h.button.disabled,true);
    assert.equal(h.button.inert,false);
});
