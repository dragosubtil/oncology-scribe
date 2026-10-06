const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const {runInNewContext} = require('node:vm');
const source = readFileSync(resolve(__dirname, '../src/clinical_asr/assets/word-audio.js'), 'utf8');

function harness(inside = true, selection = null, words = [], untimed = []) {
    let click;
    const documentHandlers = {};
    const calls = [];
    runInNewContext(source, {element: {addEventListener: (event, fn) => {if (event === 'click') click = fn;},
        contains: () => inside, isConnected: true,
        querySelectorAll: selector => selector === '.untimed-audio-word' ? untimed : words,
        querySelector: () => null}, document: {
        addEventListener: (event, fn) => {documentHandlers[event] = fn;},
        removeEventListener: () => {}, getSelection: () => selection,
    }, trigger: (...args) => calls.push(args)});
    return {click, calls, documentHandlers};
}
test('uses character identity, not spelling, to select a repeated word', () => {
    const h = harness();
    h.click({target: {closest: selector => selector === '.word-audio-cue' ?
        {dataset: {wordStart: '5', wordEnd: '9', sourceSha256: 'abc', reviewRevision: 'r1'}} : null}});
    assert.equal(h.calls[0][0], 'play_word');
    assert.deepEqual(JSON.parse(JSON.stringify(h.calls[0][1])), {start: 5, end: 9, source_sha256: 'abc', revision: 'r1'});
});
test('ignores a click on unmarked text', () => {
    const h = harness();
    h.click({target: {closest: () => null}});
    assert.equal(h.calls.length, 0);
});
test('does not accept a button outside its review component', () => {
    const h = harness(false);
    h.click({target: {closest: () => ({dataset: {}})}});
    assert.equal(h.calls.length, 0);
});

test('phrase selection sends aligned character boundaries, never client timestamps', () => {
    const words = [
        {dataset: {wordStart: '0', wordEnd: '2', sourceSha256: 'abc', reviewRevision: 'r1'}},
        {dataset: {wordStart: '3', wordEnd: '6', sourceSha256: 'abc', reviewRevision: 'r1'}},
        {dataset: {wordStart: '7', wordEnd: '12', sourceSha256: 'abc', reviewRevision: 'r1'}},
    ];
    const selection = {isCollapsed: false, rangeCount: 1,
        getRangeAt: () => ({startContainer: {}, endContainer: {}, intersectsNode: () => true})};
    const h = harness(true, selection, words);
    assert.equal(typeof h.documentHandlers.selectionchange, 'function');
    h.documentHandlers.selectionchange();
    h.click({target: {closest: selector => selector === '.audio-selection-play' ? {} : null}});
    assert.equal(h.calls.length, 1);
    assert.deepEqual(JSON.parse(JSON.stringify(h.calls[0][1])),
        {start: 0, end: 12, source_sha256: 'abc', revision: 'r1'});
});

test('mixed-revision selection cannot request an excerpt', () => {
    const words = [
        {dataset: {wordStart: '0', wordEnd: '2', sourceSha256: 'abc', reviewRevision: 'r1'}},
        {dataset: {wordStart: '3', wordEnd: '6', sourceSha256: 'abc', reviewRevision: 'r2'}},
    ];
    const selection = {isCollapsed: false, rangeCount: 1,
        getRangeAt: () => ({startContainer: {}, endContainer: {}, intersectsNode: () => true})};
    const h = harness(true, selection, words);
    assert.equal(typeof h.documentHandlers.selectionchange, 'function');
    h.documentHandlers.selectionchange();
    h.click({target: {closest: selector => selector === '.audio-selection-play' ? {} : null}});
    assert.equal(h.calls.length, 0);
});

for (const boundary of ['leading negation', 'trailing qualifier', 'middle word']) {
    test(`selection cannot silently drop an untimed ${boundary}`, () => {
        const words = [{dataset: {wordStart: '3', wordEnd: '6',
            sourceSha256: 'abc', reviewRevision: 'r1'}}];
        const selection = {isCollapsed: false, rangeCount: 1,
            getRangeAt: () => ({startContainer: {}, endContainer: {}, intersectsNode: () => true})};
        const h = harness(true, selection, words, [{}]);
        h.documentHandlers.selectionchange();
        h.click({target: {closest: selector => selector === '.audio-selection-play' ? {} : null}});
        assert.equal(h.calls.length, 0);
    });
}
