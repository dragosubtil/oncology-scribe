const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const {runInNewContext} = require('node:vm');
const source = readFileSync(resolve(__dirname, '../src/clinical_asr/assets/task-clock.js'), 'utf8');

function harness() {
    let now = 0, watcher, sequence = 0;
    const timers = new Map();
    let clock;
    const element = {
        isConnected: true,
        querySelector: () => clock,
        set innerHTML(value) {
            const data = JSON.parse(value);
            clock = {
                dataset: data.dataset,
                elapsed: {textContent: data.elapsed || ''},
                eta: {textContent: data.eta || ''},
                querySelector(selector) {
                    return selector === '[data-clock-elapsed]' ? this.elapsed : this.eta;
                }
            };
        },
        get innerHTML() {
            return JSON.stringify({dataset: clock.dataset, elapsed: clock.elapsed.textContent,
                                   eta: clock.eta.textContent});
        }
    };
    const snapshot = (generation, status, elapsed = 0, processed = 0, expected = '') => JSON.stringify({
        dataset: {generation: String(generation), status, elapsed: String(elapsed),
                  processed: String(processed), expected: String(expected)}
    });
    element.innerHTML = snapshot(0, 'idle');
    runInNewContext(source, {element, watch: (_, callback) => {watcher = callback;},
        performance: {now: () => now}, window: {addEventListener() {}},
        clearInterval: id => timers.delete(id),
        setInterval: callback => {timers.set(++sequence, callback); return sequence;}});
    return {
        accept(generation, status, elapsed, processed, expected) {
            element.innerHTML = snapshot(generation, status, elapsed, processed, expected);
            watcher();
        },
        advance(seconds) {
            now += seconds * 1000;
            for (const callback of [...timers.values()]) callback();
        },
        get clock() {return clock;},
        get intervals() {return timers.size;}
    };
}

test('processing advances elapsed and estimated remaining time locally', () => {
    const h = harness();
    h.accept(1, 'processing', 2, 0, 20);
    h.advance(5);
    assert.equal(h.clock.elapsed.textContent, 'Elapsed 00:07');
    assert.equal(h.clock.eta.textContent, 'ETA ~00:15');
    h.advance(16);
    assert.equal(h.clock.eta.textContent, 'Taking longer than estimated');
});

test('late queued snapshot does not stop or rewind the running clock', () => {
    const h = harness();
    h.accept(1, 'processing', 2, 0, 20);
    h.advance(5);
    h.accept(1, 'queued', 0);
    assert.equal(h.clock.dataset.status, 'processing');
    h.advance(4);
    assert.equal(h.clock.elapsed.textContent, 'Elapsed 00:11');
    assert.equal(h.clock.eta.textContent, 'ETA ~00:11');
});

test('completion freezes the clock and rejects a late processing snapshot', () => {
    const h = harness();
    h.accept(1, 'processing', 0, 0, 20);
    h.advance(5);
    h.accept(1, 'completed', 5);
    h.accept(1, 'processing', 2, 2, 20);
    assert.equal(h.clock.dataset.status, 'completed');
    assert.equal(h.intervals, 0);
});

test('source reset rejects completion from the previous request', () => {
    const h = harness();
    h.accept(1, 'processing', 0, 0, 20);
    h.accept(2, 'idle');
    h.accept(1, 'completed', 20);
    assert.equal(h.clock.dataset.status, 'idle');
    assert.equal(h.clock.dataset.generation, '2');
    assert.equal(h.intervals, 0);
});
