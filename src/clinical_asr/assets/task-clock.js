// Only status snapshots cross the network; the live countdown runs locally.
let interval = null;
let latestGeneration = -1;
let latestRank = -1;
let latestMarkup = '';
const ranks = {idle: 0, queued: 1, processing: 2, completed: 3, failed: 3};
const duration = seconds => {
    const value = Math.max(0, Math.floor(seconds));
    return String(Math.floor(value / 60)).padStart(2, '0') + ':' + String(value % 60).padStart(2, '0');
};
const update = () => {
    const clock = element.querySelector('[data-task-clock]');
    if (!clock) return;
    const generation = Number(clock.dataset.generation);
    const rank = ranks[clock.dataset.status];
    if (generation < latestGeneration || (generation === latestGeneration && rank < latestRank)) {
        element.innerHTML = latestMarkup;
        update();
        return;
    }
    clearInterval(interval);
    interval = null;
    latestGeneration = generation;
    latestRank = rank;
    latestMarkup = element.innerHTML;
    window.__oncologyWorkflowClock?.(element, clock.dataset.status);
    if (!['queued', 'processing'].includes(clock.dataset.status)) return;
    const received = performance.now();
    const elapsed = Number(clock.dataset.elapsed);
    const processed = Number(clock.dataset.processed);
    const expected = clock.dataset.expected === '' ? null : Number(clock.dataset.expected);
    const paint = () => {
        if (!element.isConnected) {
            clearInterval(interval);
            interval = null;
            return;
        }
        const extra = (performance.now() - received) / 1000;
        clock.dataset.elapsed = String(elapsed + extra);
        clock.dataset.processed = String(processed + extra);
        const elapsedLabel = clock.querySelector('[data-clock-elapsed]');
        if (elapsedLabel) elapsedLabel.textContent = 'Elapsed ' + duration(elapsed + extra);
        const etaLabel = clock.querySelector('[data-clock-eta]');
        if (etaLabel && expected !== null && clock.dataset.status === 'processing') {
            const remaining = expected - processed - extra;
            etaLabel.textContent = remaining > 0 ? 'ETA ~' + duration(remaining) : 'Taking longer than estimated';
        }
        latestMarkup = element.innerHTML;
    };
    paint();
    interval = setInterval(paint, 250);
};
watch('value', update);
update();
window.addEventListener('pagehide', () => clearInterval(interval), {once: true});
