// Server updates can arrive out of order. Active clocks also guard the rendered UI.
const taskStates = new Map();
const ariaBeforeLock = new Map();
const inertBeforeLock = new Map();
const paintWorkflowLock = () => {
    const busy = [...taskStates.values()].some(status => ['queued', 'processing'].includes(status));
    if (busy) {
        for (const group of document.querySelectorAll('.workflow-controlled')) {
            if (!inertBeforeLock.has(group)) inertBeforeLock.set(group, group.inert);
            if (!group.inert) group.inert = true;
            if (!ariaBeforeLock.has(group)) ariaBeforeLock.set(group, group.getAttribute('aria-disabled'));
            if (group.getAttribute('aria-disabled') !== 'true') group.setAttribute('aria-disabled', 'true');
        }
    } else {
        for (const [group, previous] of inertBeforeLock) group.inert = previous;
        for (const [group, previous] of ariaBeforeLock) {
            if (previous === null) group.removeAttribute('aria-disabled');
            else group.setAttribute('aria-disabled', previous);
        }
        inertBeforeLock.clear();
        ariaBeforeLock.clear();
    }
};
window.__oncologyWorkflowClock = (key, status) => {
    taskStates.set(key, status);
    paintWorkflowLock();
};
const workflowObserver = new MutationObserver(paintWorkflowLock);
workflowObserver.observe(document.body, {
    subtree: true, childList: true, attributes: true, attributeFilter: ['disabled', 'inert', 'aria-disabled']
});
paintWorkflowLock();
window.addEventListener('pagehide', () => {
    workflowObserver.disconnect();
    taskStates.clear();
}, {once: true});
