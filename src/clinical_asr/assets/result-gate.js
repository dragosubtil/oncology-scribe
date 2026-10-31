(scope, indexes, correctionIndex = -1) => frame => {
    const skip = () => indexes.map(() => ({__type__: 'update'}));
    if (!Array.isArray(frame) || indexes.some(index => index >= frame.length)) return skip();
    const terminal = frame.at(-1);
    let kind, order, status = 'completed';
    if (typeof terminal === 'string') {
        const template = document.createElement('template');
        template.innerHTML = terminal;
        const clock = template.content.querySelector('[data-task-clock]');
        if (clock) {
            kind = clock.dataset.taskKind;
            order = Number(clock.dataset.eventOrder);
            status = clock.dataset.status;
        }
    }
    if (!kind) {
        const evidence = frame.find(value => value && typeof value === 'object' && value.workflow);
        kind = evidence?.workflow.kind;
        order = Number(evidence?.workflow.event_order);
    }
    const latest = window.__oncologySourceOrders?.[scope];
    if (kind !== scope || !Number.isSafeInteger(order) || order <= 0 || latest !== order) return skip();
    if (status !== 'completed') {
        const result = skip();
        if (['failed', 'queued', 'processing'].includes(status) && indexes.at(-1) === frame.length - 1) {
            result[result.length - 1] = terminal;
        }
        return result;
    }
    const result = indexes.map(index => frame[index]);
    if (correctionIndex >= 0) {
        const proposals = frame[1]?.proposals;
        if (!Array.isArray(proposals)) return skip();
        result[correctionIndex] = {__type__: 'update', value: [], choices: proposals
            .filter(proposal => !proposal.protected)
            .map(proposal => [`${proposal.id}: ${proposal.before} -> ${proposal.after}`, proposal.id])};
    }
    return result;
}
