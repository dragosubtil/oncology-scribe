(scope, force = false) => (tracker, order, ...values) => {
    window.__oncologySourceOrders ||= {};
    window.__oncologySourceInputs ||= {};
    const snapshot = JSON.stringify(values);
    if (force || window.__oncologySourceInputs[scope] !== snapshot || !window.__oncologySourceOrders[scope]) {
        window.__oncologySourceEventOrder = (window.__oncologySourceEventOrder || 0) + 1;
        window.__oncologySourceOrders[scope] = window.__oncologySourceEventOrder;
        window.__oncologySourceInputs[scope] = snapshot;
    }
    return [tracker, window.__oncologySourceOrders[scope], ...values];
}
