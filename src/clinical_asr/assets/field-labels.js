// Associate the visible labels with textareas, including lazily mounted tabs.
const labels = {
    'raw-transcript-field': 'Raw transcript',
    'medical-transcript-field': 'Transcript for medical review',
    'reviewed-transcript-field': 'Reviewed transcript draft',
    'note-transcript-field': 'Transcript for note'
};
const attach = () => {
    for (const [id, label] of Object.entries(labels)) {
        const input = document.querySelector(`#${id} textarea`);
        if (input && input.getAttribute('aria-label') !== label) input.setAttribute('aria-label', label);
    }
};
const observer = new MutationObserver(attach);
observer.observe(document.body, {childList: true, subtree: true});
attach();
window.addEventListener('pagehide', () => observer.disconnect(), {once: true});
