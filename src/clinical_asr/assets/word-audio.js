let selected = null;

function payload(words) {
    const first = words[0]?.dataset, last = words[words.length - 1]?.dataset;
    if (!first || !last || !first.sourceSha256 || !first.reviewRevision ||
        words.some(word => word.dataset.sourceSha256 !== first.sourceSha256 ||
            word.dataset.reviewRevision !== first.reviewRevision)) return null;
    return {start: Number(first.wordStart), end: Number(last.wordEnd),
        source_sha256: first.sourceSha256, revision: first.reviewRevision};
}

function selectionChanged() {
    if (!element.isConnected) {
        document.removeEventListener('selectionchange', selectionChanged);
        return;
    }
    const selection = document.getSelection();
    selected = null;
    if (selection && !selection.isCollapsed && selection.rangeCount === 1) {
        const range = selection.getRangeAt(0);
        if (element.contains(range.startContainer) && element.contains(range.endContainer)) {
            const words = Array.from(element.querySelectorAll('.word-audio-cue'))
                .filter(word => range.intersectsNode(word));
            const hasUntimedText = Array.from(element.querySelectorAll('.untimed-audio-word'))
                .some(word => range.intersectsNode(word));
            if (!hasUntimedText) selected = payload(words);
        }
    }
    const play = element.querySelector('.audio-selection-play');
    if (play) play.disabled = !selected;
}

element.addEventListener('mousedown', event => {
    if (event.target.closest('.audio-selection-play')) event.preventDefault();
});

element.addEventListener('click', event => {
    const selectionButton = event.target.closest('.audio-selection-play');
    if (selectionButton && element.contains(selectionButton)) {
        if (selected) trigger('play_word', selected);
        return;
    }
    const button = event.target.closest('.word-audio-cue');
    if (!button || !element.contains(button)) return;
    const selection = document.getSelection();
    if (selection && !selection.isCollapsed) return;
    element.querySelectorAll('.word-audio-cue').forEach(word => {
        word.setAttribute?.('aria-pressed', String(word === button));
    });
    const data = payload([button]);
    if (data) trigger('play_word', data);
});
document.addEventListener('selectionchange', selectionChanged);
