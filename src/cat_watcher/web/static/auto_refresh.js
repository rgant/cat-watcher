// Reload the page when the tab regains focus so new clips, alerts, and heartbeats appear without a
// manual refresh. Pages whose primary content is user-edited (free-text inputs that would be
// clobbered by a reload) opt out via `<body data-no-auto-refresh>`; the focus check below
// additionally protects in-progress edits on pages that stay opted in.
(function () {
  'use strict';

  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState !== 'visible') {
      return;
    }
    if (document.body.hasAttribute('data-no-auto-refresh')) {
      return;
    }
    var active = document.activeElement;
    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.tagName === 'SELECT')) {
      return;
    }
    window.location.reload();
  });
})();
